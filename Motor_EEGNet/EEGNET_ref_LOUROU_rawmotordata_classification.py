# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:22:05 2019

@author: OWNER
"""
#LOU,ROU RE_REFERENCE
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D,MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt
import numpy as np
import mne
import os
#import glob
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

def get_data_from(file):   
    #band-pass filtering in the range 4 Hz - 38 Hz
    fs = 250
    left_chans = ['LF', 'LB', 'LOD', 'LOU']
    right_chans = ['RF', 'RB', 'ROD', 'ROU']
    data_ear = [] #for left/right ear

    raw = mne.io.read_raw_curry(file,preload = True, verbose = False) 
    raw.resample(fs, npad="auto")  
#     raw.filter(4, 30, method ="iir") 

    #events (time, duration, description)
    events = mne.events_from_annotations(raw, verbose = False)[0]

    print(events[0,0])


    for set_chan in [left_chans, right_chans]:  
        new_raw = raw.copy()
        new_raw.filter(4, 38, method ="iir") 
        new_raw.pick_channels(set_chan)
        data = new_raw.get_data()
        data = data - data[-1,:]
        data = data[0:-1,:]    
        
        m = np.mean(data[:,0:1000], axis = 1) #m0 is the first 1000 datapoints mean values
        v = np.var(data[:,0:1000],axis = 1) #v0 is the first 1000 datapoints variance values
        normalize_data = np.zeros((data.shape[0],data.shape[1]))
      
        for i in range(1000,data.shape[1]):
            m = 0.001*data[:,i] + 0.999*m
            v = 0.001*((data[:,i]-m)**2) + 0.999*v
            normalize_data[:,i] = (data[:,i]-m)/np.sqrt(v)
        
        data_ear.append(normalize_data)

    data_ear = np.vstack([arr for arr in data_ear])
    
    trial_num = len(events)
    data_trial = []
    labels = []
    desc = raw.annotations.description
    for i in range(trial_num):
        if desc[i] == '1':
            labels.append(0)
            events_start = events[i,0]-250
            events_end = events[i,0]+500
            data_trial.append(data_ear[:,events_start:events_end]) 
            
        if desc[i] == '2':
            labels.append(1)
            events_start = events[i,0]-250
            events_end = events[i,0]+500
            data_trial.append(data_ear[:,events_start:events_end]) 
            
            
    data_trial_arr = np.stack([arr for arr in data_trial], axis = 0)
    labels = np.stack([num for num in labels], axis = 0)
    return data_trial_arr, labels

def EEGNet(nb_classes, Chans = 6, Samples = 750, 
             dropoutRate = 0.5, kernLength = 64, F1 = 25, 
             D = 2, F2 = 50, norm_rate = 0.25, dropoutType = 'Dropout'):
#    Inputs:
#        
#      nb_classes      : int, number of classes to classify
#      Chans, Samples  : number of channels and time points in the EEG data
#      dropoutRate     : dropout fraction
#      kernLength      : length of temporal convolution in first layer. We found
#                        that setting this to be half the sampling rate worked
#                        well in practice. For the SMR dataset in particular
#                        since the data was high-passed at 4Hz we used a kernel
#                        length of 32.     
#      F1, F2          : number of temporal filters (F1) and number of pointwise
#                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
#      D               : number of spatial filters to learn within each temporal
#                        convolution. Default: D = 2
#      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = ( Chans, Samples,1))
    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples,1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization(axis = 3)(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = 3)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization(axis = 3)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

folder = 'D://motorraw//'
subfolders = os.listdir(folder)

all_data = []
all_labels = []
all_subjects = []
subject_idx = 0

for subfolder in subfolders:
    print(subfolder)
    files = os.listdir(folder+ subfolder)
    for dat_file in files:        
        if '.dat' in dat_file:
            print(dat_file)
            data_trial_arr, labels = get_data_from(folder+ subfolder+'//'+dat_file)
            print(np.shape(data_trial_arr))
            all_data.append(data_trial_arr)
            all_labels.append(labels)
            print(labels[0:10])
            all_subjects.append(subject_idx * np.ones((len(labels,))))
            
    subject_idx += 1  
    
all_data = np.concatenate([arr for arr in all_data], axis = 0)
all_data = np.array(all_data, np.float32)

all_labels = np.concatenate([arr for arr in all_labels], axis = 0)
all_labels = np.array(all_labels, np.float32)

all_subjects = np.concatenate([arr for arr in all_subjects], axis = 0)
all_subjects = np.array(all_subjects, np.float32)   


print('the shape of all-data')
print(np.shape(all_data))
print('the shape of labels')
print(np.shape(all_labels))


X_train, X_test,  y_train, y_test = train_test_split(
            all_data, all_labels, test_size=0.2, random_state=42,stratify=all_labels) 

X_train = np.expand_dims(X_train,axis=3) 
X_test = np.expand_dims(X_test,axis=3)

model = EEGNet(nb_classes=2, Chans = 6, Samples = 750, 
             dropoutRate = 0.5, kernLength = 64, F1 = 25, 
             D = 2, F2 = 50, norm_rate = 0.25, dropoutType = 'Dropout')
model.summary()

# checkpoint
filepath="D:/motorraw/weights/20191119-deepconv-motor-ear-8chan-raw-2class-weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]



#Compile and train the model
model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.005),
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['acc'])

history = model.fit(X_train, y_train, epochs=300,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks_list)
model.load_weights(filepath)
#Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(test_loss, test_acc )
#test_loss, test_acc = model.evaluate(re_train_data, re_train_labels, verbose=2)


plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label ='val_loss' )
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

model.load_weights(filepath)

y_test_pred=tf.argmax(model.predict(X_test),1)
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_test_pred).numpy()

#y_test_pred=model.predict_classes(X_test)
#con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_test_pred).numpy()
import pandas as pd
import seaborn as sns
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
 
con_mat_df = pd.DataFrame(con_mat_norm,
                     index = [0,1], 
                          
                     columns = [0,1])

figure = plt.figure(figsize=(12, 12))

#bug in matplotlib 3.1.1 version
ax = sns.heatmap(con_mat_df, annot=True, square = True, cmap=plt.cm.Reds,annot_kws={"size": 20}, vmin = 0, vmax = 1)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.axes.set_title("Confusion Matrix",fontsize=12)
ax.set_xlabel("Predicted N-back",fontsize=12)
ax.set_ylabel("True N-back",fontsize=12)
#plt.tight_layout()
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()