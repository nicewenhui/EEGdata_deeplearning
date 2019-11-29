# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:13:45 2019

@author: OWNER
"""

#this script used raw data for motor classification(left or right hand)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
#from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
#from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

def newdatalabels(filename):
    raw_data = mne.io.read_raw_eeglab(filename,preload=True)
#    mne.set_log_level("WARNING")
    raw_data.resample(128,npad='auto') #resampling
    #band-pass filtering in the range 4 Hz - 38 Hz
    raw_data.filter(8, 30, method ="iir")   
    data=raw_data.get_data()
#    chs = raw_data.pick_channels(['FC6','FCZ','FC5','FC3','FC4','CP3','CP4','C5','C3','C1','CZ','C2','C4','C6'])
    
#    chs = raw_data.pick_channels(['FCZ','FC5','FC1','FC2','FC6','FC3','FC4','C5',
#                          'C3','C1','CZ','C2','C4','C6','CP3','CP1','CP2','CP4'])
#    data = chs.get_data()
    data = data[122:130,:]

    #electrode-wise exponential moving standarlization
    m = np.mean(data[:,0:1000], axis = 1) #m is the first 1000 datapoints mean values
    v = np.var(data[:,0:1000],axis = 1) #v is the first 1000 datapoints variance values
    sd = np.zeros((data.shape[0],data.shape[1])) #standarlized data
    
    for i in range(1000,data.shape[1]):
        m = 0.001*data[:,i] + 0.999*m
        v = 0.001*((data[:,i]-m)**2) + 0.999*v
        sd[:,i] = (data[:,i]-m)/np.sqrt(v)
    
    events_from_annot, event_dict = mne.events_from_annotations(raw_data)
    #new standarlized data with labels
    newsd = []
    labels = []
    print(filename)
    for i in  range(len(events_from_annot)):
        if events_from_annot[i,2]==1 or events_from_annot[i,2]==2:
            st = events_from_annot[i,0]-128 #start time -1s
            et = events_from_annot[i,0]+256 #end time 2s
            labels.append(events_from_annot[i,2])
            #newsd.append(sd[:,st:et])            
            tmp = sd[:,st:et]
            if tmp.shape[1] != 0:
                newsd.append(tmp)
    
    newsd = np.stack([arr for arr in newsd], axis = 0)
    labels = np.array(labels)
    return(newsd,labels)

def EEGNet(nb_classes, Chans = 8, Samples = 384, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
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
#    block1       = BatchNormalization(axis = 3)(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
#    block1       = BatchNormalization(axis = 3)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
#    block2       = BatchNormalization(axis = 3)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

data_folder ="D:\\allmotordata"
filenames = glob.glob(os.path.join(data_folder, '*REST.set')) #filenames list
#print all filenames
for filename in filenames:
    filename = filename.split("\\")
    print("filename: " +filename[-1])  

allnewsds = np.empty((0,8,384),float)
alllabels = np.array([])

for file in filenames:
    (newsd,labels) = newdatalabels(file)
    
    allnewsds = np.concatenate((allnewsds, newsd), axis=0)
    alllabels = np.concatenate((alllabels, labels), axis=0)
   
allnewsds_train, allnewsds_test,  alllabels_train, alllabels_test = train_test_split(
            allnewsds, alllabels, test_size=0.2, random_state=42,stratify=alllabels) 

allnewsds_train = np.expand_dims(allnewsds_train,axis=3) 
allnewsds_test = np.expand_dims(allnewsds_test,axis=3) 
 #replace labels values
palette = [1,2]
key = np.array([0,1])
index = np.digitize(alllabels_train.ravel(), palette, right=True)
alllabels_train = key[index]
palette = [1,2]
key = np.array([0,1])
index = np.digitize(alllabels_test.ravel(), palette, right=True)
alllabels_test = key[index]

print(np.shape(allnewsds_train),np.shape(allnewsds_test),np.shape(alllabels_train),np.shape(alllabels_test))


model = EEGNet(nb_classes=2, Chans = 8, Samples = 384, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout')
model.summary()

# checkpoint
filepath="D:\\allmotordata\\weights\\motorallsubjects_testweightsbest.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(allnewsds_train, alllabels_train, epochs=20,
                    validation_data=(allnewsds_test, alllabels_test),callbacks=callbacks_list)

model.load_weights(filepath)
test_loss, test_acc = model.evaluate(allnewsds_test, alllabels_test, verbose=2)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
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
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print(test_acc)

#labels = alllabels_test
#predictions = model.predict(allnewsds_test)
#predictions = tf.argmax(predictions,1)
#print(tf.math.confusion_matrix(labels,predictions,num_classes=2))

model.load_weights(filepath)

y_test_pred=tf.argmax(model.predict(allnewsds_test),1)
con_mat = tf.math.confusion_matrix(labels=alllabels_test, predictions=y_test_pred).numpy()
import pandas as pd
import seaborn as sns
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
 
con_mat_df = pd.DataFrame(con_mat_norm,
                     index = [0,1], 
                          
                     columns = [0,1])

figure = plt.figure(figsize=(6, 6))

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