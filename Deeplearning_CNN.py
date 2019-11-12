# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:53:27 2019

@author: OWNER
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.layers.convolutional import Conv2D
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

def newdatalabels(filename):
    raw_data = mne.io.read_raw_eeglab(filename,preload=True)
    mne.set_log_level("WARNING")
    raw_data.resample(128,npad='auto') #resampling
    #band-pass filtering in the range 4 Hz - 38 Hz
    raw_data.filter(4, 38, method ="iir")   
#    data=raw_data.get_data()
#    chs = raw_data.pick_channels(['FC6','FCZ','FC5','FC3','FC4','CP3','CP4','C5','C3','C1','CZ','C2','C4','C6'])
    
    chs = raw_data.pick_channels(['FCZ','FC5','FC1','FC2','FC6','FC3','FC4','C5',
                          'C3','C1','CZ','C2','C4','C6','CP3','CP1','CP2','CP4'])
    data = chs.get_data()
#    data = data[122:130,:]

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
            st = events_from_annot[i,0]-256 #start time
            et = events_from_annot[i,0]+768 #end time
            labels.append(events_from_annot[i,2])
            #newsd.append(sd[:,st:et])            
            tmp = sd[:,st:et]
            if tmp.shape[1] != 0:
                newsd.append(tmp)
    
    newsd = np.stack([arr for arr in newsd], axis = 0)
    labels = np.array(labels)
    return(newsd,labels)

data_folder ="C:\\Users\OWNER\Desktop\\allmotordata"
filenames = glob.glob(os.path.join(data_folder, '*REST.set')) #filenames list
#print all filenames
for filename in filenames:
    filename = filename.split("\\")
    print("filename: " +filename[-1])  

allnewsds = np.empty((0,18,1024),float)
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

#CNN model denifition
#CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size. 
input_shape = (18, 1024, 1) #chn = 18, trial_length = 1024 

model = models.Sequential() 
#conv_pool block1     
model.add(layers.Conv2D(25, (1, 10), activation="relu", input_shape=(18, 1024, 1),
                        kernel_constraint = max_norm(2., axis=(0,1,2)))) #kernel_size(height,width) 
model.add(layers.Conv2D(25, (18,1), activation="relu", kernel_constraint = max_norm(2., axis=(0,1,2))))
model.add(layers.BatchNormalization(axis=3,epsilon=1e-05, momentum=0.1))
model.add(layers.MaxPooling2D((1, 3),strides = (1,3)))
model.add(layers.Dropout(0.5))

#conv_pool block2
model.add(layers.Conv2D(50, (1, 10), activation='relu',kernel_constraint = max_norm(2., axis=(0,1,2))))
model.add(layers.BatchNormalization(axis=3,epsilon=1e-05, momentum=0.1))
model.add(layers.MaxPooling2D((1, 3),strides = (1,3)))
model.add(layers.Dropout(0.5))
##conv_pool block3
model.add(layers.Conv2D(100, (1, 10), activation='relu',kernel_constraint = max_norm(2., axis=(0,1,2))))
model.add(layers.BatchNormalization(axis=3,epsilon=1e-05, momentum=0.1))
model.add(layers.MaxPooling2D((1, 3),strides = (1,3)))
model.add(layers.Dropout(0.5))
##conv_pool block4
model.add(layers.Conv2D(200, (1, 10), activation='relu',kernel_constraint = max_norm(2., axis=(0,1,2))))
model.add(layers.BatchNormalization(axis=3,epsilon=1e-05, momentum=0.1))
model.add(layers.MaxPooling2D((1, 3),strides = (1,3)))
model.add(layers.Dropout(0.5))

model.summary()
#classification layer
model.add(layers.Flatten())
#model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.summary()

#Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #not sure loss function
history = model.fit(allnewsds_train, alllabels_train, epochs=50,
                    validation_data=(allnewsds_test, alllabels_test))
#Evaluate the model
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

labels = alllabels_test
predictions = model.predict(allnewsds_test)
predictions = tf.argmax(predictions,1)
print(tf.math.confusion_matrix(labels,predictions,num_classes=4))
