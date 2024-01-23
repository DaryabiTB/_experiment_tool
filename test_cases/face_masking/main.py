#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Face Detectors
import cv2
from mtcnn import MTCNN

import torch
import torchvision.ops.boxes as bop

import json
import os
import matplotlib.pyplot as plt
import random
import seaborn as sns
from keras.models import Sequential
from keras import optimizers
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
annotations = "./annotations"
images = "./images"
df = pd.read_csv("train.csv")
df_test = pd.read_csv("submission.csv")


# sample data from training set.

# In[40]:


from helper import getJSON


files = []
for i in os.listdir(annotations):
    files.append(getJSON(os.path.join(annotations,i)))
files[0]


# In[41]:


df.head()


# use mask label and non_mask label to extract bounding box data from json files.
# 
# store faces from any particular image in the train list along with its label for the training process.

# In[42]:


train = []
img_size = 124
mask = ['face_with_mask']
non_mask = ["face_no_mask"]
labels={'mask':0,'without mask':1}
for i in df["name"].unique():
    f = i+".json"
    for j in getJSON(os.path.join(annotations,f)).get("Annotations"):
        if j["classname"] in mask:
            x,y,w,h = j["BoundingBox"]
            img = cv2.imread(os.path.join(images,i),1)
            img = img[y:h,x:w]
            img = cv2.resize(img,(img_size,img_size))
            train.append([img,labels["mask"]])
        if j["classname"] in non_mask:
            x,y,w,h = j["BoundingBox"]
            img = cv2.imread(os.path.join(images,i),1)
            img = img[y:h,x:w]
            img = cv2.resize(img,(img_size,img_size))    
            train.append([img,labels["without mask"]])
random.shuffle(train)  
len(train)


# In[43]:


train[:]


# prepare test data and split data and labels.

# In[44]:


X = []
Y = []
for features,label in train:
    X.append(features)
    Y.append(label)
    
print(X[0].shape)
print(Y[0])
print(Y[1])


# In[45]:


X = np.array(X)/255.0
X = X.reshape(-1,124,124,3)

Y = np.array(Y)
print(np.unique(Y))
Y.shape


# Model training
# 

# In[46]:


model = Sequential()

model.add(Conv2D(32, (3, 3), padding = "same", activation='relu', input_shape=(124,124,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[47]:


model.summary()


# compile the model:

# In[48]:


model.compile(loss='binary_crossentropy', optimizer='adam' ,metrics=['accuracy'])


# split training data set to traqin and validation:

# In[49]:


xtrain,xval,ytrain,yval=train_test_split(X, Y,train_size=0.8,random_state=0)

print(type(X))
print(type(Y))


# In[50]:


print(xtrain.shape)
print(ytrain.shape)


# generate tensor make data using ImageDataGenerator:

# In[51]:


tensordata = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,    
        rotation_range=15,    
        width_shift_range=0.1,
        height_shift_range=0.1,  
        horizontal_flip=True,  
        vertical_flip=False)
tensordata.fit(xtrain)


# In[1]:


# from codegreen.decorators import init_experiment, time_shift, upload_cc_report
# from codecarbon import track_emissions
# from codegreen.queries import get_location_prediction, get_data
import numpy as np
from datetime import datetime


#  fit our model:

# In[52]:


# @init_experiment(estimated_runtime_hours=1,estimated_runtime_minutes=30,percent_renewable=10,allowed_delay_hours=24,area_code="ES-9",log_request=True,experiment_name="my_experiment",codecarbon_logfile="experiment.log",nextflow_logfile="nextflow.log",overwrite=False)
# @time_shift("my_experiment")
# @upload_cc_report("my_experiment")
# @track_emissions(output_file='experiment.log')

history = model.fit(
    tensordata.flow(xtrain, ytrain, batch_size=32),
    steps_per_epoch=len(xtrain) // 32,
    epochs=50,
    verbose=1,
    validation_data=(xval, yval)
)


# In[ ]:




