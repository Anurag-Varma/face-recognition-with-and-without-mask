import sys, os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

from keras.utils import np_utils


#import matplotlib.pyplot as plt
import cv2

from keras.utils import to_categorical
from keras.layers import MaxPool2D


from sklearn.model_selection import train_test_split



train_images = []       
train_labels = []
shape = (200,200)  
train_path = "C:/Users/panur/facedetection/data/train"

for filename in os.listdir(train_path):
    for images in os.listdir("C:/Users/panur/facedetection/data/train/"+filename):
        
        img = cv2.imread(train_path+"/"+filename+"/"+images)
        
        # Spliting file names and storing the labels for image in list
        train_labels.append(filename)
        
        # Resize all images to a specific shape
        img = cv2.resize(img,shape)
        
        train_images.append(img)


# Converting labels into One Hot encoded sparse matrix
train_labels = pd.get_dummies(train_labels).values

# Converting train_images to array
train_images = np.array(train_images)

# Splitting Training data into train and validation dataset
x_train,x_val,y_train,y_val = train_test_split(train_images,train_labels,random_state=1)




test_images = []
test_labels = []
shape = (200,200)
test_path = "C:/Users/panur/facedetection/data/validation"
for filename in os.listdir(test_path):
    for images in os.listdir("C:/Users/panur/facedetection/data/validation/"+filename):
        
        img = cv2.imread(test_path+"/"+filename+"/"+images)
        
        # Spliting file names and storing the labels for image in list
        test_labels.append(filename)
        
        # Resize all images to a specific shape
        img = cv2.resize(img,shape)
        
        test_images.append(img)
        
# Converting test_images to array
test_images = np.array(test_images)




model = Sequential()
# convolutional layer
model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(200,200,3)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# output layer
model.add(Dense(20, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])






model.summary()






model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))






# Testing predictions and the actual label
checkImage = test_images[0:1]
checklabel = test_labels[0:1]

predict = model.predict(np.array(checkImage))

#output = { 0:'apple',1:'banana',2:'mixed',3:'orange'}

print("Actual :- ",checklabel)
print("Predicted :- ",np.argmax(predict))






from sklearn.metrics import accuracy_score

for i in range(0,555):
    checkImage = test_images[i:i+1]
    checklabel = test_labels[i:i+1]

    predict = model.predict(np.array(checkImage))
    print("Actual :- ",checklabel)
    print("Predicted :- ",np.argmax(predict))
