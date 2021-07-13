# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:01:17 2020

@author: panur
"""


import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')


detection_model_path="C:/Users/panur/facedetection/haarcascade_frontalface_default.xml"
face_detection = cv2.CascadeClassifier(detection_model_path)
video="C:/Users/panur/Downloads/withmask_1602-18-737-006.mp4"

ret=1
flag=True
cap = cv2.VideoCapture(1)
#frameRate = cap.get(30)

while(ret!=0 and cap.isOpened()):

 ret, fm=cap.read()
 cv2.imwrite('live_test_img.jpg', fm)
 fm = cv2.resize(fm, (200, 200))
 file = cv2.cvtColor(fm, cv2.COLOR_BGR2RGB)
 
 orig_frame = file
 frame = file
 faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
 
 i=0
 test=""
 if (len(faces)) :

    faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (200, 200),3)
    roi = frame.astype("float") / 255.0
    
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    
    preds=model.predict_classes(roi)[0]
    print(preds)
    if preds==0:
      print("withmask_anurag"+str(i))
      test='withmask_anurag'
    elif preds==10:
      print("withoutmask_anurag"+str(i))
      test='withoutmask_anurag'
    i=i+1  
    cv2.putText(fm,test, (fX-15, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(fm, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
 file=fm

 
   
 cv2.imshow("Live Video", fm)

 k=cv2.waitKey(25)
 if k == 27: 
    ret=0        
    break

print("closed")
cap.release()   
cv2.destroyAllWindows()