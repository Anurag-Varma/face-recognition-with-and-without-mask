# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 22:35:27 2020

@author: panur
"""

import cv2
import os


cap = cv2.VideoCapture(r"C:\Users\panur\Downloads\withoutmask_1602-18-737-006.mp4")

ret=1
flag=True
i=0
while(ret!=0 and flag):

 ret, fm=cap.read()
 cv2.imwrite("C:/Users/panur/facedetection/dataset/withoutmask_1602-18-737-006/"+"image"+str(i)+".jpg", fm)
 print(i)
 i=i+1

 
 