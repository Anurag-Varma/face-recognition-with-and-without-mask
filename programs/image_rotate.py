import cv2


mypath="C:/Users/panur/facedetection/dataset/withoutmask_1602-18-737-054"

ret=1
flag=True
i=0

import os

arr=os.listdir(mypath)


for x in range(0,len(arr)):

 img = cv2.imread("C:/Users/panur/facedetection/dataset/withoutmask_1602-18-737-054/"+"image"+str(i)+".jpg")
 
 img_rotate_90_counterclockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
 
 cv2.imwrite("C:/Users/panur/facedetection/dataset/withoutmask_1602-18-737-054/"+"image"+str(i)+".jpg", img_rotate_90_counterclockwise)

 print(i)
 i=i+1

 

