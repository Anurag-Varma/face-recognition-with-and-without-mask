import cv2,os

rollno="1602-18-737-107"

mypath="C:/Users/panur/facedetection/dataset/withmask_"+rollno


mypath1="C:/Users/panur/facedetection/dataset/withoutmask_"+rollno


i=0
arr=os.listdir(mypath)


for x in range(0,len(arr)):

 img = cv2.imread(mypath+"/withmask"+rollno+"_"+str(i)+".jpg")
  
 cv2.imwrite(mypath+"/withmask"+"_"+rollno+"_"+str(i)+".jpg", img)
 
 os.remove(mypath+"/withmask"+rollno+"_"+str(i)+".jpg")

 print(i)
 i=i+1




arr=os.listdir(mypath1)
i=0 
for x in range(0,len(arr)):

 img = cv2.imread(mypath1+"/withoutmask"+rollno+"_"+str(i)+".jpg")
  
 cv2.imwrite(mypath1+"/withoutmask"+"_"+rollno+"_"+str(i)+".jpg", img)
 
 os.remove(mypath1+"/withoutmask"+rollno+"_"+str(i)+".jpg")

 print(i)
 i=i+1

