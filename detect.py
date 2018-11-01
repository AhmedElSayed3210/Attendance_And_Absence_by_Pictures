import cv2, os
import numpy as np
import sqlite3
from PIL import Image
import pickle

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.read('recognizer/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
eye_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier(cascadePath);

def getperson(id):
    conn=sqlite3.connect('faceBase.db')
    cmd="SELECT * from people WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile 


cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        profile=getperson(Id)
        if(conf<50):
            if(profile!=None):
               cv2.putText(im,profile[1],(x,y+h+30), font, 1,(255,255,255),2,cv2.LINE_AA)    
               cv2.putText(im,str(profile[2]),(x,y+h+60), font, 1,(255,255,255),2,cv2.LINE_AA)    
               cv2.putText(im,profile[3],(x,y+h+90), font, 1,(255,255,255),2,cv2.LINE_AA)
               cv2.putText(im,profile[4],(x,y+h+120), font, 1,(255,255,255),2,cv2.LINE_AA)
        else:
            cv2.putText(im,"UNKNOWN",(x,y+h+30), font, 1,(255,255,255),2,cv2.LINE_AA)
       
       # cv2.cv.PutText(cv2.cv.fromarray(im),str(Id), (x,y+h),font, 255)
    cv2.imshow('im',im) 
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
