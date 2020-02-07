"""
Created on Sun May  5 20:33:18 2019


@author: RAJITH
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:49:56 2019

@author: RAJITH
"""

import numpy as np
import cv2
import os
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

id_name =input('Enter User Name :')
strt=os.chdir("C:\Spyder\Improvisation/Users")
for i in os.listdir(strt):
    n,id_num,str_num,ext=i.split('.')
    if id_name==n:
        print("This user name already exists")
        id_name =input('Enter New User Name :')     
    else:
        break;
s = 0;
v=[];
l=[]
id =0;
j=[]
k=0
p=os.chdir("C:\Spyder\Improvisation/Users")
for i in os.listdir(p):
    t=int(os.path.split(i)[-1].split('.')[1])
    t+=1;
    v.append(t)
if v==[]:
    id =1;
else:
    id = max(v);


for i in range(id):
    k+=1
    j.append(k)  
     
print(j)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        s+=1; 
        cv2.imwrite(str(id_name)+"."+str(id)+"."+str(s)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0), 2)
        cv2.waitKey(100)
        
    cv2.imshow('img',frame)
    cv2.waitKey(1)
    if (s>9):
        break
X= []


if X==[]:
    for i in os.listdir(p):
        n,id_num,str_num,ext=i.split('.')
        if id_num==str_num:
            X.append(n +'.'+ id_num)




print("x=:",X)   
   
     
cap.release()
cv2.destroyAllWindows

###############################################################
#STEP:2

rec = cv2.face.LBPHFaceRecognizer_create()
p=os.chdir("C:\Spyder\Improvisation")
path = 'Users'

l=[]
def getid(path):
    imgpath = [os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for i in imgpath:
        faceimg=Image.open(i).convert('L');
        facenp= np.array(faceimg,'uint8')
        ID = int(os.path.split(i)[-1].split('.')[1])
        faces.append(facenp)
        ids.append(ID)
        cv2.imshow("training",facenp)
        cv2.waitKey(10)
    return ids,faces

ids,faces = getid(path)
rec.train(faces,np.array(ids))
os.chdir("C:\Spyder\Improvisation")
rec.save('recognizer/trainingData.xml')
cv2.destroyAllWindows()
#####################################################################
#STEP:3

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainingData.xml")
id=0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)
i=0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h)in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,0,255), 2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        for i in range(0,len(X)):
            print("X=:",X[i])
            n,id_num=(X[i]).split('.')
            print("n:",n)
            print("id:",id)
            print("id_num:",id_num)
        
            if (str(id) == id_num):
                cv2.putText(frame,str(n),(x,y+h),fontface, fontscale, fontcolor)
    cv2.imshow('img',frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
    
        break;
cap.release()
cv2.destroyAllWindows
