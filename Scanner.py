import cv2
from IPython.display import clear_output
import numpy as np
import pandas as pd
import os
class Detect:
    def __init__(self,xml_path):
        self.classification=cv2.CascadeClassifier(xml_path)

    def detec(self,image,biggest_only=True):
        scale_factor=1.4
        min_size=(30,30)
        min_neighbors=5
        flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | \
               cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
               cv2.CASCADE_SCALE_IMAGE
        faces_coord=self.classification.detectMultiScale(image,
                                                    scaleFactor=scale_factor,
                                                    minNeighbors=min_neighbors,
                                                    minSize=min_size,
                                                    flags=flags)
        return faces_coord                          

class VideoCamera:
    def __init__(self,index=0):
        self.webcam=cv2.VideoCapture(index)
        self.index=index
        print(self.webcam.isOpened())

    def camera(self):
        data,frame=self.webcam.read()
        return frame
    def det(self):
        self.webcam.destroyAllWindows()
    def __del__(self):
        self.webcam.release()

def cut_photo(frame,coord):
    pict=[]
    for (x,y,w,h) in coord:
        rm=int(0.2*w/2)
        pict.append(frame[y:y+h,x+rm:x+w-rm])
    return pict

def normalize(images):
    ima=[]
    for image in images:
        is_color=len(image.shape)==3
        if is_color:
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ima.append(cv2.equalizeHist(image))
    return ima

def resize(images,size=(50,50)):
    images_norm=[]
    for image in images:
        if image.shape<size:
            image_norm=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
        else:
            image_norm=cv2.resize(image,size,interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)
    return images_norm

def normalize_faces(frame,faces_coord):
    face=cut_photo(frame,faces_coord)
    face=normalize(face)
    face=resize(face)
    return face

def draw_rectangle(image,coords):
    for (x,y,w,h) in coords:
        w_rm=int(0.2*w/2)
        cv2.rectangle(image,(x+w_rm,y),(x+w-w_rm,y+h),(150,150,0),8)

def collect_dataset():
    images=[]
    labels=[]
    labels_dic={}
    person=[person for person in os.listdir("C:/Users/Prashant/Desktop/people/")]
    for i,people in enumerate(person):
        labels_dic[i]=people
        count=int(0)
        for image in os.listdir("C:/Users/Prashant/Desktop/people/"+people):
            if count<10:
                  images.append(cv2.imread("C:/Users/Prashant/Desktop/people/"+people+"/"+image,0))
                  labels.append(i)
                  count=count+1
    return (images,np.array(labels),labels_dic)
cam=VideoCamera()
det=Detect("C:/Users/Prashant/Downloads/haarcascade_frontalface_default.xml")

webcam=cv2.VideoCapture(0)
cv2.namedWindow("Pam Developers",cv2.WINDOW_NORMAL)
data,frame=webcam.read()
detector=Detect("C:/Users/Prashant/Downloads/haarcascade_frontalface_default.xml")
cv2.namedWindow("Pam Developers",cv2.WINDOW_AUTOSIZE)
eigen=cv2.face.LBPHFaceRecognizer_create()
images,labels,labels_dic=collect_dataset()

eigen.train(images,labels)
while True:
    da,frame=webcam.read()
    faces_coord=detector.detec(frame,True)
    if len(faces_coord):
        faces=normalize_faces(frame,faces_coord)
        for i,face in enumerate(faces):
            trp=eigen.predict(face)
            id=trp[0]
            threshold=140
            
            clear_output(wait=True)
            if(trp[1]<threshold):
                cv2.putText(frame,labels_dic[id].capitalize(),(faces_coord[i][0],faces_coord[i][1]-10),cv2.FONT_HERSHEY_PLAIN,3,(66,53,243),2,cv2.LINE_AA)
            else:
                cv2.putText(frame,"Unknown",(faces_coord[i][0],faces_coord[i][1]-10),cv2.FONT_HERSHEY_PLAIN,3,(66,53,243),2,cv2.LINE_AA)
        draw_rectangle(frame,faces_coord)
    cv2.imshow("Pam Developers",frame)
    if cv2.waitKey(40)&0xFF==27:
        break

