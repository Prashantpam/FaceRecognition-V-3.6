import cv2
import os

def normalize_faces(frame,faces_coord):
    face=cut_photo(frame,faces_coord)
    face=normalize(face)
    face=resize(face)
    return face

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

def draw_rectangle(image,coord):
    for (x,y,w,h) in coord:
        w_rm=int(0.2*w/2)
        cv2.rectangle(image,(x+w_rm,y),(x+w-w_rm,y+h),(150,150,0),8)

def cut_photo(frame,coord):
    pict=[]
    for (x,y,w,h) in coord:
        rm=int(0.2*w/2)
        pict.append(frame[y:y+h,x+rm:x+w-rm])
    return pict

webcam=cv2.VideoCapture(0)
cv2.namedWindow("Pam Developers",cv2.WINDOW_NORMAL)
biggest_only=True
detector=cv2.CascadeClassifier("C:/Users/Prashant/Downloads/haarcascade_frontalface_alt.xml")
scale_factor=1.2
min_size=(30,30)
min_neighbors=5
flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | \
       cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
       cv2.CASCADE_SCALE_IMAGE
folder="C:/Users/Prashant/Desktop/people/"+input('Person: ').lower()
if not os.path.exists(folder):
    os.makedirs(folder)
    counter=0
    timer=0
    while counter<10:
        data,frame=webcam.read()
        faces_coord=detector.detectMultiScale(frame,scaleFactor=scale_factor,
                                              minNeighbors=min_neighbors,
                                              minSize=min_size,
                                              flags=flags)
        
        if len(faces_coord):
            print("enter")
            faces=normalize_faces(frame,faces_coord)
            cv2.imwrite(folder+"/"+str(counter)+".jpg",faces[0])
            cv2.imshow("Pam Developers",faces[0])
            
        draw_rectangle(frame,faces_coord)
        cv2.imshow("Pam Developers",frame)
        cv2.waitKey(50)
        counter+=1
    cv2.destroyAllWindows()
else:
    print("Already exist")



    
        
