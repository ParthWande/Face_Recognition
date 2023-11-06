import cv2
import face_recognition
import numpy as np
import os

#path of images stored
path = 'Images'
images = []
personNames = []

#grab the list of images in the folder
myList = os.listdir(path)
print(myList)

#copy all images from path in image list
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
print(personNames)

#encode each image function
def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#encodings of image present with us
encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

#start the camera
cap = cv2.VideoCapture(0)
cap.set(3,2000)#width
cap.set(4,2000)#height

while True:
    #image from camera
    success,frame = cap.read()
    #reduce size of image
    faces = cv2.resize(frame,(0,0),None,0.25,0.25)
    #convert to rgb
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    #encodings of the web cam
    #locations of all faces
    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    #checking
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        #minimum value from list of faceDis(lowest element from image list)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 0,255), 2)
            
            cv2.putText(frame, name, (x2 + 1, y1+2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

#close the camera

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()

