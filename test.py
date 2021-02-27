import cv2
import numpy as np
from tensorflow import keras



# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = keras.models.load_model('my_model') #loading our model 

# To capture video from webcam. 
cap = cv2.VideoCapture(0)


while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)


    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_img = gray[y:y+w,x:x+w]     # face for our testing section   
        new_arry1 = cv2.resize(face_img,(90,90)) # making image same size like our training 
        pc = np.array(new_arry1).reshape(-1,90,90,1)

        ypred = np.argmax(model.predict(pc))

        if ypred ==0:
            text = 'Jack Ma'
        elif ypred ==1:
            text = 'Elon Musk'
        elif ypred ==2:
            text = 'Bill Gates'
        elif ypred ==3:
            text = 'Tural Karimov'

    

        fontScale = 1
        org = (50, 50)
        color = (255, 0, 0) 
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX 


        cv2.putText(img, text, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 


    

    # Show
    cv2.imshow('img', img)

    
    

    # close
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        

cap.release()