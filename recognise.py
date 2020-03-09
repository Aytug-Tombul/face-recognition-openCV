import numpy as np
import cv2
import pickle
import json
import os
import random


exec(open('face-training.py').read())
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("data_file.json", "r") as read_file:
    data = json.load(read_file)
    temp = data['persons']
labels = {"person_name": 1 }
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
cap = cv2.VideoCapture(0)

id_=""

while(True):
    ret , frame= cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        roi_gray= gray[y:y+h,x:x+w]
        roi_color = frame[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi_gray)
        if conf>=75:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            nfav = labels[id_]
            color=(255, 255, 255)
            stroke = 2
            cv2.putText(frame,nfav,(x,y) , font , 1 ,color,stroke,cv2.LINE_AA)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, "Unknown", (x, y), font, 1, color, stroke, cv2.LINE_AA)
            id_="Unknown"
        color = (255, 200, 0)
        stroke=2
        width = x + w
        height = y+h
        cv2.rectangle(frame, (x,y),(width , height) ,color,stroke)
    cv2.imshow('frame',frame)

    if (id_ == "Unknown"):
        name = input("Welcome i cant remember You Whats your name :")
        #crop_img = frame[y: y + h, x: x + w]  # Crop from x, y, w, h -> 100, 200, 300, 400
        #cv2.imwrite("images/" + name + ".jpg", crop_img)
        fav=input("What is your drink: ");
        print("Nice to meet you "+name+" I served your "+fav+" Enjoy your Drink :)")
        if not os.path.exists('images/'+name):
            os.makedirs('images/'+name)
        crop_img = frame[y: y + h, x: x + w]
        cv2.imwrite("images/" + name +"/"+name+str(random.randint(1,100))+".jpg", crop_img)
        with open("data_file.json", "r") as read_file:
            data = json.load(read_file)
            x = {
                "name": name,
                "fav": fav
            }
            data['persons'].append(x)
            #print(data)
        with open("data_file.json", 'w') as f:
            json.dump(data, f, indent=4)
        break
    else:
        #print(temp)
        for per in temp:
            print(per)
            #print(per['name'])
            if per['name'] == id_:
                ans = input("Welcome your Favourite Drink " + per['fav'] + " Would you like another drink (Y/N) : ")
                if (ans == "Y"):
                    print("I Served your Drink Enjoy Your Drink :)")
                else:
                    newFav = input("What is your order ? ")
                    per['fav'] = newFav
                    print(temp)
                    data['persons'] = temp
                    with open("data_file.json", 'w') as f:
                        json.dump(data, f, indent=4)
    #break
#cap.release()
#cv2.destroyAllWindows()