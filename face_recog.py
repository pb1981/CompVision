import cv2
import numpy as np
import pickle


haar_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained.yml")

labels = {"name": 1}
with open("labels.pickle",'rb') as f:
    orig_labels = pickle.load(f)
    labels = {v:k for k,v in orig_labels.items()}

vidcap = cv2.VideoCapture(0)

while (True):
        ret,frame = vidcap.read()
        grayconv = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(grayconv, scaleFactor = 2.5, minNeighbors = 3)

        for (x, y, w, h) in faces:
            #print(x,y,w,h)
            gray_roi = grayconv[y:y + h,x:x + w]
      
            color = (255, 0,0)
            stroke = 5
            font = cv2.FONT_HERSHEY_SIMPLEX
            #Identifying faces
            cv2.rectangle(frame, (x,y), (x + w,y + h), color, stroke)

            # Recognizing known faces
            #id_, conf = recognizer.predict(gray_roi)
            prediction = recognizer.predict(gray_roi)

            if prediction[1] <= 100:
                image_id = "face_image.png"
                cv2.imwrite(image_id, gray_roi)
                #print(id_, ":", labels[id_])
                #name = labels[id_]
                #print(id_, ":", labels[id_])
                #cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                print("You are identified as: ", labels[prediction[0]], ":", prediction[1], "%")
                cv2.putText(frame, '% s - %.0f' % (labels[prediction[0]], prediction[1]), (x,y), font, 1, color, stroke, cv2.LINE_AA)
            else:
                new_image_id = "new_face_image.png"
                cv2.imwrite(new_image_id, gray_roi)
                #newid_ = labels.tolist().count + 1
                #labels[newid_] = "newface" + 'newid_'
                #new_name = labels[newid_]
                #print(new_name)
                #cv2.putText(frame, new_name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                cv2.putText(frame, 'New_Face', (x,y), font, 1, color, stroke, cv2.LINE_AA)



        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
vidcap.release()
cv2.destroyAllWindows()

