import cv2
import pickle
import os
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
# Using Local Binary Pattern Histogram(LBPH) Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Trying Eigen or Fisher  Recognizer , it needs all the image zie to be of 7225 pixels
#recognizer = cv2.face.FisherFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            print(label_ids)
            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            print(image_array)
            #faces = face_cascade.detectMultiScale(image_array, scaleFactor = 2.5, minNeighbors = 3 )
            faces = face_cascade.detectMultiScale(image_array, minNeighbors = 3 )

            for (x,y,w,h) in faces:
                roi = image_array[y:y + h, x :x + w]
                x_train.append(roi)
                y_labels.append(id_)
print(y_labels)
print(x_train)
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trained.yml")