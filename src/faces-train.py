import cv2
import os
import numpy as np
from PIL import Image
import pickle

# Specifica il Path da dove prendere le immagini, in questo caso dal src dove e presente nel sistema
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the image directory path
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    # Iteriamo tra i file
    for file in files:
        #Verifichiamo se sono imaggini
        if file.endswith("png") or file.endswith("jpg"):
            # Prendiamo il path del file immagine
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # Get label name
            label = os.path.basename(root).replace(" ", "-").lower()
            # Tampiamo la label e il path per la label
            print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #y_labels.append(label) # some numbers
            #x_train.append(path) # Verify this image, turn into a NUMPY array, GRAY
            # Give me the image
            pil_image = Image.open(path).convert("L") # grayscale
            # Resize the image
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            # Take every pixel value and make it into a number array
            image_array = np.array(pil_image, "uint8")
            print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

# x_train is the images array, convert the y_labels to a numpy array
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")