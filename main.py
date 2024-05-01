import cv2
import face_recognition
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load and preprocess the dataset
dataset_dir = "C:/Users/sowmi/OneDrive/Desktop/mini project/dataset"
known_face_encodings = []
known_face_names = []

for person_dir in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_dir)
    for image_file in os.listdir(person_path):
        image_path = os.path.join(person_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(person_dir)

# Create a k-d tree for efficient nearest neighbor search
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(known_face_encodings)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, (320, 240))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        distances, indices = neigh.kneighbors([face_encoding], return_distance=True)
        name = known_face_names[indices[0][0]]

        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
