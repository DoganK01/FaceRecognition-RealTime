import cv2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open("model.json", "r") as file:
    model_json = file.read()

loaded_model = model_from_json(model_json)

loaded_model.load_weights("weights.h5")


def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255

    return img


def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((300, 300), np.uint8), img
    for (x, y, w, h) in faces:
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = img[y:y + h, x:x + w]
    try:
        roi = cv2.resize(roi, (300, 300), interpolation=cv2.INTER_AREA)
    except:
        return (x, w, y, h), np.zeros((300, 300), np.uint8), img
    return (x, w, y, h), roi, img


acc = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rect, face, image = face_detector(frame)

    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        roi = roi.reshape(-1, 300, 300, 1)

        predictions = loaded_model.predict(roi)
        probVal = np.max(predictions)
        classIndex = classNo[0]

        if probVal > 0:
            confidence = probVal * 100
            dis = str(probVal * 100) + "% similar"
            pkq = "class: " + str(classIndex)

        font = cv2.FONT_HERSHEY_DUPLEX
        color = (250, 120, 255)
        name = dis + " " + pkq
        cv2.putText(image, name, (100, 120), font, 1, color, 2)

        if confidence > 80:
            font = cv2.FONT_HERSHEY_DUPLEX
            color = (0, 255, 0)
            name = "Welcome"
            stroke = 2
            cv2.putText(image, name, (250, 450), font, 1, color, 2)
            cv2.imshow('Face Cropper', image)

        else:
            font = cv2.FONT_HERSHEY_DUPLEX
            color = (0, 0, 255)
            name = "Device Locked"
            stroke = 2
            cv2.putText(image, name, (250, 450), font, 1, color, 2)
            cv2.imshow('Face Cropper', image)
            
    else:
        font = cv2.FONT_HERSHEY_DUPLEX
        color = (0, 0, 255)
        name = "Cannot detect Face"
        stroke = 2
        cv2.putText(image, name, (250, 450), font, 1, color, 2)
        cv2.imshow('Face Cropper', image)

    if cv2.waitKey(20) & 0XFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
