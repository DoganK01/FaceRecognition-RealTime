import numpy as np
import cv2

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def extract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5)

    if faces is():
        return None

    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()

    if extract(frame) is not None:
        count += 1

        face = cv2.resize(extract(frame), (300, 300))



        path = "Dogan"+str(count)+".jpg"

        cv2.imwrite(path, face)

        name = str(count)
        font = cv2.FONT_HERSHEY_DUPLEX
        color = (0, 255, 0)
        stroke = 2
        cv2.putText(face, name, (50, 50), font, 1, color, stroke)
        cv2.imshow("Cropper", face)

    else:
        print("NOT FOUND")

        if cv2.waitKey(20) & 0XFF == ord('q') or count == 100:
            break

print("Complete")
cap.release()
cv2.destroyAllWindows()
