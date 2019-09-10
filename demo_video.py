import sys
import dex
from glob import glob

import cv2
from time import sleep
import logging as log
import datetime as dt

# setup model
dex.eval()

# read landscape
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam_predict_age_sex.log',level=log.INFO)

if __name__ == '__main__':

    video_capture = cv2.VideoCapture(0)
    anterior = 0

    while True:
        if not video_capture.isOpened():
            print("Unable to load camera.")
            sleep(5)
            pass

        # capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: "+str(len(faces))+ " at " + str(dt.datetime.now()) )

            age, female, male = dex.estimage_frame(frame)
            print("age of the person = {:.3f}".format(age))
            print("women: {:.3f}, man: {:.3f}\n".format(female, male))

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('Video', frame)

    video_capture.release()
    cv2.destroyAllWindows()