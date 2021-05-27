# -*- coding: utf-8 -*-
"""
Created on Wed May 26 19:48:55 2021

@author: PC
"""


import cv2 as cv
import os

def convertToRGB(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def detect_faces(cascade, test_image, scaleFactor = 1.1):
    
   
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        cv.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image_copy
#haar cascade frontal face
cv2_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
cascade= os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
haar_cascade_face = cv.CascadeClassifier(cascade)
#opening camera
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
while True:
    #reading frames 
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    face= detect_faces(haar_cascade_face,frame) 
    cv.imshow("face detector    press 'q' to exit",face)
    
    if cv.waitKey(1) == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()