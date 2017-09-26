#!/usr/bin/env python
from classifier_standalone import TLClassifierStandalone
import cv2

dir = './combined_pics'
color = "red"
if color == "green":
    image_path = dir + '/GREEN/1ut30.jpg' 
if color == "yellow":
    image_path = dir + '/YELLOW/1ut69.jpg' 
if color == "red":
    image_path = dir + '/RED/1ut0.jpg'            

image = cv2.imread(image_path)
#image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
classifier = TLClassifierStandalone()
result = classifier.get_classification(image)
print(result)