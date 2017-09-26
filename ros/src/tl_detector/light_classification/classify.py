#!/usr/bin/env python
from classifier_standalone import TLClassifierStandalone
import cv2

dir = './harvested'
color = "green"
if color == "green":
    image_path = dir + '/GREEN/24035.jpg' 
if color == "yellow":
    image_path = dir + '/YELLOW/13222.jpg' 
if color == "red":
    image_path = dir + '/RED/27907.jpg'            

image = cv2.imread(image_path)
#image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
classifier = TLClassifierStandalone()
result = classifier.get_classification(image)