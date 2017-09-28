#!/usr/bin/env python
from classifier_standalone import TLClassifierStandalone
import cv2, os

eval_green = True
eval_yellow = False
eval_red = False

dir = './test_images'
green_file_path = dir + '/GREEN/'
yellow_file_path = dir + '/YELLOW/'
red_file_path = dir + '/RED/'

num_images = 0
num_incorrect = 0

classifier = TLClassifierStandalone()

if eval_green:
    green_images=os.listdir(green_file_path) 
    for green_image_path in green_images:
        if type(green_image_path)==type("string"):
            image = cv2.imread(green_file_path + green_image_path)
            result = classifier.get_classification(image)
            num_images += 1
            if result != 'GREEN':
                num_incorrect += 1
            print('Expected GREEN - detected: ', result)

if eval_yellow:
    yellow_images=os.listdir(yellow_file_path) 
    for yellow_image_path in yellow_images:
        if type(yellow_image_path)==type("string"):
            image = cv2.imread(yellow_file_path + yellow_image_path)
            result = classifier.get_classification(image)
            num_images += 1
            if result != 'YELLOW':
                num_incorrect += 1
            print('Expected YELLOW - detected: ', result)

if eval_red:
    red_images=os.listdir(red_file_path) 
    for red_image_path in red_images:
        if type(red_image_path)==type("string"):
            image = cv2.imread(red_file_path + red_image_path)
            result = classifier.get_classification(image)
            num_images += 1
            if result != 'RED':
                num_incorrect += 1
            print('Expected RED - detected: ', result)

print('No Images: ' + str(num_images) + ' incorrect: ' + str(num_incorrect) + ' success rate: ' + str(100.0-100.0*(float(num_incorrect)/float(num_images))) + ' %')
