#!/usr/bin/env python
from classifier_standalone import TLClassifierStandalone
import cv2, os

def normalize_image(image):
    r, g, b = cv2.split(image)
    r = (r - 128)/128
    g = (g - 128)/128
    b = (b - 128)/128
    return cv2.merge((r, g, b))


dir = './test_images'
green_file_path = dir + '/GREEN/'
yellow_file_path = dir + '/YELLOW/'
red_file_path = dir + '/RED/'

num_images = 0
num_incorrect = 0

classifier = TLClassifierStandalone()

yellow_images=os.listdir(yellow_file_path) 
for yellow_image_path in yellow_images:
    if type(yellow_image_path)==type("string"):
        image = cv2.imread(yellow_file_path + yellow_image_path)
        image = normalize_image(image)
        result = classifier.get_classification(image)
        num_images += 1
        if result != 'YELLOW':
            num_incorrect += 1
        print('YELLOW - detected: ', result)



print('No Images: ' + str(num_images) + ' incorrect: ' + str(num_incorrect) + ' success rate: ' + str(100.0-100.0*(float(num_incorrect)/float(num_images))) + ' %')
