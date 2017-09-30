#!/usr/bin/env python
import numpy as np
import cv2
from keras.models import load_model

class TLClassifierKerasStandalone:
    def __init__(self):
        #TODO DONE load classifier
        self.debug = False
        self.capture_images = False
        self.verbose = False
        
        self.model = load_model('keras_model.h5')
        if self.debug:
            print('[TL Classifier] constructor completed: ')

    def normalize_image(self, image):
        r, g, b = cv2.split(image)
        r = (r - 128)/128
        g = (g - 128)/128
        b = (b - 128)/128
        return cv2.merge((r, g, b))
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        choices = {0: "GREEN", 1: "YELLOW", 2: "RED", 3: "UNKNOWN"}

        if self.capture_images:
            cv2.imwrite(self.imgPath+str(int(time.clock()*1000))+'.jpg', image)
            print('[TLClassifier] Saved Image ... ')

        if self.debug:
            print('[TL Classifier] invoked... ')

        if image.shape != (300, 200, 3):
            print('[TL Classifier] image shape NOK: ' + str(image.shape))
            return "UNKNOWN shape"
            
        assert image.shape == (300, 200, 3)
        if self.debug:
            print('[TL Classifier] assertion ok: ')

        image = self.normalize_image(image)
        res = None
        res = cv2.resize(image, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        image = res.reshape(1, 150, 100, 3)                  
        classification = self.model.predict_classes(image, verbose=0)
        result = choices.get(classification[0], "UNKNOWN")

        if self.verbose:
            print('[TL Classifier] ' + result + ' detected.')

        return  result
