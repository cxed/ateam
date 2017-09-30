#!/usr/bin/env python
from styx_msgs.msg import TrafficLight, TrafficLightArray
from std_msgs.msg import Int32
import numpy as np
import cv2, rospkg, rospy, time
from keras.models import load_model
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
        #TODO DONE load classifier
        self.debug = False
        self.capture_images = False
        self.verbose = True

        rospack = rospkg.RosPack()
        self.imgPath = str(rospack.get_path('tl_detector'))+'/light_classification/pics/'
        self.model_path = str(rospack.get_path('tl_detector'))+'/light_classification/'
        self.model = load_model(self.model_path + 'keras_model.h5')
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        if self.verbose:
            self.waypoint = None
            self.traffic_waypoint_sub = rospy.Subscriber('/traffic_waypoint', Int32, self.get_traffic_waypoint)

        if self.debug:
            rospy.loginfo('[TL Classifier] constructor completed: ')
    
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
        #model = load_model(self.model_path + 'keras_model.h5')
        save_image = image
        choices = {0: TrafficLight.GREEN, 1: TrafficLight.YELLOW, 2: TrafficLight.RED}

        if self.debug:
            rospy.loginfo('[TL Classifier] invoked... ')

        if image.shape != (300, 200, 3):
            rospy.loginfo('[TL Classifier] image shape NOK: ' + str(image.shape))
            return TrafficLight.UNKNOWN
            
        assert image.shape == (300, 200, 3)
        if self.debug:
            rospy.loginfo('[TL Classifier] assertion ok: ')

        image = self.normalize_image(image)
        res = None
        res = cv2.resize(image, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        image = res.reshape(1, 150, 100, 3)                  
        #classification = model.predict_classes(image)[0]
        with self.graph.as_default():
            classification = self.model.predict_classes(image)[0]
        result = choices.get(classification, TrafficLight.UNKNOWN)

        #result = TrafficLight.GREEN

        if self.capture_images:
            strings = {0: "GREEN/", 1: "YELLOW/", 2: "RED/"}
            path = strings.get(classification, "UNKNOWN/")
            savePath = self.imgPath + path
            cv2.imwrite(savePath+str(int(time.clock()*1000))+'.jpg', save_image)
            rospy.loginfo('[TLClassifier] Saved Image ... ' + savePath)

        if self.verbose:
            strings = {0: "GREEN", 1: "YELLOW", 2: "RED"}
            classification_result = strings.get(classification, "UNKNOWN")
            wp = "None Published"
            rospy.loginfo('[TL Classifier] waypoint: ' + str(self.waypoint))
            if self.waypoint != None:
                wp = str(self.waypoint)
            rospy.loginfo('[TL Classifier] Classifier: ' + classification_result)
            rospy.loginfo('[TL Classifier] /traffic/waypoint: ' + wp)

        return result

    def get_traffic_waypoint(self, msg):
        self.waypoint = msg.data