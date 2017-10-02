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

        # Use is_sim_launch to determine if we are launched in sim or site
        self.is_sim_launch = rospy.get_param("~sim_launch", False)

        rospack = rospkg.RosPack()
        self.imgPath = str(rospack.get_path('tl_detector'))+'/light_classification/pics/'
        self.model_path = str(rospack.get_path('tl_detector'))+'/light_classification/'
        
        # determine which model to load
        if(self.is_sim_launch):
            self.model = load_model(self.model_path + 'keras_model_sim.h5')
        else:
            self.model = load_model(self.model_path + 'keras_model_real.h5')

        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        if self.verbose:
            self.waypoint = None
            self.traffic_waypoint_sub = rospy.Subscriber('/traffic_waypoint', Int32, self.get_traffic_waypoint)

        if self.debug:
            rospy.loginfo('[TL Classifier] constructor completed: ')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        save_image = image
        choices = {0: TrafficLight.GREEN, 1: TrafficLight.YELLOW, 2: TrafficLight.RED}

        if self.debug:
            rospy.loginfo('[TL Classifier] invoked... ')

        if(self.is_sim_launch):
            if image.shape != (300, 200, 3):
                rospy.loginfo('[TL Classifier] image shape NOK: ' + str(image.shape))
                return TrafficLight.UNKNOWN
            
            assert image.shape == (300, 200, 3)
            if self.debug:
                rospy.loginfo('[TL Classifier] assertion ok: ')
        else:
            if image.shape != (125, 350, 3):
                rospy.loginfo('[TL Classifier] image shape NOK: ' + str(image.shape))
                return TrafficLight.UNKNOWN
            
            assert image.shape == (125, 350, 3)
            if self.debug:
                rospy.loginfo('[TL Classifier] assertion ok: ')

        res = None
        res = cv2.resize(image, (32,32), interpolation = cv2.INTER_CUBIC)
        image = res.reshape(1, 32, 32, 3)
        with self.graph.as_default():
            classification = self.model.predict_classes(image)[0]
        result = choices.get(classification, TrafficLight.UNKNOWN)

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
