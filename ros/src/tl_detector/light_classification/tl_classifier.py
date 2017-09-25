#!/usr/bin/env python
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2, rospkg, rospy, time
from tensorflow.contrib.layers import flatten

class TLClassifier(object):
    def __init__(self):
        #TODO DONE load classifier
        self.debug = False
        self.capture_images = False
        self.verbose = False

        rospack = rospkg.RosPack()
        self.imgPath = str(rospack.get_path('tl_detector'))+'/light_classification/pics/'
        modelCheckpointFile = str(rospack.get_path('tl_detector'))+'/light_classification'
        rospy.loginfo('[TL Classifier] model checkpoint file: ' + modelCheckpointFile)
        
        self.x = tf.placeholder(tf.float32, (None, 150, 100, 3))
        self.y = tf.placeholder(tf.int32, (None))
        self.logits = self.LeNet(tf.cast(self.x, tf.float32))
        saver = tf.train.Saver()
        self.sess = tf.Session()        
        saver.restore(self.sess, tf.train.latest_checkpoint(modelCheckpointFile))
        if self.debug:
            rospy.loginfo('[TL Classifier] constructor completed: ')

    def LeNet(self, x):    
        # Hyperparameters
        mu = 0
        sigma = 0.01
        Padding='VALID'
        W_lambda = 3.0

        if self.debug:
            print('[TL Classifier] input shape: ', x.shape)
    
        conv1_W = tf.Variable(tf.truncated_normal(shape=(60, 40, 3, 8), mean = mu, stddev = sigma))
        conv1_b = tf.Variable(tf.zeros(8))
        conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding=Padding) + conv1_b

        # L2 Regularization
        conv1_W = -W_lambda*conv1_W
    
        # Activation.
        conv1 = tf.nn.relu(conv1)
    
        # Pooling...
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=Padding)
        
        # Layer 2: Convolutional...
        conv2_W = tf.Variable(tf.truncated_normal(shape=(30, 20, 8, 32), mean = mu, stddev = sigma))
        conv2_b = tf.Variable(tf.zeros(32))
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding=Padding) + conv2_b
        
        # L2 Regularization
        conv2 = -W_lambda*conv2
        
        # Activation.
        conv2 = tf.nn.relu(conv2)
        # Pooling...
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=Padding)
        
        # Flatten...
        fc0   = flatten(conv2)
    
        # Layer 3: Fully Connected...
        fc1_W = tf.Variable(tf.truncated_normal(shape=(1280, 120), mean = mu, stddev = sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        
        fc1   = tf.matmul(fc0, fc1_W) + fc1_b
           
        # Activation.
        fc1    = tf.nn.relu(fc1)
           
        # Layer 4: Fully Connected...
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
        fc2_b  = tf.Variable(tf.zeros(84))
        fc2    = tf.matmul(fc1, fc2_W) + fc2_b
            
        # Activation.
        fc2    = tf.nn.relu(fc2)

        # Layer 5: Fully Connected. Input = 84. Output = 3.
        fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 3), mean = mu, stddev = sigma))
        fc3_b  = tf.Variable(tf.zeros(3))
        logits = tf.matmul(fc2, fc3_W) + fc3_b

        return logits

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        if self.capture_images:
            path = '/home/max/Pictures/'
            cv2.imwrite(self.imgPath+str(int(time.clock()*1000))+'.jpg', image)
            rospy.loginfo('[TLClassifier] Saved Image ... ')

        if self.debug:
            rospy.loginfo('[TL Classifier] invoked... ')

        if image.shape != (300, 200, 3):
            rospy.loginfo('[TL Classifier] image shape NOK: ' + str(image.shape))
            return TrafficLight.UNKNOWN
            
        assert image.shape == (300, 200, 3)
        if self.debug:
            rospy.loginfo('[TL Classifier] assertion ok: ')

        res = None
        res = cv2.resize(image, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        image = res.reshape(1, 150, 100, 3)
        assert image.shape == (1, 150, 100, 3)
        if self.debug:
            rospy.loginfo('[TL Classifier] reshape ok: ')

        prediction = self.sess.run(self.logits, feed_dict={self.x: image})
            
        # get certainty of classification
        probability=tf.nn.softmax(self.logits)
        certainties = self.sess.run([probability], feed_dict={self.x: image})

        # get class
        classification = np.argmax(prediction)
        certainty = certainties[0][0][classification]

        # in case the classifier is unsure, return unknown
        if certainty < 0.6:
            classification = 4

        choices = {0: TrafficLight.GREEN, 1: TrafficLight.YELLOW, 2: TrafficLight.RED}
        result = choices.get(classification, TrafficLight.UNKNOWN)

        if self.verbose:
            strings = {0: "GREEN", 1: "YELLOW", 2: "RED"}
            result = strings.get(classification, TrafficLight.UNKNOWN)
            rospy.loginfo('[TL Classifier] ' + str(result) + ' detected. Certainty: ' + str(certainty))

        return  result
