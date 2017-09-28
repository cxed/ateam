#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.contrib.layers import flatten

class TLClassifierStandalone:
    def __init__(self):
        #TODO DONE load classifier
        self.debug = False
        self.capture_images = False
        self.verbose = False
        
        self.x = tf.placeholder(tf.float32, (None, 150, 100, 3))
        self.y = tf.placeholder(tf.int32, (None))
        self.logits = self.LeNet(tf.cast(self.x, tf.float32))
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.saver.restore(self.sess, './model')
        if self.debug:
            print('[TL Classifier] constructor completed: ')

    def __del__(self):
        #TODO DONE load classifier
        self.sess.close()

    def normalize_image(self, image):
        r, g, b = cv2.split(image)
        r = (r - 128)/128
        g = (g - 128)/128
        b = (b - 128)/128
        return cv2.merge((r, g, b))

    def LeNet(self, x):  
 
        # Hyperparameters
        mu = 0
        sigma = 0.1
        Padding='VALID'
        W_lambda = 5.0
    
        conv1_W = tf.Variable(tf.truncated_normal(shape=(6, 4, 3, 3), mean = mu, stddev = sigma))
        conv1_b = tf.Variable(tf.zeros(3))
        conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding=Padding) + conv1_b
        if self.debug:
            print("x shape: ", x.shape)
            print("conv1_W shape: ", conv1_W.shape)
            print("conv1_b shape: ", conv1_b.shape)
            print("conv1 shape: ", conv1.shape)
    
        # L2 Regularization
        conv1_W = -W_lambda*conv1_W
        if self.debug:
            print("conv1_W (after L2 1) shape: ", conv1_W.shape)
    
        # Activation.
        conv1 = tf.nn.relu(conv1)
        if self.debug:
            print("conv1 (after Activiateion) shape: ", conv1.shape)
    
        # Pooling...
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=Padding)
        if self.debug:
            print("conv1 (after Pooling 1) shape: ", conv1.shape)
    
        # Layer 2: Convolutional...
        conv2_W = tf.Variable(tf.truncated_normal(shape=(6, 4, 3, 6), mean = mu, stddev = sigma))
        conv2_b = tf.Variable(tf.zeros(6))
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding=Padding) + conv2_b
        if self.debug:
            print("conv2_W shape: ", conv2_W.shape)
            print("conv2_b shape: ", conv2_b.shape)
            print("conv2 shape: ", conv2.shape)
    
        # L2 Regularization
        conv2 = -W_lambda*conv2
        if self.debug:
            print("conv2 shape after L2: ", conv2.shape)
    
        # Activation.
        conv2 = tf.nn.relu(conv2)
        if self.debug:
            print("conv2 shape after activation: ", conv2.shape)
    
        # Pooling...
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=Padding)
        if self.debug:
            print("conv2 shape after pooling: ", conv2.shape)

        # Flatten...
        fc0   = flatten(conv2)
    
        # Layer 3: Fully Connected...
        fc1_W = tf.Variable(tf.truncated_normal(shape=(4356, 60), mean = mu, stddev = sigma))
        fc1_b = tf.Variable(tf.zeros(60))
        
        if self.debug:
            print("fc0", fc0.shape)
            print("fc1_W", fc1_W.shape)
            print("fc1_b", fc1_b.shape)
        fc1   = tf.matmul(fc0, fc1_W) + fc1_b
        if self.debug:
            print("fc1", fc1.shape)
    
        # Activation.
        fc1    = tf.nn.relu(fc1)
        if self.debug:
            print("fc1 after Activation", fc1.shape)
    
        # Layer 4: Fully Connected...
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(60, 30), mean = mu, stddev = sigma))
        fc2_b  = tf.Variable(tf.zeros(30))
        fc2    = tf.matmul(fc1, fc2_W) + fc2_b
        if self.debug:
            print("fc2_W shape: ", fc2_W.shape)
            print("fc2_b shape: ", fc2_b.shape)
            print("fc2 shape: ", fc2.shape)
    
        # Activation.
        fc2    = tf.nn.relu(fc2)
        if self.debug:
            print("fc2 shape after activation: ", fc2.shape)
    
        # Layer 5: Fully Connected. Input = 30. Output = 3.
        fc3_W  = tf.Variable(tf.truncated_normal(shape=(30, 3), mean = mu, stddev = sigma))
        fc3_b  = tf.Variable(tf.zeros(3))
        logits = tf.matmul(fc2, fc3_W) + fc3_b
        if self.debug:
            print("fc3_W shape: ", fc3_W.shape)
            print("fc3_b shape: ", fc3_b.shape)
            print("logits shape: ", logits.shape)
    
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
        
        choices = {0: "GREEN", 1: "YELLOW", 2: "RED", 3: "UNKNOWN"}

        image = self.normalize_image(image)
        res = None
        res = cv2.resize(image, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        image = res.reshape(1, 150, 100, 3)

        assert image.shape == (1, 150, 100, 3)
        if self.debug:
            print('[TL Classifier] reshape ok: ')

        
        retval = self.sess.run(self.logits,feed_dict={self.x: image})
        pred = np.argmax(retval[0])
        result = choices.get(pred, "UNKNOWN")

        if self.verbose:
            print('[TL Classifier] ' + result + ' detected.')

        return  result
