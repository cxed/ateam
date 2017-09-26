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
        self.verbose = True
        
        self.x = tf.placeholder(tf.float32, (1, 150, 100, 3))
        self.y = tf.placeholder(tf.int32, (None))
        self.logits = self.LeNet(tf.cast(self.x, tf.float32))
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        if self.debug:
            print('[TL Classifier] constructor completed: ')

    def __del__(self):
        #TODO DONE load classifier
        self.sess.close()

    def CanonicalLeNet(self, x):    
        # Hyperparameters
        mu = 0
        sigma = 0.1
    
        # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 3), mean = mu, stddev = sigma))
        conv1_b = tf.Variable(tf.zeros(3))
        conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

        # Activation.
        conv1 = tf.nn.relu(conv1)

        # Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Layer 2: Convolutional. Output = 10x10x16.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 5), mean = mu, stddev = sigma))
        conv2_b = tf.Variable(tf.zeros(5))
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
        # Activation.
        conv2 = tf.nn.relu(conv2)

        # Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flatten. Input = 5x5x16. Output = 400.
        fc0   = flatten(conv2)
    
        # Layer 3: Fully Connected. Input = 400. Output = 200.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(3740, 50), mean = mu, stddev = sigma))
        fc1_b = tf.Variable(tf.zeros(50))
        fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
        # Activation.
        fc1    = tf.nn.relu(fc1)

        # Layer 4: Fully Connected. Input = 200. Output = 150.
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(50, 25), mean = mu, stddev = sigma))
        fc2_b  = tf.Variable(tf.zeros(25))
        fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
        # Activation.
        fc2    = tf.nn.relu(fc2)

        # Layer 5: Fully Connected. Input = 150. Output = 10.
        fc3_W  = tf.Variable(tf.truncated_normal(shape=(25, 3), mean = mu, stddev = sigma))
        fc3_b  = tf.Variable(tf.zeros(3))
        logits = tf.matmul(fc2, fc3_W) + fc3_b

        return logits
               
    def LeNet(self, x):  
 
        # Hyperparameters
        mu = 0
        sigma = 0.01
        Padding='VALID'
        W_lambda = 3.0
    
        conv1_W = tf.Variable(tf.truncated_normal(shape=(60, 40, 3, 8), mean = mu, stddev = sigma))
        conv1_b = tf.Variable(tf.zeros(8))
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
        conv2_W = tf.Variable(tf.truncated_normal(shape=(30, 20, 8, 32), mean = mu, stddev = sigma))
        conv2_b = tf.Variable(tf.zeros(32))
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
        fc1_W = tf.Variable(tf.truncated_normal(shape=(1280, 120), mean = mu, stddev = sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        
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
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
        fc2_b  = tf.Variable(tf.zeros(84))
        fc2    = tf.matmul(fc1, fc2_W) + fc2_b
        if self.debug:
            print("fc2_W shape: ", fc2_W.shape)
            print("fc2_b shape: ", fc2_b.shape)
            print("fc2 shape: ", fc2.shape)
    
        # Activation.
        fc2    = tf.nn.relu(fc2)
        if self.debug:
            print("fc2 shape after activation: ", fc2.shape)
    
        # Layer 5: Fully Connected. Input = 84. Output = 3.
        fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 3), mean = mu, stddev = sigma))
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
        
        a = np.max(image)
        b = np.min(image)
        ra = 0.9
        rb = 0.1
        try: 
            image = (((ra-rb) * (image - a)) / (b - a)) + rb
        except:
            print('catched..')

        res = None
        res = cv2.resize(image, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        image = res.reshape(1, 150, 100, 3)
        assert image.shape == (1, 150, 100, 3)
        if self.debug:
            print('[TL Classifier] reshape ok: ')

        prediction = self.sess.run(self.logits, feed_dict={self.x: image})
    
        # get certainty of classification
        probability=tf.nn.softmax(self.logits)
        certainties = self.sess.run([probability], feed_dict={self.x: image})

        # get class
        classification = np.argmax(prediction)
        certainty = certainties[0][0][classification]

        # in case the classifier is unsure, return unknown
        if certainty < 0.4:
            classification = 4

        choices = {0: "GREEN", 1: "YELLOW", 2: "RED", 3: "UNKNOWN"}
        result = choices.get(classification, "UNKNOWN")

        if self.verbose:
            print('[TL Classifier] ' + result + ' ('  +  str(classification) + ') ' + ' detected with ' + str(certainty) + ' certainty')

        return  result
