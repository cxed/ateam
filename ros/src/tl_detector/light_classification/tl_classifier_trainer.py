import os, cv2
import numpy as np
import random as rnd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

class TLClassifier_Trainer:
    def __init__(self):
        self.debug = False
        self.show_epochs = True

        self.X_train = np.ndarray(shape=(0, 150, 100, 3))
        self.Y_train = np.ndarray(shape=(0))
        self.X_test = np.ndarray(shape=(0, 150, 100, 3))
        self.Y_test = np.ndarray(shape=(0))
        self.X_val = np.ndarray(shape=(0, 150, 100, 3))
        self.Y_val = np.ndarray(shape=(0))

        self.set_image_paths()
        self.EPOCHS = 100
        self.BATCH_SIZE = 2048

    # scale images depending on extension/image type
    def load_image(self, image_path):
        
        image = cv2.imread(image_path) 
        image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        image = self.normalize_image(image)
     
        # scale image (0-255)
        if image_path[-4:] == '.png':
            image = image.astype(np.float32)*255
      
        # remove alpha channel if present
        if image.shape[2] == 4:
            b, g, r, a = cv2.split(image)
            image = np.dstack((r,g,b))
    
        return image

    def normalize_image(self, image):
        r, g, b = cv2.split(image)
        r = (r - 128)/128
        g = (g - 128)/128
        b = (b - 128)/128
        return cv2.merge((r, g, b))

    def set_image_paths(self):
        
        # load training data
        dir = './sim_images'
        green_file_path = dir + '/GREEN/'
        yellow_file_path = dir + '/YELLOW/'
        red_file_path = dir + '/RED/'

        # add training images
        images = []
        labels = []
        green_images=os.listdir(green_file_path) 
        for green_image_path in green_images:
            if type(green_image_path)==type("string"):
                image = self.load_image(green_file_path + green_image_path)
                images.append(image)
                labels.append(0)
    
        yellow_images=os.listdir(yellow_file_path) 
        for yellow_image_path in yellow_images:
            if type(yellow_image_path)==type("string"):
                image = self.load_image(yellow_file_path + yellow_image_path)
                images.append(image)
                labels.append(1)
                            
        red_images=os.listdir(red_file_path) 
        for red_image_path in red_images:
            if type(red_image_path)==type("string"):
                image = self.load_image(red_file_path + red_image_path)
                images.append(image)
                labels.append(2)
        
        self.X_train = np.array(images)  
        self.Y_train = np.array(labels)

        # load test data
        dir = './test_images'
        green_file_path = dir + '/GREEN/'
        yellow_file_path = dir + '/YELLOW/'
        red_file_path = dir + '/RED/'

        # add test images
        images = []
        labels = []
        X_val = []
        Y_val = []
        green_images=os.listdir(green_file_path) 
        for green_image_path in green_images:
            if type(green_image_path)==type("string"):
                image = self.load_image(green_file_path + green_image_path)
                images.append(image)
                labels.append(0)
    
        yellow_images=os.listdir(yellow_file_path) 
        for yellow_image_path in yellow_images:
            if type(yellow_image_path)==type("string"):
                image = self.load_image(yellow_file_path + yellow_image_path)
                images.append(image)
                labels.append(1)
                X_val.append(image)
                Y_val.append(1)
                            
        red_images=os.listdir(red_file_path) 
        for red_image_path in red_images:
            if type(red_image_path)==type("string"):
                image = self.load_image(red_file_path + red_image_path)
                images.append(image)
                labels.append(2)

        self.X_test = np.array(images)
        self.Y_test = np.array(labels)

        self.X_val = np.array(X_val)
        self.Y_val = np.array(Y_val)
               
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
    
    def train(self):
        
        def evaluate(X_data, Y_data):
            num_examples = len(X_data)
            total_accuracy = 0
            sess = tf.get_default_session()
            for offset in range(0, num_examples, self.BATCH_SIZE):
                batch_x, batch_y = X_data[offset:offset+self.BATCH_SIZE], Y_data[offset:offset+self.BATCH_SIZE]
                accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
                total_accuracy += (accuracy * len(batch_x))
            return total_accuracy / num_examples
    
        # split new training set
        X_train, Y_train = shuffle(self.X_train, self.Y_train)
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=1024)
        #X_test, Y_test = shuffle(self.X_test, self.Y_test)

        ### Train model
        x = tf.placeholder(tf.float32, (None, 150, 100, 3))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(Y_train, 3)

        rate = 0.001

        logits = self.LeNet(tf.cast(X_train, tf.float32))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        logits_test = self.LeNet(tf.cast(x, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train)
    
            print("Training...")
            print()
            avg_accuracy=[]
            for i in range(self.EPOCHS):
                X_train, Y_train = shuffle(X_train, Y_train)
                for offset in range(0, num_examples, self.BATCH_SIZE):
                    end = offset + self.BATCH_SIZE
                    batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
                    sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
                validation_accuracy = evaluate(X_test, Y_test)
                if self.show_epochs:
                    print("EPOCH {} ...".format(i+1), "Eval Accuracy = {:.6f}".format(validation_accuracy))

                test_accuracy = evaluate(self.X_val, self.Y_val)
                #if self.show_epochs:
                    #print("EPOCh {} ...".format(i+1), "Test Accuracy = {:.6f}".format(test_accuracy))
            
            dir = './test_images'
            green_file_path = dir + '/GREEN/'
            yellow_file_path = dir + '/YELLOW/'
            red_file_path = dir + '/RED/'
            choices = {0: "GREEN", 1: "YELLOW", 2: "RED", 3: "UNKNOWN"}

            eval_green = True
            eval_yellow = False
            eval_red = False
            
            num_images = 0
            num_incorrect = 0
            
            if eval_green:
                green_images=os.listdir(green_file_path)
                for green_image_path in green_images:
                    if type(green_image_path)==type("string"):
                        image = cv2.imread(green_file_path + green_image_path)
                        image = self.normalize_image(image)
                        res = None
                        res = cv2.resize(image, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
                        image = res.reshape(1, 150, 100, 3)                  
                        retval = sess.run(logits_test,feed_dict={x: image})
                        pred = np.argmax(retval[0])
                        result = choices.get(pred, "UNKNOWN")
                        #print('Retval: ', str(retval))
                        print('Result: Expected GREEN - Detected: ', result)
                        num_images += 1
                        if result != 'GREEN':
                            num_incorrect += 1

            if eval_yellow:
                yellow_images=os.listdir(yellow_file_path) 
                for yellow_image_path in yellow_images:
                    if type(yellow_image_path)==type("string"):
                        image = cv2.imread(yellow_file_path + yellow_image_path)
                        image = self.normalize_image(image)
                        res = None
                        res = cv2.resize(image, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
                        image = res.reshape(1, 150, 100, 3)                     
                        retval = sess.run(logits_test,feed_dict={x: image})
                        pred = np.argmax(retval[0])
                        result = choices.get(pred, "UNKNOWN")
                        #print('Retval: ', str(retval))
                        print('Result: Expected YELLOW - Detected: ', result)
                        num_images += 1
                        if result != 'YELLOW':
                            num_incorrect += 1

            if eval_red:
                red_images=os.listdir(red_file_path) 
                for red_image_path in red_images:
                    if type(red_image_path)==type("string"):
                        image = cv2.imread(red_file_path + red_image_path)
                        image = self.normalize_image(image)
                        res = None
                        res = cv2.resize(image, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
                        image = res.reshape(1, 150, 100, 3)                     
                        retval = sess.run(logits_test,feed_dict={x: image})
                        pred = np.argmax(retval[0])
                        result = choices.get(pred, "UNKNOWN")
                        #print('Retval: ', str(retval))
                        print('Result: Expected RED - Detected: ', result)
                        num_images += 1
                        if result != 'RED':
                            num_incorrect += 1

            success_rate = 0.0
            if num_incorrect > 0:
                success_rate = str(100.0-100.0*(float(num_incorrect)/float(num_images)))
            if num_incorrect == 0:
                success_rate = 100

            print('No Images: ' + str(num_images) + ' incorrect: ' + str(num_incorrect) + ' success rate: ' + str(success_rate) + ' %')

            saver.save(sess, './model')
            print("Model saved")
        sess.close()

trainer = TLClassifier_Trainer()
trainer.train()