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
        self.green_images = []
        self.yellow_images = []
        self.red_images = []
        self.X_data = np.ndarray(shape=(0, 150, 100, 3))
        self.Y_data = np.ndarray(shape=(0))
        dir = './sim_pics'
        self.set_image_paths(dir + '/GREEN/', 
                dir + '/YELLOW/', 
                dir + '/RED/')
        self.EPOCHS = 70
        self.BATCH_SIZE = 128

    # scale images depending on extension/image type
    def load_image(self, image_path):
        
        image = cv2.imread(image_path) 
        image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

        # scale image (0-255)
        if image_path[-4:] == '.png':
            image = image.astype(np.float32)*255
      
        # remove alpha channel if present
        if image.shape[2] == 4:
            b, g, r, a = cv2.split(image)
            image = np.dstack((r,g,b))
    
        return image

    def set_image_paths(self, green_file_path, yellow_file_path, red_file_path):
        
        # add images
        images = []
        labels = []
        green_images=os.listdir(green_file_path) 
        for green_image_path in green_images:
            if type(green_image_path)==type("string"):
                image = self.load_image(green_file_path + green_image_path)
                self.green_images.append(image)
                images.append(image)
                labels.append(0)
    
        yellow_images=os.listdir(yellow_file_path) 
        for yellow_image_path in yellow_images:
            if type(yellow_image_path)==type("string"):
                image = self.load_image(yellow_file_path + yellow_image_path)
                self.yellow_images.append(image)
                images.append(image)
                labels.append(1)
                            
        red_images=os.listdir(red_file_path) 
        for red_image_path in red_images:
            if type(red_image_path)==type("string"):
                image = self.load_image(red_file_path + red_image_path)
                self.red_images.append(image)
                images.append(image)
                labels.append(2)

        self.X_data = np.array(images)
        # Max-Min Normalization: https://www.mathworks.com/matlabcentral/answers/25759-normalizing-data-for-neural-networks
        a = np.max(self.X_data)
        b = np.min(self.X_data)
        ra = 0.9
        rb = 0.1
        pa = (((ra-rb) * (self.X_data - a)) / (b - a)) + rb

        # zero center
        #self.X_train = (self.X_train - self.X_train.mean())
        self.Y_data = np.array(labels)

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
        sigma = 0.1
        Padding='VALID'
        W_lambda = 5.0
    
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
        X_, Y_ = shuffle(self.X_data, self.Y_data)
        X_train, X_val, Y_train, Y_val = train_test_split(X_, Y_, test_size=0.2, random_state=1024)
        
        ### Train model
        x = tf.placeholder(tf.float32, (None, 150, 100, 3))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(Y_train, 3)

        rate = 0.0001

        logits = self.LeNet(tf.cast(X_train, tf.float32))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

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
            
                validation_accuracy = evaluate(X_val, Y_val)
                if self.show_epochs:
                    print("EPOCH {} ...".format(i+1), "Accuracy = {:.6f}".format(validation_accuracy))
                #if i > self.EPOCHS*2/3:
                    #rate = 0.00001
        
            saver.save(sess, './model')
            print("Model saved")
        sess.close()

trainer = TLClassifier_Trainer()
trainer.train()

