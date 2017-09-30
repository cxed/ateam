import os, cv2
import numpy as np
import random as rnd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2

    # scale images depending on extension/image type
def load_image(image_path):
        
    image = cv2.imread(image_path) 
    image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    image = normalize_image(image)
     
        # scale image (0-255)
    if image_path[-4:] == '.png':
        image = image.astype(np.float32)*255
      
        # remove alpha channel if present
    if image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        image = np.dstack((r,g,b))
    
    return image

def normalize_image(image):
    r, g, b = cv2.split(image)
    r = (r - 128)/128
    g = (g - 128)/128
    b = (b - 128)/128
    return cv2.merge((r, g, b))

def set_image_paths():
        
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
            image = load_image(green_file_path + green_image_path)
            images.append(image)
            labels.append(0)
    
    yellow_images=os.listdir(yellow_file_path) 
    for yellow_image_path in yellow_images:
        if type(yellow_image_path)==type("string"):
            image = load_image(yellow_file_path + yellow_image_path)
            images.append(image)
            labels.append(1)
                            
    red_images=os.listdir(red_file_path) 
    for red_image_path in red_images:
        if type(red_image_path)==type("string"):
            image = load_image(red_file_path + red_image_path)
            images.append(image)
            labels.append(2)
        
    X_train = np.array(images)
    Y_train = np.array(labels)

        # load test data
    dir = './test_images'
    green_file_path = dir + '/GREEN/'
    yellow_file_path = dir + '/YELLOW/'
    red_file_path = dir + '/RED/'

        # add test images
    test_images = []
    test_labels = []
    X_val = []
    Y_val = []
    green_images=os.listdir(green_file_path) 
    for green_image_path in green_images:
        if type(green_image_path)==type("string"):
            image = load_image(green_file_path + green_image_path)
            test_images.append(image)
            test_labels.append(0)
    
    yellow_images=os.listdir(yellow_file_path) 
    for yellow_image_path in yellow_images:
        if type(yellow_image_path)==type("string"):
            image = load_image(yellow_file_path + yellow_image_path)
            test_images.append(image)
            test_labels.append(1)
                            
    red_images=os.listdir(red_file_path) 
    for red_image_path in red_images:
        if type(red_image_path)==type("string"):
            image = load_image(red_file_path + red_image_path)
            test_images.append(image)
            test_labels.append(2)

    X_test = np.array(test_images)
    Y_test = np.array(test_labels)

    return X_train, Y_train, X_test, Y_test


debug = False
show_epochs = True

X_train = np.ndarray(shape=(0, 150, 100, 3))
Y_train = np.ndarray(shape=(0))
X_test = np.ndarray(shape=(0, 150, 100, 3))
Y_test = np.ndarray(shape=(0))
X_val = np.ndarray(shape=(0, 150, 100, 3))
Y_val = np.ndarray(shape=(0))

X_train, Y_train, X_test, Y_test = set_image_paths()
x_train, y_train = shuffle(X_train, Y_train)

label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)
assert y_one_hot.shape == (1075, 3), 'y_one_hot is not the correct shape.  It\'s {}, it should be (1075, 3)'.format(y_one_hot.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 100, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_train, y_one_hot, epochs=50, validation_split=0.2, verbose=2)


# Evaluate
y_one_hot_test = label_binarizer.fit_transform(Y_test)
metrics = model.evaluate(X_test, y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

from keras.models import load_model
model.save('./keras_model.h5')
            
dir = './test_images'
green_file_path = dir + '/GREEN/'
yellow_file_path = dir + '/YELLOW/'
red_file_path = dir + '/RED/'
choices = {0: "GREEN", 1: "YELLOW", 2: "RED", 3: "UNKNOWN"}

eval_green = False
eval_yellow = False
eval_red = False
            
num_images = 0
num_incorrect = 0
            
if eval_green:
    green_images=os.listdir(green_file_path)
    for green_image_path in green_images:
        if type(green_image_path)==type("string"):
            image = cv2.imread(green_file_path + green_image_path)
            image = normalize_image(image)
            res = None
            res = cv2.resize(image, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
            image = res.reshape(1, 150, 100, 3)                  
            prediction = model.predict(image, verbose=0)
            result = choices.get(prediction[0], "UNKNOWN")
            #print('Result: Expected GREEN - Detected: ', result)
            num_images += 1
            if result != 'GREEN':
                num_incorrect += 1

if eval_yellow:
    yellow_images=os.listdir(yellow_file_path) 
    for yellow_image_path in yellow_images:
        if type(yellow_image_path)==type("string"):
            image = cv2.imread(yellow_file_path + yellow_image_path)
            image = normalize_image(image)
            res = None
            res = cv2.resize(image, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
            image = res.reshape(1, 150, 100, 3)                     
            prediction = model.predict(image, verbose=0)
            result = choices.get(prediction[0], "UNKNOWN")
            #print('Result: Expected YELLOW - Detected: ', result)
            num_images += 1
            if result != 'YELLOW':
                num_incorrect += 1

if eval_red:
    red_images=os.listdir(red_file_path)
    for red_image_path in red_images:
        if type(red_image_path)==type("string"):
            image = cv2.imread(red_file_path + red_image_path)
            image = normalize_image(image)
            res = None
            res = cv2.resize(image, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
            image = res.reshape(1, 150, 100, 3)                     
            prediction = model.predict(image, verbose=0)
            result = choices.get(prediction[0], "UNKNOWN")
            #print('Result: Expected RED - Detected: ', result)
            num_images += 1
            if result != 'RED':
                num_incorrect += 1

success_rate = 0.0
if num_incorrect > 0:
    success_rate = str(100.0-100.0*(float(num_incorrect)/float(num_images)))
if num_incorrect == 0:
    success_rate = 100

if (eval_green or eval_yellow or eval_red):
    print('No Images: ' + str(num_images) + ' incorrect: ' + str(num_incorrect) + ' success rate: ' + str(success_rate) + ' %')
