# Import useful packeges:
import numpy as np
import cv2, os, pickle, glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from sklearn.preprocessing import LabelBinarizer

EPOCHS = 30

def load_image(image_path):
    
    image = cv2.imread(image_path) 
    image = cv2.resize(image,(32,32), interpolation = cv2.INTER_CUBIC)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #image = normalize_image(image)

    '''
    # scale image (0-255)
    if image_path[-4:] == '.png':
        image = image.astype(np.float32)*255
    '''

    # remove alpha channel if present
    if image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        image = np.dstack((r,g,b))

    return image

tr_red_images = glob.glob('train_images/RED/*')
tr_yellow_images = glob.glob('train_images/YELLOW/*')
tr_green_images = glob.glob('train_images/GREEN/*')

# print out how many images there were in the data
print('# red images: ', len(tr_red_images))
print('# yellow images: ', len(tr_yellow_images))
print('# green images: ', len(tr_green_images))

images = []
labels = []

# Set how many images you want to use on training and testing
# should replace this DataGenerator because this is clumsy and I am neglecting lots of data
# This is for each colour
train_num = 10000
test_num = 10000

#randomly select images from red set and remove them so that they cannot be selected again
for i in range(train_num):
    choice = random.randint(0, len(tr_red_images) - 1)
    image = load_image(tr_red_images[choice])
    images.append(image)
    labels.append(2)
    del tr_red_images[choice]
    
# yellow
for i in range(train_num):
    choice = random.randint(0, len(tr_yellow_images) - 1)
    image = load_image(tr_yellow_images[choice])
    images.append(image)
    labels.append(1)
    del tr_yellow_images[choice]
    
# green
for i in range(train_num):
    choice = random.randint(0, len(tr_green_images) - 1)
    image = load_image(tr_green_images[choice])
    images.append(image)
    labels.append(0)
    del tr_green_images[choice]
    
X_train = np.array(images)
y_train = np.array(labels)

print('Shape of X_train: ', X_train.shape)
print('Shape of y_train: ', y_train.shape)

# Now continue to work on the test set, ignoring images that were previously added    
    
images = []
labels = []

# Test set    
# red
for i in range(test_num):
    choice = random.randint(0, len(tr_red_images) - 1)
    image = load_image(tr_red_images[choice])
    images.append(image)
    labels.append(2)
    del tr_red_images[choice]
    
# yellow
for i in range(test_num):
    choice = random.randint(0, len(tr_yellow_images) - 1)
    image = load_image(tr_yellow_images[choice])
    images.append(image)
    labels.append(1)
    del tr_yellow_images[choice]
    
# green
for i in range(test_num):
    choice = random.randint(0, len(tr_green_images) - 1)
    image = load_image(tr_green_images[choice])
    images.append(image)
    labels.append(0)
    del tr_green_images[choice]
    
X_test = np.array(images)
y_test = np.array(labels)

print('Shape of X_test: ', X_test.shape)
print('Shape of y_test: ', y_test.shape)

X_train, y_train = shuffle(X_train, y_train)
#X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=1024)
X_test, y_test = shuffle(X_test, y_test)
print('Done with shuffle and split')

label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_one_hot, epochs=EPOCHS, validation_split=0.2, verbose=2)

# Evaluate
y_one_hot_test = label_binarizer.fit_transform(y_test)
metrics = model.evaluate(X_test, y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

from keras.models import load_model
model.save('./keras_model_new.h5')

# check the length of the remaining images for just to be sure that we are in fact removing images
print('# red images: ', len(tr_red_images))
print('# yellow images: ', len(tr_yellow_images))
print('# green images: ', len(tr_green_images))

# Load the model and check the probabilities
# pick a red, yellow, green
r_img = load_image(tr_red_images[0])
y_img = load_image(tr_yellow_images[0])
g_img = load_image(tr_green_images[0])

r_img = r_img.reshape(1, 32, 32, 3) 
y_prob = model.predict(r_img) 
y_classes = y_prob.argmax(axis=-1)
print(y_classes)

y_img = y_img.reshape(1, 32, 32, 3) 
y_prob = model.predict(y_img) 
y_classes = y_prob.argmax(axis=-1)
print(y_classes)

g_img = g_img.reshape(1, 32, 32, 3) 
y_prob = model.predict(g_img) 
y_classes = y_prob.argmax(axis=-1)
print(y_classes)

choices = {0: "GREEN", 1: "YELLOW", 2: "RED", 3: "UNKNOWN"}

num_images = 0
num_incorrect = 0

#green_length = len(tr_green_images)
#yellow_length = len(tr_yellow_images)
#red_length = len(tr_red_images)

green_length = 5000
yellow_length = 5000
red_length = 5000

for i in range(green_length):
    img = load_image(tr_green_images[i])
    res = None    
    image = img.reshape(1, 32, 32, 3) 
    classification = model.predict_classes(image, verbose=0)[0]
    result = choices.get(classification, "UNKNOWN")
    num_images += 1
    if result != 'GREEN':
        num_incorrect += 1

success_rate = 0.0
if num_incorrect > 0:
    success_rate = str(100.0-100.0*(float(num_incorrect)/float(num_images)))
if num_incorrect == 0:
    success_rate = 100

print('No Images: ' + str(num_images) + ' incorrect: ' + str(num_incorrect) + ' success rate: ' + str(success_rate) + ' %')