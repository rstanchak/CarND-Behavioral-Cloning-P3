import csv
import sys
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Lambda, Cropping2D, Convolution2D
from sklearn.model_selection import train_test_split
from random import shuffle
import sklearn


assert(len(sys.argv) > 1)

driving_log = sys.argv[1]
datadir = os.path.dirname(driving_log)

lines = []
with open(driving_log) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

samples = []
measurements = []
steering_adjustment = 0.12


def lrflip(im, x):
    return cv2.flip(im, 1), -x

for line in lines[1:]:  # skip header line
    measurement = float(line[3])
    # center, mirrored
    samples.append((line[0].strip(), False, measurement))
    samples.append((line[0].strip(), True, -measurement))
    # left
    samples.append((line[1].strip(), False, measurement + steering_adjustment))
    samples.append((line[1].strip(), True, -measurement - steering_adjustment))
    # right
    samples.append((line[2].strip(), False, measurement - steering_adjustment))
    samples.append((line[2].strip(), True, -measurement + steering_adjustment))

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, dirname, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = os.path.sep.join([dirname, batch_sample[0]])
                flip = batch_sample[1]
                image = cv2.imread(name)
                if flip:
                    image = cv2.flip(image, 1)
                angle = batch_sample[2]
                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, datadir, batch_size=32)
validation_generator = generator(validation_samples, datadir, batch_size=32)

image = cv2.imread(os.path.sep.join([datadir, samples[0][0]]))
input_shape = image.shape
del image

crop=((64,22),(0,0))
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=crop, input_shape=input_shape))
model.add(Convolution2D(8,5,1,subsample=(2,1),activation='relu'))
model.add(Convolution2D(8,1,5,subsample=(1,2),activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(p=0.25))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=7)

model.save('model.h5')
