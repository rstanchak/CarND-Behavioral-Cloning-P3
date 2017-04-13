import csv
import sys
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda


assert(len(sys.argv) > 1)

driving_log = sys.argv[1]

lines = []
with open(driving_log) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []


def hflip(im, x):
    return cv2.flip(im, 1), -x

for line in lines:
    source_path = line[0]
    filename = os.path.sep.join([os.path.dirname(driving_log), source_path])
    image = cv2.imread(filename)
    if image is None:
        print("skiping line {}".format(line))
        continue
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

    xy = hflip(image, measurement)
    images.append(xy[0])
    measurements.append(xy[1])

X_train = np.array(images)
y_train = np.array(measurements)

input_shape = X_train.shape[1:]

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
