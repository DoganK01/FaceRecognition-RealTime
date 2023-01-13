import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import models, layers
import cv2
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle
import glob

path = "Images"

myList = os.listdir(path)
no0fClasses = len(myList)

images = []
classNo = []

for i in range(no0fClasses):
    myImageList = os.listdir(path + "\\" + str(i))
    for j in myImageList:
        img = cv2.imread(path + "\\" + str(i) + "\\" + j)
        img = cv2.resize(img, (300, 300))
        images.append(img)
        classNo.append(i)

images = np.array(images)
classNo = np.array(classNo)

x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=0.5, random_state=42)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.5, random_state=42)


# fig, axes = plt.subplots(3,1,figsize(7,7))
# fig.subplots_adjust(hspace = 0.5)
# sns.countplot(y_train, ax= axes[0])
# axes[0].set_title("y_train")


# sns.countplot(y_test, ax=axes[1])
# axes[1].set_title(y_test)


# sns.countplot(y_validation, ax=axes[2])
# axes[2].set_title(y_validation)


def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255

    return img


x_train = np.array(list(map(preProcess, x_train)))
x_test = np.array(list(map(preProcess, x_test)))
x_validation = np.array(list(map(preProcess, x_validation)))

x_train = x_train.reshape(-1, 300, 300, 1)
x_test = x_test.reshape(-1, 300, 300, 1)
x_validation = x_validation.reshape(-1, 300, 300, 1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1,
                             rotation_range=10)

dataGen.fit(x_train)

y_train = to_categorical(y_train, no0fClasses)
y_test = to_categorical(y_test, no0fClasses)
y_validation = to_categorical(y_validation, no0fClasses)

model = Sequential()

model.add(Conv2D(32, (3,3), padding = 'same', kernel_initializer="he_normal", input_shape = (img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding = 'same', kernel_initializer="he_normal", input_shape = (img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


model.add(Conv2D(64, (3,3), padding = 'same', kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding = 'same', kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))



model.add(Conv2D(128, (3,3), padding = 'same', kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding = 'same', kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))




model.add(Conv2D(256, (3,3), padding = 'same', kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding = 'same', kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))



model.add(Conv2D(512, (3,3), padding = 'same', kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), padding = 'same', kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))



model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))



model.add(Dense(no0fClasses, kernel_initializer='he_normal'))
model.add(Activation('sigmoid'))


# model = Sequential()

# model.add(Conv2D(input_shape=(300, 300, 1), filters=8, kernel_size=(5, 5), activation="relu", padding="same"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(units=32, activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(units=no0fClasses, activation="sigmoid"))


model.compile(loss="binary_crossentropy", optimizer=("Adam"), metrics=["accuracy"])

batch_size = 10

hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=batch_size),
                           validation_data=(x_validation, y_validation),
                           epochs=5, steps_per_epoch=x_train.shape[0] // batch_size, shuffle=1)

model_json = model.to_json()
with open("model.json", "w") as file:
    file.write(model_json)

model.save_weights("weights.h5")
