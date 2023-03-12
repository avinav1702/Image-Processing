#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Importing Modules
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Sequential

# Model (AlexNet architecture)
model = Sequential()
# 1st convolution layer
model.add(Conv2D(filters=96, input_shape=(255, 255, 3),
                 kernel_size=(11, 11), strides=(4, 4), padding='valid'))
model.add(Activation('relu'))
# Max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                       padding='valid'))

# Batch Normalization
model.add(BatchNormalization())

# 2nd Convolution layer
model.add(Conv2D(filters=256, kernel_size=(11, 11),
                 strides=(4, 4), padding='valid'))
model.add(Activation('relu'))
# Max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                       padding='valid'))

# Batch Normalization
model.add(BatchNormalization())

# 3rd Convolution layer
model.add(Conv2D(filters=384, kernel_size=(3, 3),
                 strides=(2, 2), padding='valid'))
model.add(Activation('relu'))
# Batch Normalization
model.add(BatchNormalization())

# 4th Convolution layer
model.add(Conv2D(filters=384, kernel_size=(1, 1),
                 strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalization
model.add(BatchNormalization())

# 5th Convolution layer
model.add(Conv2D(filters=256, kernel_size=(1, 1),
                 strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Max pooling layer
model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1),
                       padding='valid'))

# Batch Normalization
model.add(BatchNormalization())

# Flattening layer
model.add(Flatten())

# 1st Dense layer
model.add(Dense(4096, input_shape=(225 * 225 * 3, )))
model.add(Activation('relu'))

# Dropout layer to avoid overfitting
model.add(Dropout(0.5))

# Batch Normalization
model.add(BatchNormalization())

# 2nd Dense layer
model.add(Dense(4096))
model.add(Activation('relu'))

# Dropout layer
model.add(Dropout(0.5))

# Batch Normalization
model.add(BatchNormalization())

# Output Softmax layer
model.add(Dense(39))
model.add(Activation('softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Summary of the model
model.summary()

# Saving the model
model.save('/home/tragedy/Image_Processing/Models/AlexNet_Untrained.h5')
