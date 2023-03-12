#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Importing Modules
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import concatenate
from keras.models import Model
from keras.layers import Input

# Convolution Module
def conv_module(input, No_of_filters, filtersizeX, filtersizeY, stride, chanDim, padding="same"):
  input = Conv2D(No_of_filters, (filtersizeX, filtersizeY), strides=stride, padding=padding)(input)
  input = BatchNormalization(axis=chanDim)(input)
  input = Activation("relu")(input)
  return input

# Inception Module
def inception_module(input, numK1x1, numK3x3, numk5x5, numPoolProj, chanDim):
  conv_1x1 = conv_module(input, numK1x1, 1, 1,(1, 1), chanDim)
  conv_3x3 = conv_module(input, numK3x3, 3, 3,(1, 1), chanDim)
  conv_5x5 = conv_module(input, numk5x5, 5, 5,(1, 1), chanDim)
  pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
  pool_proj = Conv2D(numPoolProj, (1, 1), padding='same', activation='relu')(pool_proj)
  input = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=chanDim)
  return input

def downsample_module(input, No_of_filters, chanDim):
  conv_3x3=conv_module(input, No_of_filters, 3, 3, (2, 2), chanDim, padding="valid")
  pool = MaxPooling2D((3, 3), strides=(2, 2))(input)
  input = concatenate([conv_3x3,pool], axis=chanDim)
  return input

def GoogleNet(width, height, depth, classes):
  inputShape = (height, width, depth)
  chanDim = -1

  # (Step 1) Define the model input
  inputs = Input(shape = inputShape)

  # First CONV module
  x = conv_module(inputs, 96, 3, 3, (1, 1), chanDim)

  # (Step 2) Two Inception modules followed by a downsample module
  x = inception_module(x, 32, 32, 32, 32, chanDim)
  x = inception_module(x, 32, 48, 48, 32, chanDim)
  x = downsample_module(x, 80, chanDim)
  
  # (Step 3) Five Inception modules followed by a downsample module
  x = inception_module(x, 112, 48, 32, 48, chanDim)
  x = inception_module(x, 96, 64, 32, 32, chanDim)
  x = inception_module(x, 80, 80, 32, 32, chanDim)
  x = inception_module(x, 48, 96, 32, 32, chanDim)
  x = inception_module(x, 112, 48, 32, 48, chanDim)
  x = downsample_module(x, 96, chanDim)

  # (Step 4) Two Inception modules followed
  x = inception_module(x, 176, 160,96, 96, chanDim)
  x = inception_module(x, 176, 160, 96, 96, chanDim)
  
  # Global POOL and dropout
  x = AveragePooling2D((7, 7))(x)
  x = Dropout(0.5)(x)

  # (Step 5) Softmax classifier
  x = Flatten()(x)
  x = Dense(classes)(x)
  x = Activation("softmax")(x)

  # Create the model
  model = Model(inputs, x, name="googlenet")
  return model

# Function Calling
model = GoogleNet(width=255, height=255, depth=3, classes = 39)

# Compiling the Model
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

# Summary of the Model
model.summary()

# Saving the model
model.save('/home/tragedy/Image_Processing/Models/GoogleNet_Untrained.h5')