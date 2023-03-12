#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Importing Modules
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential


# Model (LeNET 5 Architecture)
model = Sequential ()

# 1st Convolution Layer
model.add (Conv2D (20, 5, 
                   padding = "same", 
                   input_shape = (255, 255, 3)))
model.add (Activation ('relu'))

# MaxPooling Layer
model.add (MaxPooling2D (pool_size = (2, 2), 
                         strides = (2, 2)))

# 2nd Convolution Layer
model.add (Conv2D (50, 5, padding = "same"))
model.add (Activation ('relu'))

# MaxPooling Layer
model.add (MaxPooling2D (pool_size = (2, 2), 
                         strides = (2, 2)))

# Flatten Layer
model.add (Flatten())

# 1st Dense Layer
model.add (Dense (500))
model.add (Activation ('relu'))

# 2nd Dense Layer
model.add (Dense (39))
model.add (Activation ("softmax"))

# Compiling the model
model.compile(optimizer = 'adam', 
	loss = 'categorical_crossentropy', 
	metrics = ['accuracy'])

# Summary of the Model
model.summary ()

# Saving the model
model.save('/home/tragedy/Image_Processing/Models/LeNet_Untrained.h5')