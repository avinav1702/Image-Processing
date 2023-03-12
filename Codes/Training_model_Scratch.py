#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Importing modules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf

# Importing Data
data_dir = '/home/tragedy/Image_Processing/Dataset/Unaugmented/train/'

train_datagen = ImageDataGenerator (rescale = 1./255, 
	validation_split = 0.2)

image_size = (255, 255)
batch_size = 10

# Creating Training and Validation batches
train_batches = train_datagen.flow_from_directory (
	data_dir, 
	target_size = image_size, 
	batch_size = batch_size, 
	class_mode = 'categorical', 
	subset = 'training')

validation_batches = train_datagen.flow_from_directory (
	data_dir, 
	target_size = image_size, 
	batch_size = batch_size, 
	class_mode = 'categorical', 
	subset = 'validation')

# Creating a Tensorboard Callback
tb_callback = tf.keras.callbacks.TensorBoard (log_dir = "/home/tragedy/Image_Processing/Logs/", 
                                              histogram_freq = 1)

# Loading the model
model = load_model ('/home/tragedy/Image_Processing/Models/AlexNet_Untrained.h5')

# Summary of the model
model.summary ()

# Setting up Callback
call_back = [tb_callback]

# Training the model
history = model.fit (train_batches, 
                     steps_per_epoch = len (train_batches) // 4, 
                     epochs = 5, 
                     validation_data = validation_batches, 
                     validation_steps = len (validation_batches) // 4, 
                     verbose = 1, 
                     callbacks = call_back)

# Plotting Figure
fig = plt.figure (figsize = (16, 4))
ax = fig.add_subplot (121)
ax.plot (history.history ["val_loss"])
ax.set_title ("validation_loss")
ax.set_xlabel ("epochs")

ax2 = fig.add_subplot (122)
ax2.plot (history.history ["val_accuracy"])
ax2.set_title ("validation_accuracy")
ax2.set_xlabel ("epochs")
ax2.set_ylim (0, 1)

plt.show()
