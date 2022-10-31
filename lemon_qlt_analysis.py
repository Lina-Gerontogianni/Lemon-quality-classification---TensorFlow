## Importing libraries
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# for reproducibility purposes 
from numpy.random import seed
seed(42)
tf.random.set_seed(42)

## Load the lemon dataset
data_path = '.../kaggle_datasets/lemon_dataset/dataset'

# Define parameters for the loader
batch_size = 32
img_height = 180
img_width = 180


## Define Training set - 80% of the images
training_set = tf.keras.utils.image_dataset_from_directory(
  data_path,
  validation_split=0.2,
  subset="training",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

## Define Validation set - 20% of the images
validation_set = tf.keras.utils.image_dataset_from_directory(
  data_path,
  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Categories' names
categ_names = training_set.class_names
print(categ_names)

## Plot 16 random images from the training set
plt.figure(figsize=(11, 11))
for images, labels in training_set.take(1):
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(categ_names[labels[i]])
    plt.axis("off")
    
## Configure the dataset for performance
autotune = tf.data.AUTOTUNE

training_set = training_set.cache().prefetch(buffer_size=autotune)
validation_set = validation_set.cache().prefetch(buffer_size=autotune)

## Create the CNN 
num_categories = len(categ_names)

cnn = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(20, 3, activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(40, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    ])

## Complile the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

# Model summary
cnn.summary()

## Train the CNN
epochs = 20

cnn_training = cnn.fit(training_set, validation_data = validation_set, 
                       batch_size = 32, epochs = epochs)

## Store and visualise the training results
# accuracy
acc = cnn_training.history['accuracy']
val_acc = cnn_training.history['val_accuracy']
# loss
loss = cnn_training.history['loss']
val_loss = cnn_training.history['val_loss']

epochs_range = range(epochs)

## Accuracy plot
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
