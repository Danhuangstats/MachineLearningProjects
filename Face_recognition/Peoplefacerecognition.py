from time import time
import matplotlib.pyplot as plt
import sklearn
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import os

# libraries to support custom function for copying.

import errno
import shutil


def copy(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


src = '/Face_recognition/faces/'
dest = '/Users/danhuang/Desktop/Desktop/Face_cognition/faces/LFW/lfw_home'
copy(src, dest)

print(os.listdir('../Face_cognition/faces/LFW/lfw_home'))

path = '/Users/danhuang/Desktop/Desktop/Face_cognition/faces/LFW/'

imageDataset = fetch_lfw_people(data_home=path, min_faces_per_person=200, download_if_missing=False)

imageDataset.images.shape

X = imageDataset.images
X.shape
num_features = X.shape[1]

Y = imageDataset.target

target_name = imageDataset.target_names
num_class = target_name.shape[0]

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

####

plt.figure()
plt.imshow(X[4])
plt.colorbar()
plt.grid(False)
plt.show()

###

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i], cmap=plt.cm.binary)
    plt.xlabel(target_name[Y[i]])
plt.show()

#### train and test data sets

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=28)

X_train.shape
len(X_train[0])
Y_train.shape
### build model

model = keras.Sequential([
    layers.Flatten(input_shape=(62, 47)),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)
])

## model compiling
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fitting the Model
model.fit(X_train, Y_train, epochs=10)
# Evaluating Accuracy
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Make Predictions
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(X_test)
predictions[0]

np.argmax(predictions[0])


##########


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(target_name[predicted_label],
                                         100 * np.max(predictions_array),
                                         target_name[true_label]),
               color=color)


plot_image(3, predictions_array=predictions, true_label=Y_test, img=X_test)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 5
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], Y_test, X_test)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], Y_test)
plt.tight_layout()
plt.show()

# class lfwPeopleDataset():
#
#     def __init__(self, min_faces_per_person):
#         self.faces = fetch_lfw_people(data_home=path, min_faces_per_person=min_faces_per_person)
#
#     def draw_sample(self):
#         fig, ax = plt.subplots(3, 5)
#         for i, axi in enumerate(ax.flat):
#             axi.imshow(self.faces.images[i], cmap='bone')
#             axi.set(xticks=[], yticks=[],
#                     xlabel=self.faces.target_names[self.faces.target[i]])
#
#     def get_features_labels(self):
#         return self.faces.data, self.faces.target, self.faces.target_names
#
#
# data = lfwPeopleDataset(min_faces_per_person=100)
#
# data.draw_sample()

from torchvision import models

dir(models)

# Alexnet
alexnet = models.alexnet(pretrained=True)

X
plt.imshow(imageDataset.images[25])

with open('faces/pairsDevTrain.txt') as f:
    classes = [line.strip() for line in f.readlines()]

import torch

# Resnet
resnet = models.resnet101(pretrained=True)

from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
