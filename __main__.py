#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hickle as hkl
import numpy as np

np.random.seed(9 ** 10)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from config import * 
from sys import stdout

import argparse
import math
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data = np.load(data_save_path + '35.npy')
    # data = tf.image.rgb_to_grayscale(data)
    print("{}".format(data.shape))

    return data


def train():

    data = load_data()
    # input image shape
    input_img = keras.Input(shape=(336, 336, 3))
    # number of parms = (size of kernels**2) * number of channels of the input image * number of kernels
    x = layers.Conv2D(512, (10, 10), activation='relu', padding='same')(input_img) # number of kernels, size of kernels
    x = layers.MaxPooling2D((4, 4), padding='same')(x)
    x = layers.Conv2D(216, (10, 10), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((4, 4), padding='same')(x)
    x = layers.Conv2D(128, (10, 10), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)
    x = layers.Conv2D(32, (10, 10), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((3, 3), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = layers.Conv2D(32, (10, 10), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((3, 3))(x)
    x = layers.Conv2D(128, (10, 10), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((4, 4))(x)
    x = layers.Conv2D(216, (10, 10), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((4, 4))(x)
    decoded = layers.Conv2D(3, (10, 10), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # plot_model(autoencoder, to_file='model.png')
    autoencoder.summary()

    # (x_train, _), (x_test, _) = mnist.load_data()
    
    x_train = data.astype('float32') / 255.
    x_test = data.astype('float32') / 255.
    # x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    # x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    
    # autoencoder.fit(x_train, x_train,
    #                 epochs=50,
    #                 batch_size=8,
    #                 shuffle=True,
    #                 validation_data=(x_test, x_test),
    #                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    # decoded_imgs = autoencoder.predict(x_test)

    # n = 10
    # plt.figure(figsize=(20, 4))
    # for i in range(1, n + 1):
    #     # Display original
    #     ax = plt.subplot(2, n, i)
    #     plt.imshow(x_test[i])
    #     # plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    #     # Display reconstruction
    #     ax = plt.subplot(2, n, i + n)
    #     plt.imshow(decoded_imgs[i])
    #     # plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.show()

if __name__ == "__main__":
    train()