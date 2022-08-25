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
# from custom_layers import AttnLossLayer

import argparse
import math
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

VIDEO_LENGTH = 10

def encoder_model():
    model = Sequential()

    # 10x128x128
    model.add(layers.Conv3D(filters=128,
                     strides=(1, 4, 4),
                     kernel_size=(3, 11, 11),
                     padding='same',
                     input_shape=(int(VIDEO_LENGTH/2), 128, 128, 3)))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.LeakyReLU(alpha=0.2)))
    model.add(layers.TimeDistributed(layers.Dropout(0.5)))

    # 10x32x32
    model.add(layers.Conv3D(filters=64,
                     strides=(1, 2, 2),
                     kernel_size=(3, 5, 5),
                     padding='same'))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.LeakyReLU(alpha=0.2)))
    model.add(layers.TimeDistributed(layers.Dropout(0.5)))

    # 10x16x16
    model.add(layers.Conv3D(filters=64,
                     strides=(1, 1, 1),
                     kernel_size=(3, 3, 3),
                     padding='same'))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.LeakyReLU(alpha=0.2)))
    model.add(layers.TimeDistributed(layers.Dropout(0.5)))

    return model


def decoder_model():
    inputs = keras.Input(shape=(int(VIDEO_LENGTH/2), 16, 16, 64))

    # 10x16x16
    convlstm_1 = layers.ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(convlstm_1)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    out_1 = layers.TimeDistributed(layers.Dropout(0.5))(x)

    flat_1 = layers.TimeDistributed(layers.Flatten())(out_1)
    aclstm_1 = layers.GRU(units=16 * 16,
                   recurrent_dropout=0.2,
                   return_sequences=True)(flat_1)
    x = layers.TimeDistributed(layers.BatchNormalization())(aclstm_1)
    dense_1 = layers.TimeDistributed(layers.Dense(units=16 * 16, activation='softmax'))(x)
    a1_reshape = layers.Reshape(target_shape=(int(VIDEO_LENGTH/2), 16, 16, 1))(dense_1)
    a1 = AttnLossLayer()(a1_reshape)
    dot_1 = layers.multiply([out_1, a1])

    convlstm_2 = layers.ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(dot_1)
    x = layers.TimeDistributed(layers.BatchNormalization())(convlstm_2)
    h_2 = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    out_2 = layers.UpSampling3D(size=(1, 2, 2))(h_2)

    skip_upsamp_1 = layers.UpSampling3D(size=(1, 2, 2))(out_1)
    res_1 = layers.concatenate([out_2, skip_upsamp_1])

    # 10x32x32
    convlstm_3 = layers.ConvLSTM2D(filters=128,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(res_1)
    x = layers.TimeDistributed(layers.BatchNormalization())(convlstm_3)
    h_3 = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    out_3 = layers.UpSampling3D(size=(1, 2, 2))(h_3)

    skip_upsamp_2 = layers.UpSampling3D(size=(1, 2, 2))(out_2)
    res_2 = layers.concatenate([out_3, skip_upsamp_2])

    # 10x64x64
    convlstm_4 = layers.ConvLSTM2D(filters=32,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(res_2)
    x = layers.TimeDistributed(layers.BatchNormalization())(convlstm_4)
    h_4 = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    out_4 = layers.UpSampling3D(size=(1, 2, 2))(h_4)

    # 10x128x128
    convlstm_5 = layers.ConvLSTM2D(filters=3,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            return_sequences=True,
                            recurrent_dropout=0.2)(out_4)
    predictions = layers.TimeDistributed(layers.Activation('tanh'))(convlstm_5)

    model = keras.Model(inputs=inputs, outputs=predictions)

    return model

def train():
    print ("Creating models...")
    encoder = encoder_model()
    encoder.summary()
    decoder = decoder_model()
    decoder.summary()

if __name__ == "__main__":
    train()