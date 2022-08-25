#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hickle as hkl
import numpy as np
import random 

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
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(0)

OPTIM_A = Adam(lr=0.001, beta_1=0.5)
FRAME_LENGTH = 1
FOR_NEW_DATA_LOAD = 1
Training = 0.7
Validation = 0.2
Test = 0.3
_data = os.listdir(data_save_path)
time = 4
height, width = 336, 336
color_channels = 3
batch_size = 10
it = int(10/10)
number_of_hiddenunits = 32

def batch_dispatch(mode):
    num_of_train = int(len(_data) * 0.8) 
    num_of_val = int(len(_data) - num_of_train) 
    train_data = _data[:num_of_train]
    val_data = _data[num_of_val:]
    random.shuffle(_data)
    if mode == "train":
        counter = 0
        while counter<=num_of_train:
            image_seqs = np.empty((0,time,height,width,color_channels))
            labels = np.empty((0,height,width,color_channels))
            print("{}".format(image_seqs.shape))
    
            for i in range(it):
                np_data = np.load(os.path.join(data_save_path, train_data[counter]))
                if len(np_data['arr_0']) == 0:
                    continue

                for i in range(np_data['arr_0'].shape[0] - time):
                    t = np_data['arr_0'][i:i+time, :, :, :].reshape(1, time,height,width,color_channels)
                    t_label = np_data['arr_0'][i + time, :, :, :].reshape(1, height , width, color_channels)
                    image_seqs = np.vstack((image_seqs, t/255))
                    labels = np.vstack((labels, t_label/255))
                            
                counter += 1
                if counter>=num_of_train:
                    counter = 0
                    random.shuffle(train_data)
            return image_seqs, labels
    else:
        counter = 0
        while counter<=num_of_val:
            image_seqs = np.empty((0,time,height,width,color_channels))
            labels = np.empty((0,height,width,color_channels))
            print("{}".format(image_seqs.shape))
    
            for i in range(it):
                np_data = np.load(os.path.join(data_save_path, val_data[counter]))
                if len(np_data['arr_0']) == 0:
                    continue

                for i in range(np_data['arr_0'].shape[0] - time):
                    t = np_data['arr_0'][i:i+time, :, :, :].reshape(1, time,height,width,color_channels)
                    t_label = np_data['arr_0'][i + time, :, :, :].reshape(1, height , width, color_channels)
                    image_seqs = np.vstack((image_seqs, t/255))
                    labels = np.vstack((labels, t_label/255))
                            
                counter += 1
                if counter>=num_of_val:
                    counter = 0
                    random.shuffle(val_data)
            return image_seqs, labels

def load_data():
    print("Load Data...")
    frames = np.zeros((1, 1, height, width, color_channels))    
    for i in range(len(glob.glob(f'{data_save_path}/*'))-1):
        # data = np.load(data_save_path + str(i+1) + '.npy')
        data = np.load(data_save_path + str(i+1) + '.npz')
        
        if len(data['arr_0']) == 0:
            continue
        folder_file_size = data['arr_0'].shape[0]
        iw = data['arr_0'].shape[1]
        ih = data['arr_0'].shape[2]
        ch = data['arr_0'].shape[3]
        tmp_data = data['arr_0'].reshape(int(folder_file_size / FRAME_LENGTH), int(1 * FRAME_LENGTH), iw, ih, ch)
    
        # normalize
        tmp_data = tmp_data.astype('float32') / 255.
        frames = np.concatenate((frames, tmp_data), axis = 0)
        data.close()
    
    # preprocess except first zero tensor
    data_x = frames[1:frames.shape[0]-1, :, :, :, :]
    data_y = frames[2:frames.shape[0], :, :, :, :]
    new = np.concatenate((data_x, data_y), axis=1)

    training_data_num = int(data_x.shape[0] * 0.7)
    test_data_num = data_x.shape[0] - training_data_num

    train_x = new[:training_data_num, 0, ...]
    train_y = new[:training_data_num, 1, ...]

    test_x = new[training_data_num:, 0, ...]
    test_y = new[training_data_num:, 1, ...]


    print("train x data shape : {}".format(train_x.shape))
    print("train y data shape : {}".format(train_y.shape))
    print("test x data shape : {}".format(test_x.shape))
    print("test y data shape : {}".format(test_y.shape))
    return train_x, train_y, test_x, test_y

def get_model():
    input_img = keras.Input(shape=(time,height,width,color_channels))
    seq = Sequential()
    seq.add(layers.TimeDistributed(layers.Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, time, height, width, color_channels)))
    seq.add(layers.LayerNormalization())
    seq.add(layers.TimeDistributed(layers.Conv2D(64, (5, 5), strides=2, padding="same")))
    seq.add(layers.LayerNormalization())
    # # # # #
    seq.add(layers.ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True)) # temporal encoder
    seq.add(layers.LayerNormalization())
    seq.add(layers.ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True)) # bottleneck
    seq.add(layers.LayerNormalization())
    seq.add(layers.ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True)) # temporal decoder
    seq.add(layers.LayerNormalization())
    # # # # #
    seq.add(layers.TimeDistributed(layers.Conv2DTranspose(64, (5, 5), strides=2, padding="same")))
    seq.add(layers.LayerNormalization())
    seq.add(layers.TimeDistributed(layers.Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
    seq.add(layers.LayerNormalization())
    # seq.add(layers.TimeDistributed(layers.Conv2D(3, (11, 11), activation="sigmoid", padding="same")))
    seq.add(layers.Conv2D(3, (11, 11), activation="sigmoid", padding="same"))
    seq.summary()
    seq.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))
   
    return seq

def new_encoder_model():
    input_img = keras.Input(shape=(time,height,width,color_channels))
    conv_model = layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu') )(input_img)
    conv_model = layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
    conv_model = layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME',strides=(2,2)))(conv_model)

    conv_model = layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu') )(conv_model)
    conv_model = layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
    conv_model = layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME',strides=(2,2)))(conv_model)

    conv_model = layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu') )(conv_model)
    conv_model = layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu') )(conv_model)
    conv_model = layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
    conv_model = layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME',strides=(2,2)))(conv_model)

    conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu') )(conv_model)
    conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu') )(conv_model)
    conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu') )(conv_model)
    conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(conv_model)
    conv_model = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='SAME',strides=(2,2)))(conv_model)

    #embedded
    image_features = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(conv_model)

    lstm_network = tf.keras.layers.LSTM(number_of_hiddenunits, return_sequences=True,dropout=0.5,recurrent_dropout=0.5)(image_features)
    lstm_network = tf.keras.layers.LSTM(number_of_hiddenunits, return_sequences=False,dropout=0.5,recurrent_dropout=0.5)(lstm_network)
    # lstm_network = tf.keras.layers.Dense(1024,activation='relu')(lstm_network)
    # lstm_network = tf.keras.layers.BatchNormalization()(lstm_network)
    # lstm_network = tf.keras.layers.Dropout(0.5)(lstm_network)
    # lstm_network = tf.keras.layers.Dense(512,activation='relu')(lstm_network)
    # lstm_network = tf.keras.layers.Dropout(0.5)(lstm_network)
    # lstm_network = tf.keras.layers.Dense(64,activation='relu')(lstm_network)
    # lstm_network = tf.keras.layers.Dropout(0.5)(lstm_network)    
    # lstm_network = tf.keras.layers.Dense(n_classes,activation='softmax')(lstm_network)

    model = keras.Model(inputs = input_img, outputs = lstm_network)

    return model

def encoder_model(): 
    input_img = keras.Input(shape=(time, 336, 336, 3))
    # number of parms = (size of kernels**2) * number of channels of the input image * number of kernels

    conv000 = layers.Conv3D(filters = 512,
                        kernel_size = (1, 10, 10), 
                        activation='relu', 
                        padding='same')(input_img)
    x = layers.TimeDistributed(layers.BatchNormalization())(conv000)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    out000 = layers.TimeDistributed(layers.Dropout(0.5))(x)

    conv00 = layers.Conv3D(filters = 512,
                        kernel_size = (1, 10, 10), 
                        activation='relu', 
                        padding='same')(out000)
    x = layers.TimeDistributed(layers.BatchNormalization())(conv00)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    out00 = layers.TimeDistributed(layers.Dropout(0.5))(x)

    conv0 = layers.Conv3D(filters = 128,
                        kernel_size = (1, 10, 10), 
                        activation='relu', 
                        padding='same')(out00)
    x = layers.TimeDistributed(layers.BatchNormalization())(conv0)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    out0 = layers.TimeDistributed(layers.Dropout(0.5))(x)
    # out0 = layers.MaxPool3D((1, 2, 2), padding='same')(x)

    conv1 = layers.Conv3D(filters = 128,
                        kernel_size = (1, 10, 10), 
                        activation='relu', 
                        padding='same')(out0)
    x = layers.TimeDistributed(layers.BatchNormalization())(conv1)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.5))(x)
    out1 = layers.MaxPool3D((1, 2, 2), padding='same')(x)

    conv2 = layers.Conv3D(filters = 64,
                        kernel_size = (1, 10, 10),
                        activation='relu', 
                        padding='same')(out1)
    x = layers.TimeDistributed(layers.BatchNormalization())(conv2)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.5))(x)
    out2 = layers.MaxPool3D((1, 2, 2), padding='same')(x)

    conv3 = layers.Conv3D(filters = 64, 
                        kernel_size = (1, 10, 10), 
                        activation='relu', 
                        padding='same')(out2)
    x = layers.TimeDistributed(layers.BatchNormalization())(conv3)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.5))(x)
    out3 = layers.MaxPool3D((1, 2, 2), padding='same')(x)

    conv4 = layers.Conv3D(filters = 32, 
                        kernel_size = (1, 10, 10), 
                        activation='relu', 
                        padding='same')(out3)
    x = layers.TimeDistributed(layers.BatchNormalization())(conv4)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.5))(x)
    out4 = layers.MaxPool3D((1, 2, 2), padding='same')(x)

    conv5 = layers.Conv3D(filters = 32, 
                        kernel_size = (1, 10, 10), 
                        activation='relu', 
                        padding='same')(out4)
    x = layers.TimeDistributed(layers.BatchNormalization())(conv5)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    encoded = layers.TimeDistributed(layers.Dropout(0.5))(x)
    # encoded = layers.MaxPool3D((1, 2, 2), padding='same')(x)

    model = keras.Model(inputs = input_img, outputs = encoded)

    return model

def decoder_model():
    inputs = keras.Input(shape=(time, 21, 21, 32))

    convlstm_1 = layers.ConvLSTM2D(filters = 32,
                                kernel_size= (10, 10), 
                                padding='same', 
                                return_sequences=True, 
                                recurrent_dropout=0.2)(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(convlstm_1)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.5))(x)
    out1 = layers.UpSampling3D(size = (1, 2, 2))(x)

    convlstm_2 = layers.ConvLSTM2D(filters = 32,
                                kernel_size= (10, 10), 
                                padding='same', 
                                return_sequences=True, 
                                recurrent_dropout=0.2)(out1)
    x = layers.TimeDistributed(layers.BatchNormalization())(convlstm_2)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.5))(x)
    out2 = layers.UpSampling3D(size = (1, 2, 2))(x)

    convlstm_3 = layers.ConvLSTM2D(filters = 64,
                                kernel_size= (10, 10), 
                                padding='same', 
                                return_sequences=True, 
                                recurrent_dropout=0.2)(out2)
    x = layers.TimeDistributed(layers.BatchNormalization())(convlstm_3)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.5))(x)
    out3 = layers.UpSampling3D(size = (1, 2, 2))(x)

    convlstm_3 = layers.ConvLSTM2D(filters = 64,
                                kernel_size= (10, 10), 
                                padding='same', 
                                return_sequences=True, 
                                recurrent_dropout=0.2)(out3)
    x = layers.TimeDistributed(layers.BatchNormalization())(convlstm_3)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.5))(x)
    out4 = layers.UpSampling3D(size = (1, 2, 2))(x)

    convlstm_4 = layers.ConvLSTM2D(filters = 128,
                                kernel_size= (10, 10), 
                                padding='same', 
                                return_sequences=True, 
                                recurrent_dropout=0.2)(out4)
    x = layers.TimeDistributed(layers.BatchNormalization())(convlstm_4)
    x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.2))(x)
    out5 = layers.TimeDistributed(layers.Dropout(0.5))(x)
    # out5 = layers.UpSampling3D(size = (1, 2, 2))(x)

    convlstm_5 = layers.ConvLSTM2D(filters = 3,
                                kernel_size= (10, 10), 
                                padding='same', 
                                return_sequences=True, 
                                recurrent_dropout=0.2)(out5)

    predictions = layers.TimeDistributed(layers.Activation('sigmoid'))(convlstm_5)

    model = keras.Model(inputs = inputs, outputs = predictions)

    return model

def autoencoder_model(encoder, decoder):
    model = Sequential()
    model.add(encoder)
    model.add(decoder)
    return model

def new_train():
    print ("Creating models...")
    # encoder = new_encoder_model()
    # encoder.summary()
    encoder = encoder_model()
    encoder.summary()
    decoder = decoder_model()
    decoder.summary()
    # autoencoder = autoencoder_model(encoder, decoder)
    # autoencoder.compile(loss="binary_crossentropy", optimizer=OPTIM_A)

    # autoencoder.fit(batch_dispatch("train"),
    #                 epochs=100,
    #                 batch_size= batch_size,
    #                 shuffle=True,
    #                 validation_data = batch_dispatch("val"),
    #                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=1)])

    # autoencoder.save('autoencoder_v2.h5')
     # # # Load Model
    # # new_model = tf.keras.models.load_model('autoencoder_v1.h5')

    # # Prediction ...
    # predicted_frames = np.zeros((1,336,336,3)) 
    # for i in range(10):
    #     img = x_train[i, :, :, :, :]
    #     img = np.expand_dims(img, axis=0)
    #     decoded_imgs = autoencoder.predict(img)
    #     predicted_frame = np.reshape(decoded_imgs, (1, 336, 336, 3))
    #     predicted_frames = np.concatenate((predicted_frames, predicted_frame), axis=0)

    # n = 10
    # plt.figure(figsize=(20, 4))
    # for i in range(1, n + 1):
    #     # Display original
    #     ax = plt.subplot(2, n, i)
    #     img = x_train[i, :, :, :, :]
    #     img = np.expand_dims(img, axis=0)
    #     plt.imshow(np.reshape(img, (336, 336, 3)))
    #     # plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    #     # Display reconstruction
    #     ax = plt.subplot(2, n, i + n)
    #     plt.imshow(predicted_frames[i])
    #     # plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.show()

def train(x_train, y_train, x_test, y_test):
    # Build the Spatio-temporal Autoencoder
    print ("Creating models...")
    encoder = encoder_model()
    encoder.summary()
    # decoder = decoder_model()
    # decoder.summary()
    # autoencoder = autoencoder_model(encoder, decoder)
    # autoencoder.compile(loss="binary_crossentropy", optimizer=OPTIM_A)
    
    # plot_model(autoencoder, to_file='model.png')
    # autoencoder.summary()

    # Start Training....
    # autoencoder.fit(x_train, y_train,
    #                 epochs=100,
    #                 batch_size= 4,
    #                 shuffle=True,
    #                 validation_data=(x_test, y_test),
    #                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=1)])

    

    # # Save Model
    # autoencoder.save('autoencoder_v2.h5')
    # # # Load Model
    # # new_model = tf.keras.models.load_model('autoencoder_v1.h5')

    # # Prediction ...
    # predicted_frames = np.zeros((1,336,336,3)) 
    # for i in range(10):
    #     img = x_train[i, :, :, :, :]
    #     img = np.expand_dims(img, axis=0)
    #     decoded_imgs = autoencoder.predict(img)
    #     predicted_frame = np.reshape(decoded_imgs, (1, 336, 336, 3))
    #     predicted_frames = np.concatenate((predicted_frames, predicted_frame), axis=0)

    # n = 10
    # plt.figure(figsize=(20, 4))
    # for i in range(1, n + 1):
    #     # Display original
    #     ax = plt.subplot(2, n, i)
    #     img = x_train[i, :, :, :, :]
    #     img = np.expand_dims(img, axis=0)
    #     plt.imshow(np.reshape(img, (336, 336, 3)))
    #     # plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    #     # Display reconstruction
    #     ax = plt.subplot(2, n, i + n)
    #     plt.imshow(predicted_frames[i])
    #     # plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.show()

if __name__ == "__main__":

    # image_seqs = batch_dispatch("train")

    # if FOR_NEW_DATA_LOAD:
        # train_x, train_y, test_x, test_y = load_data()
        # np.savez(data_save_path + 'data_1.npz', train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)

    # data = np.load(data_save_path + 'data_1.npz')
    # x_train = data['train_x']
    # y_train = data['train_y']
    # x_test = data['test_x']
    # y_test = data['test_y']

    # x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2], x_train.shape[3])
    # y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1], y_train.shape[2], y_train.shape[3])
    # x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2], x_test.shape[3])
    # y_test = y_test.reshape(y_test.shape[0], 1, y_test.shape[1], y_test.shape[2], y_test.shape[3])
    # x_train = x_train[1:]
    # y_train = y_train[1:]
    # print("traning data shape : {}".format(x_train.shape))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Use Multi GPU
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # train(x_train, y_train, x_test, y_test)
        new_train()
        # get_model()