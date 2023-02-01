#python version = 3.9.12
from scipy.io import loadmat
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gc
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

"""Data loading"""
X_train = np.array(np.load('X_train.npz')['arr_0'], dtype=np.float16)
X_val = np.array(np.load('X_val.npz')['arr_0'], dtype=np.float16)
y_train = np.array(np.load('y_train.npz')['arr_0'])
y_val = np.array(np.load('y_val.npz')['arr_0'])

is_64 = False
eeg_band = 5
se_band_type = 'max'
eeg_channel_new = 64 
se_channel_type = 'avg'
window_time = 1
fcn_input_num = 10
fs_data = 128
window_length = fs_data * window_time 
wav_channel = 1
wav_band = 5
eeg_channel = 64
eeg_s_band = 0
eeg_start = wav_channel * wav_band + eeg_channel * eeg_s_band
eeg_end = eeg_start + eeg_channel * eeg_band
is_se = True
is_se_band = is_se and True  # band attention
is_se_channel = is_se and True 


class mySE(tf.keras.layers.Layer):
    def __init__(self, se_weight_num, se_type, se_fcn_squeeze, conv_num):
        super(mySE, self).__init__()
        se_fcn_num_dict = {'avg': se_weight_num, 'max': se_weight_num, 'mix': se_weight_num*2}
        se_fcn_num = se_fcn_num_dict.get(se_type)
        
        self.se_conv = tf.keras.Sequential([
            tf.keras.layers.Conv3D(filters=1, kernel_size=(1, conv_num, 1), strides=(1, 1, 1)),
            tf.keras.layers.ELU(),
        ])
        
        self.se_fcn = tf.keras.Sequential([
            tf.keras.layers.Dense(units=se_fcn_num, activation='tanh'),
            # tf.keras.activations.tanh(),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(units=se_fcn_squeeze, activation='tanh'),
            # tf.keras.activations.tanh(),
        ])
        
    # def forward(self, se_data, se_type):
    #     se_weight = se_data
    #     se_weight = self.se_conv(se_weight)
    
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.se_band = mySE(eeg_band, se_band_type, 5, eeg_channel_new)
        self. se_channel = mySE(eeg_channel_new, se_channel_type, 8, eeg_band)
        
        self.cnn_conv_eeg = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=eeg_band, kernel_size=(eeg_channel_new, 9), strides=(eeg_channel_new, 1), activation='relu'),
            # tf.keras.activations.relu(),
            tf.keras.layers.GlobalMaxPooling2D(data_format='channels_last'),
        ])
        
        self.cnn_fcn = tf.keras.Sequential([
            tf.keras.layers.Dense(units=fcn_input_num*window_time, activation='sigmoid'),
            # tf.keras.activations.sigmoid(),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(units=fcn_input_num, activation='sigmoid'),
            # tf.keras.activations.sigmoid(),
        ])
        
    def call(self, x):
        # eeg = x[:, :, eeg_start:eeg_end, :]
        eeg = x
        
        # #if frequency attnetion
        # if is_se_band:
        #     eeg = tf.reshape(eeg, shape=(eeg_band, eeg_channel_new, window_length))
        #     eeg = self.se_band(eeg, se_band_type)
        
        #   # channel attention
        # if is_se_channel:
        #     eeg = tf.reshape(eeg, shape=(eeg_band, eeg_channel_new, window_length)).transpose(0, 1)
        #     eeg = self.se_channel(eeg, se_channel_type).transpose(0, 1)

        # normalization
        eeg = tf.reshape(eeg, shape=(1, eeg_band, eeg_channel_new, window_length))
        
        # convolution
        y = eeg
        y = self.cnn_conv_eeg(y)
        y = y.reshape(1, -1)
        
        output = self.cnn_fcn(y)
        
        return output

#initialization
myNet = CNN()
opt = tf.keras.optimizers.SGD(learning_rate=0.001)
myNet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
myNet.fit(X_train, y_train, batch_size=128, epochs=2, validation_data=(X_val, y_val))
print("============================CNN Summary============================")
print(myNet.summary())
        
        
        