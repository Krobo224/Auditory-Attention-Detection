#python version = 3.9.12
from scipy.io import loadmat
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gc
import math
from tensorflow import keras
from tensorflow.keras import layers

X_train = np.array(np.load('X_train.npz',allow_pickle=True)['arr_0'], dtype=np.float16)
X_val = np.array(np.load('X_val.npz', allow_pickle=True)['arr_0'], dtype=np.float16)
y_train = np.array(np.load('y_train.npz',allow_pickle=True)['arr_0'])
y_val = np.array(np.load('y_val.npz', allow_pickle=True)['arr_0'])

print("X_train shape", X_train.shape)
print("Y_train shape", y_train.shape)
print("X_val shape", X_val.shape)
print("Y_val shape", y_val.shape)

verbose, epochs, batch_size = 0, 200, 32
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], 1

model = keras.Sequential()
model.add(layers.Conv1D(filters=64, kernel_size=3,
          input_shape=(n_timesteps, n_features)))
model.add(layers.Conv1D(filters=64, kernel_size=3))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='Adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

accuracy1 = model.evaluate(X_val, y_val, verbose=2)
accuracy2 = model.evaluate(X_train, y_train, verbose=2)

print("Accuracy I got on val", accuracy1)
print("Accuracy I got on train", accuracy2)

a = dict(zip(model.metrics_names, accuracy1))
print(a)
print(model.summary())
