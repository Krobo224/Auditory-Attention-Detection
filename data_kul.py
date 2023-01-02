#python version = 3.9.12
from scipy.io import loadmat
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gc
import math
import matplotlib.pyplot as plt
import random

# data = loadmat('preprocessed_data/S1.mat')
# print(type(data['preproc_trials'][0][1]['experiment'][0][0][0][0]))

#Data loader :
sampling_rate = 128 #Hz
window = 1 #sec 1, 0.1, 2
overlapp = 0.5 # 0.8
window_size = math.ceil(sampling_rate*window)
per_window_overlapp = 1
overlap_status = True 

if overlap_status == True:
    rem = 1 - overlapp
    per_window_overlapp = int(rem*window_size) # for getting to the first index of next window

def divide_into_segments(window_size, per_window_overlapp, overlap_status, trial_train, trial_test, attended_ear_value):
    k=0
    while k < trial_train.shape[0] :
        temp = trial_train[k: k+window_size]
        # print("temp: ", temp.shape)
        if temp.shape == (window_size, trial_train.shape[1]):
            X_train.append(temp)
            y_train.append(attended_ear_value)

        if overlap_status == True:
            k = k+per_window_overlapp    
        else:
            k = k+window_size
    
    k=0
    while k < trial_test.shape[0] :
        temp = trial_test[k: k+window_size]
        # print("temp: ", temp.shape)
        if temp.shape == (window_size, trial_test.shape[1]):
            X_test.append(temp)
            y_test.append(attended_ear_value)

        if overlap_status == True:
            k = k+per_window_overlapp    
        else:
            k = k+window_size
    

def read_eeg(data_path, fl, overlapp_status, per_window_overlapp):
    data = loadmat(data_path)
    for i in range (8):
        eeg_matrix = data['preproc_trials'][0][i]['RawData'][0][0][0][0]['EegData'] # e.g. (12448, 64)
        # print("EEG_MATRIX_SHAPE: ", eeg_matrix.shape)
        attended = data['preproc_trials'][0][i]['attended_ear'][0][0] #L/R
        
        attended_ear_value = 0
        if attended == 'R':
            attended_ear_value = 1
        elif attended == 'L':
            attended_ear_value = 0
            
        train_index = int(eeg_matrix.shape[0]*0.8)
        trial_train = eeg_matrix[:train_index]
        trial_test = eeg_matrix[train_index:]
        
        # print("Train_trial", trial_train.shape)
        # print("Test_trial", trial_test.shape)
        
        divide_into_segments(window_size, per_window_overlapp, overlap_status, trial_train, trial_test, attended_ear_value)
        

X_train = []
y_train = []
X_test = []
y_test = []

for i in range(16):
    print('======================================================')
    if i%2 == 0:
        print('Current Subject Index:'+str(i+1))
        read_eeg('preprocessed_data/S'+str(i+1)+'.mat', 0, overlap_status, per_window_overlapp)
    else:
        print("Current Subject Index:"+str(i+1))
        read_eeg('preprocessed_data/S'+str(i+1)+'.mat', 1, overlap_status, per_window_overlapp)
    
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_test)
y_val = np.array(y_test)
# print(y_val)


# Shuffling the data 
X_train, y_train = shuffle(X_train, y_train, random_state=2)
# X_val, y_val = shuffle(X_val, y_val, random_state=35)

print("X_train shape: ", X_train.shape)    
print("y_train shape: ", y_train.shape)
print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)

print(y_train)


##### Saving Numpy Arrays
np.savez_compressed('X_train.npz',np.array(X_train))
np.savez_compressed('y_train.npz',np.array(y_train))
np.savez_compressed('X_val.npz',np.array(X_val))
np.savez_compressed('y_val.npz',np.array(y_val))
