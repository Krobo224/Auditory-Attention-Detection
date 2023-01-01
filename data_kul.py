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
window = 0.5 #sec 1, 0.1, 2
overlapp = 0.6 # 0.8
window_size = math.ceil(sampling_rate*window)
per_window_overlapp = 1
overlap_status = False 

if overlap_status == True:
    rem = 1 - overlapp
    per_window_overlapp = int(rem*window_size) # for getting to the first index of next window

def divide_into_segments(window_size, per_window_overlapp, overlap_status, eeg_matrix, attended_ear_value, division_type):
    k=0
    while k < eeg_matrix.shape[0] :
        temp = eeg_matrix[k: k+window_size]
        # print("temp: ", temp.shape)
        if temp.shape == (window_size, 64):
            if division_type == 'train':
                X_train.append(temp)
                y_train.append(attended_ear_value)
            elif division_type == 'val':
                X_val.append(temp)
                y_val.append(attended_ear_value)    
        if overlap_status == True:
            k = k+per_window_overlapp    
        else:
            k = k+window_size
    
    

def read_eeg(data_path, fl, overlapp_status, per_window_overlapp):
    data = loadmat(data_path)
    l1 = [0, 1, 2, 3, 4, 5, 6, 7] #trials
    l2 = random.sample(range(0, 7), 5)
    l3 = list(set(l1).symmetric_difference(l2))
    for i in l2:
        eeg_matrix = data['preproc_trials'][0][i]['RawData'][0][0][0][0]['EegData'] # e.g. (12448, 64)
        attended = data['preproc_trials'][0][i]['attended_ear'][0][0] #L/R
  
        #considering left attended value '0' and right attended value '1'
      
        attended_ear_value = 0
        if(attended == 'R'):
            attended_ear_value = 1
        else:
            attended_ear_value = 0
        
        divide_into_segments(window_size, per_window_overlapp, overlap_status, eeg_matrix, attended_ear_value, 'train')

        
                
    for i in l3:
        eeg_matrix = data['preproc_trials'][0][i]['RawData'][0][0][0][0]['EegData'] # e.g. (12448, 64)
        attended = data['preproc_trials'][0][i]['attended_ear'][0][0] #L/R
  
        #considering left attended value '0' and right attended value '1'
      
        attended_ear_value = 0
        if(attended == 'R'):
            attended_ear_value = 1
        else:
            attended_ear_value = 0

        divide_into_segments(window_size, per_window_overlapp, overlap_status, eeg_matrix, attended_ear_value, 'val')

X_train = []
y_train = []
X_val = []
y_val = []

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
X_val = np.array(X_val)
y_val = np.array(y_val)
# print(y_val)


# Shuffling the data 
X_train, y_train = shuffle(X_train, y_train, random_state=2)
# X_val, y_val = shuffle(X_val, y_val, random_state=35)

print("X_train shape: ", X_train.shape)    
print("y_train shape: ", y_train.shape)
print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)

print(y_train)


# ##### Saving Numpy Arrays
np.savez_compressed('X_train.npz',np.array(X_train))
np.savez_compressed('y_train.npz',np.array(y_train))
np.savez_compressed('X_val.npz',np.array(X_val))
np.savez_compressed('y_val.npz',np.array(y_val))
