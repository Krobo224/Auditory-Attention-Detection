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
import biosppy
from biosppy import signals as biosig
from scipy.fftpack import fft, ifft

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
    
def eeg_decompose(eeg):
    # Get the number of timesteps and channels
    timesteps = eeg.shape[0]
    channels = eeg.shape[1]
    
    delta = [0.5, 4]
    theta = [4, 7]
    alpha = [8, 13]
    beta = [13, 30]
    gamma = [30, 100]
    
    # Initialize the decomposed signals
    delta_signal = np.zeros_like(eeg)
    theta_signal = np.zeros_like(eeg)
    alpha_signal = np.zeros_like(eeg)
    beta_signal = np.zeros_like(eeg)
    gamma_signal = np.zeros_like(eeg)
    
    for i in range(channels):
        # compute the fft of EEG signal
        spectrum = fft(eeg[:, i])
        
        # defining frequency axis
        N = timesteps
        freq = np.arange(0, N) / N*sampling_rate
        
        #Creating mask for each frequncy band
        delta_mask = (freq >= delta[0]) & (freq <= delta[1])
        theta_mask = (freq >= theta[0]) & (freq <= theta[1])
        alpha_mask = (freq >= alpha[0]) & (freq <= alpha[1])
        beta_mask = (freq >= beta[0]) & (freq <= beta[1])
        gamma_mask = (freq >= gamma[0]) & (freq <= gamma[1])
        
        # Create a copy of the FFT representation for each frequency band
        delta_spectrum = spectrum.copy()
        theta_spectrum = spectrum.copy()
        alpha_spectrum = spectrum.copy()
        beta_spectrum = spectrum.copy()
        gamma_spectrum = spectrum.copy()
        
        # Mask out the non-relevant frequency components for each frequency band
        delta_spectrum[~delta_mask] = 0
        theta_spectrum[~theta_mask] = 0
        alpha_spectrum[~alpha_mask] = 0
        beta_spectrum[~beta_mask] = 0
        gamma_spectrum[~gamma_mask] = 0
        
        # Compute the inverse FFT for each frequency band
        delta_signal[:, i] = ifft(delta_spectrum).real
        theta_signal[:, i] = ifft(theta_spectrum).real
        alpha_signal[:, i] = ifft(alpha_spectrum).real
        beta_signal[:, i] = ifft(beta_spectrum).real
        gamma_signal[:, i] = ifft(gamma_spectrum).real
        
    return_arr = np.stack((delta_signal, theta_signal, alpha_signal, beta_signal, gamma_signal))
    return return_arr

def divide_into_segments(window_size, per_window_overlapp, overlap_status, trial_train, trial_test, attended_ear_value):
    k=0
    while k < trial_train.shape[0] :
        temp = trial_train[k: k+window_size]
        # print("temp: ", temp.shape)
        if temp.shape == (window_size, trial_train.shape[1]):
            # band separation
            # print("Temp shape", temp.shape)
            temp_decompose = eeg_decompose(temp)
            # print("Decomposed shape", temp_decompose.shape)
            X_train.append(temp_decompose)
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
            
            temp_decompose = eeg_decompose(temp)
            
            X_test.append(temp_decompose)
            y_test.append(attended_ear_value)
        
        if overlap_status == True:
            k = k+per_window_overlapp    
        else:
            k = k+window_size
    z = np.array(X_train)
    # print("Z shape: ", z.shape)

def read_eeg(data_path, overlapp_status, per_window_overlapp):
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
    
    print('Current Subject Index:'+str(i+1))
    read_eeg('preprocessed_data/S'+str(i+1)+'.mat', overlap_status, per_window_overlapp)

    
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_test)
y_val = np.array(y_test)
# print(y_val)

print("X_train.shape ", X_train.shape)
# exit()
# Shuffling the data 
X_train, y_train = shuffle(X_train, y_train, random_state=2)
# X_val, y_val = shuffle(X_val, y_val, random_state=35)

print("X_train shape: ", X_train.shape)    
print("y_train shape: ", y_train.shape)
print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)

# print(y_train)


##### Saving Numpy Arrays
np.savez_compressed('X_train_cnn.npz',np.array(X_train))
np.savez_compressed('y_train_cnn.npz',np.array(y_train))
np.savez_compressed('X_val_cnn.npz',np.array(X_val))
np.savez_compressed('y_val_cnn.npz',np.array(y_val))
