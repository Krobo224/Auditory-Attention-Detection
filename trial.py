#python version = 3.9.12
from scipy.io import loadmat
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gc
import math

# #function to load subject file
# def load_subject_eeg(filepath):
#     data = loadmat(filepath)
#     return data

# data_sub1 = load_subject_eeg('preprocessed_data/s1.mat')
# print(data_sub1['preproc_trials'].shape) #(1, 20)
# s1 = data_sub1['preproc_trials']
# # a = first_trial['RawData']
# # print(type(first_trial))
# # print(type(a['EegData']))
# # # print(first_trial['RawData']['EegData'])
# # a = prep
# # # print(type(first_trial)) #<class 'numpy.ndarray'>

# a = s1[0][0]['RawData']
# attended = s1[0][0]['attended_ear'][0][0]
# if(attended == 'R'):
#     print("Its right")
# print(attended)
# print("Shape of channels: ", a.shape)
# print(a[0][0][0][0]['EegData']) #(12448, 64)

#"""

#Data loader :

sampling_rate = 128 #Hz
window = 1 #sec
overlapp = 0.6
window_size = sampling_rate*window
per_window_overlapp = 51 # 40%

def read_eeg(data_path):
    data = loadmat(data_path)
    
    final_eeg_subject = [] #list of 20 trials for a single subject
    final_attended_ear = [] #list of 20 attended ears for a single subject
    for i in range (20): #no. of trials
        eeg_matrix = data['preproc_trials'][0][i]['RawData'][0][0][0][0]['EegData'] #(12448, 64)
        # eeg_matrix = eeg_matrix.transpose() #(64, 12448)
        # for i in range(64):
            # tf.keras.utils.pad_sequences(eeg_matrix[:, i], maxlen=12500, dtype='float64', padding='post', truncating='post', value=0.0)
        # temp = []
        # for j in range (64): #no. of channels
        #     temp.append(list(eeg_matrix[j]))
        # szs.append(eeg_matrix.shape[0])
        # eeg_matrix = np.reshape(eeg_matrix, (1, eeg_matrix.shape[1], eeg_matrix.shape[0]))
        # print(len(temp))

        k=0
        
        #considering left attended value '0' and right attended value '1'
        attended = data['preproc_trials'][0][i]['attended_ear'][0][0]
        attended_ear_value = 0
        if(attended == 'R'):
            attended_ear_value = 1
        else:
            attended_ear_value = 0
        
        X.append(eeg_matrix)
        y.append(attended_ear_value)
        # szs.append(eeg_matrix.shape[2])
        
        # while k < eeg_matrix.shape[0] :
        #     temp = eeg_matrix[k: k+window_size]
        #     # print("temp: ", temp.shape)
        #     if temp.shape == (128, 64):
        #         X.append(temp)
        #         y.append(attended_ear_value)
            
        #     k = k+per_window_overlapp
        
        


X = []
y = []
szs = []
for i in range(2):
    print('======================================================')
    print('Current Subject Index:'+str(i+1))
    read_eeg('preprocessed_data/S'+str(i+1)+'.mat')
    
X = np.array(X)
y = np.array(y)
print("X[0] shape: ", X[0].shape)
print("y shape", y.shape)


# Shuffling the data 
X, y = shuffle(np.array(X),np.array(y),random_state=35)

def train_val_split(X, y):
    z = int((60/100)*X.shape[0]) #60% train, 40% val 
    X_train = X[:z]
    y_train = y[:z]
    
    X_val = X[z:]
    y_val = y[z:]
    
    return (X_train, y_train, X_val, y_val)

X_train, y_train, X_val, y_val = train_val_split(X, y)

print("X_train shape: ", X_train.shape)    
print("y_train shape: ", y_train.shape)
print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)
            
X_train,y_train = shuffle(np.array(X_train),np.array(y_train),random_state=35)
X_val,y_val = shuffle(np.array(X_val),np.array(y_val),random_state=35)

###### Saving Numpy Arrays
np.savez_compressed('X_train.npz',np.array(X_train))
np.savez_compressed('y_train.npz',np.array(y_train))
np.savez_compressed('X_val.npz',np.array(X_val))
np.savez_compressed('y_val.npz',np.array(y_val))
#np.savez_compressed('EBT_X_train_bispectrum.npz',np.array(X_bispectrum))

# train_val_split(X, y)
# # print(type(X_train[0][0]))
# # print(X_train[0][0].shape)
# # print(len(X_train))
# # import matplotlib.pyplot as plt

# # fig, ax = plt.subplots(figsize =(10, 7))
# # ax.hist(szs)
# # plt.show()
# # print(max(szs))


# BATCH_SIZE = 8

# train_batches_X = []
# train_batches_y = []

# val_batches_X = []
# val_batches_y = []

# def make_batches(features, labels, type_of_data):
#     if type_of_data == 'train':
#         for i in range(0, 192, 8):
#             temp_features = []
#             temp_labels = []
#             for j in range(8):
#                 temp_features.append(features[i+j])
#                 temp_labels.append(labels[i+j])

#             train_batches_X.append(temp_features)
#             train_batches_y.append(temp_labels)
#     elif type_of_data == 'validation':
#         for i in range(0, 128, 8):
#             temp_features = []
#             temp_labels = []
#             for j in range(8):
#                 temp_features.append(features[i+j])
#                 temp_labels.append(labels[i+j])

#             val_batches_X.append(temp_features)
#             val_batches_y.append(temp_labels)

# make_batches(X_train, y_train, 'train')
# make_batches(X_val, y_val, 'validation')


# # Transformer

# class BaseAttention(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
#         self.layernorm = tf.keras.layers.LayerNormalization()
#         self.add = tf.keras.layers.Add()

# class CrossAttention(BaseAttention):
#     def call(self, x, context):
#         attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True)
        
#         # Cache the attention scores for plotting later.
#         self.last_attn_scores = attn_scores

#         x = self.add([x, attn_output])
#         x = self.layernorm(x)

#         return x
    
# class GlobalSelfAttention(BaseAttention): 
#     def call(self, x):
#         attn_output = self.mha(query=x, value=x, key=x)
#         x = self.add([x, attn_output])
#         x = self.layernorm(x)
#         return x
    


# class CausalSelfAttention(BaseAttention):
#     def call(self, x):
#         attn_output = self.mha(query=x, value=x, key=x, use_causal_mask = True)
#         x = self.add([x, attn_output])
#         x = self.layernorm(x)
#         return x
    



# """FEED FORWARD NETWORK"""

# class FeedForward(tf.keras.layers.Layer):
#     def __init__(self, d_model, dff, dropout_rate=0.1):
        
#         super().__init__()
#         self.seq = tf.keras.Sequential([
#         tf.keras.layers.Dense(dff, activation='relu'),
#         tf.keras.layers.Dense(d_model),
#         tf.keras.layers.Dropout(dropout_rate)
#         ])
        
#         self.add = tf.keras.layers.Add()
#         self.layer_norm = tf.keras.layers.LayerNormalization()

#     def call(self, x):
#         x = self.add([x, self.seq(x)])
#         x = self.layer_norm(x) 
#         return x
    
# #THE ENCODER layer
# class EncoderLayer(tf.keras.layers.Layer):
#     def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
#         super().__init__()

#         self.self_attention = GlobalSelfAttention(
#             num_heads=num_heads,
#             key_dim=d_model,
#             dropout=dropout_rate)

#         self.ffn = FeedForward(d_model, dff)

#     def call(self, x):
#         x = self.self_attention(x)
#         x = self.ffn(x)
#         return x
    
# """ENCODER"""
# class Encoder(tf.keras.layers.Layer):
#     def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
#         super().__init__()

#         self.d_model = d_model
#         self.num_layers = num_layers

#         # # self.pos_embedding = PositionalEmbedding(
#         #     # vocab_size=vocab_size, d_model=d_model)

#         self.enc_layers = [
#             EncoderLayer(d_model=d_model,
#                         num_heads=num_heads,
#                         dff=dff,
#                         dropout_rate=dropout_rate)
#             for _ in range(num_layers)]
#         self.dropout = tf.keras.layers.Dropout(dropout_rate)

#     def call(self, x):
#         # `x` is token-IDs shape: (batch, seq_len)
#         x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

#         # Add dropout.
#         x = self.dropout(x)

#         for i in range(self.num_layers):
#             x = self.enc_layers[i](x)

#         return x  # Shape `(batch_size, seq_len, d_model)`.
    

# #Decoder layer
# class DecoderLayer(tf.keras.layers.Layer):
#     def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
#         super().__init__()

#         self.causal_self_attention = CausalSelfAttention(
#             num_heads=num_heads,
#             key_dim=d_model,
#             dropout=dropout_rate)

#         self.cross_attention = CrossAttention(
#             num_heads=num_heads,
#             key_dim=d_model,
#             dropout=dropout_rate)

#         self.ffn = FeedForward(d_model, dff)

#     def call(self, x, context):
#         x = self.causal_self_attention(x=x)
#         x = self.cross_attention(x=x, context=context)

#         # Cache the last attention scores for plotting later
#         self.last_attn_scores = self.cross_attention.last_attn_scores

#         x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
#         return x   

# """DECODER"""    
# class Decoder(tf.keras.layers.Layer):
#     def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
#         super().__init__()

#         self.d_model = d_model
#         self.num_layers = num_layers

#         # self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
#         #                                         d_model=d_model)
#         self.dropout = tf.keras.layers.Dropout(dropout_rate)
#         self.dec_layers = [DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate) for _ in range(num_layers)]

#         self.last_attn_scores = None

#     def call(self, x, context):
#         # `x` is token-IDs shape (batch, target_seq_len)
#         x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

#         x = self.dropout(x)

#         for i in range(self.num_layers):
#             x  = self.dec_layers[i](x, context)

#         self.last_attn_scores = self.dec_layers[-1].last_attn_scores

#         # The shape of x is (batch_size, target_seq_len, d_model).
#         return x
    
    
# """TRANSFORMER"""
# class Transformer(tf.keras.Model):
#     def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
#         super().__init__()
#         self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
#                             num_heads=num_heads, dff=dff,
#                             dropout_rate=dropout_rate)

#         self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
#                             num_heads=num_heads, dff=dff,
#                             dropout_rate=dropout_rate)

#         self.final_layer = tf.keras.layers.Dense(2, activation='sigmoid') # replacing by 2

#     def call(self, inputs):
#         # To use a Keras model with `.fit` you must pass all your inputs in the
#         # first argument.
#         context, x  = inputs

#         context = self.encoder(context)  # (batch_size, context_len, d_model)

#         x = self.decoder(x, context)  # (batch_size, target_len, d_model)

#         # Final linear layer output.
#         logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

#         # try:
#         #     # Drop the keras mask, so it doesn't scale the losses/metrics.
#         #     # b/250038731
#         #     del logits._keras_mask
            
#         # except AttributeError:
#         #     pass

#         # Return the final output and the attention weights.
#         return logits
   
   


# """Calling Transformer"""
# transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)

# # attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
# # print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)


# """Customer schedule"""

# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=4000):
#         super().__init__()

#         self.d_model = d_model
#         self.d_model = tf.cast(self.d_model, tf.float32)

#         self.warmup_steps = warmup_steps

#     def __call__(self, step):
#         step = tf.cast(step, dtype=tf.float32)
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps ** -1.5)

#         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)   
    

# learning_rate = CustomSchedule(d_model)

# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


# #Since the target sequences are padded, it is important to apply a padding mask when calculating the loss. Use the cross-entropy loss function
# def masked_loss(label, pred):
#     mask = label != 0
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#         from_logits=True, reduction='none')
#     loss = loss_object(label, pred)

#     mask = tf.cast(mask, dtype=loss.dtype)
#     loss *= mask

#     loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
#     return loss


# def masked_accuracy(label, pred):
#     pred = tf.argmax(pred, axis=2)
#     label = tf.cast(label, pred.dtype)
#     match = label == pred

#     mask = label != 0

#     match = match & mask

#     match = tf.cast(match, dtype=tf.float32)
#     mask = tf.cast(mask, dtype=tf.float32)
#     return tf.reduce_sum(match)/tf.reduce_sum(mask)


# transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])

# # print("============================Transformer Summary============================")
# # print(transformer.summary())
# transformer.fit(train_batches_X, train_batches_y, epochs=2)
    
      
  

            
    

    





    
     
     
     
     
    