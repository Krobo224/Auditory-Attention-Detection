#python version = 3.9.12
from scipy.io import loadmat
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gc
import math

#Data loading
X_train = np.array(np.load('X_train.npz')['arr_0'], dtype=np.float16)
X_val = np.array(np.load('X_val.npz')['arr_0'], dtype=np.float16)
y_train = np.array(np.load('y_train.npz')['arr_0'])
y_val = np.array(np.load('y_val.npz')['arr_0'])

num_layers = 4
d_model = 64
dff = 64
num_heads = 8

print(X_train.shape)
print(X_val.shape)
print(y_val.shape)
print(y_train[34])
print(y_train[56], y_train[75])

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class GlobalSelfAttention(BaseAttention): 
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


"""FEED FORWARD NETWORK"""
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):

        super().__init__()
        self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model),
        ])

        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        # print("Shape of input after add: ", x.shape)
        x = self.layer_norm(x)
        # print("Shape of input after normalization: ", x.shape)
        return x


#THE ENCODER layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        # print("Shape of input after Attention: ", x.shape)
        x = self.ffn(x)
        # print("Shape of input after feedforward: ", x.shape)
        return x


"""ENCODER"""
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                        num_heads=num_heads,
                        dff=dff)
            for _ in range(num_layers)]
        # self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # # Add dropout.
        # x = self.dropout(x)
        # print("Shape of input after dropout: ", x.shape)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
            # print("Shape of input after encoder layer ", i,": ", x.shape)

        return x  # Shape `(batch_size, seq_len, d_model)`


"""TRANSFORMER"""
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff):
        super().__init__()
        # self.intial_layer = tf.keras.layers.Conv1D(filters=64, kernel_size = (3, 3))
        self.convLayers = tf.keras.Sequential([ tf.keras.layers.Conv1D(filters=64, kernel_size=3),
                                                tf.keras.layers.MaxPool1D(pool_size=2),
                                                tf.keras.layers.Conv1D(filters=64, kernel_size=3)
                                               ])
        
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff)


        self.final_layer =  tf.keras.Sequential([ tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        eeg_data = inputs
        eeg_data = self.convLayers(eeg_data)
        eeg_data = self.encoder(eeg_data)  # (batch_size, eeg_data_len, d_model)

        # Final linear layer output.
        final_output = self.final_layer(eeg_data)  # (batch_size, target_len, target_vocab_size)
        # print("FINAL OUTPUT SHAPE: ", final_output.shape)

        # Return the final output and the attention weights.
        return final_output


"""Calling Transformer"""
transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff)

# attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
# print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)

transformer.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy']) #SGD #learning_rate


"""MODEL CHECKPOINTING"""
checkpoint_filepath = 'bestmodel.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, 
                                                               monitor='val_accuracy',
                                                               mode='max', save_best_only=True)


transformer.fit(X_train, y_train, batch_size=32, epochs=32)

print("============================Transformer Summary============================")
print(transformer.summary())

accuracy1 = transformer.evaluate(X_val, y_val, batch_size=32)
accuracy2 = transformer.evaluate(X_train, y_train, batch_size=32)
print(accuracy1)
print(accuracy2)
