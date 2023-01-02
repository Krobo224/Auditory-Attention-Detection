#python version = 3.9.12
from scipy.io import loadmat
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gc
import math

"""Data loading"""
X_train = np.array(np.load('X_train.npz')['arr_0'], dtype=np.float16)
X_val = np.array(np.load('X_val.npz')['arr_0'], dtype=np.float16)
y_train = np.array(np.load('y_train.npz')['arr_0'])
y_val = np.array(np.load('y_val.npz')['arr_0'])

num_layers = 4
d_model = 64
dff = 128
num_heads = 8

print(X_train.shape)
print(X_val.shape)
print(y_val.shape)
print(y_train[34])
print(y_train[56], y_train[75])


""" Sinusoidal Positional Embedding """
class Sinusoidal_PE(tf.keras.layers.Layer):
    
    def __init__(self, maxlen, embed_dim):
        
        #### Defining Essentials
        super().__init__()
        self.maxlen = maxlen # Maximum Sequence Length
        self.embed_dim = embed_dim # Embedding Dimensions of the Positional Encodings                                           

        #### Defining Layers
        position_embedding_matrix = self.get_position_encoding(self.maxlen, self.embed_dim)
        self.position_embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.maxlen, output_dim=self.embed_dim,
            weights=[position_embedding_matrix],
            trainable=False
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen, 
            'embed_dim': self.embed_dim 
        })
        return config 
             
    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P
 
    def call(self, inputs):        
        position_indices = tf.range(tf.shape(inputs)[-2])
        embedded_indices = self.position_embedding_layer(position_indices)
        return inputs+embedded_indices
    
    
"""Attention"""
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


"""THE ENCODER layer"""
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

        

        self.enc_layers = [EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff) for _ in range(num_layers)]
        
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
    def __init__(self, *, num_layers, d_model, num_heads, dff, timesteps_each_segment):
        super().__init__()
        # self.intial_layer = tf.keras.layers.Conv1D(filters=64, kernel_size = (3, 3))
        self.convLayers = tf.keras.Sequential([ tf.keras.layers.Conv1D(filters=64, kernel_size=3),
                                                tf.keras.layers.Conv1D(filters=64, kernel_size=3)
                                               ])
        
        self.pos_encoding = Sinusoidal_PE(timesteps_each_segment, d_model)
        
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
        eeg_data = self.pos_encoding(eeg_data)
        eeg_data = self.encoder(eeg_data)  # (batch_size, eeg_data_len, d_model)

        # Final linear layer output.
        final_output = self.final_layer(eeg_data)  # (batch_size, target_len, target_vocab_size)
        # print("FINAL OUTPUT SHAPE: ", final_output.shape)

        # Return the final output and the attention weights.
        return final_output

"""Calling Transformer"""
transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, timesteps_each_segment=128)

# attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
# print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)
opt = tf.keras.optimizers.SGD(learning_rate=0.001)
transformer.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) #SGD #learning_rate


"""MODEL CHECKPOINTING"""
checkpoint_path = "/transformer_model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1)

transformer.fit(X_train, y_train, batch_size=32, epochs=60)

print("============================Transformer Summary============================")
print(transformer.summary())

accuracy1 = transformer.evaluate(X_val, y_val)
accuracy2 = transformer.evaluate(X_train, y_train)

print("Accuracy on val: ", accuracy1)
print("Accuracy on train: ", accuracy2)

# Basic model instance
model_transformer = tf.keras.Model()
model_transformer.load_weights(checkpoint_path)

# evaluate the model
loss, acc = model_transformer.evaluate(X_val, y_val, verbose=2)
print("Validation model, accuracy: {:5.2f}%".format(100 * acc))

