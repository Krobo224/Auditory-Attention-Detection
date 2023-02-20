### IMPORTING THE LIBRARIES ###
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


### LOADING THE DATA ###
"""Data loading"""
X_train = np.array(np.load('X_train.npz')['arr_0'], dtype=np.float16)
X_val = np.array(np.load('X_val.npz')['arr_0'], dtype=np.float16)
y_train = np.array(np.load('y_train.npz')['arr_0'])
y_val = np.array(np.load('y_val.npz')['arr_0'])

print("X_train Shape", X_train.shape)
print("X_test.shape", X_val.shape)


### HYPERPARAMETERS ###
num_layers = 6 # number of encoder layers
d_model = 64
dff = 128
num_heads = 8 # number of attention heads


"""TRANSFORMER"""

### SINUSOIDAL POSITIONAL EMBEDDING ###
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
    
    
### ATTENTION ###
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


### FEED FORWARD NETWORK ###
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


### THE ENCODER LAYER ###
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff):
        super().__init__()

        self.convLayer = tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='same', activation='relu')
        
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.convLayer(x)
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


### ENCODER ###
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
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x  # Shape `(batch_size, seq_len, d_model)`


### TRANSFORMER ###
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, timesteps_each_segment):
        super().__init__()
        # self.intial_layer = tf.keras.layers.Conv1D(filters=64, kernel_size = (3, 3))
        # self.convLayers = tf.keras.Sequential([ tf.keras.layers.Conv1D(filters=64, kernel_size=3),
        #                                         tf.keras.layers.Conv1D(filters=64, kernel_size=3)
        #                                        ])
        self.pos_encoding = Sinusoidal_PE(timesteps_each_segment, d_model)
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff)
        self.final_layer =  tf.keras.Sequential([ tf.keras.layers.Flatten(),
                                                tf.keras.layers.Dense(1, activation='sigmoid')
                                            ])

    def call(self, inputs):
        eeg_data = inputs
        # eeg_data = self.convLayers(eeg_data)
        eeg_data = self.pos_encoding(eeg_data)
        eeg_data = self.encoder(eeg_data)  # (batch_size, eeg_data_len, d_model)
        # Final linear layer output.
        final_output = self.final_layer(eeg_data)  # (batch_size, target_len, target_vocab_size)
        return final_output


### CALLING TRANSFORMER ###
transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, timesteps_each_segment=128)
opt = tf.keras.optimizers.SGD(learning_rate=0.001)
transformer.compile(loss='SGD', optimizer=opt, metrics=['accuracy'])


### MODEL CHECKPOINTING ###
filepath= "./Models/AAD_Transformer.ckpt"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True)

### TRAINING ###
transformer.fit(X_train, y_train, batch_size=32, epochs=150, validation_data=(X_val, y_val), callbacks=[checkpoint])
print("============================Transformer Summary============================")
print(transformer.summary())


### CHECKING ACCURACY ###
accuracy1 = transformer.evaluate(X_val, y_val)
accuracy2 = transformer.evaluate(X_train, y_train)
print("Accuracy on val: ", accuracy1)
print("Accuracy on train: ", accuracy2)

### LOADING THE TRANSFORMER MODEL ###
transformer.load_weights(filepath)
accuracy1 = transformer.evaluate(X_val, y_val)
accuracy2 = transformer.evaluate(X_train, y_train)
print("Accuracy on val: ", accuracy1)
print("Accuracy on train: ", accuracy2)

# y_pred1 = []
# for k in y_pred:
#     if k > 0.5:
#         y_pred1.append(1)
#     else:
#         y_pred1.append(0)

# y_pred1 = np.array(y_pred1)


# """CONFUSION MATRIX""""
# conf_matrix = confusion_matrix(y_true=y_val, y_pred=y_pred1)
# print("---------------Confusion Matrix---------------")
# fig, ax = plt.subplots(figsize=(7.5, 7.5))
# ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(conf_matrix.shape[0]):
#     for j in range(conf_matrix.shape[1]):
#         ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()


