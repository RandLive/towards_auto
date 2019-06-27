# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:29:22 2019

@author: m02li
"""

# Generate dummy data
import numpy as np
import keras
X_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(2, size=(1000, 1)), num_classes=10)
X_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(2, size=(100, 1)), num_classes=10)

y=y_train
X=X_train

X_pred = np.array(X_train)

input_shape = X.shape[1]
output_shape = y.shape[1]

# In[Keras model classes]
from keras.models import Sequential
from keras.layers import Embedding, Lambda, Input, Dense, Dropout, Activation, Conv1D, Conv2D, LSTM, GRU, MaxPooling2D, Reshape, Flatten, Layer, AveragePooling2D
from keras import activations
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

from common import squash, softmax, margin_loss, Capsule, Attention

from keras.layers import merge
from keras.layers.core import *
from keras.models import *

# In[All model class]
class NN_Models(object):
    
    def __init__(self, input_shape, output_shape):
        # initialization
        self.input_shape = input_shape
        self.output_shape = output_shape
        if output_shape>=2:
            self.act_func = 'softmax'
        else:
            self.act_func = 'sigmoid'
        return

    def callback(self):        
        lr_reduced = ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=5, verbose=0, min_delta=1e-5, mode='min')       
        callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=100),
                     ModelCheckpoint(filepath="NN_ml.hdf5", monitor='val_loss', mode='min', save_best_only=True),
                     lr_reduced]        
        return callbacks
        
    def mlp(self):
        input_arg = Input(shape=(self.input_shape,))
        x = Dense(256, activation='relu')(input_arg)
        x = Dense(64, activation='relu')(x)
        output_arg = Dense(self.output_shape, activation=self.act_func)(x)
        model = Model(input_arg, output_arg)         
        return model

    def conv1d(self):
        input_arg = Input(shape=(self.input_shape,))
        x = Reshape((self.input_shape, 1))(input_arg)
        x = Conv1D(filters=32, kernel_size=10)(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Flatten()(x)
        output_arg = Dense(self.output_shape, activation=self.act_func)(x)
        model = Model(input_arg, output_arg)          
        return model

    def conv_lstm(self):
        input_arg = Input(shape=(self.input_shape,))
        x = Reshape((self.input_shape, 1))(input_arg)
        x = Conv1D(filters=32, kernel_size=10)(x)
        x = LSTM(128)(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output_arg = Dense(self.output_shape, activation=self.act_func)(x)
        model = Model(input_arg, output_arg)          
        return model

    def lstm(self):
        input_arg = Input(shape=(self.input_shape,))
        x = Reshape((self.input_shape, 1))(input_arg)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(128)(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output_arg = Dense(self.output_shape, activation=self.act_func)(x)
        model = Model(input_arg, output_arg)          
        return model
    
    def gru(self):
        input_arg = Input(shape=(self.input_shape,))
        x = Reshape((self.input_shape, 1))(input_arg)
        x = GRU(128)(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output_arg = Dense(self.output_shape, activation=self.act_func)(x)
        model = Model(input_arg, output_arg)       
        return model

    def cnn_capsule(self):
        input_arg = Input(shape=(self.input_shape,))
        x = Reshape((self.input_shape, 1, 1))(input_arg)
        x = Conv2D(self.input_shape, (1, 1), activation='relu')(x)
        x = AveragePooling2D((self.input_shape, 1))(x)
        x = Conv2D(self.input_shape, (1, 1), activation='relu')(x)
        x = Conv2D(self.input_shape, (1, 1), activation='relu')(x)      
        x = Reshape((-1, self.input_shape))(input_arg)
        capsule = Capsule(10, 16, 3, True)(x)
        output_arg = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
        model = Model(input_arg, output_arg)
        return model

    def capsule(self):
        input_arg = Input(shape=(self.input_shape,))    
        x = Reshape((-1, self.input_shape))(input_arg)
        capsule = Capsule(10, 16, 3, True)(x)
        output_arg = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
        model = Model(input_arg, output_arg)
        return model

    def attention_lstm(self):
        input_arg = Input(shape=(self.input_shape,))
        x = Reshape((self.input_shape, 1))(input_arg)
        x = Attention(self.input_shape)(x)
        x = Reshape((1, 1))(x)
        x = LSTM(64, return_sequences=True)(x)
        x = LSTM(64)(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output_arg = Dense(self.output_shape, activation=self.act_func)(x)
        model = Model(input_arg, output_arg)          
        return model  

Loss = 'mean_squared_error'
#Loss = 'mean_absolute_error'
#Loss = 'categorical_crossentropy'
#Loss = 'binary_crossentropy'

models = NN_Models(input_shape, output_shape)
model = models.attention_lstm()
model.compile(optimizer='adam', loss=Loss, metrics=['accuracy'])   

model.fit(X_train, y_train, 
          epochs = 750,
          batch_size = 512, 
          callbacks = models.callback(),
          shuffle = True,
#          validation_data=(X_test, y_test),
          validation_split=(0.8),
          verbose=1)

model.load_weights("NN_ml.hdf5")
predictions = model.predict(X_pred)

score = model.evaluate(X_test, y_test, batch_size=128)
print(score)