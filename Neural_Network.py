#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:20:50 2020

@author: rahulgupta
"""
import tensorflow as tf
from Data_Preprocessing import training_data_set, training_tar_vector, val_data_set, val_tar_vector, testing_data_set

import numpy as np

tf.random.set_seed(20)

#Set parameters

LSTM = 60
LSTM2 = 0

BATCH_SIZE = 10
EPOCHS = 10
V_STEPS = 20

#Save file naming

if LSTM2 != 0:
    model_desp = "Stacked-LSTM-(" + str(LSTM) + ","+str(LSTM2)+"), batch " + str(BATCH_SIZE) + " epochs " + str(EPOCHS)
else:
    model_desp = "LSTM-(" + str(LSTM) + "), batch " + str(BATCH_SIZE) + " epochs " + str(EPOCHS)
    


train_uni = tf.data.Dataset.from_tensor_slices((training_data_set, training_tar_vector))
train_uni = train_uni.cache().batch(BATCH_SIZE).repeat()
val_uni = tf.data.Dataset.from_tensor_slices((val_data_set, val_tar_vector))
val_uni = val_uni.batch(BATCH_SIZE).repeat()

tf.keras.backend.clear_session()

#Sequential Model

if LSTM2 != 0:
    simple_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(LSTM, input_shape = (1,14), return_sequences = True),
        tf.keras.layers.LSTM(LSTM2),
        tf.keras.layers.Dense(20, input_shape = (1,14)),
        tf.keras.layers.Dense(20),
        tf.keras.layers.Dense(20),
        tf.keras.layers.Dense(1),
        ])
else: 
    simple_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape = (1,14)),
        tf.keras.layers.LSTM(LSTM, input_shape = (1,10)),
        tf.keras.layers.Dense(20),
        tf.keras.layers.Dense(20),
        tf.keras.layers.Dense(20),  
        tf.keras.layers.Dense(1)
        ])    

simple_model.compile(optimizer = 'adam', loss = 'mae')

#Fit Model

steps = len(training_data_set)/BATCH_SIZE
simple_model.fit(train_uni, epochs = EPOCHS, steps_per_epoch = steps, validation_data = val_uni, validation_steps = V_STEPS)


        
# %%

predict = []

#Generate predictions based on testing dataset
testing_data_set = testing_data_set.reshape(testing_data_set.shape[0],1,14)
dataset = tf.data.Dataset.from_tensors(testing_data_set)

predict = simple_model.predict(dataset)
predict = predict.reshape(predict.shape[0],1)
print(predict.shape)
print(predict)
remain = predict

#Save predictions

np.savetxt('test.txt', remain)

simple_model.save(model_desp)

tf.keras.backend.clear_session()


