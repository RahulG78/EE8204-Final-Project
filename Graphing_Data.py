#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:27:43 2020

@author: rahulgupta
"""

from Data_Preprocessing import Test1, testing_tar_vector, y_mean, y_std, testing_data_set_for_graph
import pandas as pd
import csv 
import matplotlib.pyplot as plt
import numpy as np

read_file = pd.read_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/test.txt')
read_file.to_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/Excel Files/test.csv', index=None)

test = list(csv.reader(open('CMAPSSData/Excel Files/test.csv'))); 

test = np.array(test, dtype = 'float')

testing_data_set = testing_data_set_for_graph

pred = []

#gather only required RUL predictions (the predictions after engine has ended)
for i in range(Test1.shape[0]):
    if i < np.shape(testing_data_set)[0] - 1:
        if testing_data_set[i,0] != testing_data_set[i+1,0]:
            pred.append(test[i])
            
    else: 
        pred.append(test[i])


pred = pred
actual = testing_tar_vector[:100,]

print(len(pred))
print(len(actual))

#convert normalize data 

y_pred = pred*y_std + y_mean
actual = actual*y_std + y_mean


#Plot data

x = range(0, len(y_pred))
plt.plot(x,y_pred, color = 'red', label = 'Prediction RUL')
plt.plot(x,actual, color = 'blue', label = 'Actual RUL')
plt.legend(loc = 'upper right')
plt.grid()
plt.show()

error = []

add = 0
sqr = 0

#Determine RMSE

for i in range(len(actual)):
    sub = actual[i] - y_pred[i]
    sqr = (sub)**2
    add = add + sqr
    
    
RMSE = np.sqrt((add)/len(actual))    
print(RMSE)

# %% 





















