#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 21:52:29 2020
Initinal project file
@author: rahulgupta
"""
# %% 


#Convert txt file to excel file
# import pandas as pd
# import csv

# read_file = pd.read_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/RUL_FD001.txt')
# read_file.to_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/Excel Files/RUL_FD001.csv', index=None)

# read_file = pd.read_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/RUL_FD002.txt')
# read_file.to_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/Excel Files/RUL_FD002.csv', index=None)

# read_file = pd.read_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/RUL_FD003.txt')
# read_file.to_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/Excel Files/RUL_FD003.csv', index=None)

# read_file = pd.read_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/RUL_FD004.txt')
# read_file.to_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/Excel Files/RUL_FD004.csv', index=None)

# read_file = pd.read_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/test_FD001.txt', delimiter = " ")
# read_file.to_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/Excel Files/test_FD001.csv', index=None)

# read_file = pd.read_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/test_FD002.txt', delimiter = " ")
# read_file.to_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/Excel Files/test_FD002.csv', index=None)

# read_file = pd.read_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/test_FD003.txt', delimiter = " ")
# read_file.to_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/Excel Files/test_FD003.csv', index=None)

# read_file = pd.read_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/test_FD004.txt', delimiter = " ")
# read_file.to_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/Excel Files/test_FD004.csv', index=None)

# read_file = pd.read_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/train_FD001.txt', delimiter = " ")
# read_file.to_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/Excel Files/train_FD001.csv', index=None)


# read_file = pd.read_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/train_FD002.txt', delimiter = " ")
# read_file.to_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/Excel Files/train_FD002.csv', index=None)

# read_file = pd.read_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/train_FD003.txt', delimiter = " ")
# read_file.to_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/Excel Files/train_FD003.csv', index=None)

# read_file = pd.read_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/train_FD004.txt', delimiter = " ")
# read_file.to_csv (r'/Users/rahulgupta/Desktop/University/masters/Neural Networks/Spyder Files/Project/CMAPSSData/Excel Files/train_FD004.csv', index=None)


# %% 



import csv
import numpy as np

#Read in excel file data

RUL1 = list(csv.reader(open('CMAPSSData/Excel Files/RUL_FD001.csv'))); 
RUL2 = list(csv.reader(open('CMAPSSData/Excel Files/RUL_FD002.csv'))); 
RUL3 = list(csv.reader(open('CMAPSSData/Excel Files/RUL_FD003.csv'))); 
RUL4 = list(csv.reader(open('CMAPSSData/Excel Files/RUL_FD004.csv'))); 
Test1 = list(csv.reader(open('CMAPSSData/Excel Files/test_FD001.csv')));
Test2 = list(csv.reader(open('CMAPSSData/Excel Files/test_FD002.csv')));
Test3 = list(csv.reader(open('CMAPSSData/Excel Files/test_FD003.csv'))); 
Test4 = list(csv.reader(open('CMAPSSData/Excel Files/test_FD004.csv'))); 
Train1 = list(csv.reader(open('CMAPSSData/Excel Files/train_FD001.csv'))); 
Train2 = list(csv.reader(open('CMAPSSData/Excel Files/train_FD002.csv')));
Train3 = list(csv.reader(open('CMAPSSData/Excel Files/train_FD003.csv'))); 
Train4 = list(csv.reader(open('CMAPSSData/Excel Files/train_FD004.csv'))); 

def delete_rows(p):
    for x in p: 
        del x[26:28]
    return p


Test1 = delete_rows(Test1)
Test2 = delete_rows(Test2)
Test3 = delete_rows(Test3)
Test4 = delete_rows(Test4)
Train1 = delete_rows(Train1)
Train2 = delete_rows(Train2)
Train3 = delete_rows(Train3)
Train4 = delete_rows(Train4)

RUL1 = np.array(RUL1, dtype = 'float')
RUL2 = np.array(RUL2, dtype = 'float') 
RUL3 = np.array(RUL3, dtype = 'float')
RUL4 = np.array(RUL4, dtype = 'float')
Test1 = np.array(Test1, dtype = 'float')
Test2 = np.array(Test2, dtype = 'float')
Test3 = np.array(Test3, dtype = 'float')
Test4 = np.array(Test4, dtype = 'float')
Train1 = np.array(Train1, dtype = 'float')
Train2 = np.array(Train2, dtype = 'float')
Train3 = np.array(Train3, dtype = 'float')
Train4 = np.array(Train4, dtype = 'float')


RUL1 = RUL1.astype(np.int)
RUL2 = RUL2.astype(np.int)
RUL3 = RUL3.astype(np.int)
RUL4 = RUL4.astype(np.int)


# %% 

#generate target based on data provided from files for each dataset
def gen_tar_vector(y):
    start = 0
    end = 0
    tar = np.zeros((np.shape(y)[0],1))

    for i in range(np.shape(y)[0]):
        if i < np.shape(y)[0] - 1: 
            if y[i,0] != y[i+1,0]:
                end = i
            
                tar[start:end + 1, 0] = abs(y[i,1] - y[start:end + 1, 1])
                start = i+1
        else:
            end = np.shape(y)[0]
            tar[start:end + 1, 0] = abs(y[i,1] - y[start:end + 1, 1])
        
    return tar    
    

tar1 = gen_tar_vector(Train1[:,0:2])
tar2 = gen_tar_vector(Train2[:,0:2])
tar3 = gen_tar_vector(Train3[:,0:2])
tar4 = gen_tar_vector(Train4[:,0:2])


#%% 

#Combine dataset
training_data_set = np.concatenate((Train1, Train2, Train3, Train4))
training_tar_vector = np.concatenate((tar1,tar2,tar3,tar4))
testing_data_set = np.concatenate((Test1, Test2, Test3, Test4))
testing_tar_vector = np.concatenate((RUL1, RUL2, RUL3, RUL4))

testing_data_set_for_graph = testing_data_set

#Delete unrequired columns

training_data_set = np.delete(training_data_set, [0,1,2,3,4,5,9,10,14,20,22,23],1)
testing_data_set = np.delete(testing_data_set,[0,1,2,3,4,5,9,10,14,20,22,23],1)


#normalize dataset
max_train = len(training_data_set)
uni_data = np.concatenate((training_data_set, testing_data_set))
uni_data_mean = training_data_set.mean(axis = 0)
uni_data_std = training_data_set.std(axis = 0)
uni_data = (uni_data - uni_data_mean)/uni_data_std
training_data_set = uni_data[:max_train, :]
testing_data_set = uni_data[max_train:,:]

y_max = len(training_tar_vector)
y_uni_data = np.concatenate((training_tar_vector, testing_tar_vector))
y_mean = training_tar_vector.mean(axis=0)
y_std = training_tar_vector.std(axis=0)
y_uni_data = (y_uni_data - y_mean)/y_std
training_tar_vector = y_uni_data[:y_max,]
testing_tar_vector = y_uni_data[y_max:]

val_data_set = training_data_set[60994:, :]
val_tar_vector = training_tar_vector[60994:,:]

#reshape data for neural network
training_data_set = training_data_set.reshape(training_data_set.shape[0],1, training_data_set.shape[1])
training_tar_vector = training_tar_vector.reshape(training_tar_vector.shape[0],1, training_tar_vector.shape[1])
val_data_set = val_data_set.reshape(val_data_set.shape[0],1, val_data_set.shape[1])
val_tar_vector = val_tar_vector.reshape(val_tar_vector.shape[0],1, val_tar_vector.shape[1])




