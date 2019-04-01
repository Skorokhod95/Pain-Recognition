import csv
from multiprocessing import  Pool
import numpy as np
from numpy import genfromtxt 
import random
import pandas
#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout,RepeatVector,SimpleRNN, GRU
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn import datasets
from sklearn import metrics

#import matplotlib.pyplot as plt
import h5py

import gc

# Naive LSTM to learn one-char to one-char mapping
np.random.seed(7)

def Get_data(i):
    
    data_Data = genfromtxt('/home/askorokhod/projects/PSOPB_bServer/Data.csv', delimiter=',') 
    data_Labels = genfromtxt('/home/askorokhod/projects/PSOPB_bServer/Labels.csv', delimiter=',') 
  
    train_x, test_x, train_y, test_y = train_test_split(data_Data, data_Labels, test_size=0.2, random_state=42)
    train_Y = np.array(train_y)
    test_Y = np.array(test_y)
    train_X = np.array(train_x)
    test_X = np.array(test_x)

    return train_X,train_Y,test_X,test_Y


def ANnet (train_data, traint_labels, test_data, test_labels):
    model = Sequential()
    model.add(RepeatVector(3, input_shape=(len(train_data.loc[0]),)))
    model.add(LSTM(40))
    model.add(Dense(traint_labels.shape[1], activation = "softmax"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data, traint_labels, epochs=1, batch_size=32, verbose=0)
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
    del model
    del loss
    return accuracy
    '''del accuracy
    return loss'''
    

#x=np.array([random.randint(0, 1) for i in range(len(train_X[0]))])
  
def Model (x):
    '''h5 = h5py.File ("/mnt/eshare/DatasetsRW/SENSEmotion/SENSEmotion_Patrick_Features.mat", 'r')
    N = len (h5['/dataset/right'])   # Count of participants'''
    acc=[]
    for i in range(5):
        train_X,train_Y,test_X,test_Y=Get_data(i)
        train_data = pandas.DataFrame(train_X)
        test_data = pandas.DataFrame(test_X)
        itr=0
        for j in x:
            if (j==0):
                del train_data[itr]
                del test_data[itr]
            itr=itr+1
        res=ANnet(train_data, train_Y, test_data, test_Y)
        acc.append(res) 
        del train_data 
        del test_data
        del train_X
        del train_Y
        del test_X
        del test_Y
    return (sum(acc)/5)
    #acc=np.array(acc)
    #return (0.5)
    #print(acc.mean())
    #return(acc.mean())
    #return 1.0/(1.0+acc.mean())

def func(x):
    # Count of participants
    gc.collect()
    pool=Pool(processes=20)
    results = pool.map(Model,x)
    return results
   
  