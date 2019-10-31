#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:47:16 2019

@author: kushagra
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from numpy.linalg import inv

dataset = pd.read_excel('data.xlsx', header  = None)
dataset[3] = np.array([1.0]*349)

dataset['temp'] = dataset[2]
dataset[2] = dataset[3]
dataset[3] = dataset['temp']

X = dataset.iloc[:, :3].values
y = dataset.iloc[:, 3:4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.fit_transform(X_test)

for x in X_train:
    x[2] = 1.0
    
for x in X_test:
    x[2] = 1.0
    
class VectorizedLinearRegressor:
    def __init__(self, X, y):
        self.__X = X
        self.__y = y
        self.__W = []
        self.__update()
        
    def __update(self):
        X = np.matrix(self.__X)
        XT = np.transpose(X)
        y = np.matrix(self.__y)
        XTy = np.dot(XT, y)
        XTX = np.dot(XT, X)
        XTX = inv(XTX)
        XTy = np.dot(XTX, XTy)
        self.__W = np.array(XTy).tolist()
        
    def printweights(self):
        print("W0: " + str(self.__W[2]))
        print("W1: " + str(self.__W[0]))
        print("W2: " + str(self.__W[1]))
        
VLR = VectorizedLinearRegressor(X_train, y_train)
VLR.printweights()