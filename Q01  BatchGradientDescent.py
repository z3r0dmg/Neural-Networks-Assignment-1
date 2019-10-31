#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:33:31 2019

@author: kushagra
"""
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

dataset = pd.read_excel('data.xlsx', header  = None)
X = dataset.iloc[:, :2].values
y = dataset.iloc[:, 2:].values

from sklearn.model_selection import train_test_split
seed = 0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.preprocessing import StandardScaler
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.fit_transform(X_test)

W1_array = []
W2_array = []
J_array = []

class BatchLinearRegressor:
    '''Linear Regressor class 
    that uses Batch Gradient Descent'''
    
    def __init__(self, X, Y, epochs):
        self.__X = X
        self.__Y = Y
        self.__epochs = epochs
        self.__W = [0.3]*(X.shape[1]+1)
        self.__alpha = 0.005
        self.__cost = 0.0
        self.__gradientDescent()
    
    
    def __hypothesis(self):
        '''Method to calculate the hypothesis
        for each X vector'''
        
        h = [0]*(self.__X.shape[0])
        for i in range(0, len(self.__X)):
            h[i] = self.__W[0]
            for j in range(1, len(self.__W)):
                h[i] = h[i] + self.__W[j]*self.__X[i][j-1]
        return h
    
    def __costFunction(self):
        '''Method to calculate the
        cost function'''
        
        self.__cost = 0.0
        hx = self.__hypothesis()
        for i in range(0, len(hx)):
            temp = hx[i] - self.__Y[i]
            temp = temp * temp
            self.__cost = self.__cost + temp
        self.__cost = 0.5*self.__cost
        return self.__cost

    def __gradientDescent(self):
        '''Gradient descent that runs 
        for specified number of epochs'''
        
        for t in range(1, self.__epochs + 1):
            sum_of_values = [0]*len(self.__W)
            hx = self.__hypothesis()
            for i in range(0, len(self.__X)):
                temp = hx[i] - self.__Y[i]
                sum_of_values[0] = sum_of_values[0] + temp
                for j in range(1, len(self.__W)):
                    sum_of_values [j] = sum_of_values[j] + temp*self.__X[i][j-1]
            for i in range(0, len(self.__W)):
                self.__W[i] = self.__W[i] - self.__alpha*sum_of_values[i]
            new_cost = self.__costFunction()
            
            W1_array.append(self.__W[1])
            W2_array.append(self.__W[2])
            J_array.append(new_cost)
            
            
            print("Epoch: " + str(t) + " Cost: " + str(new_cost))
            
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            pred = self.__W[0]
            for j in range(0, len(x)):
                pred = pred + (self.__W[j+1]*x[j])
            predictions.append(pred)
        return predictions
    
    def printweights(self):
        print("W0: " + str(self.__W[0]))
        print("W1: " + str(self.__W[1]))
        print("W2: " + str(self.__W[2]))
            

BGD = BatchLinearRegressor(X_train, y_train, 100)
predictions = BGD.predict(X_test)
BGD.printweights()