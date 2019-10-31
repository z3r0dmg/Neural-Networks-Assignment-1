#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 18:57:30 2019

@author: kushagra
"""

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

dataset = pd.read_excel('data.xlsx', header  = None)
X = dataset.iloc[:, :2].values
y = dataset.iloc[:, 2:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.fit_transform(X_test)

cost_array = []

class StochasticLinearRegressor:
    '''Linear Regressor class 
    that uses Stochastic Gradient Descent'''
    
    def __init__(self, X, y, epochs):
        self.__X = X
        self.__y = y
        self.__epochs = epochs
        self.__W = [0.3]*(X.shape[1]+1)
        self.__alpha = 0.0002
        self.__cost = 0.0
        self.__gradientDescent()
    
    
    def __costFunction(self):
        '''Method to calculate the
        cost function'''
        
        self.__cost = 0.0
        for i in range(0,len(self.__X)):
            temp = self.__W[0]
            for j in range(1,len(self.__W)):
                temp = temp + (self.__W[j])*(self.__X[i][j-1])
            temp = temp - self.__y[i]
            temp = (temp)*(temp)
            self.__cost = (self.__cost) + temp
        self.__cost = 0.5*(self.__cost)
        return self.__cost

    def __gradientDescent(self):
        '''Gradient descent that runs 
        for specified number of epochs'''
        
        new_cost = self.__costFunction()
        for t in range(1,self.__epochs+1):
            for i in range(0,len(self.__X)):
                temp = self.__W[0]
                for j in range(1,len(self.__W)):
                    temp = temp + (self.__W[j])*(self.__X[i][j-1])
                temp = temp - self.__y[i]
                self.__W[0] = self.__W[0] - (temp * self.__alpha)
                for j in range(1,len(self.__W)):
                    self.__W[j] = self.__W[j] - (temp * self.__alpha * self.__X[i][j-1])
            new_cost = self.__costFunction()
            cost_array.append(new_cost)
            print('Epoch: ' + str(t) + ' Cost: ' + str(new_cost))
            
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
            

SGD = StochasticLinearRegressor(X_train, y_train, 100)
SGD.predict(X_test)
SGD.printweights()

J = cost_array[30:]