#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:13:32 2019

@author: kushagra
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

dataset = pd.read_excel('data.xlsx', header  = None)
X = dataset.iloc[:, :2].values
y = dataset.iloc[:, 2:].values

from sklearn.model_selection import train_test_split
seed = 0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.fit_transform(X_test)

class LeastAngleRegressor:
    '''Least Angle Regressor class that is trained
    using both Batch and Stochastic Gradient Descent'''
    
    def __init__(self, X, Y, epochs, lmbd, gradientDescent):
        self.__X = X
        self.__Y = Y
        self.__epochs = epochs
        self.__lmbd = lmbd
        self.__W = [0.3]*(X.shape[1]+1)
        self.__alpha = 0.005
        self.__cost = 0.0
        if(gradientDescent == 'batch'):
            self.__batchgradientdescent()
        else:
            self.__stochasticgradientdescent()
    
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
        for i in self.__W:
            self.__cost = self.__cost + self.__lmbd*i
        self.__cost = 0.5*self.__cost
        return self.__cost
    
    def __batchgradientdescent(self):
        for t in range(1, self.__epochs + 1):
            sum_of_values = [0]*len(self.__W)
            hx = self.__hypothesis()
            for i in range(0, len(self.__X)):
                temp = hx[i] - self.__Y[i]
                sum_of_values[0] = sum_of_values[0] + temp
                for j in range(1, len(self.__W)):
                    sum_of_values [j] = sum_of_values[j] + temp*self.__X[i][j-1]
            for i in range(0, len(self.__W)):
                self.__W[i] = self.__W[i] - self.__alpha*sum_of_values[i] - self.__alpha*self.__lmbd*np.sign(self.__W[i])     
            new_cost = self.__costFunction()
            print("Epoch: " + str(t) + " Cost: " + str(new_cost))
    
    def __stochasticgradientdescent(self):
        for t in range(1, self.__epochs + 1):
            hx = self.__hypothesis()
            for i in range(0, len(self.__X)):
                self.__W[0] = self.__W[0] - self.__alpha*hx[i] - self.__alpha*self.__lmbd*np.sign(self.__W[0])
                for j in range(1, len(self.__W)):
                    self.__W[j] = self.__W[j] - self.__alpha*hx[i]*self.__X[i][j-1] - self.__alpha*self.__lmbd*np.sign(self.__W[j])
            new_cost = self.__costFunction()
            print("Epoch: " + str(t) + "\t Cost: " + str(new_cost))
            
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
        
LARbatch = LeastAngleRegressor(X_train, y_train, 100, 2, 'batch')
LARbatch.printweights()

LARstochastic = LeastAngleRegressor(X_train, y_train, 100, 2, 'stochastic')
LARstochastic.printweights()
        