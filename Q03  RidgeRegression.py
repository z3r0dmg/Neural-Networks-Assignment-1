#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:02:50 2019

@author: kushagra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

W1_array = []
W2_array = []
J_array = []

class RidgeRegressor:
    '''Ridge Regressor class that is trained
    using both Batch and Stochastic Gradient Descent'''
    
    def __init__(self, X, y, epochs, lmbd, gradientDescent):
        self.__X = X
        self.__y = y
        self.__epochs = epochs
        self.__lmbd = lmbd
        self.__W = [0.3]*(X.shape[1]+1)
        self.__alpha = 0.0008
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
        for i in range(0,len(self.__X)):
            current_cost = self.__W[0]
            for j in range(1,len(self.__W)):
                current_cost = current_cost + (self.__W[j])*(self.__X[i][j-1])
            current_cost = current_cost - self.__y[i]
            current_cost = (current_cost)*(current_cost)
            self.__cost = (self.__cost) + current_cost
        for j in self.__W:
            self.__cost = (self.__cost) + (self.__lmbd) * (j * j)
        self.__cost = 0.5*(self.__cost)
        return self.__cost
    
    def __batchgradientdescent(self):
        coeff = 1 - self.__lmbd*self.__alpha
        for t in range(1, self.__epochs + 1):
            sum_of_values = [0]*(len(self.__W))
            for i in range(0,len(self.__X)):
                temp = self.__W[0]
                for j in range(1,len(self.__W)):
                    temp = temp + (self.__W[j])*(self.__X[i][j-1])
                temp = temp - self.__y[i]
                sum_of_values[0] = sum_of_values[0] + temp
                for j in range(1,len(self.__W)):
                    sum_of_values[j] = sum_of_values[j] + (temp * self.__X[i][j-1])
            for j in range(0,len(self.__W)):
                self.__W[j] = coeff * self.__W[j] - (sum_of_values[j] * self.__alpha)
            new_cost = self.__costFunction()
            
            W1_array.append(self.__W[1])
            W2_array.append(self.__W[2])
            J_array.append(new_cost)
            
            print("Epoch: " + str(t) + " Cost: " + str(new_cost))
    
    def __stochasticgradientdescent(self):
        self.__alpha = 0.0004
        coeff = 1 - self.__lmbd*self.__alpha
        for t in range(1, self.__epochs + 1):
            for i in range(0,len(self.__X)):
                current_cost = self.__W[0]
                for j in range(1,len(self.__W)):
                    current_cost = current_cost + (self.__W[j])*(self.__X[i][j-1])
                current_cost = current_cost - self.__y[i]
                self.__W[0] = coeff * self.__W[0] - (current_cost * self.__alpha)
                for j in range(1,len(self.__W)):
                    self.__W[j] = coeff * self.__W[j] - (current_cost * self.__alpha * self.__X[i][j-1])
            new_cost = self.__costFunction()
            
            W1_array.append(self.__W[1])
            W2_array.append(self.__W[2])
            J_array.append(new_cost)
            
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
        
RRbatch = RidgeRegressor(X_train, y_train, 100, 30, 'batch')
RRstochastic = RidgeRegressor(X_train, y_train, 100, 0.1, 'stochastic')
RRbatch.printweights()
RRstochastic.printweights()

