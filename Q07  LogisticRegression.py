#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 15:10:46 2019

@author: kushagra
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

dataset = pd.read_excel('data3.xlsx', header  = None)
X = dataset.iloc[:, :4].values
y = dataset.iloc[:, 4:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

from sklearn.preprocessing import StandardScaler
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.fit_transform(X_test)

class LogisticRegressor:
    '''Logistic Regressor class that is trained
    using both Batch and Stochastic Gradient Descent'''
    
    def __init__(self, X, Y, epochs, gradientDescent):
        self.__X = X
        self.__Y = Y
        self.__epochs = epochs
        self.__W = [0.3]*(X.shape[1])
        self.__alpha = 0.00005
        self.__cost = 0.0
        for i in self.__Y:
            i = i - 1
        if(gradientDescent == 'batch'):
            self.__batchgradientdescent()
        else:
            self.__stochasticgradientdescent()
    
    def __sigmoid(self, z):
        '''Method to return sigmoid of z'''
        return 1/(1 + np.exp(-z))
    
    def __multiply_weights(self):
        '''Method to multiply weights with a feature vector'''
        
        h = [0]*(self.__X.shape[0])
        for i in range(0, len(self.__X)):
            for j in range(0, len(self.__W)):
                h[i] = h[i] + self.__W[j]*self.__X[i][j-1]
        return h
    
    def __hypothesis(self, z):
        '''Method to get value of hypothesis'''
        
        hx = []
        for i in z:
            hx.append(self.__sigmoid(i))
        return hx
    
    def __costFunction(self, h):
        '''Method to calculate the
        cost function'''
        
        cost = 0
        for i in range(len(h)):
            cost = cost + (self.__Y[i] * np.log(h[i]) - (1 - self.__Y[i]) * np.log(1 - h[i]))
        return cost
    
    def __batchgradientdescent(self):
        for t in range(1, self.__epochs + 1):
            sum_of_values = [0]*len(self.__W)
            z = self.__multiply_weights()
            hx = self.__hypothesis(z)
            for i in range(0, len(self.__X)):
                temp = self.__Y[i] - hx[i]
                sum_of_values[0] = sum_of_values[0] + temp
                for j in range(1, len(self.__W)):
                    sum_of_values [j] = sum_of_values[j] + temp*self.__X[i][j-1]
            for i in range(0, len(self.__W)):
                self.__W[i] = self.__W[i] - self.__alpha*sum_of_values[i]
            new_cost = self.__costFunction(hx)
            print("Epoch: " + str(t) + "\tCost: " + str(new_cost))
    
    def __stochasticgradientdescent(self):
        self.__alpha = self.__alpha/10
        for t in range(1, self.__epochs + 1):
            z = self.__multiply_weights()
            hx = self.__hypothesis(z)
            for i in range(0, len(self.__X)):
                for j in range(0, len(self.__W)):
                    self.__W[j] = self.__W[j] - self.__alpha*(self.__Y[i] - hx[i])
            new_cost = self.__costFunction(hx)
            print("Epoch: " + str(t) + "\t Cost: " + str(new_cost))
            
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            pred = 0
            for j in range(0, len(x)):
                pred = pred + (self.__W[j]*x[j])
            pred = self.__hypothesis(pred)
            print(pred)
            if pred[0] < 0.5:
                pred[0] = 1
            else:
                pred[0] = 2
            predictions.append(pred)
        #accuracy calculation
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(len(X_test)):
            if y_test[i] == [1]:
                if predictions[i] == [1]:
                    TN += 1
                else:
                    FP += 1
            else:
                if predictions[i] == [1]:
                    FN += 1
                else:
                    TP += 1
        
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        
        print('accuracy: ' + str(accuracy))
        print('sensitivity: ' + str(sensitivity))
        print('specificity: ' + str(specificity))
        
        return predictions
      
LRbatch = LogisticRegressor(X_train, y_train, 100, 'batch')
LRstochastic = LogisticRegressor(X_train, y_train, 100, 'stochastic')
LRbatch.predict(X_test)
LRstochastic.predict(X_test)
