#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:01:10 2019

@author: kushagra
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

dataset = pd.read_excel('data4.xlsx', header  = None)
X = dataset.iloc[:, :4].values
y = dataset.iloc[:, 7:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

from sklearn.preprocessing import StandardScaler
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.fit_transform(X_test)

def create_model(y_train, label):
    y_model = [0]*len(y_train)
    for i in range(len(y_model)):
        if y_train[i] == label:
            y_model[i] = 1
    return y_model

y_model1 = create_model(y_train, 1)

y_model2 = create_model(y_train, 2)

y_model3 = create_model(y_train, 3)

class LogisticRegressor:
    '''Logistic Regressor class that is trained
    using both Batch and Stochastic Gradient Descent'''
    
    def __init__(self, X, Y, epochs, gradientDescent):
        self.__X = X
        self.__Y = Y
        self.__epochs = epochs
        self.__W = [0.5]*(X.shape[1])
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
            pred = self.__sigmoid(pred)
            predictions.append(pred)
        return predictions

LR1 = LogisticRegressor(X_train, y_model1, 100, 'batch')
LR2 = LogisticRegressor(X_train, y_model2, 100, 'batch')
LR3 = LogisticRegressor(X_train, y_model3, 100, 'batch')
pred1 = LR1.predict(X_test)
pred2 = LR2.predict(X_test)
pred3 = LR3.predict(X_test)

labels = []

for i in range(len(y_test)):
    label = np.argmax((pred1[i], pred2[i], pred3[i])) + 1
    labels.append(label)

IA1, IA2, IA3, OA = 0, 0, 0, 0

#confusion matrix 

u = [[0 for j in range(4)] for i in range(4)]

for i in range(len(y_test)):
    u[y_test[i][0]][labels[i]] += 1
    
IA1 = u[1][1] / sum(u[1])
IA2 = u[2][2] / sum(u[2])
IA3 = u[3][3] / sum(u[3])
OA = (u[1][1] + u[2][2] + u[3][3]) / (sum(u[1]) + sum(u[2]) + sum(u[3]))

print("IA1: " + str(IA1))
print("IA2: " + str(IA2))
print("IA3: " + str(IA3))
print("OA: " + str(OA))
