#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 23:47:43 2019

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

def create_model(X, y, label1, label2):
    X_model = []
    y_model = []
    for i in range(len(y)):
        if y[i] == label1 or y[i] == label2:
            X_model.append(X[i])
            if y[i] == label1:
                y_model.append(0)
            else:
                y_model.append(1)
    return X_model, y_model

X_model1v2, y_model1v2 = create_model(X_train, y_train, 1, 2)
X_model2v3, y_model2v3 = create_model(X_train, y_train, 2, 3)
X_model3v1, y_model3v1 = create_model(X_train, y_train, 3, 1)

class LogisticRegressor:
    '''Logistic Regressor class that is trained
    using both Batch and Stochastic Gradient Descent'''
    
    def __init__(self, X, Y, epochs, gradientDescent, label1, label2):
        self.__X = X
        self.__Y = Y
        self.__epochs = epochs
        self.__W = [0.3]*(4)
        self.__alpha = 0.00005
        self.__cost = 0.0
        self.__label1 = label1
        self.__label2 = label2
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
        
        h = [0]*(len(self.__X))
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
                temp = -self.__Y[i] + hx[i]
                for j in range(0, len(self.__W)):
                    sum_of_values [j] = sum_of_values[j] + temp*self.__X[i][j]
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
            if pred<0.5:
                pred = self.__label1
            else:
                pred = self.__label2
            predictions.append(pred)
        return predictions
#
LR1 = LogisticRegressor(X_model1v2, y_model1v2, 200, 'batch', 1, 2)
LR2 = LogisticRegressor(X_model2v3, y_model2v3, 200, 'batch', 2, 3)
LR3 = LogisticRegressor(X_model3v1, y_model3v1, 200, 'batch', 3, 1)

pred1 = LR1.predict(X_test)
pred2 = LR2.predict(X_test)
pred3 = LR3.predict(X_test)

labels = []

for i in range(len(pred1)):
    temp = [pred1[i], pred2[i], pred3[i]]
    predicted = 2
    if temp.count(1) >= 2:
        predicted = 1
    elif temp.count(3) >=2:
        predicted = 3
    labels.append(predicted)
        
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
