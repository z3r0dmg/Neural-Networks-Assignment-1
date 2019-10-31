#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 23:40:32 2019

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

def get_prior(label):
    prob = 0
    for i in y_train:
        if i == label:
            prob += 1
    return prob / y_train.shape[0]
    
def get_mu_cov(label):
    label_cnt = 0
    mu = [0]*X_train.shape[1]
    cov = [0]*X_train.shape[1]
    for i in range(X_train.shape[1]):
        for j in range(X_train.shape[0]):
            if y_train[j] == label:
                mu[i] = mu[i] + X_train[j][i]
                label_cnt += 1
    label_cnt /= X_train.shape[1]
    #print(label_cnt)
    for i in range(X_train.shape[1]):
        mu[i] = mu[i]/label_cnt
    for i in range(X_train.shape[1]):
        for j in range(X_train.shape[0]):
            if y_train[j] == label:
                cov[i] = cov[i] + (X_train[j][i] - mu[i])**2
        cov[i] = np.sqrt(cov[i]) 
    return mu, cov

mu1 , cov1 = get_mu_cov(1)
mu2 , cov2 = get_mu_cov(2)

def gauss(x, mu, cov):
    log_val = 0
    for i in range(len(x)):
        coeff = 1 / (np.sqrt(2 * np.pi) * cov[i])
        log_val = log_val + np.log(coeff * np.exp(- ((x[i] - mu[i])**2)/(2 * (cov[i]**2))))
    return np.exp(log_val)

labels = []

for i in range(X_test.shape[0]):
    log_vals = [0, 0]
    log_vals[0] = gauss(X_test[i], mu1, cov1) * get_prior(1)
    log_vals[1] = gauss(X_test[i], mu2, cov2) * get_prior(2)
    label = np.argmax(log_vals) + 1
    labels.append(label)

TP, TN, FP, FN = 0, 0, 0, 0
for i in range(len(X_test)):
    if y_test[i] == [1]:
        if labels[i] == 1:
            TN += 1
        else:
            FP += 1
    else:
        if labels[i] == 1:
            FN += 1
        else:
            TP += 1

accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print('accuracy: ' + str(accuracy))
print('sensitivity: ' + str(sensitivity))
print('specificity: ' + str(specificity))