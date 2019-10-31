#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 23:30:40 2019

@author: kushagra
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

dataset = pd.read_excel('data4.xlsx', header  = None)
X = dataset.iloc[:, :4].values
y = dataset.iloc[:, 7:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

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
    label_cnt /= 4
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
mu3 , cov3 = get_mu_cov(3)

def gauss(x, mu, cov):
    log_val = 0
    for i in range(len(x)):
        coeff = 1 / (np.sqrt(2 * np.pi) * cov[i])
        log_val = log_val + np.log(coeff * np.exp(- ((x[i] - mu[i])**2)/(2 * (cov[i]**2))))
    return np.exp(log_val)

labels = []

for i in range(X_test.shape[0]):
    log_vals = [0, 0, 0]
    log_vals[0] = gauss(X_test[i], mu1, cov1) * get_prior(1)
    log_vals[1] = gauss(X_test[i], mu2, cov2) * get_prior(2)
    log_vals[2] = gauss(X_test[i], mu3, cov3) * get_prior(3)
    label = np.argmax(log_vals) + 1
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
