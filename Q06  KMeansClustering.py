import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel('data2.xlsx', header  = None)
X = dataset.iloc[:, :].values

labels = []

class KMeansClassifier:
    '''Clustering class based on K Means algorithm'''
    
    def __init__(self, X, clusters, num_iter):
        self.__X = X
        self.__K = clusters
        self.__num_iter = num_iter
        self.__class_labels = []
        self.__cluster()
    
    def __select_centers(self):
        '''Method to select any K random centers'''
        
        np.random.RandomState(10)
        perm = np.random.permutation(self.__X.shape[0])
        centers = self.__X[perm[:self.__K]]
        return centers
    
    def __get_centers(self, labels):
        centers = np.zeros((self.__K, self.__X.shape[1]))
        for k in range(self.__K):
            centers[k, :] = np.mean(self.__X[labels == k, :], axis=0)
        return centers
    
    
    def __get_distances(self, centers):
        distances = np.zeros((self.__X.shape[0], self.__K))
        for k in range(self.__K):
            row_dif = norm(self.__X - centers[k, :], axis=1)
            distances[:, k] = np.square(row_dif)
        return distances
    
    def __get_closest(self, distances):
        return np.argmin(distances, axis=1)
    
    def __get_av(self, labels, centers):
        distances = np.zeros(X.shape[0])
        for k in range(self.__K):
            distances[labels == k] = norm(X[labels == k] - centers[k], axis=1)
        return np.sum(np.square(distances))
    
    def __cluster(self):
        '''Method which decides the labels based on minimum distance 
        from cluster centers'''
        
        centers = self.__select_centers()
        for i in range(self.__num_iter):
            prev_centers = centers
            distances = self.__get_distances(prev_centers)
            self.__class_labels = self.__get_closest(distances)
            centers = self.__get_centers(self.__class_labels)
            if np.all(prev_centers == centers):
                break
    
    def print_classes(self):
        for i in range(len(self.__X)):
            labels.append(self.__class_labels[i])
            print(str(self.__X[i]) + " " + str(self.__class_labels[i] + 1))
        
KMC = KMeansClassifier(X, 2, 105)
KMC.print_classes()

#plot 6 scatter plots each with two features to show the labelling

for i in range(4):
    for j in range(i+1,4):
        plt.scatter(X[:,i], X[:,j], c=labels)
        plt.xlabel('feature ' + str(i))
        plt.ylabel('feature ' + str(j))
        plt.show()            