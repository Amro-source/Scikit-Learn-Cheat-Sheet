# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 20:42:33 2021

@author: Zikantika
"""

from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
iris = datasets.load_iris()

X, y = iris.data[:, :2], iris.target



print(X)

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

Q=accuracy_score(y_test, y_pred)

print(Q)


from sklearn.grid_search import GridSearchCV


params ={"n_neighbors" : np.arange(1,3),"metric" : ["euclidean","cityblock"]}

grid =GridSearchCV(estimator=knn, param_grid=params)

grid.fit(X_train, y_train)

print(grid.best_score_)

print(grid.best_estimator_.n_neighbors)

