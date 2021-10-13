# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 20:48:15 2021

@author: Zikantika
"""


from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


iris = datasets.load_iris()

X, y = iris.data[:, :2], iris.target



print(X)

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
lr = linear_model.LogisticRegression(C=1e5)
 
#
#Supervised learning
lr.fit(X, y)
#Fit the model to the data

knn.fit(X_train, y_train)

svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))

svc.fit(X_train, y_train) 
#Unsupervised Learning
#Fit the model to the data
#Fit to data, then transform it
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10)
y_pred = kmeans.fit_predict(X_train)

kmeans.fit(X_train)

# The lowest SSE value
print(kmeans.inertia_)
# Final locations of the centroid
print(kmeans.cluster_centers_)
# The number of iterations required to converge
print(kmeans.n_iter_)
# first five predicted labels 
print(kmeans.labels_[:5])

pca = PCA(n_components=2)
pca.fit(X)

pca_model = pca.fit_transform(X_train) 




#y_pred = svc.predict(np.random.random((2,5)))
y_pred = lr.predict(X_test)
y_pred = knn.predict_proba(X_test) 
y_pred = kmeans.predict(X_test)
