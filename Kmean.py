# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 08:38:58 2019

@author: suraj
"""
import numpy as n
import matplotlib.pyplot as plt
import pandas as pd
 
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

wcss=[]
for i in range(1,11):
    from sklearn.cluster import KMeans
    kmeans= KMeans(n_clusters=i,init='k-means++',n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of cluster')
plt.xlabel('WCSS')
plt.show()
    
#Kmeans data into mall dataset
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

#visulation of Cluster
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='pink',label='Careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='black',label='Sensable')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='Centriod')
plt.title('Clusters of clients')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.show()

    

