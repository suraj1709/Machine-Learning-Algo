# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 22:37:52 2019

@author: suraj
"""


import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt 
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x_1=LabelEncoder()
X[:,1]=labelencoder_x_1.fit_transform(X[:,1])
labelencoder_x_2=LabelEncoder()
X[:,2]=labelencoder_x_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.20,random_state =0)


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
x_test=sc_x.transform(x_test)


import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

#classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=10,epochs=100)

import tensorflow as tf
print(tf.__version__)


#create your classifier code


y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)