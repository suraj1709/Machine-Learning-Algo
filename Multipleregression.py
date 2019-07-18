# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:47:18 2019

@author: suraj
"""
import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt 
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

X=X[:,1]

#Catagorical data 

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
X[:,3]=labelencoder_x.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#Slipt into train and test set
from sklearn.cross_validation import train_test_split
X_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.2,random_state =0)

#Regression 

from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(x_test)

#Backward Elimination 
import statsmodels.formula.api as sf
X=np.append(arr= np.ones((50,1)).astype(int),values=X,axis=1)
x_otp=X[:,[0,3]]
regressor_OLS=sf.OLS(endog=Y,exog=x_otp).fit()
regressor_OLS.summary()

import numpy as np
import statsmodels.formula.api as sm
def backwardelimination(x1,sl):
    numvars=len(x1[0])
    for i in range(0,numvars):
        regressor_OLS=sm.OLS(y,x1).fit()
        maxvar=max(regressor_OLS.pvalues).astype(float)
        if maxvar>sl:
            for j in range(0,numvars-i):
                if(regressor_OLS.pvalues[i].astype(float)==maxvar):
                    x1 = np.delete(x1,j,1)
                    regressor_OLS.summary()
                    return x1


SL = 0.05
x_otp = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardelimination(x_otp, SL)

