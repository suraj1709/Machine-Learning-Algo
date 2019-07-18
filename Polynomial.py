# -*- coding: utf-8 -*-
"""
Created on Mon May 20 22:44:20 2019

@author: suraj
"""

import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt 
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
linrreg=LinearRegression()
linrreg.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
ployreg=PolynomialFeatures(degree=3)
x_poly=ployreg.fit_transform(X)
linreg2=LinearRegression()
linreg2.fit(x_poly,Y)



x_grid=np.arange(min(X),max(X),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(x_grid,linreg2.predict(ployreg.fit_transform(x_grid)),color='black')
plt.title('Salary vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

linrreg.predict(6.5)


linreg2.predict(ployreg.fit_transform(6.5))



