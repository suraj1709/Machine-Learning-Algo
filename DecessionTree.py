# -*- coding: utf-8 -*-
"""
Created on Fri May 24 22:39:48 2019

@author: suraj
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 22 20:46:46 2019

@author: suraj
"""

import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt 
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

"""from sklearn.cross_validation import train_test_split
X_train,y_train,y_test,x_test=train_test_split(X,Y,test_size=.2,random_state =0)"""

"""from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
X[:,3]=labelencoder_x.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()"""

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)
y_pred=regressor.predict(5.6)


plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='black')
plt.title('Salary vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()



x_grid=np.arange(min(X),max(X),0.01)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(X,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='black')
plt.title('Salary vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


