# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 22:34:54 2022
@author: Abdelrahman-Ahmed-Samir
"""


import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  



data_set= pd.read_csv('USA_Housing1.csv')  

X = data_set.iloc[:, :-1]
Y = data_set.iloc[:, 6]

x= data_set.iloc[:, :-1].values         # Features
y= data_set.iloc[:, 6].values           # Target




                       # Encode Features


status = pd.get_dummies(data_set['Address'])
status.head()
status = pd.get_dummies(data_set['Address'], drop_first = True)
data_set = pd.concat([data_set, status], axis = 1)
data_set.head()
data_set.drop(['Address'], axis = 1, inplace = True)

X = X.apply(pd.to_numeric, errors="coerce")
Y = Y.apply(pd.to_numeric, errors="coerce")
X.fillna(0,inplace=True)
Y.fillna(0,inplace=True)

                        # Function End




x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)          
                        
                        
regressor = LinearRegression()          # Linear Regression Model
regressor.fit(x_train,y_train)       

y_pred= regressor.predict(x_test)
regression_diff = pd.DataFrame({'Actual Value':y_test , "Predicted Value: " : y_pred})
regression_diff.head()                 
mtp.scatter(y_test,y_pred)
         

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error (MAE): ",mean_absolute_error(y_test, y_pred))


                     #R^^2 Function
def r2_score(y_test, y_pred):
        '''
        r^^2 = 1 - (rss/tss)
        rss = sum_{i=0}^{n} (y_i - y_hat)^2
        tss = sum_{i=0}^{n} (y_i - y_bar)^2
        '''
        y_values = y_test.values
        y_average = nm.average(y_values)
        residual_sum_of_squares = 0
        total_sum_of_squares = 0
        for i in range(len(y_values)):
            residual_sum_of_squares += (y_values[i] - y_pred[i])**2
            total_sum_of_squares += (y_values[i] - y_average)**2
            return 1 - (residual_sum_of_squares/total_sum_of_squares)
                    
                     #R^^2 Function End
                    
score = r2_score(y_test,y_pred)   #Calling the R2_Score Function
print("R^^2 Score: " , score)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
mse = mean_squared_error(y_test,y_pred)
rmse = (nm.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)


"Calculating MSE"
MSE = nm.square(nm.subtract(y_test, y_pred)).mean()

print("Mean Squared error is :", MSE)
print("r2: " , r2)
print("MSE: " , mse)
print("RMSE : " , rmse)

print("Y-intercept: " , regressor.intercept_)

# coefficients and how we can interpret them.
coeff_df = pd.DataFrame(regressor.coef_,X.columns,columns=['Coefficient'])


"""
# gradient descent star
def grad_desc(X, Y, rate = 0.001, iterations = 100):
    w = nm.zeros((X.shape[1], 1))
    for _ in range(iterations):
        errors = Y - X.dot(w)
        grad = -(X.T).dot(errors)
        w = w - rate*grad
    return w

w = grad_desc(X, Y)

print("Weights" , w)
"""
"""
# Gradient Descent Function
def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size  # number of training examples
    for i in range(num_iters):
        y_hat = nm.dot(X, theta)
        theta = theta - alpha * (1.0/m) * nm.dot(X.T, y_hat-y)
    return theta


#Grandient Desc. End

gradientDescent(X, y, theta, alpha, m)
"""

