# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:45:52 2022

@author: Champo
"""

import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  


df = pd.DataFrame(nm.random.randint(0,1000,size=(1000, 3)), columns=list('ABC'))
df.plot.scatter(x='A',y='B')

df['D'] = 5*df['A']+3*df['B']+1.5*df['C']+6        # Y = 5X1 + 3X2 + 1.5X3 + 6

df.plot.scatter(x='A',y='D')


X = df.iloc[:,:-1]          # Features (A,B,C)
Y = df.iloc[:,-1]           # Target (D)

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


#split data into train and test
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)  


regressor = LinearRegression()          # Linear Regression Model
regressor.fit(x_train,y_train)

                        
                        #Least Sqaure Method
def least_square (x,y):
    #calculating slope and y-intercept 
    mean_x = nm.mean(x)
    mean_y = nm.mean(y)
    n = len(x)
    numer = 0
    denom = 0
    for i in range(n):
        numer += (x[i] - mean_x) * (y[i] - mean_y)
        denom += (x[i] - mean_x) ** 2
        m = numer / denom                    #slope
        c = mean_y - (m * mean_x)            # c = y-mx : y-intercept
        
    print("Mean-Weight: " , m)
    print("Y-intercept: " , c)    
    # Y = mX+c          to predict any X in our dataset
    
    max_x = nm.max(X) 
    min_x = nm.min(X) 
 
    # Calculating line values x and y
    x = nm.linspace(min_x, max_x, 1000)
    y = c + m * x
 
    # Ploting Line
    mtp.plot(x, y, color='#58b970', label='Regression Line')
    # Ploting Scatter Points
    mtp.scatter(x, y, c='#ef5423', label='Scatter Plot')
 
    mtp.xlabel('X')
    mtp.ylabel('Y')
    mtp.legend()
    mtp.show()
                    #Least Square Function End
                    
                    
                    
                    #R^^2 Function
def r2_score(y_true, y_pred):
        '''
        r^^2 = 1 - (rss/tss)
        rss = sum_{i=0}^{n} (y_i - y_hat)^2
        tss = sum_{i=0}^{n} (y_i - y_bar)^2
        '''
        y_values = y_true.values
        y_average = nm.average(y_values)
        residual_sum_of_squares = 0
        total_sum_of_squares = 0
        for i in range(len(y_values)):
            residual_sum_of_squares += (y_values[i] - y_pred[i])**2
            total_sum_of_squares += (y_values[i] - y_average)**2
            return 1 - (residual_sum_of_squares/total_sum_of_squares)
                    
                    #R^^2 Function End
    
    
least_square(x,y)       #Calling the Least Square Method function


y_pred= regressor.predict(x_test)

score = r2_score(y_test,y_pred)   #Calling the R2_Score Function
print("R^^2 Score: " , score)
sk_score = regressor.score(x_test, y_test)
print(f'Scikit-Learn\'s Final R^2 score: {sk_score}')
print("")
"Calculating MSE"

MSE = nm.square(nm.subtract(y_test, y_pred)).mean()
print("Mean Squared error is :", MSE)

#print("Prediction for test set: " , y_pred)
print("")
regression_diff = pd.DataFrame({'Actual Value':y_test , "Predicted Value: " : y_pred})
regression_diff.head()   # difference between Actual and Predicted value


"""
print("")

print('Train Score: ', regressor.score(x_train, y_train))  
print('Test Score: ', regressor.score(x_test, y_test))  
"""


