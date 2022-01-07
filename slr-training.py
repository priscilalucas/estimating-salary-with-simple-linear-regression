# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 22:16:39 2022

@author: Priscila
"""

#SIMPLE LINEAR REGRESSION#

#DataPreprocessing
#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

#Split the dataset into training & test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)


#Training SLR on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#Predict the Test set results
y_pred = regressor.predict(X_test)

#Acurracy of the model to predict unknown observations
accuracy = regressor.score(X_test, y_test)
print(int(round(accuracy * 100)))


#Visualize the Training set results
plt.scatter(X_train, y_train, color = "gray")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


#Visualize the Test set results
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "black")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Getting the parameters of the equation.
regressor.coef_
regressor.intercept_

#Predicting a Salary for a person with 15 years of experience
y_pred = regressor.predict(np.array([15]).reshape(-1, 1))
print(y_pred)

#Predicting a Salary for a person with 12 years of experience
y_pred = regressor.predict(np.array([12]).reshape(-1, 1))
print(y_pred)
#or this simple code line does work just fine!
print(regressor.predict([[12]]))









