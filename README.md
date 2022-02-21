# Predicting new employees salaries
Simple Linear Regression algorithm trained with Python

## Introduction

This project was created for learning purposes. In order to practice ML algorithms in Python, this first and very simple ML algorithm is a regression where we want to predict the salary (dependent variable) based on years of experience (independent variable). The parameters learnt can be used on the equation Salary = b0 + b1*Years of Experience to predict how much a company should pay to a new employee based on their experience.

## Technologies

Python 3.9 (IDE: Spyder)
libraries: scikit learn, numpy and matplotlib

## Results & Illustrations
The class LinearRegression() in Python fitted the linear model on the training dataset.

The predict function was used on the test dataset.

The model accuracy score is 99%.

![Alt text](./images/trainingsetplot.png)
![Alt text](./images/testsetplot.png)

From the smart parameters we get the the equation Salary = 2678.009 + 9312.575*Years of Experience
So, if we want to predict the salary of a person with 15 years of experience based on this equation, we should pay an annual salary of about U$166,468 to this person.

## References
The whole code can be viewed on the file slr-training.
The codes were based on the course Machine Learning from A-Z: Hands-on Python & R in Data Science from Udemy.
