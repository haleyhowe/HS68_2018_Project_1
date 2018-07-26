import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

import plotly.plotly as py
import plotly.graph_objs as go

from Test import data_split as ds

def model_metrics(X_train, y_train, X_test, y_test):
    """
    This function will print out the linear model metrics including the coefficients, mean squared error, mean absolute error, root mean squared error, and Variance score.

        Parameters:
            X_train: dataframe with training set split of all independent variables 
            y_train: dataframe with training set split of the dependent variable
            X_test: dataframe with the testing set split of all the independent variables
            y_test: dataframe with the testing set split of the dependent variable
            
    """
    reg_obj = linear_model.LinearRegression()
    reg_obj.fit(X_train, y_train)
    y_pred = reg_obj.predict(X_test)

    # The coefficients
    print('Coefficient(s): \n', reg_obj.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    # The mean absolute error 
    print("Mean absolute error: %.2f"
          % mean_absolute_error(y_test, y_pred))
    # The mean squared error
    print("Root mean squared error: %.2f"
          % np.sqrt(mean_squared_error(y_test, y_pred)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))


def plot_model(X_train, y_train, X_test, y_test):
    """
    This function will plot out the simple linear regression model with the points as the actual testing set y values and the regression line to compare predicted y values. 

        Parameters:
            X_train: dataframe with training set split of all independent variables 
            y_train: dataframe with training set split of the dependent variable
            X_test: dataframe with the testing set split of all the independent variables
            y_test: dataframe with the testing set split of the dependent variable
            
    """
    reg_obj = linear_model.LinearRegression()
    reg_obj.fit(X_train, y_train)
    y_pred = reg_obj.predict(X_test)
    
    def data_to_plotly(x):
        k = []
    
        for i in range(0, len(x)):
            k.append(x[i][0])
        
        return k

    p1 = go.Scatter(x=data_to_plotly(X_test), 
                y=y_test, 
                mode='markers',
                marker=dict(color=Q6),
                name ='y_test values'
               )

    p2 = go.Scatter(x=data_to_plotly(X_test), 
                y=regr.predict(X_test),
                mode='lines',
                line=dict(color=Q7, width=3),
                name ='Linear Regression Model'
                )

    layout = go.Layout(xaxis=dict(ticks='', showticklabels=True,
                              zeroline=False, title=Q4),
                   yaxis=dict(ticks='', showticklabels=True,
                              zeroline=False, title=Q5),
                   showlegend=True, hovermode='closest',
                   title=Q2
                   )

    fig = go.Figure(data=[p1, p2], layout=layout)

    return py.iplot(fig)

## MAIN
import pandas as pd 
data = pd.read_csv('data.csv')
X = sub_data.iloc[:,0:3]
Y = sub_data['readmission_30d']

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = ds.split_ratio)

Q1= input("What type of plot output? (Enter #)"+"\n"+"\n"+"1.Countplot"+"\n"+"2.Scatterplot"+"\n"+"3.Boxplot"+"\n"+"4.Simple Linear Regression Model"+"\n"+"Multiple Linear Regression Model"+"\n"

question(Q1)


if Q1 == 4:
    model_metric(X_train, X_test, y_train, y_test)

if Q1 == 5:
    Q2= input("What do you want to title the graph?"+"\n)
    Q3 = input("What is the predictor(dependent) variable name?:"+"\n")
    Q4 = input("Name of x axis:")
    Q5 = input("Name of y axis:") 
    Q6 = input("Choose color for actual y points:"+"\n")
    Q7 = input("Choose color for line:"+"\n")
    model_metrics(X_train, X_test, y_train, y_test)
    plot_model(X_train, X_test, y_train, y_test)
    return 

 

