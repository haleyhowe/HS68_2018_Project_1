
# coding: utf-8

# In[12]:

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import pandas as pd 
from sklearn.model_selection import cross_val_predict
import sys
plotly.tools.set_credentials_file(username='hhowell', api_key='BRP1aHhfTHwkqnvalVQL')
import plotly.plotly as py
import plotly.graph_objs as go


# !{sys.executable} -m pip install plotly

# In[2]:

import os 
os.getcwd()


# In[3]:

data = pd.read_csv('data.csv')


# In[4]:

data.columns


# In[5]:

data2 = data.iloc[1:101, :]

data2.columns
len(data2)

data2.describe()


# Plots

# In[165]:

#Count- Bar Plot  
def BarPlot(x, data):
    return sns.countplot(x=x , data= data);

BarPlot("number_diagnoses",data2)


# In[173]:

#Scatter Plot
def scatter(x,y,z):
    trace1 = go.Scatter(
        x = x,
        y = y,
        mode='markers',
        marker=dict(
            size=16,
            color = data2[z], #set color equal to a variable
        )
        )
    data = [trace1]
    fig = go.Figure(data=data,layout=layout)
    return py.iplot(fig, filename='scatter-plot-with-colorscale');

y = data2['number_diagnoses']
x = data2.num_procedures
z = 'readmission_30d'
test_x = data2['readmission_30d']

scatter(x,y,z)



# In[172]:

#BoxPlot
def boxplot(x,y, y_label, x_label, chartTitle):
    data = [go.Box(x= x,
            y= y)]

    layout = go.Layout(
        title = chartTitle,
       yaxis=dict(
            title= y_label,
            zeroline=False
        ),
        xaxis=dict(
            title=x_label,
            zeroline=False
        )
    )

    fig = go.Figure(data=data,layout=layout)
    return py.iplot(fig, filename='test');
    
test_x = data2['readmission_30d']
test_y = data2['num_lab_procedures']
ylab = "Testing for y"
xlab = "Testing for x"
title = "Testing for title"


boxplot(test_x,test_y,ylab, xlab, title)

