
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

# In[6]:

#Count- Bar Plot  
plot1 = sns.countplot(x="number_diagnoses", data= data2)


# In[7]:

#Vertical Scatter Plot 
plot2 = sns.catplot(x="number_diagnoses", y="num_procedures", data=data2);


# In[18]:

sns.set_color_codes("muted")
sns.barplot(x="number_diagnoses", y="num_procedures", data=data2,
            label="Diagnoses vs Procedures", color="g")


# In[37]:

import plotly
plotly.tools.set_credentials_file(username='hhowell', api_key='BRP1aHhfTHwkqnvalVQL')



# In[67]:


def scatter(y):
    trace1 = go.Scatter(
        y = y,
        mode='markers',
        marker=dict(
            size=16,
            color = np.random.randn(500), #set color equal to a variable
            colorscale='Viridis',
            showscale=True
        )
    )
    data = [trace1]

    return py.iplot(data, filename='scatter-plot-with-colorscale');

y = data2['number_diagnoses']
x = data2.num_procedures

scatter(y)



# In[64]:


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
    
j = data2['readmission_30d']
k = data2['num_lab_procedures']
ylab = "testing for y"
xlab = "testing for x"
title = "testing for title"


boxplot(j,k,ylab, xlab, title)


# In[ ]:

def boxplot(x,y, y_label, x_label, chartTitle):

    x = data2['readmission_30d']
    y = data2['num_lab_procedures']
    data = [go.Box(x= x,
            y= y)]

    layout = go.Layout(
        title = "This is the Title",
       yaxis=dict(
            title='This is the Y axis',
            zeroline=False
        ),
        xaxis=dict(
            title='This is the X axis',
            zeroline=False
        )
    )

    fig = go.Figure(data=data,layout=layout)
    py.iplot(fig, filename='jupyter-basic_bar');


# In[ ]:



