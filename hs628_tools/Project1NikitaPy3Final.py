
# coding: utf-8

# In[179]:

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

# In[181]:

data = pd.read_csv('data.csv')


# In[183]:

data2 = data.iloc[1:101, :]

data2.describe()


# Plots

# In[184]:

#Count- Bar Plot  
def BarPlot_count(x, data):
    return sns.countplot(x=x , data= data);

BarPlot_count("number_diagnoses",data2)


# In[211]:

def histogram_count(x):
    trace1 = go.Histogram(
        x=x,
        histfunc = 'count',
        name='control',
            xbins=dict(
                start=0,
                end=9.0,
                size=0.5
            ),
        marker=dict(
            color='#FFD7E9',
        ),
            opacity=0.75
    )
    data = [trace1]
    
    layout = go.Layout(
        title='Count of Number of Diagnoses',
        xaxis=dict(
            title='Number of Diagnoses'
        ),
        yaxis=dict(
            title='Count'
        ),
        bargap=1,
        bargroupgap=1
    )
    fig = go.Figure(data=data, layout = layout)
    return py.iplot(fig, filename='styled histogram')

histogram_count(data2['number_diagnoses'])


# In[241]:



def count(variable):
    new_dic = {}
    count = 0
    for index, value in enumerate(variable):
        for index2, value2 in enumerate(variable):
            if (value in new_dic == True):
                pass
            if value == value2:
                count = count +1 
            if index2 == len(variable):
                new_dic = {value:count}
                return new_dic

                


# In[240]:

dictionary = {'orange':1}
print(dictionary)

print ('orange' in dictionary)
len(data2['number_diagnoses'])
x = data2['number_diagnoses']


# In[196]:

#Scatter Plot
def scatter(x,y,colorby,x_label,y_label,title):
    trace1 = go.Scatter(
        x = x,
        y = y,
        mode='markers',
        marker=dict(
            size=16,
            color = data2[z], #set color equal to a variable
        )
    )
    layout = go.Layout(
        title= title,
        xaxis=dict(
            title=xlabel,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
                )
            ),
        yaxis=dict(
            title= ylabel,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
                )
            )
    )
    data = [trace1]
    fig = go.Figure(data=data,layout=layout)
    return py.iplot(fig, filename='scatter-plot-with-colorscale');

y = data2['number_diagnoses']
x = data2.num_procedures
colorby = 'readmission_30d'
test_x = data2['readmission_30d']
x_label = "Number of Procedures"
y_label = "Number of Diagnoses"
title = "Number of Procedures vs Number of Diagnoses"
scatter(x,y,colorby,xlabel,ylabel,title)



# In[197]:

#BoxPlot
def boxplot(x,y, y_label, x_label, title):
    data = [go.Box(x= x,
            y= y)]

    layout = go.Layout(
        title = title,
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
ylab = "Num Lab Procedures"
xlab = "Readmission_30d"
title = "Num Lab Procedures by Readmission"


boxplot(test_x,test_y,ylab, xlab, title)

