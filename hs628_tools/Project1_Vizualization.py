
# coding: utf-8

# In[3]:

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
#plotly.tools.set_credentials_file(username='hhowell', api_key='BRP1aHhfTHwkqnvalVQL')


# In[4]:

data = pd.read_csv('data.csv')
data2 = data.iloc[1:101, :]


# In[5]:

###Bar Plot-Count

#This function counts all instances in a variable and creates a barplot to display the results 
#@param: The variable to be counted, the x_axis label, title of the graph
#@return: A count bar plot of the variable inputted 
def count_and_print(variable, x_label, title):
    def count(variable):
        new_dic = {}
        c = (len(variable)-1)
        for index, value in enumerate(variable):
            count = 0
            if((value in new_dic) == True):
                pass
            else:
                for index2, value2 in enumerate(variable):
                    if value == value2:
                        count = count + 1
                    if index2 == (len(variable)-1):
                        new_dic[value] =  count
        return new_dic

    variable = variable

    count_dictionary = count(variable)
    from collections import OrderedDict
    from operator import itemgetter

    count_ordered = (OrderedDict(sorted(count_dictionary.items(), key = itemgetter(0), reverse = False)))

    def key_list(list):
        keys_list= []
        for key in list.keys():
            keys_list.append(key)
        return keys_list

    def value_list(list):
        values_list= []
        for value in list.values():
            values_list.append(value)
        return values_list
    keys = key_list(count_ordered)
    values = value_list(count_ordered)
#Plot the variable
    f1 = (go.Bar(
                x=keys,
                y=values
        )
           )
    layout = go.Layout(
        title= title,
        xaxis=dict(
            title=x_label,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
                )
            )
    )
    
    data = [f1]
    fig = go.Figure(data=data,layout=layout)
    return py.iplot(fig, filename='basic-bar')

count_and_print(data2['number_diagnoses'], "Number of Diagnoses", "Count of Number of Diagnoses")


# In[7]:

def scatter(x,y,colorby,xlabel,ylabel,title):
    trace1 = go.Scatter(
        x = x,
        y = y,
        mode='markers',
        marker=dict(
            size=16,
            color = data2[colorby], #set color equal to a variable
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



# In[8]:

#BoxPlot
#This function creates a box plot of the two variables the user decides to examine
#@param: the x and y vales, along with the labels and the title of the graph
#@return: A box plot
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


# In[30]:


def question(Q1):
    if Q1 == 2:
        x1 = input("Please enter X variable:")
        x = data2[x1]
        x2 = input("Please enter X variable:")
        xx = data2[x2]
        y = input("Please enter Y variable:")
        x_lab = input("Please enter the label for the X-Axis:")
        y_lab = input("Please enter the label for the Y-Axis:")
        title = input("Please enter the title of the graph:")
        return scatter(x,xx, y, x_lab, y_lab, title);

    if Q1 == 3:
        x1 = input("Please enter a variable:")
        x = data2[x1]
        x_lab = input("Please enter the label for the X-Axis:")
        title = input("Please enter the title of your graph:")
        return count_and_print(x, x_lab, title);


    if Q1==4:
        x = data2[input("Please X variable:")]
        y = data2[input("Please Y variable:")]
        x_lab = input("Please enter the label for the X-Axis:")
        y_lab = input("Please enter the label for the Y-Axis:")
        title = input("Please enter the title of your graph:")
        return boxplot(x,y,x_lab,y_lab,title);
    
question(4)   
  

