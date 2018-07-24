
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

# In[ ]:

data = pd.read_csv('data.csv')
data2 = data.iloc[1:101, :]


# In[ ]:

Q1 = 2
if Q1 == 2:
    data = data2
    x = input("Please enter x variable:")
    #y = input("Please enter y variable:")
    #x_lab = input("Please enter X-axis label:")
    #y_lab = input("Please enter Y-axis label:")
    #title_1 = input("Please enter the title of your graph:")
    
    print (type(x))
    #print(data.x)
    
    #y_new= data.y
    #scatter(x_new,y_new,colorby,x_lab,y_lab,title_1)


# In[310]:

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

    count_ordered = (OrderedDict(sorted(hello.items(), key = itemgetter(0), reverse = False)))

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


# In[318]:

#Scatter Plot
#This function creates a scatter plot of two variables, the user can also color the values by
#some sort of classification variable 
#@param: Two variables the user wants to compare(variable_1, variable_2), X-axis label, Y-axis 
#label, title of the graph, and a colorby value that the user can color each point by 
#@return: Scatter plot of two variables being compared 

def scatter(x,y,colorby,x_label,y_label,title):
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

