
# coding: utf-8

# In[4]:

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import pandas as pd 
from sklearn.model_selection import cross_val_predict
import sys


# In[5]:

get_ipython().system(u'{sys.executable} -m pip install pandas')


# In[1]:

import os 
os.getcwd()


# In[6]:

data = pd.read_csv('data.csv')


# In[7]:

data.columns


# In[8]:

data2 = data.iloc[1:101, :]

data2.columns
len(data2)

data2.describe()


# Plots

# In[15]:

#Count- Bar Plot  
plot1 = sns.countplot(x="number_diagnoses", data= data2)




# In[16]:

#Vertical Scatter Plot 
plot2 = sns.catplot(x="number_diagnoses", y="num_procedures", data=data2);


