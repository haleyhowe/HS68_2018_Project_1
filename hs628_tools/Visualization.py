import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score


def plotmodel(X_train, y_train, X_test, y_test):
    reg_obj = linear_model.LinearRegression()
    reg.fit(Qa, Qb)
    y_pred = regr.predict(Qc)

    # The coefficients
    print 'Coefficients: \n', regr.coef_
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(Qd, y_pred))
    # The mean absolute error 
    print("Mean absolute error: %.2f"
          % mean_absolute_error(Qd, y_pred))
    # The mean squared error
    print("Root mean squared error: %.2f"
          % np.sqrt(mean_squared_error(Qd, y_pred)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))


def data_to_plotly(x):
    k = []
    
    for i in range(0, len(x)):
        k.append(x[i][0])
        
    return k

p1 = go.Scatter(x=data_to_plotly(X_test), 
                y=y_test, 
                mode='markers',
                marker=dict(color=Q4),
                name ='y_test values'
               )

p2 = go.Scatter(x=data_to_plotly(X_test), 
                y=regr.predict(X_test),
                mode='lines',
                line=dict(color=Q5, width=3),
                name ='Linear Regression Model'
                )

layout = go.Layout(xaxis=dict(ticks='', showticklabels=True,
                              zeroline=True),
                   yaxis=dict(ticks='', showticklabels=True,
                              zeroline=True),
                   showlegend=True, hovermode='closest'
                   title=Q2
                   )

fig = go.Figure(data=[p1, p2], layout=layout)

py.iplot(fig)


## MAIN
Q1= input("What type of plot output? (Enter #)"+"\n"+"\n"+"1.Linear Regression"+"\n"+"2.Scatterplot"+"\n"+"3.Histogram"+"\n"+"4.Correlation Plot"+"\n"+"\n")
Qa = input("Enter X_train variable name:"+"n")
Qb = input("Enter y_train variable name:"+"n")
Qc = input("Enter X_test variable name:"+"n")
Qd = input("Enter y_test variable name:"+"n")
Q2= input("What do you want to title the graph?"+"\n)
Q3 = input("What is the predictor(dependent) variable name?:"+"\n")
Q4 = input("Choose color for actual y points:"+"\n")
Q5 = input("Choose color for line:"+"\n")

Q8 = int(Q2)
i = 0
store = {}
key = 0 
while i <= (Q2-1): 
    store[key] = input("Name variable:")
    key = key+1 
    i += 1    

