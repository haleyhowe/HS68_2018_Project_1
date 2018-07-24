import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

def get_var_input():
    

def plotmodel(X_train, y_train, X_test, y_test):
    reg_obj = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    # The coefficients
    print 'Coefficients: \n', regr.coef_
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


    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())
    plt.show()
    
### Fix plotly visuals 
##    plot = go.Scatter(
##        x= Q3a,
##        y= Q3b,
##    )
##
##    execute = [plot]
##    layout = go.Layout(showlegend=True)
##    fig = go.Figure(data=execute, layout=layout)
##
##    py.iplot(fig, filename='show-legend')
##    plt.title(Q6)


## MAIN
Q1= raw_input("What type of plot output? (Enter #)"+"\n"+"\n"+"1.Linear Regression"+"\n"+"2.Scatterplot"+"\n"+"3.Histogram"+"\n"+"4.Correlation Plot"+"\n"+"\n")
Q2= input("How many independent variables do you want to use?"+"\n)
Q3a= raw_input("Name independent variables:"+"\n"+"\n"+#function#)
Q3b = raw_input("Name dependent variable:"+"\n"+"\n"+#function#)
Q4= raw_input("What do you want to label the x axis?"+"\n)
Q5= raw_input("What do you want to label the y axis?"+"\n)
Q6= raw_input("What do you want to title the graph?"+"\n)
Q7 = raw_input("Choose color scheme:"+"\n"+"\n"#Look up schemes#)
Q8 = raw_input("What do you want to label each independent variable in the legend/key?"+"\n"+"\n"+#function#)



