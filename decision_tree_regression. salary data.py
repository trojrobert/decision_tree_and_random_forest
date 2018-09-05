# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#create and fit the model decision tree 
from sklearn.tree import DecisionTreeRegressor 
deciTree = DecisionTreeRegressor(random_state = 0)
deciTree.fit(X,y)

#Predict a new value 
prediciton = deciTree.predict(6.5)

#Decision tree is non continuous so we need to plot it in high resolution
# Visualising the Decision Tree Regression results (higher resolution)

X_high = np.arange(min(X), max(X),0.01)
X_high = X_high.reshape((len(X_high), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_high, deciTree.predict(X_high), color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()