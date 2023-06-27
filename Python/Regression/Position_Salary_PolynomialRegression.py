# Import Libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Import Datasets 
df = pd.read_csv(r"C:\Users\snapp\Data-Projects\Datasets\Position_Salaries.csv")
# Independent Variables 
X = df.iloc[:,1:2].values
# X = df.iloc[:,:-1].values
# Dependent Variables 
y = df.iloc[:,-1].values

# Split Data into Train and Test is not necessary because the dataset is too small 
# Feature Scaling is not necessary beacasue the library included Feature Scaling

# Fit Linear Regression 
LR = LinearRegression()
LR.fit(X, y)

# Fit Polynomial Regression
PR = PolynomialFeatures(degree = 4)
# The model itself add one column as Intercept 
X_poly = PR.fit_transform(X)
PLR = LinearRegression()
PLR.fit(X_poly,y)

# Visualize the Linear Regression Results
plt.scatter(X, y, c = 'red', s = 10)
plt.plot(X, LR.predict(X), c = 'green')
plt.title('Linear Regression Results')
plt.xlabel("Position Level")
plt.ylabel("Salary")

# Visualize the Polynomial Regression Results
plt.scatter(X, y, c = 'red', s = 10)
plt.plot(X, PLR.predict(PR.fit_transform(X)), c = 'green')
plt.title('Polynomial Regression Results')
plt.xlabel("Position Level")
plt.ylabel("Salary")

# Visualize the Polynomial Regression Results _ Improved (Smooth Line)
X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X, y, c = 'red', s = 10)
plt.plot(X_grid, PLR.predict(PR.fit_transform(X_grid)), c = 'green')
plt.title('Polynomial Regression Results')
plt.xlabel("Position Level")
plt.ylabel("Salary")

# Predict a new Result with Linear Regression
LR.predict([[6.5]])
# Predict a new Result with Linear Regression 
PLR.predict(PR.fit_transform([[6.5]]))


