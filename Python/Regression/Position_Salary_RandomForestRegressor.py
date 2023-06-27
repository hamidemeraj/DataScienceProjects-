# Ensemble Learning : One of them is Random Forest: take Multiple algorithms or one algorithm multiple times 
# Import Libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


# Import Datasets 
df = pd.read_csv(r"C:\Users\snapp\Data-Projects\Datasets\Position_Salaries.csv")
# Independent Variables 
X = df.iloc[:,1:2].values
# Dependent Variables 
y = df.iloc[:,2:3].values

# Split Data into Train and Test is not necessary because the dataset is too small 

# Fit Random Forest  to the dataset 
# N_estimators can be diffrent numbers 
RF = RandomForestRegressor(n_estimators=300, random_state=0)
RF.fit(X,y)

# Predict a new Result with RF 
y_pred = RF.predict([[6.5]])

# Random Forest is nonlinear and noncontinous model: It is necessary to visualize in Resolution
# Visualize the Random Forest Regression Results _ Improved (Smooth Line)

X_grid = np.arange(min(X),max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X, y, c = 'red', s = 10)
plt.plot(X_grid, RF.predict(X_grid), c = 'green')
plt.title('Random Forest Results')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Adjusted_R_squared = 1 – [(1-R2)*(n-1)/(n-k-1)] : n: number of samples - k: number of regressors
Adj_R2 = 1 - (1-RF.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1)



 

