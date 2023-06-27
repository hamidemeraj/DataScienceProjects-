# Import Libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Import Datasets 
df = pd.read_csv(r"C:\Users\snapp\Data-Projects\Datasets\Position_Salaries.csv")
# Independent Variables 
X = df.iloc[:,1:2].values
# Dependent Variables 
y = df.iloc[:,2:3].values

# Split Data into Train and Test is not necessary because the dataset is too small 
# Feature Scaling is necessary because in the library it is not included 
SC_X = StandardScaler()
SC_y = StandardScaler()
X = SC_X.fit_transform(X)
y = SC_y.fit_transform(y)

# Fit SVR to the dataset 
# Kernel can be Poly - RBF 
SV = SVR(kernel = 'rbf', degree=4)
SV.fit(X, y)

# Visualize the Polynomial Regression Results
plt.scatter(X, y, c = 'red', s = 5)
plt.plot(X, SV.predict(X), c = 'blue')
plt.title('Polynomial Regression Results')
plt.xlabel("Position Level")
plt.ylabel("Salary")

# Predict a new Result with SVR Regression 
# It is necessary to inverse data to real Data 
y_pred = SV.predict(SC_X.transform(np.array([[6.5]])))
y_pred_inverse = SC_y.inverse_transform([y_pred])

# SVR is nonlinear model
# Visualize the SVR Regression Results _ Improved (Smooth Line)
X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X, y, c = 'red', s = 10)
plt.plot(X_grid, SV.predict(X_grid), c = 'green')
plt.title('SVR Results')
plt.xlabel("Position Level")
plt.ylabel("Salary")

# Adjusted_R_squared = 1 – [(1-R2)*(n-1)/(n-k-1)] : n: number of samples - k: number of regressors
Adj_R2 = 1 - (1-SV.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1)
