# Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import Dataset
df = pd.read_csv(r"C:\Users\snapp\Data-Projects\Datasets\Salary_Data.csv")
X = df.iloc[:,:-1].values
y = df.iloc[:,1].values

# Split data into train test part
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fit Simple Linear Regression 
LR = LinearRegression() 
LR.fit(X_train, y_train)

# Predict the Test set 
y_pred = LR.predict(X_test)

# Detail of the Linear Regressor 
intercept = LR.intercept_
slope = LR.coef_

# Visualising Regression - train set 
plt.scatter(X_train, y_train, c = 'red', s = 10)
plt.plot(X_train, LR.predict(X_train), c = 'green')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Between Salary and Experience(X_train)",s=8)
plt.show()

# Visualising Regression - test set 
plt.scatter(X_test, y_test, c = 'red', s = 10)
plt.plot(X_train, LR.predict(X_train), c = 'green')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Between Salary and Experience(X_test)",s=8)
plt.show()




