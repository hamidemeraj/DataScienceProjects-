"""
Multiple Linear Regression
"""
# Import Libraries 
from sklearn.datasets import load_boston
import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import math 

# Load Data
Boston = load_boston()
Boston_Data = pd.DataFrame(data=Boston.data, columns=Boston.feature_names)
Boston_Data['Target'] = Boston.target

# Splitting X , Y 
x = Boston_Data.iloc[:,:13]
y = Boston_Data.iloc[:, 13]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, train_size=0.75, random_state=70)

# Normalization 
Sc = MinMaxScaler(feature_range=(0,1))
X_train = Sc.fit_transform(X_train)
X_test = Sc.fit_transform(X_test)
y_train = y_train.values.reshape(-1,1)
y_train = Sc.fit_transform(y_train)

# Creating a Linear Model
MLR = LinearRegression()
MLR.fit(X_train,y_train)

# These are Normalized Values
y_predicted_MLR = MLR.predict(X_test)
# Converting to Actual Values
y_predicted_MLR = Sc.inverse_transform(y_predicted_MLR)

# Evaluating Model
mae = mean_absolute_error(y_test, y_predicted_MLR)
mse = mean_squared_error(y_test, y_predicted_MLR)
rmse = math.sqrt(mse)
mean_absolute_percentage_error(y_test, y_predicted_MLR)
R2 = r2_score(y_test, y_predicted_MLR)

# Define Metrics without Using Libraries
def Mean_Absolute_Percentage_Error(y_true,y_pred):
    y_true , y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100

mape = Mean_Absolute_Percentage_Error(y_test,y_predicted_MLR)


"""
Random Forest Regressor 
"""
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(random_state=33, n_estimators = 100, max_depth=20)
RFR.fit(X_train, y_train)
y_predicted_RFR = RFR.predict(X_test)

# Reverse to Actual Data 
y_predicted_RFR = y_predicted_RFR.reshape(-1,1)
y_predicted_RFR = Sc.inverse_transform(y_predicted_RFR)



# Evaluating Model
mae = mean_absolute_error(y_test, y_predicted_RFR)
mse = mean_squared_error(y_test, y_predicted_RFR)
rmse = math.sqrt(mse)
mean_absolute_percentage_error(y_test, y_predicted_RFR)
R2 = r2_score(y_test, y_predicted_RFR)

# Define Metrics without Using Libraries
def Mean_Absolute_Percentage_Error(y_true,y_pred):
    y_true , y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100

mape = Mean_Absolute_Percentage_Error(y_test,y_predicted_RFR)


"""
Support Vector Regressor
"""
from sklearn.svm import SVR
# Use kernel Function 
SVR = SVR(kernel= 'rbf')
SVR.fit(X_train, y_train)
y_predicted_SVR = SVR.predict(X_test)
y_predicted_SVR = y_predicted_SVR.reshape(-1,1)
y_predicted_SVR = Sc.inverse_transform(y_predicted_SVR)

mae = mean_absolute_error(y_test, y_predicted_SVR)
mse = mean_squared_error(y_test, y_predicted_SVR)
rmse = math.sqrt(mse)
mean_absolute_percentage_error(y_test, y_predicted_SVR)
R2 = r2_score(y_test, y_predicted_SVR)

# Define Metrics without Using Libraries
def Mean_Absolute_Percentage_Error(y_true,y_pred):
    y_true , y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100

mape = Mean_Absolute_Percentage_Error(y_test,y_predicted_SVR)


