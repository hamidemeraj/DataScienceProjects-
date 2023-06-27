import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

# Import Dataset
df = pd.read_csv(r"C:\Users\snapp\Data-Projects\Datasets\50_Startups.csv")

# Independent Variables 
X = df.iloc[:,:-1]
# X = df.iloc[:,:-1].values
# Dependent Variables 
y = df.iloc[:,-1]

# Encoding Categorical Data 
onehotencoder = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(onehotencoder.fit_transform(X[['State']]).toarray())
X = X[['R&D Spend', 'Administration', 'Marketing Spend']].join(enc_df)
X = X.iloc[:,:]

# Avoid Dummy Variable Trap (some Libraries like sikit learn.linearregression take care of dummy variables)
X = X.iloc[:, :-1]

# Build an optimal Model using Backward Elimination 
# Add a columns of onr for Constant Value because we can not use it in sm (y = b1x1 + b2x2 + b3x3 + ...)
# X['constant'] = 1 
# or 
X = X.values
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis = 1)
y = y.values

# create a matrix of optimal variables for regression model
X_opt1 = X[:, [0, 1, 2, 3, 4, 5]]

# First Step:Select significance level 
SL = 0.05

# Second Step: Fit the full model with all possible predictors 
regressor_OLS = sm.OLS(endog = y, exog = X_opt1).fit()

# Third Step: Consider the Predictor with the highest P-value, if p > SL go to step 4 otherwise finish
regressor_OLS.summary()
Adjusted_r1 = regressor_OLS.rsquared_adj

# Remove Predictor 
X_opt2 = X[:, [0, 1, 2, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt2).fit()
regressor_OLS.summary()
Adjusted_r2 = regressor_OLS.rsquared_adj

X_opt3 = X[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt3).fit()
regressor_OLS.summary()
Adjusted_r3 = regressor_OLS.rsquared_adj

X_opt4 = X[:, [0, 1, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt4).fit()
regressor_OLS.summary()
Adjusted_r4 = regressor_OLS.rsquared_adj

X_opt5 = X[:, [0, 1]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt5).fit()
regressor_OLS.summary()
Adjusted_r5 = regressor_OLS.rsquared_adj
