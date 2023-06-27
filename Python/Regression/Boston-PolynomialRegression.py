"""
Polynomial Linear Regression
"""
# Import Libraries
from sklearn.datasets import load_boston
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from  sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Load Data
Boston = load_boston()
#Extract one Feature for polynomial linear regression
Boston_Data =  pd.DataFrame(Boston.data, columns=Boston.feature_names)
x = Boston_Data.iloc[:,4]
y = Boston_Data.iloc[:,-1]
x = x.to_numpy()
y = y.to_numpy()

# Use random state like MLR to compare your results  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, train_size=0.75, random_state=70)

# Create a Polynomial Feature from Selected Feature
Poly_p = PolynomialFeatures(degree = 2)
X_train = X_train.reshape(-1,1)
Poly_X = Poly_p.fit_transform(X_train)

# Create a Linear Model
LR = LinearRegression()
PLR = LR.fit(X_train, y_train)
X_test = X_test.reshape(-1,1)
y_predicted_PLR = LR.predict(X_test)

# Evaluate Model
R2 = r2_score(y_test, y_predicted_PLR)













