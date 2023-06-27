# Multiple Linear Regression 
# Import LIbraries  
import pandas as pd 
import seaborn as sns 

# Reading Data 
companies = pd.read_csv(r"C:\Users\snapp\Desktop\AI\Dataset\Linear Regression\1000_Companies.csv")
print(companies.head(4))
sns.heatmap(companies.corr())
y = pd.DataFrame(companies.iloc[:,-1])
x = pd.DataFrame(companies.iloc[:,:4])

# Handling Categorical Data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
Lencoder = LabelEncoder()
x['State']= Lencoder.fit_transform(x['State'])

Oencoder = OneHotEncoder()
enc_data = pd.DataFrame(Oencoder.fit_transform(x[['State']]).toarray())

# Avoiding the Dummy Variable Trap
enc_data = enc_data[[0,1]]
x.drop(['State'], axis = 1, inplace=True)
x = x.join(enc_data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
MLR = LinearRegression()
MLR.fit(X_train,y_train)
y_pred = MLR.predict(X_test)

print(MLR.coef_)
print(MLR.intercept_)

# Evaluating performance of model 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math 

r2_score(y_test,y_pred)
mse = mean_squared_error(y_test, y_pred)
print(mse)    
rmse = math.sqrt(mse)
print(rmse)
mae = mean_absolute_error(y_test,y_pred)
print(mae)
