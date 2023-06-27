import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


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

# Split Data into Train and Test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit Multiple Linear Regression to the Training Set 
MLR = LinearRegression()
MLR.fit(X_train, y_train)

# Predict test set 
y_pred = MLR.predict(X_test)

# Adjusted_R_squared = 1 – [(1-R2)*(n-1)/(n-k-1)] : n: number of samples - k: number of regressors
Adj_R2 = 1 - (1-MLR.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1)
