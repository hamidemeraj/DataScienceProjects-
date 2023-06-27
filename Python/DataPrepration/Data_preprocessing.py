# Import Liabraries 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import Datasets 
df = pd.read_csv(r"C:\Users\snapp\Data-Projects\Datasets\Data_Preprocessing1.csv")
# Independent Variables 
X = df.iloc[:,:-1]
# X = df.iloc[:,:-1].values
# Dependent Variables 
y = df.iloc[:,3].values 

# Missing Data 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X.iloc[:,1:])
X.iloc[:,1:] = imputer.transform(X.iloc[:,1:])

# Encoding Categorical Data with two values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Encoding Categorical Data with more than two values
onehotencoder = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(onehotencoder.fit_transform(X[['Country']]).toarray())
X = X[['Age','Salary']].join(enc_df)
X = X.iloc[:,:].values

# Split Data into Train and Test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling: Standardisation(-1,+1) - Normalization(0,1) 
# Is it necessary to scale dummy variables? It depends

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# mm_X = MinMaxScaler()
# X_train = mm_X.fit_transform(X_train)
# X_test = mm_X.transform(X_test)
