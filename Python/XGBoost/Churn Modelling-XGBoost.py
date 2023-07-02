# XGBoost : very powerful model specially on large dataset with fast execution 

# Data Preprocessing 
# Import Libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

# Import the Dataset 
df = pd.read_csv(r"C:\Users\snapp\Data-Projects\Datasets\Churn_Modelling.csv")
X = df.iloc[:,3:13].values 
y = df.iloc[:,-1].values

# Encoding Categorical Data (Independent Variables)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])

labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(handle_unknown='ignore')
enc_data  = onehotencoder.fit_transform(X[:, 1].reshape(-1,1)).toarray()
X = np.hstack((X, enc_data[:,:2]))
X = np.delete(X, 1, axis=1)

# Split Data into Train and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# In XGBoost feature scaling is totaly unnecessary 
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
