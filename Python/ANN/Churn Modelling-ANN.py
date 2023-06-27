# ANN Churn Modelling 
# Import Libraries 
import tensorflow as tf
tf.__version__

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

# It is necessary to Scale our data in ANN
# Feature Scaling 
from sklearn.preprocessing import StandardScaler
SC_X = StandardScaler()
X_train = SC_X.fit_transform(X_train)
X_test = SC_X.transform(X_test)

# Importing the Keras Libraries and Packages
import keras  
from keras.models import Sequential
from keras.layers import Dense

# Initiliasing the ANN 
classifier = Sequential()
# Adding the input layer and the first hidden layer 
# (input node + output node)/2 : number of node in layers 
classifier.add(Dense(6, kernel_initializer= 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer 
classifier.add(Dense(6, kernel_initializer= 'uniform', activation = 'relu'))

# Adding the output the layer  
# If dependent variable has 3 or more class: output_dim = number of class, activation = 'softmax' 
classifier.add(Dense(1, kernel_initializer= 'uniform', activation = 'sigmoid'))

# Compiling the ANN 
# optimizer : find the best weight based on algorithms 
# If dependent variable has 3 or more class: loss = 'categorical_crossentropy'
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set 
classifier.fit(X_train, y_train, batch_size= 10, epochs = 100)

# Making predictions and evaluating the model 
# Predict the Test Set Results 
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred) 
