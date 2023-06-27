# Import Libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Import the Dataset 
df = pd.read_csv(r"C:\Users\snapp\Data-Projects\Datasets\Social_Network_Ads.csv")
X = df.iloc[:,[2,3]].values 
y = df.iloc[:,-1].values

# Split Data into Train and Test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling 
# In Random Forest because it is not based on Euclidean distance it is not necessary to do  Feature scaling
# But we keep it for the Visualization 
SC_X = StandardScaler()
X_train = SC_X.fit_transform(X_train)
X_test = SC_X.transform(X_test)

# Fit Random Forest to the Training set 
classifier = RandomForestClassifier(n_estimators = 10, criterion= 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predict the Test Set Results 
y_pred = classifier.predict(X_test)

# Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)

# Visualise the Train set Results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alph = 0.25, cmap = ListedColormap(('#FF9999', '#99FF99')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i , j  in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('#990000', '#009900'))(i), label = j, s = 5)
plt.title('Random Forest ')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualise the Test set Results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alph = 0.25, cmap = ListedColormap(('#FF9999', '#99FF99')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i , j  in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('#990000', '#009900'))(i), label = j, s = 5)
plt.title('Random Forest ')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()















