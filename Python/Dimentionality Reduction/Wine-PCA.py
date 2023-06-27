# PCA
# PCA is an unsupervised model 

# Import Libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Import the Dataset 
df = pd.read_csv(r"C:\Users\snapp\Data-Projects\Datasets\Wine.csv")
X = df.iloc[:,0:13].values 
y = df.iloc[:,-1].values

# Split Data into Train and Test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling 
SC_X = StandardScaler()
X_train = SC_X.fit_transform(X_train)
X_test = SC_X.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None )
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Check the variance of data to select the number of variables 
explained_variance = pca.explained_variance_ratio_

# Applying PCA
pca = PCA(n_components = 2 )
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Fit Logistic Regression to the Training set 
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predict the Test Set Results 
y_pred = classifier.predict(X_test)

# Confusion Matrix 
CM = confusion_matrix(y_test, y_pred)

# Visualise the Train set Results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alph = 0.25, cmap = ListedColormap(('#FF9999', '#99FF99','blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i , j  in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('#650011','green','#82aacf'))(i), label = j, s = 5)
plt.title('Logistic Regression')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualise the Test set Results 
# Visualise the Test set Results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alph = 0.25, cmap = ListedColormap(('#FF9999', '#99FF99','blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i , j  in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('#650011','green','#82aacf'))(i), label = j, s = 5)
plt.title('Logistic Regression')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
















