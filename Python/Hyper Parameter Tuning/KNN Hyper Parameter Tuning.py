# Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load Dataset 
iris = load_iris()
Data_iris = iris.data
Data_iris = pd.DataFrame(Data_iris, columns = iris.feature_names)
Data_iris['Labels']= iris.target



# Splitting Samples and Targets
x = Data_iris.iloc[:,:-1]
y = Data_iris.iloc[:,4]

# Splitting train and test 
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.2,
                                                    train_size = 0.8,
                                                    random_state = 22,
                                                    shuffle = True,
                                                    stratify = y
                                                    )
 
KNN_Accuracy_test = []
KNN_Accuracy_train = []
for k in range(1,50):
    KNN = KNeighborsClassifier(n_neighbors=k , metric = 'minkowski', p=1)
    KNN.fit(X_train,y_train)
    KNN_Accuracy_train.append(KNN.score(X_train, y_train))
    KNN_Accuracy_test.append(KNN.score(X_test, y_test))
    
plt.plot(range(1,50),KNN_Accuracy_train, label = 'train')   
plt.plot(range(1,50),KNN_Accuracy_test, label = 'test')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

    
    
    