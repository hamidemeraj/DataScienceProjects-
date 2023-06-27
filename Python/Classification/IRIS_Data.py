# Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score 
from sklearn.model_selection import cross_val_score

# Load Dataset 
iris = load_iris()
iris.feature_names
Data_iris = iris.data
Data_iris = pd.DataFrame(Data_iris, columns = iris.feature_names)
Data_iris['Labels']= iris.target

# Plot 2-Dimensional
plt.scatter(x= Data_iris.iloc[:,2],
            y= Data_iris.iloc[:,3],
            c= Data_iris['Labels']
            )
plt.xlabel('Petal Length(cm)')
plt.ylabel('Petal Width(cm)')
plt.show()

# Splitting Samples and Targets
x = Data_iris.iloc[:,:-1]
y = Data_iris.iloc[:,4]

# Splitting train and test 
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.2,
                                                    train_size = 0.8,
                                                    random_state = 88,
                                                    shuffle = True,
                                                    stratify = y
                                                    )
 
"""
KNN Classifier 
"""
# Manhatan Distance with p = 1 and Euclidean Distance with p = 2
KNN = KNeighborsClassifier(n_neighbors= 6, metric= 'minkowski', p = 1)
KNN.fit(x,y)
x_N = np.array([[5.6, 3.4, 1.4, 0.1]])
x_N2 = np.array([[3.5, 4, 5.5, 2]])
KNN.predict(x_N)
KNN.predict(x_N2)

KNN = KNeighborsClassifier(n_neighbors= 6, metric= 'minkowski', p= 1)
KNN.fit(X_train,y_train)
y_predicted_knn = KNN.predict(X_test)
acc_KNN = round(accuracy_score(y_test, y_predicted_knn)*100,3)

"""
Decision Tree Classifier 
"""
DT = DecisionTreeClassifier()
DT = DT.fit(X_train,y_train)
y_predicted_dt = DT.predict(X_test)
acc_DT = round(accuracy_score(y_test, y_predicted_dt)*100,3)

"""
Naive Bayes Classifier 
"""
NB = GaussianNB()
NB.fit(X_train, y_train)
y_predicted_nb = NB.predict(X_test)
acc_NB = round(accuracy_score(y_test, y_predicted_nb)*100,3)

"""
K_Fold Cross Validation  
"""
# Name of Model - CV is number of cross validation
Score_KNN = cross_val_score(KNN, x, y, cv =10)
Score_KNN.mean()
Score_DT = cross_val_score(DT, x, y, cv = 10)
Score_DT.mean()
Score_NB = cross_val_score(NB, x, y, cv = 10)
Score_NB.mean()

"""
Other Evaluation Metrics For Best Model
"""
Con_Mat = confusion_matrix(y_test, y_predicted_dt)
Class_rep = classification_report(y_test, y_predicted_dt)



