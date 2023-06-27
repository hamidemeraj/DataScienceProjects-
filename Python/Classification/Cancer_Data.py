# Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score 

# Load Dataset 
Data_cancer = load_breast_cancer()
Data_cancer.feature_names
x = Data_cancer.data
y = Data_cancer.target
# Splitting Train and test 
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                     test_size= 0.3,
                                                     train_size= 0.7,
                                                     random_state = 88)

"""
Logistic Regression (binay classifier)
"""
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_predicted_LR = LR.predict(X_test)
acc_LR = round(accuracy_score(y_test,y_predicted_LR)*100,3)

"""
KNN Classifier 
"""
KNN = KNeighborsClassifier(n_neighbors=6 , metric= 'minkowski', p=2)
KNN.fit(X_train,y_train)
y_predicted_KNN = KNN.predict(X_test)
acc_KNN = round(accuracy_score(y_test,y_predicted_KNN)*100,3)

"""
Decision Tree Classifier 
"""
DT = DecisionTreeClassifier()
DT.fit(X_train,y_train)
y_predicted_DT = DT.predict(X_test)
acc_DT = round(accuracy_score(y_test,y_predicted_DT)*100,3)

"""
Naive Bayes Classifier 
"""
NB = GaussianNB()
NB.fit(X_train,y_train)
y_predicted_NB = NB.predict(X_test)
acc_NB = round(accuracy_score(y_test,y_predicted_NB)*100,3)

#------------------------------------------------------------
"""
K fold Cross Validation  
"""
Score_LR = cross_val_score(LR, x, y, cv = 10)
Score_LR.mean()
Score_KNN = cross_val_score(KNN, x, y, cv = 10)
Score_KNN.mean()
Score_DT = cross_val_score(DT, x, y, cv = 10)
Score_DT.mean()
Score_NB = cross_val_score(NB, x, y, cv = 10)
Score_NB.mean()


"""
Other Evaluation Metrics For Best Model
"""
Conf_Mat = confusion_matrix(y_test, y_predicted_LR)
Class_rep = classification_report(y_test, y_predicted_LR)
y_prob = LR.predict_proba(X_test)
y_prob = y_prob[:, 1]

FPR, TPR, Threshlds = roc_curve(y_test, y_prob)
plt.plot(FPR, TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

AUC_Score = roc_auc_score(y_test, y_prob)





