
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

balance_data = pd.read_csv( r"C:\Users\snapp\Desktop\AI\Dataset\Decision Tree\Decision_Tree_ Dataset.csv", sep =',', header = 0)
print(balance_data.head())
print('Balance Dataset:', len(balance_data))
print('Balance Dataset:', balance_data.shape)
print(balance_data.columns)

balance_data.rename(columns = {'1':'Initial_payment','2':'Last_payment', '3':'Credit_score', '4':'House_Number','Unnamed: 5':'Result'}, inplace=True)
print(balance_data.columns)


x = balance_data.values[:,0:4]
y = balance_data.values[:,5]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)

print('Accuracy is', accuracy_score(y_test, y_pred)*100)
