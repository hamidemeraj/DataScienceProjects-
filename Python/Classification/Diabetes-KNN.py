import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r"C:\Users\snapp\Desktop\AI\Dataset\diabetes.csv")
len(df)
df.head()

df.columns
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for column in zero_not_accepted: 
    df[column] = df[column].replace(0,np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)

df.tail()

# Split Dataset
X = df.iloc[:,0:8]
y = df.iloc[:,8]
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size =0.2 )
 
# Feature Scaling 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Choose k neighbors 
import math 
math.sqrt(len(y_test))

# KNN Model 
knn = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(round(accuracy_score(y_test, y_pred)*100,2))
print(round(f1_score(y_test, y_pred)*100,2))



















