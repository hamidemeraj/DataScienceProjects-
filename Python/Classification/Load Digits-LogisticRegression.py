from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
digits = load_digits()

#Visualize Images 
plt.figure(figsize = (20,4))
for index, (image,label) in enumerate(zip(digits.data[0:8],digits.target[0:8])):
    plt.subplot(1,8,index+1)
    plt.imshow(np.reshape(image,(8,8)), cmap = plt.cm.gray)
    plt.title('Training:%i\n' %label , fontsize = 10)


X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train, y_train)

# Predicting ten elements of test data 
LR.predict(X_test[0:10])
LR.predict(X_test[0:1])

y_pred=LR.predict(X_test)
import pandas as pd 
Labels = pd.DataFrame()
Labels['test']=y_test
Labels['predict']=y_pred
Labels.head(30)

score = LR.score(X_test, y_test)*100
print('The score of logistic regression model is {}' .format(score))

# Confusion Metrics 
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(9,9))
sns.heatmap(cf_matrix, annot = True, fmt = ".3f", linewidth = .5, square = True)
plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title , size = 15 )


index = 0 
classified_index = []
for predict, actual in zip(y_pred, y_test):
    if predict == actual: 
        classified_index.append(index)
    index += 1
plt.figure(figsize =(20,3))
for plotindex, right in enumerate(classified_index[10:18]):
    plt.subplot(1,8, plotindex+1)
    plt.imshow(np.reshape(X_test[right],(8,8)),cmap= plt.cm.gray)
    plt.title("predicted: {}, Actual: {}".format(y_pred[right],y_test[right], fontsize= 5))












