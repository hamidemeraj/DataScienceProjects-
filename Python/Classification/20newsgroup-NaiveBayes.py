import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()
from sklearn.datasets import fetch_20newsgroups

# Run without VPN
data = fetch_20newsgroups()
data.target_names

Categories = ['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x','misc.forsale','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space','soc.religion.christian','talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']


# Training Data on these categories 
train = fetch_20newsgroups(subset='train', categories=Categories)
# Testing Data on these categories
test = fetch_20newsgroups(subset='test', categories=Categories)

# printing training and test sample Data 
print(train.data[5])
print(test.data[5])

# Length of train and test data 
print(len(train.data))
print(len(test.data))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Creating a model based on Multinomial Naive Bayes
model = make_pipeline(TfidfVectorizer(),MultinomialNB())

# Training the model with train data 
model.fit(train.data, train.target)

# Creating Labels for the test data 
labels = model.predict(test.data)

# Creating Confusion Matrix 
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
print(mat)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')

# Predicting category on new data 
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

predict_category('Jesus Christ')
predict_category('Iran Nuclear')
predict_category('increase your heart rate and improve your immune system')
predict_category('Sending load to International Space Station')
predict_category('BMW is better than Audi')
predict_category('President of India')








