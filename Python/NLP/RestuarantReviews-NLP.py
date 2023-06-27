# Natural Language Processing 

# Import Libraries 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

# Rad Dateset 
# Use Quoting to ignore the double quotation in data 
df = pd.read_csv(r"C:\Users\snapp\Data-Projects\Datasets\Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

# Cleaning the texts 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,df.shape[0]):
    # Clean the punctuations, numbers, ... except alphabet and spaces
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    # Change all the capital letter to lower letter
    review = review.lower()
    # Remove irrelevent words - stopwords/Stemming: Taking the root of the words/Changing the text to a list of words 
    review = review.split()
    ps = PorterStemmer()
    # Set is faster than list in python 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Joining the word with space
    review = ' '.join(review)
    # Append the cleaned text to the list 
    corpus.append(review)

# Creating the Bag of words model - create a sparse matrix(create one column for each word)
from sklearn.feature_extraction.text import CountVectorizer
## countvectorizer has a lot of feature for cleaning data but it is better clean data step by step 
cv = CountVectorizer(max_features= 1500)
# Toarray is necessary to convert it to matrix 
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,-1]

# Classification Model: the best model for NLP: NB, DT, RF 
# Split Data into Train and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling: it is not necessary because all the data is: 0,1,2, and few numbers

# Fit Naive Bayes to the Training set 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict the Test Set Results 
y_pred = classifier.predict(X_test)

# Confusion Matrix
# The Data is small  
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy 
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
