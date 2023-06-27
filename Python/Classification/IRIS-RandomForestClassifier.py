from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import numpy as np 
np.random.seed(0)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Get Labels of Data 
# df['species']= iris.target
df['species']= pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()

df['is_train'] = np.random.uniform(0,1,len(df)) <= 0.75
df.head()

train, test = df[df['is_train']==True], df[df['is_train']==False]
train.shape
test.shape

features = df.columns[:4]
features

# Converting each species name into digits 
y = pd.factorize(train['species'])[0]
y

# Creating a random Forest Classifier 
rfc = RandomForestClassifier(random_state=0, n_jobs = 2)
rfc.fit(train[features],y)

# Predicting test data 
y_pred = rfc.predict(test[features])

# Viewing the predicted probabilities of the first 20 observations (test) 
rfc.predict_proba(test[features])[0:20]

# Mapping names for each predicted 
preds = iris.target_names[y_pred]

test['species'].head()

# Create Confusion Matrix 
pd.crosstab(test['species'], preds, rownames=['Actial Species'], colnames= ['Predicted species'])

from sklearn.metrics import accuracy_score
accuracy_score(test['species'],preds)

iris.target_names[rfc.predict([[5.0,3.6,1.4,2],[5.0,3,4,2]])]







