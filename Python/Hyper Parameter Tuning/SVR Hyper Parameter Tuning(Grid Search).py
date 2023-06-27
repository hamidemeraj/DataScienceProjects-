"""
SVR Hyper Parameter Tuning
"""
# Import Libraries 
from sklearn.datasets import load_boston
import pandas as pd 

# Load Data
Boston = load_boston()
Boston_Data = pd.DataFrame(data=Boston.data, columns=Boston.feature_names)
Boston_Data['Target'] = Boston.target

# Splitting X , Y 
x = Boston_Data.iloc[:,:13]
y = Boston_Data.iloc[:, 13]

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':['rbf','linear'],
              'gamma':[1,0.1,0.01]}

Grid = GridSearchCV(estimator=SVR(), param_grid=parameters,refit =True, verbose=2, scoring='neg_mean_squared_error')
# It is not necessary to have train and test 
Grid.fit(x,y)
Best_Params = Grid.best_params_
