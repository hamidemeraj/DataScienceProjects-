# import Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Import Dataset 
df = pd.read_csv(r'C:/Users/snapp/Data-Projects/Datasets/Market_Basket_Optimisation.csv', header= None)

# Apriori needs a sparse matrix 
transactions = []
for i in range(0, df.shape[0]): 
    transactions.append([str(df.values[i,j]) for j in range(0, df.shape[1])])
    
# Training Apriori on the dataset 
# Use a Function 
import sys
sys.path.append("C:/Users/snapp/Data-Projects/Python/Association Rule/")
from apyori import apriori

""" about confidence when it is high:
    you will have some obvious rules,
    you will get rules containing products that are most purchased overall and they are not well associated
    but there is not a logical association between two products 
"""   
rules = apriori(transactions = transactions, min_support = 0.003 , min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the result 
results = list(rules)

print(results)
