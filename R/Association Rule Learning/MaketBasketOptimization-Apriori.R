# Apriori 

# Date Preprocessing 
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/Market_Basket_Optimisation.csv", header = FALSE)

# the Apriori Library takes a sparse matrix - not a csv 
# install.packages('arules')
# There is a possible that we have duplicate data in matrix (for example: one item repeated in one transaction)
library(arules)
df = read.transactions("C:/Users/snapp/Data-Projects/Datasets/Market_Basket_Optimisation.csv", sep = ',', rm.duplicates = TRUE)

summary(df)
# Visualise Frewuency of Transactions  
itemFrequencyPlot(df, topN = 50)

# Training Apriori on the dataset 
rules = apriori(df, parameter = list(support = 0.003 , confidence = 0.4))

# Visualising the Rules 
inspect(sort(rules, by = 'lift')[1:10])

# Some products presents in set of items not because they make good association but because they have high support
# For example: chocolate is in a lot of baskets

# In order to avoid finding rules that contains most purchased items 

# Training Apriori on the dataset 
rules = apriori(df, parameter = list(support = 0.003 , confidence = 0.2))

# Visualising the Rules 
inspect(sort(rules, by = 'lift')[1:10])
