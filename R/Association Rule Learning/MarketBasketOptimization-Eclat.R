# Eclat

# Date Preprocessing 
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/Market_Basket_Optimisation.csv", header = FALSE)

# the Eclat Library takes a sparse matrix - not a csv 
# install.packages('arules')
# There is a possible that we have duplicate data in matrix (for example: one item repeated in one transaction)
library(arules)
df = read.transactions("C:/Users/snapp/Data-Projects/Datasets/Market_Basket_Optimisation.csv", sep = ',', rm.duplicates = TRUE)

summary(df)
# Visualise Frewuency of Transactions  
itemFrequencyPlot(df, topN = 50)

# Training Eclat on the dataset 
rules = eclat(df, parameter = list(support = 0.003, minlen = 2))

# Visualising the Rules 
inspect(sort(rules, by = 'support')[1:10])
