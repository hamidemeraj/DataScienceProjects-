# Import Data set 
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/Data_Preprocessing1.csv")

# Missing Value 
df$Age = ifelse(is.na(df$Age),
                ave(df$Age, FUN = function(x) mean(x,na.rm = TRUE)),
                df$Age)

df$Salary = ifelse(is.na(df$Salary),
                ave(df$Salary, FUN = function(x) mean(x,na.rm = TRUE)),
                df$Salary)


# Encode Categorical Data 
df$Country = factor(df$Country,
                    levels = c('France','Spain','Germany'),
                    labels = c(1,2,3))

df$Purchased = factor(df$Purchased,
                    levels = c('No','Yes'),
                    labels = c(0,1))

# Split Dataset into train test split 
# Install a library: install.packages('caTools')
library(caTools)
# Set random state for seeing same data 
set.seed(123)
# See help with F1
split = sample.split(df$Purchased, SplitRatio = 0.8)
train_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

# Feature Scaling 
# Need Numeric Columns 

train_set[, 2:3] = scale(train_set[, 2:3]) 
test_set[, 2:3] = scale(test_set[, 2:3])
















