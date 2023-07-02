# XGBoost : High Performance - Fast Execution Speed - No Need to Feature Scaling 

# XGBoost

# Importing the df
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/Churn_Modelling.csv")
df = df[4:14]

# Encoding the categorical variables as factors
df$Geography = as.numeric(factor(df$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
df$Gender = as.numeric(factor(df$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the df into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(df$Exited, SplitRatio = 0.8)
train_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

# Fitting XGBoost to the Training set
# install.packages('xgboost')
library(xgboost)
classifier = xgboost(data = as.matrix(train_set[-11]), label = train_set$Exited, nrounds = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = as.matrix(test_set[-11]))
y_pred = (y_pred >= 0.5)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(train_set$Exited, k = 10)
cv = lapply(folds, function(x) {
  train_fold = train_set[-x, ]
  test_fold = train_set[x, ]
  classifier = xgboost(data = as.matrix(train_set[-11]), label = train_set$Exited, nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-11]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 11], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))