# K Fold Cross Validation : First Technique of Model Selection 

# If we run the model again on another test set we will get another accuracy 
# Judging our model performance only on one test set is not good. 

# Split test in n-fold. we can have n diffrent fold to train and test our model 
# Import Data set 
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/Social_Network_Ads.csv")
df = df[, 3:5]

# Encoding the Target Feature aas Factor 
df$Purchased = factor(df$Purchased, levels = c(0,1))

# Split Dataset into train test split 
library(caTools)
# Set random state for seeing same data 
set.seed(123)

split = sample.split(df$Purchased, SplitRatio = 0.75)
train_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

# Feature Scaling 
# Need Numeric Columns 
train_set[, 1:2] = scale(train_set[, 1:2]) 
test_set[, 1:2] = scale(test_set[, 1:2])

# Fit Kernel SVM To the training set and Predict the test set results  
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Purchased ~ ., 
                 data = train_set, 
                 type = 'C-classification',
                 kernel = 'radial')

# Predict Test Results 
y_pred = predict(classifier, newdata = test_set[-3])

# Make Confusion Matrix 
CM = table(test_set[,3], y_pred)
print(CM)

# Applying KFold Cross Validation 
# install.packages('caret')
library(caret)
folds = createFolds(train_set$Purchased, k = 10)
cv = lapply(folds, function(x){
  train_fold = train_set[-x,]
  test_fold = train_set[x, ]
  classifier = svm(formula = Purchased ~ ., 
                   data = train_fold, 
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-3])
  cm = table(test_fold[,3], y_pred)
  accuracies = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
  return(accuracies)
})
accuracy_mean = mean(as.numeric(cv))

# Visualising the Training Set Results 
library(ElemStatLearn)
set = train_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, '#FF9999', '#99FF99'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, '#990000', '#009900'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, '#FF9999', '#99FF99'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, '#990000', '#009900'))

