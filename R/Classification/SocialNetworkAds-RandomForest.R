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

# Fit Random Forest To the training set and Predict the test set results  
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = train_set[-3],
                          y =train_set$Purchased,
                          ntree = 500)

# Predict Test Results 
y_pred = predict(classifier, newdata = test_set[-3])

# Make Confusion Matrix 
CM = table(test_set[,3], y_pred)
print(CM)

# Visualising the Training Set Results 
library(ElemStatLearn)
set = train_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Random Forest (Training set)',
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
     main = 'Random Forest (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, '#FF9999', '#99FF99'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, '#990000', '#009900'))

