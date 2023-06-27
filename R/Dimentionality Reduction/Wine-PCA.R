# PCA 
# Import Data set 
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/Wine.csv")

# Split Dataset into train test split 
library(caTools)
# Set random state for seeing same data 
set.seed(123)

split = sample.split(df$Customer_Segment, SplitRatio = 0.8)
train_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

# Feature Scaling 
# Need Numeric Columns 
train_set[, -14] = scale(train_set[, -14]) 
test_set[, -14] = scale(test_set[, -14])

# Applying PCA 
# install.packages('caret')
library(caret)
library(e1071)

pca = preProcess(x = train_set[-14], 
                 method = 'pca',
                 pcaComp = 2)

train_set = predict(pca, train_set)
train_set = train_set[c(2,3,1)]
test_set = predict(pca, test_set)
test_set = test_set[c(2,3,1)]

library(e1071)
classifier = svm(formula = Customer_Segment ~ ., 
                 data = train_set, 
                 type = 'C-classification',
                 kernel = 'linear')

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
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse( y_grid == 1, '#FF9999', '#99FF99')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2,'blue3', ifelse(set[, 3] == 1, '#990000', '#009900')))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse( y_grid == 1, '#FF9999', '#99FF99')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2,'blue3', ifelse(set[, 3] == 1, '#990000', '#009900')))
