# Kernel PCa is good for nonlinear data (data is nonlinearly seprable)

# Import Data set 
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/Social_Network_Ads.csv")
df = df[, 3:5]

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

# Applying Kernel PCA 
# install.packages('kernelLab'): Download it and install it directly 
library(kernlab)
kpca = kpca(~.,
            data = train_set[-3],
            kernel = 'rbfdot',
            features = 2)
train_set_pca = as.data.frame(predict(kpca, train_set))
train_set_pca$Purchased = train_set$Purchased

test_set_pca = as.data.frame(predict(kpca, test_set))
test_set_pca$Purchased = test_set$Purchased  
  
# Fit Logistic Regression To the training set 
classifier = glm(formula = Purchased ~ ., 
                 family = binomial,
                 data = train_set_pca)

# Predict the Test Results 
prob_pred = predict(classifier, type = 'response', newdata = test_set_pca[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Make Confusion Matrix 
CM = table(test_set_pca[,3], y_pred)
print(CM)

# Visualising the Training Set Results 
# install.packages('ElemStatLearn')
# Visualising the Training set results
library(ElemStatLearn)
set = train_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'dodgerblue', 'salmon'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'dodgerblue3', 'salmon3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'dodgerblue', 'salmon'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'dodgerblue3', 'salmon3'))

