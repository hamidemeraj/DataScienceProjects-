# Multiple Linear Regression 

# Import Data set 
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/50_Startups.csv")

# Encode Categorical Data 
df$State = factor(df$State,
                  levels = c('New York','California','Florida'),
                  labels = c(1,2,3))

# Set random state for seeing same data 
set.seed(123)

# Split Dataset into train test split 
library(caTools)
split = sample.split(df$Profit, SplitRatio = 0.8)
train_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

# Feature Scaling: It is not necessary this will take care of with functions in model

# Fit multiple Linear Regression Model to Train set 
MLR = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
         data = train_set)
# You can choose all columns by . MLR = lm(formula = Profit ~ .,data = train_set)

summary(MLR)

# Predict the Test Set Result 
y_pred = predict(MLR, newdata = test_set)
y_pred






