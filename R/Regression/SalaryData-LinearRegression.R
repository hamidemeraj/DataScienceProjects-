# Simple Linear Regression 
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/Salary_Data.csv")

# Set random state for seeing same data 
set.seed(123)

# Split Dataset into train test split 
library(caTools)
split = sample.split(df$Salary, SplitRatio = 2/3)
train_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

# Simple Linear regression does not need to feature scaling 

# Fit Simple Linear Regression to the Training Set 
LR = lm(formula = Salary ~ YearsExperience,
        data = train_set)

print(summary(LR))

# Predict the Test Set 
y_pred = predict(LR, newdata = test_set)
print(y_pred)

# Visualise the Training set 
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = train_set$YearsExperience, y = train_set$Salary), 
             colour = 'red') + 
  geom_line(aes(x = train_set$YearsExperience, y = predict(LR, newdata = train_set)),
             colour = 'green') + 
  ggtitle('Salary vs Experience(Train set)') +
  xlab('Years of experience') + 
  ylab('Salary')

# Visualise the Training set 
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             colour = 'red') + 
  geom_line(aes(x = train_set$YearsExperience, y = predict(LR, newdata = train_set)),
            colour = 'green') + 
  ggtitle('Salary vs Experience(Train set)') +
  xlab('Years of experience') + 
  ylab('Salary')






