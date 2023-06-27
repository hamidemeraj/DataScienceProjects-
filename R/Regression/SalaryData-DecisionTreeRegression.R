# Import Data set 
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/Position_Salaries.csv")

# Remove First Column that is not necessary 
df = df[2:3]

# Split Dataset into train test split is not necessary because Dataset is too small  
# No need for Feature Scaling: DT is not based on Euclidean Distances 

# Fit DT
# install.packages('rpart')
library(rpart)
# Number of split here is important 
DT = rpart(formula = Salary ~ ., 
           data = df,
           control = rpart.control(minsplit = 1))

# Predict a new Result 
y_pred = predict(DT, data.frame(Level =6.5))

library(ggplot2)
# Visualise the Decision Tree Results 
# First Nonlinear and Noncontinous Model 

x_grid = seq(min(df$Level), max(df$Level), 0.01)
ggplot() + 
  geom_point(aes(x = df$Level, y = df$Salary ),
             colour = 'red') + 
  geom_line(aes(x = x_grid, y = predict(DT, newdata = data.frame(Level = x_grid))), 
            colour = 'blue') + 
  ggtitle('Decision Tree Model') +
  xlab('Level') + 
  ylab('Salary')
