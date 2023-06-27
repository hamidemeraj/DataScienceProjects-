# Import Data set 
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/Position_Salaries.csv")

# Remove First Column that is not necessary 
df = df[2:3]

# Split Dataset into train test split is not necessary because Dataset is too small  
# No need for Feature Scaling 

# Fit Random Forest
# install.packages('randomForest')
library(randomForest)
set.seed(1234)
RF = randomForest(x = df[1],
                  y = df$Salary,
                  ntree = 500)

# Predict a new Result 
y_pred = predict(RF, data.frame(Level = 6.5))

library(ggplot2)
# Visualise the Random Forest Results(with Higher Resolutions)
# Nonlinear and Noncontinous Model 
x_grid = seq(min(df$Level), max(df$Level), 0.01)
ggplot() + 
  geom_point(aes(x = df$Level, y = df$Salary ),
             colour = 'red') + 
  geom_line(aes(x = x_grid, y = predict(RF, newdata = data.frame(Level = x_grid))), 
            colour = 'blue') + 
  ggtitle('Random Forest Model') +
  xlab('Level') + 
  ylab('Salary')
