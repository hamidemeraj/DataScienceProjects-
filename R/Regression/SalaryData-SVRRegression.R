# Import Data set 
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/Position_Salaries.csv")

# Remove First Column that is not necessary 
df = df[2:3]

# Split Dataset into train test split is not necessary because Dataset is too small  
# No need for Feature Scaling 

# Fit SVR
# install.packages('e1071')
library(e1071)
SVR = svm(formula = Salary ~ .,
          data = df,
          type = 'eps-regression')

# Predict a new Result 
y_pred = predict(SVR, data.frame(Level =6.5))

library(ggplot2)
# Visualise the SVR Results 
ggplot() + 
  geom_point(aes(x = df$Level, y = df$Salary ),
             colour = 'red') + 
  geom_line(aes(x = df$Level, y = predict(SVR, newdata = df)), 
            colour = 'blue') + 
  ggtitle('SVR Model') +
  xlab('Level') + 
  ylab('Salary')
