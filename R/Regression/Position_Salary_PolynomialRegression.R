# Import Data set 
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/Position_Salaries.csv")

# Remove First Column that is not necessary 
df = df[2:3]

# Split Dataset into train test split is not necessary because Dataset is too small  

# Fit Linear Regression 
LR = lm(formula = Salary ~ .,
        data = df )
summary(LR)

# Fit Polynomial Regression 
df$Level2 = df$Level^2
df$Level3 = df$Level^3
df$Level4 = df$Level^4
PLR = lm(formula = Salary ~ .,
         data = df) 
summary(PLR)

library(ggplot2)
# Visualise the Linear Regression 
ggplot() + 
  geom_point(aes(x = df$Level, y = df$Salary ),
             colour = 'red') + 
  geom_line(aes(x = df$Level, y = predict(LR, newdata = df)), 
            colour = 'blue') + 
  ggtitle('Linear Regression') +
  xlab('Level') + 
  ylab('Salary')

# Visualise the Polynomial Regression 
ggplot() + 
  geom_point(aes(x = df$Level, y = df$Salary ),
             colour = 'red') + 
  geom_line(aes(x = df$Level, y = predict(PLR, newdata = df)), 
            colour = 'blue') + 
  ggtitle('Polynomial Regression') +
  xlab('Level') + 
  ylab('Salary')

# Visualising the Polynomial Model results (for higher resolution and smoother curve)
x_grid = seq(min(df$Level), max(df$Level), 0.1)
ggplot() +
  geom_point(aes(x = df$Level, y = df$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(PLR, 
                                        newdata = data.frame(Level = x_grid,
                                                             Level2 = x_grid^2,
                                                             Level3 = x_grid^3,
                                                             Level4 = x_grid^4))),
            colour = 'blue') +
  ggtitle('Polynomial Regression') +
  xlab('Level') +
  ylab('Salary')

# Predict a New Result in Linear Regression
y_pred = predict(LR, newdata = data.frame(Level = 6.5))
# Predict a New Result in Polynomial Regression 
y_pred = predict(PLR, newdata = data.frame(Level = 6.5,
                                           Level2 = 6.5^2,
                                           Level3 = 6.5^3,
                                           Level4 = 6.5^4))

