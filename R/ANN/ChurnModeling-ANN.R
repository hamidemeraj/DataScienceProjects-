# Import Data set 
df = read.csv("C:/Users/snapp/Data-Projects/Datasets/Churn_Modelling.csv")
df = df[, 4:14]

# Encoding the Target Feature aas Factor 
# It is not necessary to labelencode target column
df$Exited = factor(df$Exited, level = c(0,1))

# Encoding the Categorical  Feature as Factor 
df$Geography = as.numeric(factor(df$Geography, 
                                 levels = c('France','Spain','Germany'),
                                 labels = c(0,1,3)))
df$Gender = as.numeric(factor(df$Gender, 
                              levels = c('Female','Male'),
                              labels = c(1,2)))

# Split Dataset into train test split 
library(caTools)
# Set random state for seeing same data 
set.seed(123)

split = sample.split(df$Exited, SplitRatio = 0.8)
train_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)

# Feature Scaling 
# Need Numeric Columns 
train_set[-11] = scale(train_set[-11]) 
test_set[-11] = scale(test_set[-11])

# Fitting ANN to the training set 
# install.packages("h2o", dependencies = T)
library(h2o)
# establish a connection 
h2o.init(nthreads = -1 )
classifier = h2o.deeplearning(y = 'Exited', 
                              training_frame = as.h2o(train_set), 
                              activation = 'Rectifier', 
                              hidden = c(5,5), 
                              epochs = 100, 
                              train_samples_per_iteration = -2)
# predicting the Test Result 
y_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

# Convert as h2o object to a vector to use it in Confusion Matrix 
# Make Confusion Matrix 
print(CM)

# Disconnect from H2O
h2o.shutdown()

