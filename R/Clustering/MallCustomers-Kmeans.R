# Import Datasets 
df <- read.csv("C:/Users/snapp/Data-Projects/Datasets/Mall_Customers.csv")
X <- df[4:5]

# Using the Elbow Method to Find the Best number of clsuters 
set.seed(6)
wcss <- vector()
for (i in 1:10) wcss[i] <- sum(kmeans(X,i)$withinss)
plot(1:10,
     wcss,
     type ="b",
     main = paste("Clusters of Clients"),
     xlab = "Number of Clusters",
     ylab = "WCSS")

set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

y_kmeans = kmeans$cluster

# Visulaising the Clusters 
library(cluster)
clusplot(X, 
         y_kmeans, 
         lines = 0, 
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE, 
         main = paste('Clusters of Clients'),
         xlab = "Annual Income",
         ylab = "Spending Score"
         )