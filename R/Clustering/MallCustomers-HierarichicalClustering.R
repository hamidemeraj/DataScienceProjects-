# Import Datasets 
df <- read.csv("C:/Users/snapp/Data-Projects/Datasets/Mall_Customers.csv")
X <- df[4:5]

# Use Dendogram to find optimal number of Clusters 
dendrogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendrogram, 
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean Distance')

# Fitting the Hierarchical Clustering to the mall dataset 
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
# Define number of clusters and define cluster for each customers 
y_hc = cutree(hc, k = 5)

# Visulaizing the Clusters 
library(cluster)
clusplot(X, 
         y_hc, 
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