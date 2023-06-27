# import Libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# Reading the Datasets 
df = pd.read_csv(r"C:\Users\snapp\Data-Projects\Datasets\Mall_Customers.csv")
X = df.iloc[:, [3,4]].values

# Using Elbow Method to find optimal Number of Clusters 
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11): 
    # Init : Do not start random to not to fall into intialization trap 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss )
plt.title(" The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Applying Kmeans to the df 
kmeans = KMeans(n_clusters= 5, init = 'k-means++', max_iter= 300, n_init= 10, random_state= 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the Clusters 
# Name each clster after watching the visualization
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Clusetr1')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.title("Clusters of C=lients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()

