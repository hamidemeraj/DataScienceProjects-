import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
sns.set()

# Create Fake Data 
X, y_true = make_blobs(n_samples=400, centers=4, random_state=0, cluster_std= 0.7)
plt.scatter(X[:,0],X[:,1], s=20)

#Create kmeans Model
km = KMeans(n_clusters=4)    
km.fit(X)
y_pred = km.predict(X)
cluster_centers  = km.cluster_centers_
Cluster_labels = km.labels_

#Plot Data and their labels and Centers 
plt.scatter(X[:,0],X[:,1], s=20, cmap='viridis', c=y_pred) 
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], c='black', s=50, alpha=0.5)


#We can calculate centers like below: 
#Calculate Distance between each two data points 
from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_cluster, rseed = 2): 
    #randomly choose cluster 
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_cluster]
    centers = X[i]
    
    while True: 
        #Assign Labels Based on Closest Labels 
        labels = pairwise_distances_argmin(X,centers)
        
        #Find New Centers from Means of Points 
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_cluster)])
        
        #Check For Convergance 
        if np.all(centers == new_centers):
            break 
        centers = new_centers 
        
    return centers, labels 

centers, labels = find_clusters(X, n_cluster=4)
plt.scatter(X[:,0],X[:,1], s=20, cmap='viridis', c=y_pred) 
plt.scatter(centers[:,0], centers[:,1], c='black', s=50, alpha=0.5)           
    