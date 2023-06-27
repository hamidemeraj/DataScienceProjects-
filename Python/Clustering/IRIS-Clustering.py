import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
iris = load_iris()
Data_iris = iris.data

"""
Kmeans Clustering 
"""
from sklearn.cluster import KMeans
KMS = KMeans(n_clusters=3)
KMS.fit(Data_iris)

Labels = KMS.predict(Data_iris)
Ctn = KMS.cluster_centers_
plt.scatter(x = Data_iris[:,2], y= Data_iris[:,3], c = Labels)
plt.scatter(Ctn[:,2],Ctn[:,3], marker='o', c = 'red', s = 120)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

# Evaluating Diffrent number of Clusters 
K_inertia = []
for i in range(1,10):
    KMS = KMeans(n_clusters=i, random_state=44)
    KMS.fit(Data_iris)
    K_inertia.append(KMS.inertia_)

# Elbow Method 
plt.plot(range(1,10), K_inertia, color = 'green', marker = 'o')
plt.xlabel('The number of Cluster')
plt.ylabel('Inertia')
plt.show()

"""
DBSCAN Clustering 
"""

from sklearn.cluster import DBSCAN
DBS = DBSCAN(eps =0.9 , min_samples= 4)
DBS.fit(Data_iris)

# Negative numbers are abnormalities and outliers (potential)
Labels = DBS.labels_
plt.scatter(x = Data_iris[:,2], y= Data_iris[:,3], c = Labels)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

"""
Hierarchical Clustering 
"""
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Diffrent Method For Linkage (Single - Complete - Average)
HR_complete = linkage(Data_iris, method= 'complete', metric = 'euclidean')
DND_complete = dendrogram(HR_complete)

HR_single = linkage(Data_iris, method= 'single', metric = 'euclidean')
DND_single = dendrogram(HR_single)

Labels_4 = fcluster(HR_complete,t = 4, criterion='distance')
# If you edcrease t you have more clusters
plt.scatter(Data_iris[:,2], Data_iris[:,3], c = Labels_4)
plt.xlabel('Pethal width')
plt.ylabel('Pethal Length')
plt.show()

Labels_1 = fcluster(HR_complete,t = 1 , criterion='distance')
# If you edcrease t you have more clusters
plt.scatter(Data_iris[:,2], Data_iris[:,3], c = Labels_1)
plt.xlabel('Pethal width')
plt.ylabel('Pethal Length')
plt.show()




