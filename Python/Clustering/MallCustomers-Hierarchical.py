# Import Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
 
# Import dataset
df = pd.read_csv(r"C:\Users\snapp\Data-Projects\Datasets\Mall_Customers.csv")
X = df.iloc[:,[3,4]].values

# Use Dendogram to find the Optimal cluster
import scipy.cluster.hierarchy as sch 
# Ward is a method that tries to minimize the variance within each cluster 
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel("Customers")
plt.ylabel("Euclidean Distances")
plt.show()

# Fitting Hierarichical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualizing the clusters 
 # Name each clster after watching the visualization
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Clusetr1')


plt.title("Clusters of C=lients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()








