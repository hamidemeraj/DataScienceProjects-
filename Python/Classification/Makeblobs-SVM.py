# Import Libraries 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.datasets import make_blobs

# Create Data 
X, y = make_blobs(n_samples=40, centers=2, random_state=20)

# Fit the Model 
sv = svm.SVC(kernel='linear', C=1000)
sv.fit(X,y)

# Using Model to predict Unknown data 
new_data = [[3,4],[5,6],[10,12]]
print(sv.predict(new_data))

# Display the Data in graph
plt.scatter(X[:,0],X[:,1], c=y, s=30, cmap=plt.cm.Paired)

# Plot Decision Lines
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)

YY, XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = sv.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors ='k', levels=[-1,0,1], alpha=0.5, linestyles=['--','-','--'])

ax.scatter(sv.support_vectors_[:,0], sv.support_vectors_[:,1], s=100, linewidth=1, facecolor='none')
plt.show()
