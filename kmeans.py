import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

x,y = make_blobs(n_samples=200, centers=3)

kmeans=KMeans(n_clusters=3)
kmeans.fit(x)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.figure()
plt.scatter(x[:,0], x[:,1], c=labels)
plt.scatter(centroids[:,0], centroids[:,1], marker='*', c='red')
plt.show()

