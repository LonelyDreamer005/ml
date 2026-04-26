import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv('/content/sample_data/Iris1.csv')

x = df.select_dtypes(include=[np.number])

true_labels = None
if 'species' in df.columns:
    true_labels = df['species'].astype('category').cat.codes

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(x_scaled)
em = GaussianMixture(n_components=3, random_state=0)
em_labels = em.fit_predict(x_scaled)

print("K-Means Silhouette Score:", silhouette_score(x_scaled, kmeans_labels))
print("EM Silhouette Score:", silhouette_score(x_scaled, em_labels))

plt.figure()
plt.subplot(1,2,1)
plt.scatter(x_scaled[:,0], x_scaled[:,1], c=kmeans_labels)
plt.title('K-Means')
plt.subplot(1,2,2)
plt.scatter(x_scaled[:,0], x_scaled[:,1], c=em_labels)
plt.title('EM')
plt.show()

