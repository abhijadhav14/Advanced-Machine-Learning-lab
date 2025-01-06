import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Step 1: Generate sample data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Step 2: Visualize the data
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title("Sample Data")
plt.show()

# Step 3: Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Step 4: Extract cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Step 5: Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering")
plt.legend()
plt.show()

