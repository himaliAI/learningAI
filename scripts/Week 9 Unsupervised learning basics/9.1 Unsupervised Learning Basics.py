# Load and prepare the data
import pandas as pd
from sklearn.datasets import load_wine
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Standardize features; PCA and k-means are sensitive to non-standardized data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA and plot explained variance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

pca = PCA()
X_pca = pca.fit_transform(X_scaled) # learns directons of maximum variance
explained_var = pca.explained_variance_ratio_ # tells us how much variance each principal compoenet captures

# plot cumulutive explained variance
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

# applying k-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca[:, :2]) 
    # we clustered data using only first 2 PCA components
    # fit_predict() assigns each sample to one of 3 clusters
# Plot cluster assignments
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', edgecolor='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-means Clustering on PCA-reduced Data')
plt.grid(True)
plt.tight_layout()
plt.show()

# Run isolation Forest for anomaly detection
    # IsolationForest isolates anomalies by randomly partitionling the data
    # contamination=0.05 assumes about 5% of data are outliers
    # fit_predict() returns -1 for anomalies, 1 for normal points
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05, random_state=42)
anomaly_labels = iso.fit_predict(X_scaled)

# Visualize anomalies in PCA space
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=(anomaly_labels == -1), cmap='coolwarm', edgecolor='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Isolation Forest - Anomaly Detection')
plt.grid(True)
plt.tight_layout()
plt.show()

# Summarize cluster profiles
    # Add cluster labels to original DataFrame
X_clustered = X.copy()
X_clustered['cluster'] = kmeans_labels
    # group by cluster and compute mean feature values
cluster_summary = X_clustered.groupby('cluster').mean().round(2)
    # Display summary
print(cluster_summary)
    # You will see a table showing average values of each feature per cluster