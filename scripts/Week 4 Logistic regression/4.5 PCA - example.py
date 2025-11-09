from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load dataset
wine = load_wine()
X, y = wine.data, wine.target

# convert to DataFrame for clarity
df_wine = pd.DataFrame(X, columns=wine.feature_names)
df_wine['target'] = y

# Apply PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

'''
# Apply PCA
pca = PCA(n_components=2) # reduce to 2D for visualization
X_pca = pca.fit_transform(X_std)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Visualize in PCA space
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Wine Dataset in PCA Space')
plt.colorbar(label='Wine Class')
plt.grid(True)
plt.show()
'''

# Fit PCA
pca = PCA().fit(X_std)

# Explained variance ratios
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

# Decide cutoff (eg 95%)
cutoff = 0.95
n_components = np.argmax(cumulative_var >= cutoff) + 1 
# np.argmax(cumulative_var >= cutoff) returns index of first value in cumulative_var which is >= cutoff

print(f"Explained variance ratio: {explained_var}")
print(f"Cumulative variance: {cumulative_var}")
print("Number of components to retain (95% cutoff):", n_components)

# Scree plot
plt.figure(figsize=(6,4))
plt.plot(range(1, len(explained_var)+1), cumulative_var, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Wine Dataset Scree Plot')
plt.grid(True)
plt.show()