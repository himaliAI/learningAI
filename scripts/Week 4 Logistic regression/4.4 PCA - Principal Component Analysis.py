import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

"""
rng = np.random.default_rng(42)
x1 = rng.normal(0, 1, 100) # feature 1
x2 = 2 * x1 + rng.normal(0, 0.5, 100) # feature 2

data = pd.DataFrame({'x1': x1, 'x2': x2})

'''
plt.figure(figsize=(6,6))
plt.scatter(data['x1'], data['x2'], alpha=0.7, color='blue')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter plot of correlated data')
plt.grid(True)
plt.show()
'''

# visualize the Principal Axes
# step 1: standardize the data
scaler = StandardScaler()
data_std = scaler.fit_transform(data) # calculate mean and SD of each column and standardize it i.e mean=0, sd=1

# step 2: Fit PCA
pca = PCA(n_components=2) # PCA object with 2 principal components
pca.fit(data_std) # pca takes the standardized data and calcultes covariance matrix, direction (eigenvectors), and variances (eigenvalues)

# step 3: Get principal components (directions)
pc1 = pca.components_[0] # first principal component from array of eigenvectors
pc2 = pca.components_[1]

'''
# step 4: Plot data + principal axes
plt.figure(figsize=(6,6))
plt.scatter(data_std[:,0], data_std[:,1], alpha=0.7, color='blue')
plt.xlabel('Standardized x1')
plt.ylabel('Standardized x2')
plt.title('Data with Principal Axes')
plt.grid(True)

# Draw arrows for PC1 and PC2
origin = [0, 0]
plt.quiver(*origin, pc1[0], pc1[1], color='red', scale=3, label='PC1')
plt.quiver(*origin, pc2[0], pc2[1], color='green', scale=3, label='PC2')
# quiver is Matplotlib function that draws arrows (vectors)
# *orign mean unpack {origin} to its values
# pc1[0] x-coordinate; pc1[1] y-coordinate

plt.legend()
plt.show()
'''

print(f"Explained variance ratio: {pca.explained_variance_ratio_}") # fraction of total variance captured by each component
print(f"Explained variance: {pca.explained_variance_}") # the acutal variance (eigenvalues); larger values means more important component

# transform data to PCA space
data_pca = pca.transform(data_std) # projects each point onto principal component axes
# now data_pca has coordicates in terms of PC1 and PC2

# convert to DataFrame for clarity
data_pca_df = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
print(data_pca_df.head())

'''
# Plotting in PCA space
plt.figure(figsize=(6,6))
plt.scatter(data_pca_df['PC1'], data_pca_df['PC2'], alpha=0.7, color='purple')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Data in PCA Space')
plt.grid(True)
plt.show()
'''
"""

# load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Build pipeline with StandardScaler + PCA + Logistic regression
# so that data first get scaled (standerdized), then reduced in dimension by PCA, then classified by logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('clf', LogisticRegression(max_iter=1000))
])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit model
# pipeline ensures all steps (scaler, pca, clf) are applied consistently
pipeline.fit(X_train, y_train)

# Evaluate
score = pipeline.score(X_test, y_test)


# Compare accuracy with Vs without PCA
# Pipeline A: with PCA
pipeline_pca = Pipeline([
    ('scalar', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline_pca.fit(X_train, y_train)
score_pca = pipeline_pca.score(X_test, y_test)

# Pipeline B: without PCA
pipeline_no_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline_no_pca.fit(X_train, y_train)
score_no_pca = pipeline_no_pca.score(X_test, y_test)

print(f"Accuracy with PCA (2D): {score_pca}")
print(f"Accuracy without PCA (4D): {score_no_pca}")
