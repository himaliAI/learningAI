import numpy as np
from sklearn.model_selection import train_test_split

# 1. Arrays & Indexing
a = np.array([1, 2, 3])
print(a[0])        # indexing
print(a[1:])       # slicing

# 2. Broadcasting
a = np.array([1, 2, 3])
b = np.array([10])
print(a + b)       # [11, 12, 13]

# 3. Reshaping & Flattening
a = np.arange(1, 17)
a_reshaped = a.reshape(4, 4)
a_flattened = a_reshaped.ravel()
print(a_reshaped)
print(a_flattened)

# 4A. Stacking
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.array([[7, 8, 9], [10, 11, 12]])
print(np.vstack((b, c)))   # vertical stack
print(np.hstack((b, c)))   # horizontal stack

# 4B. Splitting
a = np.arange(12).reshape(3, 4)
print(np.split(a, 2, axis=1))          # equal split
print(np.array_split(a, 3, axis=1))    # unequal split

# 4C. Train-Test Split
X = np.arange(100).reshape(50, 2)
y = np.random.randint(0, 2, size=50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

# 5. Random Numbers
rng = np.random.default_rng(42)
print(rng.integers(1, 10, size=(2, 3)))
print(rng.normal(50, 5, 5))
arr = np.arange(10)
rng.shuffle(arr)
print(arr)
print(rng.choice(np.arange(10), size=3, replace=False))

# 6A. Feature Scaling
X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))   # Min-Max
X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)              # Z-score
print(X_scaled)
print(X_standardized)

# 6B. Distance Metrics
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
euclidean = np.linalg.norm(x - y)
cosine = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
print("Euclidean:", euclidean)
print("Cosine:", cosine)

# 6C. Vectorization
X = np.array([[1, 2, 3], [4, 5, 6]])
w = np.array([0.1, 0.2, 0.3])
y_pred = X @ w
y_true = np.array([1, 2])
mse = np.mean((y_true - y_pred)**2)
print("Predictions:", y_pred)
print("MSE:", mse)