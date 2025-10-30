# same dataset as in gradient descent
import numpy as np

rng = np.random.default_rng(seed=42)
n = 100
x = rng.uniform(0, 10, n)
y_true = 2.5 * x + 1.5
noise = rng.normal(0, 2, n)
y = y_true + noise

# Prepare design matrix for our manual approach
X_manual = np.column_stack([np.ones(n), x])  # (n,2)
y_col = y.reshape(-1, 1)

'''
print("Shapes — X_manual:", X_manual.shape, "y:", y_col.shape)
print("First 3 x, y pairs:")
for xi, yi in zip(x[:3], y[:3]):
    print(round(xi,3), "->", round(yi,3))
'''

from sklearn.linear_model import LinearRegression
# scikit-learn expects X to be 2D (n_samples, m_features)
# but our x is currently 1D
X = x.reshape(-1, 1)

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Get parameters
print(f"Intercept (b): {model.intercept_}")
print(f"Coefficient (w): {model.coef_}")

# Use those parameters to predict
y_pred = model.predict(X)

# Plot to visualize Fit
import matplotlib.pyplot as plt

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Fitted Line')
plt.legend()
plt.show()