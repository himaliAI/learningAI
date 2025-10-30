import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n = 50
x = 2.0 * np.random.rand(n)
true_w = 3.5
true_b = -1.0
noise = np.random.randn(n) * 0.8
y = true_w * x + true_b + noise

# Reshape x into (n,1)
X = np.column_stack([np.ones(n), x])   # shape (50, 2)
y = y.reshape(-1, 1)                   # shape (50, 1)

# Normal equation: θ = (Xᵀ X)⁻¹ Xᵀ y
theta = np.linalg.inv(X.T @ X) @ (X.T @ y)

# Flatten θ into 1D array (makes indexing easy)
theta = theta.flatten()

b_est, w_est = theta[0], theta[1]
print("Estimated intercept (b):", b_est)
print("Estimated slope (w):", w_est)

# Predictions
y_pred = X @ theta.reshape(-1, 1)

# Plot
plt.scatter(x, y, label="Data")
plt.plot(x, y_pred, color="red", linewidth=2, label="Fit (Normal Eq.)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
