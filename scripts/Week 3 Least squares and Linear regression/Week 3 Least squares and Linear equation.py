import numpy as np

rng = np.random.default_rng(42)
N = 200
X = rng.normal(size=(N, 2)) # (200x2) maxtix of features
w_true = np.array([2.0, -3.0]) # true weights
b_true = 0.5 # true bias

y = X @ w_true + b_true + rng.normal(scale=0.5, size=N) # target values with noise

# Add bias column (of ones) to X
Xb = np.c_[X, np.ones(N)]

# Closed-form least squares solution (normal equation)
w_hat = np.linalg.pinv(Xb) @ y
print(f"w_hat: {w_hat}") # Should be close to [2.0, -3.0, 0.5]

# Define the gradient of MSE and Optimize using simple gradient descent
def mse_grad(X, y, w):
    N = X.shape[0]
    preds = X @ w
    grad = (2.0/N) * (X.T @ (preds - y))
    return grad

Xb = np.c_[X, np.ones(N)]
w = np.zeros(Xb.shape[1]) # Initial weights [0, 0, 0]
alpha = 0.1 # Learning rate, controls step size
for t in range(500):
    g = mse_grad(Xb, y, w)
    w -= alpha * g
print(f"w (GD): {w}")

# Compare solutions and Errors
from sklearn.metrics import mean_squared_error
print(f"MSE closed-form: {mean_squared_error(y, Xb @ w_hat)}")
print(f"MSE GD: {mean_squared_error(y, Xb @ w)}")