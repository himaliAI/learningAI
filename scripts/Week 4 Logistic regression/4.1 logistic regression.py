import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
rng = np.random.default_rng(seed=0)

# Create synthetic data
X = np.linspace(0, 10, 100).reshape(-1, 1)  # study hours
y = (X.flatten() + rng.normal(0, 1, 100) > 5).astype(int)  # pass(1) if total > 5

# Define Sigmoid (logistic) function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define cost function (log-loss)
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5 # to avoid log(0)
    cost = (-1/m) * np.sum(y*np.log(h + epsilon) + (1 - y)*np.log(1 - h + epsilon))
    return cost

# Define gradient descent
def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1000):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= learning_rate * gradient
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history

# Add intercept (bias)
X_b = np.column_stack([np.ones(X.shape[0]), X])

# Initialize theta
theta = np.zeros((X_b.shape[1], 1))

# Run gradient descent
theta, cost_history = gradient_descent(X_b, y.reshape(-1, 1), theta, learning_rate=0.1, iterations=2000)

# Visualize cost funtion
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost J(θ)")
plt.title("Convergence of Gradient Descent")
plt.show()

# Visualize decision boundary
plt.scatter(X, y, c=y, cmap='bwr')
x_vals = np.linspace(0, 10, 100)
y_pred = sigmoid(theta[0] + theta[1] * x_vals)
plt.plot(x_vals, y_pred, color='green', label="Prediction curve")
plt.xlabel("Study Hours")
plt.ylabel("Predicted Probability of Passing")
plt.legend()
plt.show()
