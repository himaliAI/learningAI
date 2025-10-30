import numpy as np
import matplotlib.pyplot as plt

# generate some data
rng = np.random.default_rng(42)
n = 100

x = rng.uniform(0, 10, n)
y_true = 2.5 * x + 1.5
noise = rng.normal(0, 2, n)
y = y_true + noise

# Reshape X and y for matrix operations
X = np.column_stack([np.ones(n), x]) 
y = y.reshape(-1, 1)

# Initialize parameters
theta = np.zeros((2, 1)) #[b, w]
alpha = 0.001            # learning rate
epochs = 1000            # no of iterations
m = len(y)               # no of samples

# define the gradient descent loop
cost_history = []
for i in range(epochs):
    y_pred = X @ theta  # predictionss
    error = y_pred - y  # error vector
    cost = (1/(2*m)) * np.sum(error ** 2) # cost function
    cost_history.append(cost)

    gradient = (1/m) * (X.T @ error)    # gradient calculation
    theta = theta - alpha * gradient

'''
# view results
print(f"Estimated parameters (theta): {theta.ravel()}") # ravel flattens the array/vector withoout copying; flattens() copies it
plt.plot(range(epochs), cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost j(theta)')
plt.title('Cost Function Convergence')
plt.show()
'''

# Visualize the fitted line
plt.scatter(x, y, label='Data')
plt.plot(x, X @ theta, color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()