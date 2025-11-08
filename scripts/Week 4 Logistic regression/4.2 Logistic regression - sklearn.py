import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Create synthetic data
rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=(100, 1)) # hours studied
y = (X + rng.normal(0, 2, size=(100, 1)) > 5).astype(int).ravel() # 1 if > 5 (pass) or 0 (fail)

# Train logistic regression
model = LogisticRegression()
model.fit(X, y)

# Prediction
X_test = np.linspace(0, 10, 200).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:,1] 

# Inspect model parameters
print(f"Intercept (b): {model.intercept_}")
print(f"Coefficient (w): {model.coef_}")

# Plot
plt.scatter(X, y, c=y, cmap='bwr', edgecolor='k', label="Data")
plt.plot(X_test, y_prob, 'k-', linewidth=2, label="Sigmoid Fit")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression with scikit-learn")
plt.legend()
plt.show()