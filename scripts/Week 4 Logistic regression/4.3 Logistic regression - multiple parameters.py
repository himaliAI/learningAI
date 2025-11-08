import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

rng = np.random.default_rng(0)

# 500 samples, each with 5 features
X = rng.normal(0, 1, size=(500, 5))

# True coefficients (just for simulation)
true_theta = np.array([1.5, -2.0, 0.8, 0.0, 1.2])

# generate probabilities and binary targets
logits = X @ true_theta + rng.normal(0, 1, size=500)
probabilities = 1 / (1 + np.exp(-logits))
y = (probabilities > 0.5).astype(int)

# Split data into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
# predict on test set
y_pred = model.predict(X_test)

# calculate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

# show coefficients, each coefficient has corresponding weightage in that feature
print(f"Intercept (b): {model.intercept_}")
print(f"Coefficients (w): {model.coef_}")

# confusion matrix and report
print(f"\nConfusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n {classification_report(y_test, y_pred)}")
