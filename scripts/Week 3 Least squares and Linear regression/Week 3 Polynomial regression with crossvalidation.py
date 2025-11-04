import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

# 1️⃣ Generate data
rng = np.random.default_rng(0)
X = rng.uniform(0, 2, 100).reshape(-1, 1)
y = 4 + 3 * X.squeeze()**2 + rng.normal(0, 0.3, size=100)

# 2️⃣ Cross-validation to test polynomial degrees 1→10
degrees = range(1, 11)
mean_scores = []

for deg in degrees:
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
    mean_scores.append(np.mean(scores))

# 3️⃣ Find the best degree
best_degree = degrees[np.argmax(mean_scores)]
print(f"✅ Best degree: {best_degree} (Mean R² = {max(mean_scores):.3f})")

# 4️⃣ Refit the model using best degree on all data
poly_best = PolynomialFeatures(degree=best_degree, include_bias=False)
X_poly_best = poly_best.fit_transform(X)

model_best = LinearRegression()
model_best.fit(X_poly_best, y)

# 5️⃣ Visualization
X_plot = np.linspace(0, 2, 200).reshape(-1, 1)
X_plot_poly = poly_best.transform(X_plot)
y_plot = model_best.predict(X_plot_poly)

plt.figure(figsize=(10, 5))

# Plot data and best-fit line
plt.scatter(X, y, color="gray", label="Data points")
plt.plot(X_plot, y_plot, color="red", linewidth=2, label=f"Best fit (deg={best_degree})")

# Plot CV scores
plt.twinx()
plt.plot(degrees, mean_scores, marker='o', color='blue', label="CV Mean R²")
plt.title("Best Polynomial Fit and Cross-Validation Performance")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean R²", color="blue")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()