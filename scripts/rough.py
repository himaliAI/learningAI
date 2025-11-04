# ---------------------------------------------------------------
# POLYNOMIAL REGRESSION (VISUAL DEMO)
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ---------------------------------------------------------------
# 1. Create synthetic non-linear data
# ---------------------------------------------------------------
rng = np.random.default_rng(0)
n = 100
x = rng.uniform(0, 5, n).reshape(-1, 1)

# True relationship (a cubic equation)
y_true = 2 + 3*x - 0.5*x**2 + 0.1*x**3

# Add random noise
y = y_true + rng.normal(0, 1, size=(n, 1))

# ---------------------------------------------------------------
# 2. Fit a simple Linear Regression (just for comparison)
# ---------------------------------------------------------------
lin_model = LinearRegression()
lin_model.fit(x, y)
y_lin_pred = lin_model.predict(x)

# ---------------------------------------------------------------
# 3. Polynomial Regression (degree = 3)
# ---------------------------------------------------------------
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(x)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Predictions
x_plot = np.linspace(0, 5, 200).reshape(-1, 1)
X_plot_poly = poly.transform(x_plot)
y_poly_pred = poly_model.predict(X_plot_poly)

# ---------------------------------------------------------------
# 4. Print model parameters
# ---------------------------------------------------------------
print("Linear Regression:")
print("  Intercept:", lin_model.intercept_)
print("  Coefficient:", lin_model.coef_)
print()
print("Polynomial Regression (degree = 3):")
print("  Intercept:", poly_model.intercept_)
print("  Coefficients:", poly_model.coef_)
print()

# ---------------------------------------------------------------
# 5. Visualize Results
# ---------------------------------------------------------------
plt.figure(figsize=(12, 8))

# (a) Raw data and true function
plt.subplot(2, 2, 1)
plt.scatter(x, y, color='blue', alpha=0.6, label='Noisy Data')
plt.plot(x, y_true, color='green', label='True Function', linewidth=2)
plt.title('True Function vs Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# (b) Linear Regression Fit
plt.subplot(2, 2, 2)
plt.scatter(x, y, color='lightgray', label='Data')
plt.plot(x, y_lin_pred, color='red', linewidth=2, label='Linear Fit')
plt.title('Linear Regression (Underfits)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# (c) Polynomial Regression Fit (degree=3)
plt.subplot(2, 1, 2)
plt.scatter(x, y, color='blue', alpha=0.5, label='Data')
plt.plot(x_plot, y_poly_pred, color='red', linewidth=2, label='Polynomial Fit')
plt.plot(x_plot, 2 + 3*x_plot - 0.5*x_plot**2 + 0.1*x_plot**3, 
         color='green', linestyle='--', label='True Function')
plt.title('Polynomial Regression (degree=3)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()