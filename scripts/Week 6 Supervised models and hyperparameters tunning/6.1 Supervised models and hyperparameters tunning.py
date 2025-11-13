# Step 1: Baseline Linear Regression
# setup
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# load data
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Train test split (20% test data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Numeric preprocessing pipeline
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

# full pipeline (preprocessing + lineare regression model)
lin_pipeline = Pipeline([
    ('preprocessing', num_pipeline),
    ('model', LinearRegression())
])

# Fit and evaluate
lin_pipeline.fit(X_train, y_train)
r2 = lin_pipeline.score(X_test, y_test)

print(f"Linear regressioin R^2: {r2:.4f}")

# Step 2: Regularization with Ridge and Lasso
from sklearn.linear_model import Ridge, Lasso

# Ridge pipeline, fit and evaluate
ridge_pipeline = Pipeline([
    ('preprocessing', num_pipeline),
    ('model', Ridge(alpha=1))
])
ridge_pipeline.fit(X_train, y_train)
ridge_r2 = ridge_pipeline.score(X_test, y_test)
print(f"Ridge (alpha=0.1) R^2: {ridge_r2:.4f}")

# Lasso pipeline, fit and evaluate
lasso_pipeline = Pipeline([
    ('preprocessing', num_pipeline),
    ('model', Lasso(alpha=0.001, max_iter=5000))
])
lasso_pipeline.fit(X_train, y_train)
lasso_r2 = lasso_pipeline.score(X_test, y_test)
print(f"Lasso (alpha=0.001) R^2: {lasso_r2:.4f}")

# Step 3: Decision trees, Random forest, and Gradient Boosting
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Decision tree (basic)
tree_pipeline = Pipeline([
    ('preprocessing', num_pipeline),
    ('model', DecisionTreeRegressor(random_state=42))
])
tree_pipeline.fit(X_train, y_train)
tree_r2 = tree_pipeline.score(X_test, y_test)
print(f"Decision Tree R^2 {tree_r2:.4f}")

# Random Forest (default-ish)
rf_pipeline = Pipeline([
    ('preprocessing', num_pipeline),
    ('model', RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1))
])
rf_pipeline.fit(X_train, y_train)
rf_r2 = rf_pipeline.score(X_test, y_test)
print(f"Random Forest R^2: {rf_r2:.4f}")

# Gradient Boosting (learning rate + depth matter)
gb_pipeline = Pipeline([
    ('preprocessing', num_pipeline),
    ('model', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42))
])
gb_pipeline.fit(X_train, y_train)
gb_r2 = gb_pipeline.score(X_test, y_test)
print(f"Gradient Boosting R^2: {gb_r2:.4f}")

# Step 4: SVMs and k-NN (regression)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# SVR regression
svr_pipeline = Pipeline([
    ('preprocessing', num_pipeline),
    ('model', SVR(C=10, gamma='scale'))
])
svr_pipeline.fit(X_train, y_train)
svr_r2 = svr_pipeline.score(X_test, y_test)
print(f"SVR (RBF) R^2: {svr_r2:.4f}")

# k-NN
knn_pipeline = Pipeline([
    ('preprocessing', num_pipeline),
    ('model', KNeighborsRegressor(n_neighbors=7))
])
knn_pipeline.fit(X_train, y_train)
knn_r2 = knn_pipeline.score(X_test, y_test)
print(f"k-NN (k=7) R^2: {knn_r2:.4f}")

# Step 5: Hyperparameter tuning with GridSearchCV
from sklearn.model_selection import GridSearchCV
rf_gs_pipeline = Pipeline([
    ('preprocessing', num_pipeline),
    ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_grid = {
    'model__n_estimators': [200, 400],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5],
    'model__max_features': ['sqrt', 'log2', None]
}

search = GridSearchCV(rf_gs_pipeline, param_grid, cv=3, n_jobs=-1)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best CV R^2: {search.best_score_:.4f}")

# evaluate best model on test set
best_r2 = search.best_estimator_.score(X_test, y_test)
print(f"Test R^2 (best RF): {best_r2:.4f}")
