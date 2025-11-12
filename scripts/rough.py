from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing(as_frame=True)

# Extract features X and target y
X = data.data
y = data.target

# All colums in X are float64, i.e numericals
num_cols = X.select_dtypes(include=['float64']).columns.tolist()

# Build numerical pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Numeric pipeline
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
])

# As all data are numerical we do not need cat_pipeline and we do not need to combine them

# train test split (80% training data, 20% testing data)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit preprocessor in training data
X_train_processed = num_pipeline.fit_transform(X_train)

# transform test data using same learned paramenters
X_test_processed = num_pipeline.transform(X_test)

# build a full pipeline
from sklearn.linear_model import LinearRegression

full_pipeline = Pipeline([
    ('preprocess', num_pipeline),
    ('model', LinearRegression())
])

# fit on training data
full_pipeline.fit(X_train, y_train)

# evaluate on test data
score = full_pipeline.score(X_test, y_test)

print(f"R^^2 score on test data: {score:.4f}")