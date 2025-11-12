from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
housing = fetch_california_housing(as_frame=True)

# Extract features and target
X = housing.data
y = housing.target

# Combine into one DataFrame for convenience
df = X.copy()
df['target'] = y

# Define split index (Here 80% train, 20% test)
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

# preprocessing + model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
#from sklearn.model_selection import train_test_split

# numeric pipeline
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

# full pipeline (preprocessing + model)
full_pipeline = Pipeline([
    ('preprocess', num_pipeline),
    ('model', LinearRegression())
])

# train test split
X_train, y_train = train.drop(columns='target'), train['target']
X_test, y_test = test.drop(columns='target'), test['target']

# fit on training data
full_pipeline.fit(X_train, y_train)

# evaluate on test data
r2_score = full_pipeline.score(X_test, y_test)
print(f"R^2 score on test data: {r2_score:.4f}")
