from sklearn.datasets import fetch_openml # openml is a data repository site
import pandas as pd

# Load titanic dataset
titanic = fetch_openml('titanic', version=1, as_frame=True) # there are multiple version of 'Titanic' data; as_frame=True means DataFrame

# Extract features X and target y
X = titanic.data
y = titanic.target

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

cat_cols.append('pclass')
num_cols.remove('pclass')

# Build numerical and categoricla pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Numeric pipeline
num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='mean')), # fill missing numeric values with mean
    ('scale', StandardScaler()) # standardize (zero mean, unit variance)
])

# Categorical pipeline
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')), # fill missing categorical values with most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))   # convert categories to one-hot (binary columns) vectors
])

# Combine into single preprocessor
preprocessor = ColumnTransformer([ 
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols) # applies right pipeline to the right set of columns
])

# Train-test split
from sklearn.model_selection import train_test_split

# split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) # random_state=42 ensures reproducibility, stratify=y keeps class balance consistent between train and test

# Fit and Transform with Preprocessor
# Fit the Preprocessor on training data only
X_train_processed = preprocessor.fit_transform(X_train) # imputation, scaling, and one-hot encoding in training data

# Transform test data using same fitted preprocessor
X_test_processed = preprocessor.transform(X_test) # applies same learned rules to test data

# Step 6: Build a Full Pipeline
from sklearn.linear_model import LogisticRegression

# Full pipeline: preprocession + model
clf_pipeline = Pipeline([
    ('preprocess', preprocessor), # our numeric + categorical transformations
    ('model', LogisticRegression(max_iter=1000)) # classifier
])

# Fit on training data
clf_pipeline.fit(X_train, y_train)

# Evaluate on test data
score = clf_pipeline.score(X_test, y_test)
print(f"Test accuracy: {score}")