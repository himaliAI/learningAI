# Setup and data preparation
import pandas as pd
import numpy as np
import seaborn as sns
df = sns.load_dataset('titanic').copy()

df = df.dropna(subset='survived')
y = df['survived'].astype(int)
X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]

# Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Features engineering and pipelines
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

# custom transformers to be plugged into scikit-learn pipelines
# BaseEstimator provides sklearn conventions like get_params / set_params
# TransformerMixin supplies a default fit_transform method and enforces fit / transform interface
class LogFare(BaseEstimator, TransformerMixin):
    # here fit does nothing, just return itself, but is required by sklearn.
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Z = X.copy()
        # safe log: log1p handles zero
        # fillna(0) means fill missing fare with 0 to avoid log(NA) error
        # np.log1p(x) = log(1 + x)
        Z['fare_log'] = np.log1p(Z['fare'].fillna(0))
        return Z

class AgeFareBins(BaseEstimator, TransformerMixin):
    def __init__(self, n_fare_bins=4):
        self.n_fare_bins = n_fare_bins
    def fit(self, X, y=None):
        # compute bin edges on training set
        age = X['age']
        fare = X['fare']

        # age bins: fixed cut points, the underscore at end is a sklearn convention: attributes learned during fit() get a trailing underscore
        self.age_bins_ = [0, 12, 18, 35, 50, 80, np.inf]
        
        # fare bins: quantiles for robustness
        # fare.dropna() - missing data in fare column is removed for quantile calculation
        self.fare_bins_ = np.quantile(fare.dropna(), np.linspace(0, 1, self.n_fare_bins + 1))
        return self
    def transform(self, X):
        Z = X.copy()
        # pd.cut(data, bins=[...], right = False) => cuts data into bins; includes left edge of bin but exclude right edge of each bin
        Z['age_group'] = pd.cut(Z['age'], bins=self.age_bins_, right=False)
        Z['fare_group'] = pd.cut(Z['fare'], bins=self.fare_bins_, include_lowest=True)
        return Z

class FamilyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Z = X.copy()
        # .fillna(0) => replaces missing values with 0
        # +1 => passenger himself
        Z['family_size'] = Z['sibsp'].fillna(0) + Z['parch'].fillna(0) + 1
        Z['is_alone'] = (Z['family_size'] == 1).astype(int)
        Z['pclass_str'] = Z['pclass'].astype(str) # treat pclass as categorical string
        return Z
    
# Preprocessing definitions
num_features = ['age', 'sibsp', 'parch', 'fare', 'family_size']
cat_features = ['sex', 'embarked', 'pclass_str', 'age_group', 'fare_group']

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    # ColumnTransformer is a scikit-learn tool that lets you apply different preprocessing steps 
        # to different columns of your dataset - all in one unified pipeline
    # drop: any columns not listed in 'num' and 'cat' will be dropped
    # remainder='passthrough' => unlisted columns would be kept as is
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
], remainder="drop")

# Interaction features (optional, moderate degree to avoid blow-up)
# interaction_only=True => create only interaction terms, not powers eg x1*x2, x1*x3, not x1^2, x2^2
interaction = PolynomialFeatures(
    degree=2, interaction_only=True, include_bias=False
)

# Full pipeline: feature engineering -> preprocess -> interactions -> model
full_pipe = Pipeline([
    ("family", FamilyFeatures()),
    ("logfare", LogFare()),
    ("bins", AgeFareBins(n_fare_bins=4)),
    ("pre", preprocessor),
    ("interact", interaction),
    ("model", RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    ))
])

# Fit
full_pipe.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
y_pred = full_pipe.predict(X_test)
y_proba = full_pipe.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# Permutation importance: It measures how much suffling a feature increases error, using the fitted pipeline end to end
    # If randomly shuffling a feature drops model's performance significantly, that feature was important
from sklearn.inspection import permutation_importance
result = permutation_importance(
    full_pipe, X_test, y_test, n_repeats=10, random_state=42
)
# ColumnTransformer + PolynomialFeatures produce many derived columns, 
    # but we will map top-k importances for readability
importances = result.importances_mean # average importance per feature
indices = np.argsort(importances)[::-1]
    # np.argsort() -> returns indices that would for the array in asc order
    # [::-1] -> reverse array
top_k = 15
print(f"Top feature importances (permuted pipeline columns):")
for i in indices[:top_k]:
    print(f"Column {i}: {importances[i]:.4f}")

# Permutation importance reports importance a the transformed column level
    # To recover original feature names, we need to extract get_feature_names_out() as below

# Build a pipeline up to interactons to fetch feature names
prep_only = Pipeline([
    ("family", FamilyFeatures()),
    ("logfare", LogFare()),
    ("bins", AgeFareBins(n_fare_bins=4)),
    ("pre", preprocessor),
    ("interact", interaction)
])
prep_only.fit(X_train, y_train)

# Names
feature_names = prep_only.named_steps['pre'].get_feature_names_out()
# PolynomialFeatures expands names; we approximate by repeating names
# Sklearn doesn't auto-name interaction features; construct manually:
# A practical approach: run transform and use column indices with feature_names length
X_trans = prep_only.transform(X_test)
base_dim = len(feature_names)
print(f"Base features count before interactions: {base_dim}")
print("Note: interaction feature names are not readily available; focus on base features' importances or skip interactions when interpreting.")

# Error slicing and residual inspection
    # we will examine erros across cohorts: fare (proxy for income), age groups, and sex
# Cohort labels from engineered features (recompute using prep steps before model)
Z_test = FamilyFeatures().transform(X_test)
Z_test = LogFare().transform(Z_test)
Z_test = AgeFareBins(n_fare_bins=4).fit(X_train).transform(Z_test)

errors = (y_pred != y_test).astype(int)
res_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred,
    "proba": y_proba,
    "error": errors,
    "sex": Z_test["sex"].values,
    "pclass": Z_test["pclass"].values,
    "age_group": Z_test["age_group"].astype(str).values,
    "fare_group": Z_test["fare_group"].astype(str).values,
    "is_alone": Z_test["is_alone"].values
})

# Error rates by cohort
def cohort_error_rate(df, by):
    g = df.groupby(by)["error"].mean().sort_values(ascending=False)
    print(f"\nError rate by {by}:\n{g}\n")

cohort_error_rate(res_df, "sex")
cohort_error_rate(res_df, "pclass")
cohort_error_rate(res_df, "age_group")
cohort_error_rate(res_df, "fare_group")
cohort_error_rate(res_df, "is_alone")

# Plot error distributions (probabilities on wrong vs right)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.kdeplot(res_df.loc[res_df["error"]==1, "proba"], label="Wrong predictions", shade=True)
sns.kdeplot(res_df.loc[res_df["error"]==0, "proba"], label="Right predictions", shade=True)
plt.title("Predicted probability distribution by correctness")
plt.xlabel("Predicted probability of survival")
plt.legend()
plt.show()

# Calibration-like view: predicted proba bins vs accuracy
res_df["proba_bin"] = pd.cut(res_df["proba"], bins=np.linspace(0,1,11), include_lowest=True)
calib = res_df.groupby("proba_bin")["error"].mean()
plt.figure(figsize=(7,4))
calib.plot(kind="bar")
plt.title("Error rate by predicted probability bin")
plt.ylabel("Error rate")
plt.xticks(rotation=45)
plt.show()
