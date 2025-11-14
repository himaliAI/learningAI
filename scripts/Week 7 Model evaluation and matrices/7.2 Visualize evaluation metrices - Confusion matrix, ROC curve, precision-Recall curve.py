# Same as in 7.1 up to line 102
# Step 1: import libraries and dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score, precision_score, recall_score, roc_auc_score
)
import seaborn as sns

# load titanic dataset from seaborn
df = sns.load_dataset('titanic')

# define features (mix of numerical + categorical) and target
X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
y = df['survived']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build preprocessing pipelines
# numerical and categorical columns
num_features = ['age', 'sibsp', 'parch', 'fare']
cat_features = ['pclass', 'sex', 'embarked']

# pipelines
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

cat_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# train several models with the same pipeline
# Logistic regression, Random Forest, Gradient Boosting, SVM, k-NN
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'k-NN': KNeighborsClassifier(n_neighbors=7)
}

# fit and evaluate each model
results = {}

for name, model in models.items():
    # Full pipeline (preprocess + model)
    clf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # fit and predict
    clf_pipeline.fit(X_train, y_train)

    y_pred = clf_pipeline.predict(X_test)
    y_proba = clf_pipeline.predict_proba(X_test)[:,1] if hasattr(clf_pipeline, 'predict_proba') else None

    # metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'ROC-AUC': roc
    }

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    if roc is not None:
        print(f"ROC-AUC: {roc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# week 7.2 starts from here
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# Confusion matrix visualization
y_pred = clf_pipeline.predict(X_test)
conf_matrx = confusion_matrix(y_test, y_pred)

# plot heatmap
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrx, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived","Survived"], yticklabels=["Not Survived","Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# ROC Curve visualization
# predicted probabilities
y_proba = clf_pipeline.predict_proba(X_test)[:,1]

# plot ROC curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve - Random Forest")
plt.show()

# Precision-Recall Curve
from sklearn.metrics import PrecisionRecallDisplay

y_proba = clf_pipeline.predict_proba(X_test)[:,1]
PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.title("Precision-Recall Curve - Random Forest")
plt.show()
