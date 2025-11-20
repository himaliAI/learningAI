# End-to-end multiclass classification with preprocessing and CUDA
# Author: Anup + Copilot

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# ----------------------------
# Reproducibility
# ----------------------------
rng = np.random.default_rng(42)
torch.manual_seed(123)

n = 1000

# ----------------------------
# 1) Synthetic data creation
# ----------------------------
age = rng.integers(18, 80, size=n)
bp = rng.normal(120, 15, size=n)
chol = rng.normal(200, 30, size=n)
gender = rng.choice(['M', 'F'], size=n, p=[0.5, 0.5])
smoker = rng.choice(['Y', 'N'], size=n, p=[0.3, 0.7])

# Missing values
bp[rng.choice(n, size=50, replace=False)] = np.nan
chol[rng.choice(n, size=50, replace=False)] = np.nan
gender[rng.choice(n, size=30, replace=False)] = None
smoker[rng.choice(n, size=30, replace=False)] = None

risk_score = (
    0.02 * age +
    0.03 * np.nan_to_num(bp, nan=120) +
    0.02 * np.nan_to_num(chol, nan=200) +
    0.5 * (smoker == 'Y').astype(float) +
    0.1 * (gender == 'M').astype(float) +
    rng.normal(0, 1, size=n)
)

q1, q2 = np.quantile(risk_score, [0.33, 0.66])
y = np.where(risk_score < q1, 0, np.where(risk_score < q2, 1, 2))

df = pd.DataFrame({
    'age': age,
    'bp': bp,
    'chol': chol,
    'gender': gender,
    'smoker': smoker,
    'y': y
})

num_features = ['age', 'bp', 'chol']
cat_features = ['gender', 'smoker']
target = 'y'

# ----------------------------
# 2) Preprocessing pipelines
# ----------------------------
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # consider sparse_output=False to avoid sparse
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_features),
    ('cat', categorical_pipeline, cat_features)
])

# ----------------------------
# 3) Train-test split
# ----------------------------
X = df[num_features + cat_features]
y = df[target].values

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

# Fit on train, transform both
X_train = preprocessor.fit_transform(X_train_df)
X_test = preprocessor.transform(X_test_df)

input_dim = X_train.shape[1]
n_classes = len(np.unique(y))
print(f"Features: {input_dim}; Classes: {n_classes}")

# ----------------------------
# 4) Tensors and CUDA
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Handle sparse matrices from OneHotEncoder
X_train_arr = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_test_arr = X_test.toarray() if hasattr(X_test, "toarray") else X_test

X_train_t = torch.tensor(X_train_arr, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test_arr, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

# ----------------------------
# 5) Model
# ----------------------------
model = nn.Sequential(
    nn.Linear(input_dim, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, n_classes)  # logits for 3 classes
)
model = model.to(device)

# ----------------------------
# 6) Loss and optimizer
# ----------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# ----------------------------
# 7) Training loop
# ----------------------------
def train_epoch(model, X_t, y_t, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    logits = model(X_t)
    loss = loss_fn(logits, y_t)
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_logits(model, X_t):
    model.eval()
    with torch.no_grad():
        return model(X_t)

n_epochs = 100
for epoch in range(1, n_epochs + 1):
    train_loss = train_epoch(model, X_train_t, y_train_t, optimizer, loss_fn)
    val_logits = eval_logits(model, X_test_t)
    val_loss = loss_fn(val_logits, y_test_t).item()
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ----------------------------
# 8) Evaluation: accuracy, precision, recall, F1, AUC
# ----------------------------
val_logits_np = val_logits.detach().cpu().numpy()
val_probs_np = torch.softmax(torch.from_numpy(val_logits_np), dim=1).numpy()
val_preds_np = np.argmax(val_probs_np, axis=1)
y_test_np = y_test

acc = accuracy_score(y_test_np, val_preds_np)

prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test_np, val_preds_np, average='macro', zero_division=0
)
prec_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_test_np, val_preds_np, average='weighted', zero_division=0
)

y_test_ovr = np.eye(n_classes)[y_test_np]
auc_ovr = roc_auc_score(y_test_ovr, val_probs_np, average='macro', multi_class='ovr')

print("\nEvaluation Metrics:")
print(f"- Accuracy: {acc:.4f}")
print(f"- Precision (macro): {prec_macro:.4f}")
print(f"- Recall    (macro): {recall_macro:.4f}")
print(f"- F1        (macro): {f1_macro:.4f}")
print(f"- Precision (weighted): {prec_weighted:.4f}")
print(f"- Recall    (weighted): {recall_weighted:.4f}")
print(f"- F1        (weighted): {f1_weighted:.4f}")
print(f"- AUC (macro, OvR): {auc_ovr:.4f}")
