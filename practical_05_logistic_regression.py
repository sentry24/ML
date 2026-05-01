# ============================================================
# Practical 5 — Logistic Regression
# Dataset: load_breast_cancer() — Classification
# ============================================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (helps convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Metrics
acc         = accuracy_score(y_test, y_pred)
error       = 1 - acc
recall      = recall_score(y_test, y_pred)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1          = f1_score(y_test, y_pred)
y_prob      = model.predict_proba(X_test_scaled)[:, 1]
auc         = roc_auc_score(y_test, y_prob)

print("---- 5. Logistic Regression ----")
print(f"Accuracy: {acc:.4f} | Error: {error:.4f}")
print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
print(f"Recall: {recall:.4f} | Specificity: {specificity:.4f} | F1-Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy')
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f}")
