# ============================================================
# Practical 2 — Simple & Multiple Linear Regression
# Dataset: load_diabetes() — Regression
# ============================================================

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ── 2a. Simple Linear Regression (single feature) ──────────────
X_simple_train = X_train[:, 0].reshape(-1, 1)
X_simple_test  = X_test[:, 0].reshape(-1, 1)

slr = LinearRegression()
slr.fit(X_simple_train, y_train)
y_pred_slr = slr.predict(X_simple_test)

print("---- 2a. Simple Linear Regression ----")
print(f"MSE: {mean_squared_error(y_test, y_pred_slr):.4f} | R2 Score: {r2_score(y_test, y_pred_slr):.4f}")
cv_slr = cross_val_score(slr, X_simple_train, y_train, cv=kf, scoring='neg_mean_squared_error')
print(f"5-Fold CV MSE: {-cv_slr.mean():.4f}")

# ── 2b. Multiple Linear Regression (all features) ──────────────
mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_test)

print("\n---- 2b. Multiple Linear Regression ----")
print(f"MSE: {mean_squared_error(y_test, y_pred_mlr):.4f} | R2 Score: {r2_score(y_test, y_pred_mlr):.4f}")
cv_mlr = cross_val_score(mlr, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
print(f"5-Fold CV MSE: {-cv_mlr.mean():.4f}")
