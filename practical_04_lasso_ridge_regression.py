# ============================================================
# Practical 4 — Lasso & Ridge Regression
# Moksh Singh (230439) | BSc(Hons) CS, Section B | Sem VI
# Dataset: load_diabetes() — Regression
# ============================================================

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

# Load dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (best practice for regularised models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ── 4a. Lasso Regression ───────────────────────────────────────
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

print("---- 4a. Lasso Regression ----")
print(f"MSE: {mean_squared_error(y_test, y_pred_lasso):.4f} | R2 Score: {r2_score(y_test, y_pred_lasso):.4f}")
cv_lasso = cross_val_score(lasso, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
print(f"5-Fold CV MSE: {-cv_lasso.mean():.4f}")

# ── 4b. Ridge Regression ───────────────────────────────────────
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

print("\n---- 4b. Ridge Regression ----")
print(f"MSE: {mean_squared_error(y_test, y_pred_ridge):.4f} | R2 Score: {r2_score(y_test, y_pred_ridge):.4f}")
cv_ridge = cross_val_score(ridge, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
print(f"5-Fold CV MSE: {-cv_ridge.mean():.4f}")
