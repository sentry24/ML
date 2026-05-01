# ============================================================
# Practical 3 — Polynomial Regression
# Moksh Singh (230439) | BSc(Hons) CS, Section B | Sem VI
# Dataset: load_diabetes() — Regression
# ============================================================

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Load dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Degree-2 Polynomial Regression
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)
y_pred = poly_model.predict(X_test)

print("---- 3. Polynomial Regression (degree=2) ----")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f} | R2 Score: {r2_score(y_test, y_pred):.4f}")

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(poly_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
print(f"5-Fold CV MSE: {-cv_scores.mean():.4f}")
