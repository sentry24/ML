# ============================================================
# Practical 10 — K-Means Clustering
# Dataset: load_breast_cancer() — Unsupervised (labels ignored)
# ============================================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
cancer = load_breast_cancer()
X = cancer.data   # Labels NOT used — unsupervised learning

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means — 2 clusters (matching the 2 cancer classes)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

print("---- 10. K-Means Clustering ----")
print(f"Number of Clusters  : 2")
print(f"Inertia (WCSS)      : {kmeans.inertia_:.4f}")
print(f"Silhouette Score    : {silhouette_score(X_scaled, labels):.4f}")
print(f"Cluster sizes       : {dict(zip(*np.unique(labels, return_counts=True)))}")
