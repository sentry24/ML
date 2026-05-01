# ============================================================
# Practical 11 — Hierarchical (Agglomerative) Clustering
# Moksh Singh (230439) | BSc(Hons) CS, Section B | Sem VI
# Dataset: load_breast_cancer() — Unsupervised (labels ignored)
# ============================================================

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Load dataset
cancer = load_breast_cancer()
X = cancer.data   # Labels NOT used — unsupervised learning

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Dendrogram (optional visual — uses first 50 samples for clarity) ──
plt.figure(figsize=(12, 5))
dendrogram = sch.dendrogram(sch.linkage(X_scaled[:50], method='ward'))
plt.title("Dendrogram (first 50 samples)")
plt.xlabel("Sample Index")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.savefig("dendrogram.png", dpi=150)
plt.show()
print("Dendrogram saved as dendrogram.png")

# ── Agglomerative Clustering ────────────────────────────────────
model = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = model.fit_predict(X_scaled)

print("\n---- 11. Hierarchical Clustering ----")
print(f"Number of Clusters  : 2")
print(f"Silhouette Score    : {silhouette_score(X_scaled, labels):.4f}")
print(f"Cluster sizes       : {dict(zip(*np.unique(labels, return_counts=True)))}")
