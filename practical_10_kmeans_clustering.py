from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

print("── 10. K-Means Clustering ──")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_clf_train_scaled)
print(f"Silhouette Score: {silhouette_score(X_clf_train_scaled, kmeans_labels):.4f}\n")

print("── 11. Hierarchical Clustering ──")
hierarchical = AgglomerativeClustering(n_clusters=2)
hier_labels = hierarchical.fit_predict(X_clf_train_scaled)
print(f"Silhouette Score: {silhouette_score(X_clf_train_scaled, hier_labels):.4f}")
