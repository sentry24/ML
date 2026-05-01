from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
evaluate_classification(knn_model, X_clf_train_scaled, y_clf_train,
                        X_clf_test_scaled, y_clf_test, "7. K-Nearest Neighbors (K-NN)")
