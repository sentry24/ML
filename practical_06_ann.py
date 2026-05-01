from sklearn.neural_network import MLPClassifier

ann_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
evaluate_classification(ann_model, X_clf_train_scaled, y_clf_train,
                        X_clf_test_scaled, y_clf_test, "6. Artificial Neural Network")
