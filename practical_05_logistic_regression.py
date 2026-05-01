from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
evaluate_classification(log_model, X_clf_train_scaled, y_clf_train,
                        X_clf_test_scaled, y_clf_test, "5. Logistic Regression")
