from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)
evaluate_classification(dt_model, X_clf_train, y_clf_train,
                        X_clf_test, y_clf_test, "8. Decision Tree Classifier")
