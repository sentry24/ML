from sklearn.svm import SVC

svm_model = SVC(probability=True, random_state=42)
evaluate_classification(svm_model, X_clf_train_scaled, y_clf_train,
                        X_clf_test_scaled, y_clf_test, "9. Support Vector Machine (SVM)")
