from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
evaluate_classification(nb_model, X_clf_train, y_clf_train,
                        X_clf_test, y_clf_test, "1. Naive Bayes Classifier")
