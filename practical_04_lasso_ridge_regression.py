from sklearn.linear_model import Lasso, Ridge

lasso_model = Lasso(alpha=0.1)
evaluate_regression(lasso_model, X_reg_train_scaled, y_reg_train,
                    X_reg_test_scaled, y_reg_test, "4a. Lasso Regression")

ridge_model = Ridge(alpha=1.0)
evaluate_regression(ridge_model, X_reg_train_scaled, y_reg_train,
                    X_reg_test_scaled, y_reg_test, "4b. Ridge Regression")
