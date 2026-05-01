from sklearn.linear_model import LinearRegression

# Simple (1 feature)
X_reg_simple_train = X_reg_train[:, 0].reshape(-1, 1)
X_reg_simple_test  = X_reg_test[:, 0].reshape(-1, 1)
slr_model = LinearRegression()
evaluate_regression(slr_model, X_reg_simple_train, y_reg_train,
                    X_reg_simple_test, y_reg_test, "2a. Simple Linear Regression")

# Multiple (all features)
mlr_model = LinearRegression()
evaluate_regression(mlr_model, X_reg_train, y_reg_train,
                    X_reg_test, y_reg_test, "2b. Multiple Linear Regression")
