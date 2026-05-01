from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
evaluate_regression(poly_model, X_reg_train, y_reg_train,
                    X_reg_test, y_reg_test, "3. Polynomial Regression")
