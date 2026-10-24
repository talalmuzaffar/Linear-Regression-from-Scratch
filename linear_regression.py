import numpy as np  # noqa: F401 (used by the test code below)

# The model lives in linear_regression_from_scratch/model.py, it is imported here so that
# `from linear_regression import LinearRegression` keeps working
from linear_regression_from_scratch.model import LinearRegression  # noqa: F401


# Kindly uncomment this code to test the code

# X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
# y_train = np.array([2, 3, 5, 6])

# regressor = LinearRegression(0.001)
# regressor.train(X_train, y_train)
# X_test = np.array([[5, 6], [6, 7]])
# predictions = regressor.test(X_test)
# print("Predictions: ",predictions)
# print("Trained Parameters: ",regressor.get_parameters())
