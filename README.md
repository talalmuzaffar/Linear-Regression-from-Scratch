# Linear Regression Model

This is a simple implementation of a linear regression model in Python using NumPy. The model can be trained on a given dataset and used to make predictions on new data.

## Features

- **Simple Implementation**: The code provides a straightforward implementation of linear regression using gradient descent optimization.
- **Customizable Parameters**: You can specify the learning rate and the number of iterations for training the model.
- **Training and Testing**: It includes methods for training the model with labeled data and making predictions on new data.
- **Get Parameters**: You can retrieve the trained parameters (thetas) of the model.

## Usage

To use the linear regression model, follow these steps:

1. Import the `LinearRegression` class from the provided code.
2. Prepare your dataset with features (`X_train`) and labels (`y_train`).
3. Create an instance of the `LinearRegression` class, optionally specifying the learning rate and the number of iterations.
4. Train the model by calling the `train()` method with your training data.
5. Optionally, test the trained model using the `test()` method with new data.
6. Retrieve the trained parameters using the `get_parameters()` method.

## Requirements

- Python 3.x
- NumPy
