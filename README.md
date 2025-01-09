# Linear Regression from Scratch 📊

A pure Python implementation of Linear Regression using only NumPy, built from scratch. This implementation provides a clear understanding of the underlying mathematics and optimization techniques used in linear regression.

## 🌟 Features

- **Pure Implementation**: Built using only NumPy, no machine learning libraries
- **Gradient Descent**: Custom implementation of optimization algorithm
- **Configurable Training**: Adjustable learning rate and iteration count
- **Model Parameters**: Access to trained parameters (thetas)
- **Prediction Support**: Methods for training and testing on new data

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- NumPy

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/talalmuzaffar/Linear-Regression-from-Scratch.git
   cd Linear-Regression-from-Scratch
   ```

2. Install NumPy if not already installed:
   ```bash
   pip install numpy
   ```

## 💡 Usage

1. Import the LinearRegression class:
   ```python
   from linear_regression import LinearRegression
   ```

2. Prepare your dataset:
   ```python
   # X_train: features
   # y_train: labels
   ```

3. Create and train the model:
   ```python
   model = LinearRegression(learning_rate=0.01, iterations=1000)
   model.train(X_train, y_train)
   ```

4. Make predictions:
   ```python
   predictions = model.test(X_test)
   ```

5. Access model parameters:
   ```python
   parameters = model.get_parameters()
   ```

## 📁 Project Structure

```
Linear-Regression-from-Scratch/
├── linear_regression.py  # Main implementation
└── README.md            # Project documentation
```

## 👥 Authors

- Talal Muzaffar - [GitHub](https://github.com/talalmuzaffar)

## 📞 Support

For issues and questions, please [open an issue](https://github.com/talalmuzaffar/Linear-Regression-from-Scratch/issues) on GitHub.
