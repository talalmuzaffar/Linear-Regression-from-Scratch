import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000): # Defualt Learning Rate and Number of Ietrations
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.thetas = None
    
    def train(self, X, y): 

        if len(X.shape) != 2 or len(y.shape) != 1: # Dimention Check
            print("Invalid Dimensions")
            return
        
        if X.shape[0] != y.shape[0]: # Same number of rows in features and labels array
            print("Rows Mismatch")
            return        
        
        X = np.hstack((X, np.ones((X.shape[0], 1)))) # Adding an extra column for the sake of dot product as explained in the class
        y = y[:, np.newaxis] # taking transpose as it was taking it as a row
        self.thetas = np.random.rand(X.shape[1], 1) # Randomally initializing the thetas/parameters
        
        prev_loss = 10000000000000000000000 ### Could be any big number
        
        for iteration in range(self.num_iterations): # Loop for the Summation
            y_pred = np.dot(X, self.thetas) # Calculating y pred (01x+00 but no theta 0 as we have added a column of one for that sake)
            error = y - y_pred # y_actual-y-pred term
            loss = np.sum(error ** 2) / X.shape[0] #(y_actual - y_pred)^2/total_number_of_rows or n
            
            if iteration > 0: #Calculating Relative Loss so so that we can stop training if the relative loss is very small
                relative_loss = (prev_loss - loss) / prev_loss
                print("Iteration: " ,iteration, "Loss: ", loss, "Relative Loss: " ,relative_loss)

                if relative_loss < 0.0001:
                    print("Training stopped as relative loss is below 0.0001")
                    break
            else:
                print("Iteration: ",iteration, "Loss: ", loss)
            
            prev_loss = loss # Updating the previous loss
            
            partial_derivatives = -(2 / X.shape[0]) * np.dot(X.T, error) # Taking partial derivatives, we are multiplying the error(yact-ypred) with every features
            self.thetas -= self.learning_rate * partial_derivatives # New thetas 
        
        print("Final Thetas:", self.thetas)
    
    def test(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1)))) 
        y_pred = np.dot(X, self.thetas)
        return y_pred

    def get_parameters(self):
        if self.thetas is not None:
            return self.thetas
        else:
            print("Model not trained yet.")
            return None


# Kindly uncomment this code to test the code

# X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
# y_train = np.array([2, 3, 5, 6])

# regressor = LinearRegression(0.001)
# regressor.train(X_train, y_train)
# X_test = np.array([[5, 6], [6, 7]])
# predictions = regressor.test(X_test)
# print("Predictions: ",predictions)
# print("Trained Parameters: ",regressor.get_parameters())
