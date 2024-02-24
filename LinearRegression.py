import pandas as pd
import numpy as np
import time

# At the moment this is a VERY naive approach to gradient descent.
# I would like to extend this class to support batch and stochastic gradient descent 
# and also the most efficient solution: the normal equations.
class LinearRegression:
    def __init__(self):
        # Placeholder for model coefficients
        self.coef = None
        self.training_rate = 0.000001

    def fit(self, X, y):
        start_time = time.time()
        
        # Raise an exception if X is a Pandas Series
        if isinstance(X, pd.Series):
            raise ValueError("X must be a Pandas DataFrame, not a Series.")
        # Convert X and y to NumPy arrays if they are Pandas DataFrames/Series
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        # Initialize the coefficients vector to zeros (+1 length for the bias term)
        self.coef = np.zeros(X.shape[1] + 1)
        # Perform Naive Gradient Descent
        for _ in range(1000):  # Assuming a fixed number of iterations for simplicity
            for i in range(len(X)):
                # Dynamically add the bias term for each prediction
                prediction = self.predict(X[i], add_bias_term=True)
                # Update each coefficient (including bias)
                for j in range(len(self.coef)):
                    if j == 0:  # Update for bias term
                        self.coef[j] = self.coef[j] + self.training_rate * (y[i] - prediction)
                    else:  # Update for each feature
                        self.coef[j] = self.coef[j] + self.training_rate * ((y[i] - prediction) * X[i][j-1])
        
        elapsed_time = time.time() - start_time
        print("Training took {:.2f} seconds".format(elapsed_time))

    def predict(self, feature_vector, add_bias_term=True):
        # Check if the input is a scalar by trying to iterate over it
        # If it's a scalar, it will raise TypeError, then convert it to a 1D array        
        if add_bias_term:
            # Dynamically add the bias term for prediction
            # If feature_vector was a scalar, now it becomes [feature_vector]
            # Then we insert the bias term, resulting in [1, feature_vector]
            feature_vector = np.insert(feature_vector, 0, 1)
        
        return np.dot(feature_vector, self.coef)