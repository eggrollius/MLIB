
import sys
sys.path.append('../')
import LinearRegression

import os # for high level paths

import pandas as pd
import numpy as np


# For generating and manipulating test data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Example of using this class with NumPy arrays
if __name__ == "__main__":
  # Read the data
  path = os.path.join('datasets', 'Walmart_sales.csv')
  data = pd.read_csv(path)

  # get some preliminary info on the data
  print(data.describe())

  # Split into features and  the dependant variable
  X = data[['Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag', 'Temperature']]
  # print(X.head())
  y = data['Weekly_Sales']  # Only the 'Weekly_Sales' column

  # Split into traingin and validation sets
  X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

  # Fit the model with default parameters
  model = LinearRegression.LinearRegression()
  model.fit(X_train, y_train)

  print(model.coef)
  
  # Manually predict for each example in X_validation
  y_pred = np.array([model.predict(X_validation.iloc[[i]].to_numpy(), add_bias_term=True) for i in range(len(X_validation))])


  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

  # Calculate performance metrics
  mse = mean_squared_error(y_validation, y_pred)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(y_validation, y_pred)
  r2 = r2_score(y_validation, y_pred)

  print(f"Range of y: {y.min()} to {y.max()}")
  print(f"Standard deviation of y: {y.std()}")

  print(f"Mean Squared Error (MSE): {mse}")
  print(f"Root Mean Squared Error (RMSE): {rmse}")
  print(f"Mean Absolute Error (MAE): {mae}")
  print(f"R-squared (R^2): {r2}")

