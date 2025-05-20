# Gradient Descent for Linear Regression

## Overview

This project demonstrates a simple implementation of **Gradient Descent** to perform linear regression from scratch using Python. It fits a line to a set of data points by minimizing the error iteratively and visualizes the progression of the regression line during the optimization.

## Features

- Manual gradient descent implementation without relying on machine learning libraries.
- Visualization of the regression line at each iteration to show how the model converges.
- Adjustable learning rate and number of iterations.
- Clear plotting of original data points alongside the final fitted line.

## How It Works

1. Initialize slope (`m`) and intercept (`b`) to zero.
2. For a fixed number of iterations:
   - Predict `y` values based on current parameters.
   - Calculate the gradients (slopes of the error function).
   - Update `m` and `b` by stepping against the gradient.
   - Plot the regression line for each iteration to visualize improvement.
3. Display the final regression line with the original data points.

## Requirements

- Python 3.x
- numpy
- matplotlib

Install dependencies with:

```bash
pip install numpy matplotlib
Usage
Run the script or notebook containing this code:


import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([1.5, 3.7, 4.1, 6.0, 8.2])

# Hyperparameters
alpha = 0.01  # learning rate
iterations = 50
m = 0  # initial slope
b = 0  # initial intercept

plt.figure()

# Gradient Descent Loop
for _ in range(iterations):
    y_pred = m * x + b
    error = y - y_pred

    # Plot current line (in green)
    plt.plot(x, y_pred, color='green', alpha=0.3)

    # Gradient calculation
    m_grad = -(2 / len(x)) * sum(x * error)
    b_grad = -(2 / len(x)) * sum(error)

    # Update parameters
    m -= alpha * m_grad
    b -= alpha * b_grad

# Final plot with data points
plt.scatter(x, y, color='red')  # actual data
plt.title("Gradient Descent for Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


#Contribution
#Contributions and suggestions are welcome! Feel free to open issues or submit pull requests.
