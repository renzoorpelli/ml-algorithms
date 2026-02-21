from typing import Union

import numpy as np

"""
Implements the MSE cost for linear regression.

Functions:
  1) compute_cost
     - Purpose: compute MSE cost using a simple loop.
     - Output: total_cost (float)

  2) compute_cost_optimized
     - Purpose: compute MSE cost using vectorized NumPy operations.
     - Output: total_cost (float)
"""


def compute_cost(x: np.ndarray, y: np.ndarray, w: Union[int, float], b: Union[int, float]) -> float:
    """
    Computes the cost function for linear regression.

    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        # model prediction
        f_wb = w * x[i] + b
        # cost
        cost_sum += (f_wb - y[i]) ** 2

    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


def compute_cost_opt(x: np.ndarray, y: np.ndarray, w: Union[int, float], b: Union[int, float]) -> float:
    """
        Compute the cost function for linear regression using nmpy vectorization
    """
    m = x[0].shape
    model_predictions = w * x + b
    squared_errors = (model_predictions - y) ** 2
    return float(np.sum(squared_errors) / (2 * m))
