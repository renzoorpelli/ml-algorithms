from typing import Callable, List, Tuple, Union
import math
from typing import Tuple, Union

import numpy as np

"""
Implements the gradients for linear regression.

Functions:
  1) compute_gradient
     - Purpose: compute gradients using a simple loop (equations 4 & 5).
     - Outputs: (dj_dw, dj_db) both floats.

  2) compute_gradient_optimized
     - Purpose: compute the same gradients using vectorized NumPy (faster).
     - Outputs: (dj_dw, dj_db) both floats.
"""


def compute_gradient(
    x: np.ndarray, y: np.ndarray, w: Union[int, float], b: Union[int, float]
) -> Tuple[float, float]:
    """
    Compute the gradient for linear regression (scalar implementation).

    Args:
      x: ndarray (m, ) input data
      y: ndarray (m, ) target values
      w, b: model parameters

    Returns:
      (dj_dw, dj_db): gradients w.r.t. w and b
    """
    m = x.shape[0]
    dj_dw = 0.0
    dj_db = 0.0

    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]
        dj_dw += error * x[i]
        dj_db += error

    dj_dw /= m
    dj_db /= m
    return float(dj_dw), float(dj_db)


def compute_gradient_optimized(
    x: np.ndarray, y: np.ndarray, w: Union[int, float], b: Union[int, float]
) -> Tuple[float, float]:
    """
    Vectorized gradient computation using NumPy.
    """
    m = x.shape[0]
    predictions = w * x + b
    errors = predictions - y
    dj_dw = float(np.dot(errors, x) / m)
    dj_db = float(np.sum(errors) / m)
    return dj_dw, dj_db


"""
We will need three functions:

  1) compute_gradient
     - Purpose: compute gradients for linear regression (equations 4 & 5).
     - Output: tuple (dj_dw, dj_db) where both are floats representing
       the gradients of the MSE cost w.r.t. w and b.

  2) compute_cost
     - Purpose: compute the MSE cost for linear regression (equation 2).
     - Output: total_cost (float): the cost value for parameters w,b.

  3) gradient_descent
     - Purpose: perform iterative gradient descent using compute_gradient and compute_cost.
     - Outputs:
         w (float): learned weight after gradient descent
         b (float): learned bias after gradient descent
         J_history (List[float]): cost at each saved iteration
         p_history (List[List[float]]): saved parameter pairs [w, b] at each saved iteration
"""


def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    w_in: Union[int, float],
    b_in: Union[int, float],
    alpha: float,
    num_iters: int,
    cost_function: Callable[[np.ndarray, np.ndarray, Union[int, float], Union[int, float]], float],
    gradient_function: Callable[
        [np.ndarray, np.ndarray, Union[int, float],
            Union[int, float]], Tuple[float, float]
    ],
) -> Tuple[float, float, List[float], List[List[float]]]:
    """
    Performs gradient descent to fit w,b. Updates w,b by taking num_iters gradient steps.

    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w_in, b_in (scalar): initial model parameters
      alpha (float): Learning rate
      num_iters (int): number of iterations to run gradient descent
      cost_function: function(x, y, w, b) -> float
      gradient_function: function(x, y, w, b) -> (dj_dw, dj_db)

    Returns:
      w (float): Updated weight after running gradient descent
      b (float): Updated bias after running gradient descent
      J_history (List[float]): History of cost values (saved up to cap)
      p_history (List[List[float]]): History of parameters [w, b] (saved up to cap)
    """

    J_history: List[float] = []
    p_history: List[List[float]] = []

    w = float(w_in)
    b = float(b_in)

    print_every = max(1, math.ceil(num_iters / 10))

    save_cap = 100000

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        if i < save_cap:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        if i % print_every == 0:
            current_cost = J_history[-1] if J_history else float("nan")
            print(
                f"Iteration {i:4}: Cost {current_cost:0.2e} ",
                f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                f"w: {w: 0.3e}, b:{b: 0.5e}",
            )

    return w, b, J_history, p_history
