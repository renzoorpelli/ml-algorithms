from typing import Tuple

import numpy as np
from sigmoid import sigmoid


def compute_cost_linear_reg(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 1
) -> float:
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost
    """
    # 1st term
    m = X.shape[0]
    n = len(w)
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # (n,) (n,) scalar, see np.dot()
        cost = cost + (f_wb_i - y[i]) ** 2  # scalar
    cost = cost / (2 * m)  # scalar

    # 2st term
    reg_cost = 0
    for j in range(n):
        reg_cost += w[j] ** 2  # scalar
    reg_cost = (lambda_ / (2 * m)) * reg_cost  # scalar

    return cost + reg_cost  # total_cost = scalar


def compute_cost_logistic_reg(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 1
) -> float:
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost
    """

    # 1st term
    m, n = X.shape
    cost = 0.0
    for i in range(m):
        # (n,) (n,) scalar, see np.dot()
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)  # scalar
        cost = -y[i] * np.log(f_wb_i) - (1 - y[i]) * \
            np.log(1 - f_wb_i)  # scalar

    cost = cost / m  # scalar

    # 2st term
    reg_cost = 0
    for j in range(n):
        reg_cost += w[j] ** 2  # scalar
    reg_cost = (lambda_ / (2 * m)) * reg_cost  # scalar

    return cost + reg_cost  # total_cost = scalar


def compute_gradient_linear_reg(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 1
) -> Tuple[float, np.ndarray]:
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.0  # int

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]

    return dj_db, dj_dw


def compute_gradient_logistic_reg(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 1
) -> Tuple[float, np.ndarray]:
    """
    Computes the gradient for linear regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape  # (n,)
    dj_dw = np.zeros((n,))
    dj_db = 0.0  # float, scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  # scalar
            dj_db = dj_db + err_i

    dj_dw = dj_dw / m  # (n,)
    dj_db = dj_db / m  # scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]

    return dj_db, dj_dw


def predict_linear(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    Predict using linear regression

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      w (ndarray (n,)): model parameters
      b (scalar): model parameter

    Returns:
      p (ndarray (m,)): predictions
    """
    m = X.shape[0]
    p = np.zeros(m)

    for i in range(m):
        p[i] = np.dot(X[i], w) + b

    return p


def predict_logistic(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    Predict using logistic regression

    For each example, the prediction follows:
    1. Compute linear combination: z = w·x + b = Σ(w_j * x_j) + b
    2. Apply sigmoid function: f(x) = σ(z) = 1 / (1 + e^(-z))
    3. Apply decision boundary at 0.5: predict 1 if f(x) >= 0.5, else 0

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      w (ndarray (n,)): model parameters (weights)
      b (scalar): model parameter (bias)

    Returns:
      p (ndarray (m,)): predictions (0 or 1 after threshold)
    """
    m, n = X.shape
    p = np.zeros(m)

    for i in range(m):
        z_wb = 0.0
        for j in range(n):
            z_wb += w[j] * X[i, j]

        z_wb += b

        f_wb = sigmoid(z_wb)
    return p
