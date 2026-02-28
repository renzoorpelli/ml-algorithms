import numpy as np

def sigmoid(z: np.ndarray):
    return 1/(1+np.exp(-z))

g = sigmoid

def sequential(
        X: np.ndarray, W1: np.ndarray, b1: np.ndarray,
        W2: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    a1 = dense(X,  W1, b1)
    a2 = dense(a1, W2, b2)
    return a2

def dense(
    a_in: np.ndarray, W: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units

      capital letter represents a Matrix, lowercase vectors or scalars.
    Returns
      a_out (ndarray (j,))  : j units|
    """

    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]  # column of a matrix
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out

def dense_vectorized(
    A_in: np.matrix, W: np.matrix, B: np.matrix
) -> np.matrix:
    """
    matmul implementation
    """
    Z = np.matmul(A_in, W) + B
    A_out = g(Z)
    return A_out
