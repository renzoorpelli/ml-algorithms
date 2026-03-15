import numpy as np 

def softmax(z: np.ndarray, axis =-1):
    """
        Computes a numerically stable softmax
    """
    z_max = np.max(z, axis=axis, keepdims=True)
    e_z   = np.exp(z - z_max)
    sum_e_z = np.sum(e_z, axis=axis, keepdims=True)

    return e_z / sum_e_z
