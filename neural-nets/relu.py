import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)
