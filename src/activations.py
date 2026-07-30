import numpy as np


def leaky_relu(x, alpha=0.01):
    """
    Apply the Leaky ReLU activation function.

    Positive values remain unchanged.
    Negative values are multiplied by alpha.
    """
    return np.where(x > 0, x, alpha * x)
def leaky_relu_backward(x, alpha=0.01):
    """
    Gradient of Leaky ReLU.
    """
    gradient = np.ones_like(x)
    gradient[x < 0] = alpha
    return gradient