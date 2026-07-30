import numpy as np


def softmax(x):
    """
    Compute Softmax probabilities.
    """

    shifted = x - np.max(x)

    exp_values = np.exp(shifted)

    probabilities = exp_values / np.sum(exp_values)

    return probabilities