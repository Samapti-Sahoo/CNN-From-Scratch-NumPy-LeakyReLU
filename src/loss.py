import numpy as np


def cross_entropy_loss(probabilities, label):
    """
    Cross Entropy Loss
    """

    epsilon = 1e-10

    return -np.log(
        probabilities[label] + epsilon
    )


def cross_entropy_gradient(
    probabilities,
    label
):
    """
    Gradient of Cross Entropy Loss
    """

    gradient = probabilities.copy()

    gradient[label] -= 1

    return gradient