import numpy as np


def flatten(feature_maps):
    """
    Convert feature maps into a 1D vector.
    """

    original_shape = feature_maps.shape

    flattened = feature_maps.flatten()

    return flattened, original_shape


def flatten_backward(
    d_output,
    original_shape
):
    """
    Restore the flattened gradient
    back to the original shape.
    """

    return d_output.reshape(original_shape)