import numpy as np


def conv_backward(
    image,
    kernels,
    d_feature_maps,
    learning_rate
):
    """
    Backward pass for convolution layer.
    Updates convolution kernels.
    """

    num_filters, kernel_height, kernel_width = kernels.shape

    kernel_gradients = np.zeros_like(kernels)

    for f in range(num_filters):

        for i in range(kernel_height):

            for j in range(kernel_width):

                region = image[
                    i:i + d_feature_maps.shape[1],
                    j:j + d_feature_maps.shape[2]
                ]

                kernel_gradients[f, i, j] = np.sum(
                    region * d_feature_maps[f]
                )

    kernels -= learning_rate * kernel_gradients

    return kernels