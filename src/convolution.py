import numpy as np


def conv2d(image, kernels):
    """
    Perform 2D convolution.

    Parameters
    ----------
    image : (H, W)
        Input image.

    kernels : (N, KH, KW)
        Convolution filters.

    Returns
    -------
    feature_maps : (N, OH, OW)
    """

    image_height, image_width = image.shape

    num_filters, kernel_height, kernel_width = kernels.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    feature_maps = np.zeros(
        (
            num_filters,
            output_height,
            output_width
        ),
        dtype=np.float32
    )

    for f in range(num_filters):

        kernel = kernels[f]

        for i in range(output_height):

            for j in range(output_width):

                region = image[
                    i:i + kernel_height,
                    j:j + kernel_width
                ]

                feature_maps[f, i, j] = np.sum(
                    region * kernel
                )

    return feature_maps