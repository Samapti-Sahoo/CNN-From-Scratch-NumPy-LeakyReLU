import numpy as np


def max_pool2d(feature_maps, pool_size=2, stride=2):

    num_filters, input_height, input_width = feature_maps.shape

    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1

    pooled = np.zeros(
        (
            num_filters,
            output_height,
            output_width
        ),
        dtype=np.float32
    )

    for f in range(num_filters):

        for i in range(output_height):

            for j in range(output_width):

                start_row = i * stride
                start_col = j * stride

                region = feature_maps[
                    f,
                    start_row:start_row + pool_size,
                    start_col:start_col + pool_size
                ]

                pooled[f, i, j] = np.max(region)

    return pooled


def max_pool2d_backward(
    d_output,
    feature_maps,
    pool_size=2,
    stride=2
):

    num_filters, input_height, input_width = feature_maps.shape

    d_input = np.zeros_like(feature_maps)

    output_height = d_output.shape[1]
    output_width = d_output.shape[2]

    for f in range(num_filters):

        for i in range(output_height):

            for j in range(output_width):

                start_row = i * stride
                start_col = j * stride

                region = feature_maps[
                    f,
                    start_row:start_row + pool_size,
                    start_col:start_col + pool_size
                ]

                max_index = np.unravel_index(
                    np.argmax(region),
                    region.shape
                )

                d_input[
                    f,
                    start_row + max_index[0],
                    start_col + max_index[1]
                ] = d_output[f, i, j]

    return d_input