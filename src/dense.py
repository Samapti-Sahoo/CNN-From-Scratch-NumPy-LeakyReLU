import numpy as np


class Dense:

    def __init__(self, input_size, output_size):

        self.weights = (
            np.random.randn(input_size, output_size).astype(np.float32)
            * 0.01
        )

        self.bias = np.zeros(
            output_size,
            dtype=np.float32
        )

    def forward(self, inputs):

        self.inputs = inputs

        output = np.dot(
            inputs,
            self.weights
        ) + self.bias

        return output

    def backward(
        self,
        d_output,
        learning_rate
    ):

        d_weights = np.outer(
            self.inputs,
            d_output
        )

        d_bias = d_output

        d_inputs = np.dot(
            self.weights,
            d_output
        )

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        return d_inputs