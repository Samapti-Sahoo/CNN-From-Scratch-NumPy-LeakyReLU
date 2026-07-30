import numpy as np

from sklearn.datasets import fetch_openml

from src.activations import (
    leaky_relu,
    leaky_relu_backward
)
from src.conv_backward import conv_backward
from src.convolution import conv2d
from src.pooling import (
    max_pool2d,
    max_pool2d_backward
)
from src.flatten import (
    flatten,
    flatten_backward
)
from src.dense import Dense
from src.softmax import softmax
from src.loss import (
    cross_entropy_loss,
    cross_entropy_gradient
)

print("Loading MNIST dataset...")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False,
    parser="auto"
)

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

X = X.reshape(-1, 28, 28)

print("Dataset Loaded!")
print("Images :", X.shape)
print("Labels :", y.shape)

# Use only first 1000 images
X = X[:1000]
y = y[:1000]

print("Training Images :", len(X))

# -------------------------------
# CNN Parameters
# -------------------------------

kernels = np.random.randn(4, 3, 3).astype(np.float32) * 0.1

dense = Dense(
    input_size=676,
    output_size=10
)

learning_rate = 0.01
epochs = 20

print("\nCNN Initialized Successfully!")
print("Kernels Shape :", kernels.shape)
print("Dense Weights Shape :", dense.weights.shape)

# -------------------------------
# Training Loop
# -------------------------------

for epoch in range(epochs):
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    epoch_loss = 0
    correct = 0

    print(f"\n========== Epoch {epoch + 1}/{epochs} ==========")

    for image, label in zip(X, y):

        # Convolution
        conv_output = conv2d(
    image,
    kernels
)

        feature_maps = leaky_relu(
    conv_output
)

        # Pooling
        pooled_maps = max_pool2d(
            feature_maps,
            pool_size=2,
            stride=2
        )

        # Flatten
        flattened, original_shape = flatten(
            pooled_maps
        )

        # Dense Layer
        dense_output = dense.forward(
            flattened
        )

        # Softmax
        probabilities = softmax(
            dense_output
        )
        prediction = np.argmax(probabilities)
        if prediction == label:
            correct += 1

        # Loss
        loss = cross_entropy_loss(
            probabilities,
            label
        )

        epoch_loss += loss

        # Backpropagation
        gradient = cross_entropy_gradient(
            probabilities,
            label
        )

        d_flatten = dense.backward(
             gradient,
    learning_rate
)

        d_pool = flatten_backward(
    d_flatten,
    original_shape
)

        d_relu = max_pool2d_backward(
    d_pool,
    feature_maps,
    pool_size=2,
    stride=2
)

        d_conv = d_relu * leaky_relu_backward(
    conv_output
)

        kernels = conv_backward(
    image,
    kernels,
    d_conv,
    learning_rate
)

        average_loss = epoch_loss / len(X)
    accuracy = (correct / len(X)) * 100

    print(f"Epoch {epoch + 1} Average Loss: {average_loss:.4f}")
    print(f"Epoch {epoch + 1} Accuracy: {accuracy:.2f}%")

    np.savez(
    "trained_model.npz",
    weights=dense.weights,
    bias=dense.bias,
    kernels=kernels
)

print("\nModel saved successfully!")