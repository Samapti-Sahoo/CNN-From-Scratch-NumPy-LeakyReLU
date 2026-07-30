import numpy as np

from sklearn.datasets import fetch_openml
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

X = mnist.data
y = mnist.target

print("Dataset loaded successfully!")
print("Images shape:", X.shape)
print("Labels shape:", y.shape)

import matplotlib.pyplot as plt

# Take the first image
first_image = X[0]
first_label = y[0]

print("\nFirst image original shape:", first_image.shape)

# Convert 784 values back into a 28 x 28 image
image_2d = first_image.reshape(28, 28)

print("After reshape:", image_2d.shape)
print("Actual label:", first_label)

# Display the image
plt.imshow(image_2d, cmap="gray")
plt.title(f"MNIST Digit - Label: {first_label}")
plt.axis("off")
print("\nMinimum pixel value:", image_2d.min())
print("Maximum pixel value:", image_2d.max())

# Normalize pixel values from 0-255 to 0-1
X = X.astype("float32") / 255.0

print("After normalization:")
print("Minimum value:", X.min())
print("Maximum value:", X.max())

# Reshape all images for CNN input
X = X.reshape(-1, 28, 28)

print("\nCNN input shape:", X.shape)
print("Shape of one image:", X[0].shape)
plt.show()
#temorarily added to test the leaky relu function
from src.activations import leaky_relu

test_values = np.array([-10, -2, 0, 2, 10], dtype=np.float32)

activated_values = leaky_relu(test_values)

print("\nLeaky ReLU Test")
print("Input :", test_values)
print("Output:", activated_values)

#convolution test

from src.convolution import conv2d

# Simple 5x5 test image
test_image = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
], dtype=np.float32)
# Four 3x3 kernels
test_kernels = np.array([
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ],
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ],
    [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ],
    [
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ]
], dtype=np.float32)

feature_maps = conv2d(test_image, test_kernels)


print("\n--- Convolution Test ---")
print("Input shape:", test_image.shape)
print("Kernel shape:", test_kernels.shape)
print("Output shape:", feature_maps.shape)

print("\nFeature Map:")
print(feature_maps[0]) # Displaying the first feature map



mnist_feature_maps = conv2d(X[0], test_kernels)

# Apply our Leaky ReLU
activated_feature_maps = leaky_relu(mnist_feature_maps)

print("\n--- MNIST Convolution ---")
print("Original image shape:", X[0].shape)
print("After convolution:", mnist_feature_maps.shape)
print("After Leaky ReLU:", activated_feature_maps.shape)

print("Before activation range:",
      mnist_feature_maps.min(),
      "to",
      mnist_feature_maps.max())

print("After activation range:",
      activated_feature_maps.min(),
      "to",
      activated_feature_maps.max())
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

axes[0].imshow(X[0], cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(mnist_feature_maps[0], cmap="gray")
axes[1].set_title("After Convolution")
axes[1].axis("off")

axes[2].imshow(activated_feature_maps[0], cmap="gray")
axes[2].set_title("After Leaky ReLU")
axes[2].axis("off")

plt.tight_layout()
plt.show()

from src.pooling import max_pool2d

# Simple 4x4 feature map for testing
test_feature_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [7, 2, 8, 3],
    [1, 4, 2, 9]
], dtype=np.float32)

pooled_output = max_pool2d(
    test_feature_map,
    pool_size=2,
    stride=2
)

print("\n--- Max Pooling Test ---")
print("Input Feature Map:")
print(test_feature_map)

print("\nInput shape:", test_feature_map.shape)

print("\nPooled Output:")
print(pooled_output)

print("Output shape:", pooled_output.shape)

# Apply Max Pooling to the activated MNIST feature map
pooled_feature_maps = max_pool2d(
    activated_feature_maps,
    pool_size=2,
    stride=2
)

print("\n--- MNIST Max Pooling ---")
print("Before pooling:", activated_feature_maps.shape)
print("After pooling:", pooled_feature_maps.shape)
print("Values before pooling:", activated_feature_maps.size)
print("Values after pooling:", pooled_feature_maps.size)

from src.flatten import flatten

flattened_output = flatten(
    pooled_feature_maps
)

print("\n--- Flatten Layer ---")
print("Before Flatten:", pooled_feature_maps.shape)
print("After Flatten:", flattened_output.shape)
print("Total Values:", flattened_output.size)

from src.dense import Dense

dense = Dense(
    input_size=676,
    output_size=10
)
true_label = int(y[0])
# Save weights before training
old_weights = dense.weights.copy()


# ---------------------------------
# Backward Pass
# ---------------------------------
epochs = 5
for epoch in range(epochs):

    dense_output = dense.forward(
        flattened_output
    )

    print("\n--- Dense Layer ---")
    print("Input shape :", flattened_output.shape)
    print("Weights shape :", dense.weights.shape)
    print("Bias shape :", dense.bias.shape)
    print("Output shape :", dense_output.shape)

    print("\nDense Output:")
    print(dense_output)

    probabilities = softmax(dense_output)

    print("\n--- Softmax Layer ---")
    print("Output shape:", probabilities.shape)

    print("\nProbabilities:")
    print(probabilities)

    print("\nSum of probabilities:",
          np.sum(probabilities))

    print("Predicted Digit:",
          np.argmax(probabilities))

    loss = cross_entropy_loss(
        probabilities,
        true_label
    )

    print("\n--- Cross Entropy Loss ---")
    print("Actual Label :", true_label)
    print("Predicted Label :", np.argmax(probabilities))
    print("Probability of Actual Label :",
          probabilities[true_label])
    print("Loss :", loss)

    learning_rate = 0.01

    gradient = cross_entropy_gradient(
        probabilities,
        true_label
    )

    dense.backward(
        gradient,
        learning_rate
    )

    weight_change = np.sum(
        np.abs(dense.weights - old_weights)
    )

    print("\nTotal Weight Change :", weight_change)

    print("\n--- Backward Pass ---")
    print("Learning Rate :", learning_rate)
    print("Gradient Shape :", gradient.shape)

    print("\nGradient:")
    print(gradient)

    print("\nWeights Updated Successfully!")