import numpy as np

from sklearn.datasets import fetch_openml

from src.activations import leaky_relu
from src.convolution import conv2d
from src.pooling import max_pool2d
from src.flatten import flatten
from src.softmax import softmax

print("Loading model...")

model = np.load("trained_model.npz")

kernels = model["kernels"]
weights = model["weights"]
bias = model["bias"]

print("Model Loaded Successfully!")

print("Loading MNIST...")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False,
    parser="auto"
)

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

X = X.reshape(-1, 28, 28)

# Same images used during training
X = X[:1000]
y = y[:1000]
image = X[0]
label = y[0]

feature_maps = conv2d(image, kernels)
feature_maps = leaky_relu(feature_maps)
pooled_maps = max_pool2d(feature_maps)
flattened, _ = flatten(pooled_maps)

output = np.dot(flattened, weights) + bias
probabilities = softmax(output)
prediction = np.argmax(probabilities)

print("Index 0")
print("Actual     :", label)
print("Prediction :", prediction)
print("Confidence :", probabilities[prediction] * 100)
print()
correct = 0

for image, label in zip(X, y):

    feature_maps = conv2d(
        image,
        kernels
    )

    feature_maps = leaky_relu(
        feature_maps
    )

    pooled_maps = max_pool2d(
        feature_maps
    )

    flattened, _ = flatten(
        pooled_maps
    )

    output = np.dot(
        flattened,
        weights
    ) + bias

    probabilities = softmax(output)

    prediction = np.argmax(probabilities)

    if prediction == label:
        correct += 1

accuracy = (correct / len(X)) * 100

print("\n========== Test Result ==========")
print("Total Images :", len(X))
print("Correct :", correct)
print("Wrong :", len(X) - correct)
print(f"Accuracy : {accuracy:.2f}%")