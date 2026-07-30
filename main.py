import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

from src.activations import leaky_relu
from src.convolution import conv2d
from src.pooling import max_pool2d
from src.flatten import flatten
from src.softmax import softmax

print("Loading trained model...")

model = np.load("trained_model.npz")

kernels = model["kernels"]
weights = model["weights"]
bias = model["bias"]

print("Model loaded successfully!")

print("\nLoading MNIST dataset...")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False,
    parser="auto"
)

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

X = X.reshape(-1, 28, 28)

# Change this number to test different images
index = 0

image = X[index]
actual_label = y[index]

# -----------------------------
# Forward Pass
# -----------------------------

feature_maps = conv2d(
    image,
    kernels
)

feature_maps = leaky_relu(
    feature_maps
)

pooled_maps = max_pool2d(
    feature_maps,
    pool_size=2,
    stride=2
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

confidence = probabilities[prediction] * 100

# -----------------------------
# Print Result
# -----------------------------
print("Actual Label:", actual_label)
print("Predicted Label:", prediction)
print("Probabilities:", probabilities)
print("\n========== Prediction ==========")
print("Image Index     :", index)
print("Actual Label    :", actual_label)
print("Predicted Label :", prediction)
print(f"Confidence      : {confidence:.2f}%")

print("\nTop 3 Predictions")

top3 = np.argsort(probabilities)[::-1][:3]

for i in top3:
    print(f"Digit {i} : {probabilities[i] * 100:.2f}%")

# -----------------------------
# Show Image
# -----------------------------

plt.figure(figsize=(4,4))
plt.imshow(image, cmap="gray")

plt.title(
    f"Actual : {actual_label}\n"
    f"Predicted : {prediction}\n"
    f"Confidence : {confidence:.2f}%"
)

plt.axis("off")

plt.show()