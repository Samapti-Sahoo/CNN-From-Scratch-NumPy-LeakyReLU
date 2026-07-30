import time
import numpy as np

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

print("Model Loaded Successfully!")

print("Loading MNIST dataset...")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False,
    parser="auto"
)

X = mnist.data.astype(np.float32) / 255.0
X = X.reshape(-1, 28, 28)

# Number of images for benchmark
num_images = 1000

X = X[:num_images]

print(f"Benchmark Images : {len(X)}")

start_time = time.perf_counter()

for image in X:

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

    probabilities = softmax(
        output
    )

    prediction = np.argmax(
        probabilities
    )

end_time = time.perf_counter()

total_time = end_time - start_time

latency = total_time / num_images

throughput = num_images / total_time

print("\n========== CNN Performance Benchmark ==========")
print(f"Images Processed        : {num_images}")
print(f"Total Processing Time   : {total_time:.4f} seconds")
print(f"Average Latency/Image   : {latency * 1000:.4f} ms")
print(f"Throughput              : {throughput:.2f} images/second")