#  CNN from Scratch using NumPy with Leaky ReLU

A Convolutional Neural Network (CNN) built completely from scratch using **NumPy** without using deep learning frameworks like TensorFlow, Keras, or PyTorch. This project implements both forward propagation and backpropagation for handwritten digit classification on the MNIST dataset.

---

##  Features

- Convolution Layer
- Leaky ReLU Activation
- Max Pooling Layer
- Flatten Layer
- Dense (Fully Connected) Layer
- Softmax Classifier
- Cross Entropy Loss
- Backpropagation
- Gradient Descent
- Model Save & Load
- MNIST Digit Classification
- Performance Benchmark

---

##  Technologies

- Python
- NumPy
- Matplotlib
- Scikit-learn (for loading the MNIST dataset only)

---

## 📂 Project Structure

```
LeakyReLU_CNN/
│
├── src/
│   ├── activations.py
│   ├── convolution.py
│   ├── conv_backward.py
│   ├── pooling.py
│   ├── flatten.py
│   ├── dense.py
│   ├── softmax.py
│   └── loss.py
│
├── train.py
├── test.py
├── main.py
├── benchmark.py
├── trained_model.npz
├── requirements.txt
└── README.md
```

---

##  How to Run

### Train the model

```bash
python train.py
```

### Test the model

```bash
python test.py
```

### Predict a single image

```bash
python main.py
```

### Run benchmark

```bash
python benchmark.py
```

---

##  Results

- Training Accuracy: **99.60%**
- Evaluation Accuracy: **99.90%** *(on the current 1000-image evaluation used in the project)*
- Average Latency: **29.21 ms/image**
- Throughput: **34.23 images/second**

---

##  Learning Outcomes

This project demonstrates the implementation of:

- Convolution Operation
- Feature Extraction
- Leaky ReLU
- Max Pooling
- Flattening
- Dense Neural Network
- Softmax Classification
- Cross Entropy Loss
- Backpropagation
- Gradient Descent Optimization

---

## 👩‍💻 Author

**Samapti Sahoo**