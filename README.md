# MNIST Neural Network Project

This project implements a neural network model using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## Project Overview

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9) and is commonly used for training image classification models. In this project, I have built a simple feedforward neural network to classify these digits using TensorFlow.

## Requirements

Make sure you have the following installed:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib (for visualizations)

You can install the required dependencies using `pip`:

```bash
pip install tensorflow numpy matplotlib 
```
## Comparison: From Scratch Model vs. TensorFlow Model

In this project, I implemented two versions of a neural network to classify handwritten digits from the MNIST dataset: one using a "from scratch" approach and another utilizing the TensorFlow framework.

###  1. Model Architecture:

From Scratch Model:
In the "from scratch" implementation, I manually defined the neural network's architecture, including the initialization of weights, biases, and activation functions. The forward and backward propagation were implemented using basic matrix operations and vectorized calculations to optimize performance. The model included a simple two-layer architecture with ReLU activation in the hidden layer and softmax activation in the output layer.

TensorFlow Model:
The TensorFlow implementation used the high-level Keras API to define the same architecture. Here, TensorFlow took care of the initialization, forward propagation, and backpropagation automatically, allowing for a much faster development process. The layers were defined using built-in functions, and the optimizer handled weight updates internally. This approach reduced the need for manual coding of complex mathematical operations.

### 2. Training Process:

From Scratch Model:
In the manual implementation, the training loop involved iterating over the dataset, calculating the forward pass, performing backpropagation, and updating the parameters using the computed gradients. The optimizer was manually implemented to adjust the weights and biases after each iteration. This gave me full control over the learning process, but it also made the training process slower and more error-prone.

TensorFlow Model:
The TensorFlow model leveraged efficient, optimized operations under the hood. TensorFlow’s built-in optimizers (such as Adam) handled gradient calculations and weight updates, which made the training process significantly faster and less error-prone. Additionally, TensorFlow automatically takes care of issues like gradient clipping and batch normalization when required.

### 3. Accuracy and Performance:

From Scratch Model:
The accuracy of the from-scratch model was around 89% (insert your actual accuracy here). Although the model performed reasonably well, there were some challenges in optimizing the training process, which may have led to slightly lower performance compared to the TensorFlow model.

TensorFlow Model:
The TensorFlow model achieved an accuracy of 97% (insert your actual accuracy here). The use of optimized built-in functions and TensorFlow’s automatic handling of backpropagation and weight updates likely contributed to a higher performance in terms of both accuracy and training time.

### 4. Advantages and Challenges:

From Scratch Model:
Advantages: Full control over the model and learning process. Great for understanding the underlying mechanics of neural networks and deep learning.
Challenges: More time-consuming, prone to errors in implementing complex operations like backpropagation, and may require extensive optimization to reach competitive accuracy levels.
TensorFlow Model:
Advantages: Faster to implement and easier to experiment with. TensorFlow’s optimizations allow for faster training times, and the high-level API simplifies model design and evaluation.
Challenges: Less control over the inner workings of the network. It can be more difficult to debug or modify low-level operations without diving deeper into the framework.

Conclusion:

While both models reached a satisfactory level of accuracy, the TensorFlow model performed better in terms of training efficiency and final accuracy. This is expected, as TensorFlow is designed to optimize neural network training, leveraging advanced techniques and hardware acceleration. However, the "from scratch" model provided valuable insights into the mechanics of neural networks and how various components like forward propagation, backpropagation, and optimization work in practice. It’s clear that while building a neural network from scratch is a great learning experience, using TensorFlow or other similar frameworks is much more efficient for developing production-ready models. 


## From scratch model overview 

The script will load the MNIST dataset, preprocess the data, define the neural network, train it for 10 epochs, and then evaluate its performance on the test dataset.

### Model Architecture

Our neural network (NN) will have a simple two-layer architecture:

### Layers:

#### Input Layer (a[0]):
This layer has 784 units corresponding to the 784 pixels in each 28x28 input image.
Hidden Layer (a[1]):
This layer has 10 units with ReLU activation.
Output Layer (a[2]):
This layer has 10 units corresponding to the ten digit classes with softmax activation.

#### Forward Propagation:

Z[1] = W[1] * X + b[1]
A[1] = gReLU(Z[1])
Z[2] = W[2] * A[1] + b[2]
A[2] = gsoftmax(Z[2])

#### Backward Propagation:

dZ[2] = A[2] − Y
dW[2] = (1/m) * dZ[2] * A[1]^T
dB[2] = (1/m) * ΣdZ[2]
dZ[1] = W[2]^T * dZ[2] .* g'[1](Z[1])
dW[1] = (1/m) * dZ[1] * A[0]^T
dB[1] = (1/m) * ΣdZ[1]

#### Parameter Updates:

W[2] := W[2] − α * dW[2]
b[2] := b[2] − α * dB[2]
W[1] := W[1] − α * dW[1]
b[1] := b[1] − α * dB[1]

#### Vars and Shapes:

#### Forward Propagation:

A[0] = X : 784 x m
Z[1] ~ A[1] : 10 x m
W[1] : 10 x 784 (as W[1] * A[0] ~ Z[1])
B[1] : 10 x 1
Z[2] ~ A[2] : 10 x m
W[2] : 10 x 10 (as W[2] * A[1] ~ Z[2])
B[2] : 10 x 1

#### Backward Propagation:

dZ[2] : 10 x m
dW[2] : 10 x 10
dB[2] : 10 x 1
dZ[1] : 10 x m
dW[1] : 10 x 10
dB[1] : 10 x 1

## License

This project does not currently have an explicit license. Feel free to use, modify, and distribute it as you see fit, but please give credit to the original sources of inspiration where appropriate.

## Acknowledgments

TensorFlow and Keras for the deep learning framework.
MNIST dataset for providing a widely used and easy-to-use image classification task.
Samson Zhang for the inspiration behind the model architecture and implementation.
