# FC-NN and CNN Implementation Using Numpy

This repository contains the implementation of a **Fully Connected Neural Network (FC-NN)** and a **Convolutional Neural Network (CNN)** from scratch using the **Numpy** library. These models were developed as part of a machine learning assignment for the DDA3020 course at The Chinese University of Hong Kong, Shenzhen.

## Overview

The project implements two types of neural networks:

1. **Fully Connected Neural Network (FC-NN)**: The FC-NN consists of a multi-layer perceptron (MLP) with one hidden layer. The implementation includes both forward and backward propagation processes, as well as the loss function and activation functions.
  
2. **Convolutional Neural Network (CNN)**: The CNN consists of multiple convolution layers followed by subsampling layers, all implemented from scratch using Numpy. The forward propagation process includes convolutional layers, max-pooling layers, and fully connected layers, while the backward propagation is not implemented as part of the assignment.

The networks were trained using the **FashionMNIST** dataset, which consists of 28x28 grayscale images classified into 10 categories.

## Repository Structure

- **119010148_homework3.ipynb**: Jupyter notebook containing the implementation of both the Fully Connected Neural Network (FC-NN) and the Convolutional Neural Network (CNN) from scratch. This file includes code for forward and backward propagation (for FC-NN), as well as all the required steps to run the models on the FashionMNIST dataset.
  
- **119010148_homework3.pdf**: Written report that covers the theoretical aspects of the assignment, including decision tree construction, computational graph, backpropagation, and CNN-related questions.

- **Task description.pdf**: Original task description for the assignment, outlining the details and requirements for both the written and coding parts of the assignment.

- **README.md**: This file.

## Task Description

The assignment involves both theoretical and coding tasks:

### 1. Written Part

This part involves answering several theoretical questions about decision trees, computational graphs, backpropagation, and CNNs.

- **Problem 1**: Construction of a decision tree to predict user behavior on online advertisements.
- **Problem 2**: Deriving a computational graph for an MLP and computing the derivatives for backpropagation.
- **Problem 3**: Understanding CNN architecture and computing the necessary parameters, including the receptive field and number of neurons required.
- **Problem 4**: Discussing the advantages of convolutional layers and performing additional CNN-related calculations.

### 2. Coding Part

This part involves implementing the following from scratch using Numpy:

- **Fully Connected Neural Network (FC-NN)**: Implement forward and backward propagation processes. Define the loss function, activation function (ReLU), and training process (gradient descent).
  
- **Convolutional Neural Network (CNN)**: Implement the forward propagation process, which includes convolution layers, max pooling, and linear layers. This model is trained on the FashionMNIST dataset.

## FashionMNIST Dataset

The FashionMNIST dataset is used to train both the FC-NN and CNN models. It consists of 60,000 training images and 10,000 testing images of 28x28 pixels, each associated with a label from 10 different clothing categories.

- **Training Data**: 60,000 examples, each labeled with a class from 10 categories.
- **Testing Data**: 10,000 examples used to evaluate the performance of the trained models.

Each image is flattened into a vector of 784 pixels, and the dataset is used for both training and testing the neural networks.
