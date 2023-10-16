# Documentation

As mentioned before, this project originally originates from a JetBrains Academy project. The idea is to build a neural network from scratch and train it on the Fashion-MNIST dataset. The project is split into seven stages:

-   **Stage 1**: Basic functions (xavier, sigmoid, etc.)
-   **Stage 2**: Feedforward
-   **Stage 3**: Metrics and backpropagation
-   **Stage 4**: Training and visualization
-   **Stage 5**: Two-layer neural network
-   **Stage 6**: Backpropagation with two layers
-   **Stage 7**: Training and visualization with two layers

The stages can be reviewed individually using this project's git history. Also, part of the project - the automatic data download process - was provided by JetBrains Academy.

## Stage 1 - Basic functions

At the beginning of the project, we need to implement some basic functions. These functions are used throughout the project and will come in handy later on. The functions are:

-   `scale()`: Rescale the data
-   `xavier()`: Xavier initialization
-   `sigmoid()`: Sigmoid activation function

## Stage 2 - Feedforward

In this stage, we implement the first layer of the fully connected neural network. It contains no hidden layers; all of the input-layer neurons are connected with neurons of the output layer. The functions and classes we implement in this stage are:

-   `OneLayerNeural`: Class for the neural network
-   `forward()`: Feedforward function

## Stage 3 - Metrics and backpropagation

We previously implemented the feedforward method. Now it is time for the actual training. We need to implement the backpropagation algorithm and the metrics to evaluate the model's performance. The functions and classes we implement in this stage are:

-   `backprop()`: Backpropagation algorithm
-   `sigmoid_prime()`: Derivative of the sigmoid function
-   `mse()`: Mean squared error
-   `mse_prime()`: Derivative of the mean squared error

## Stage 4 - Training and visualization

Neural networks use batch education. A dataset is split into subsets with a fixed number of rows. Then you feed your neural network with each subset, one by one. Once a neural network has finished working with every batch, we can say that an epoch is completed.

We need to implement a function to perform one learning epoch with a determined batch. In this stage, these are:

-   `train()`: Train the neural network
-   `accuracy()`: Calculate the accuracy of the neural network

If you want to visualize the learning process, you can use the provided `plot()` function.

## Stage 5 - Two-layer neural network

We have successfully built a one-layer neural network. However, this concept is very basic and not fit for solving difficult tasks. Let us complicate the model by adding a hidden layer of neurons and making it a two-layer neural network. We will use 64 neurons in the hidden layer. The functions and classes we implement in this stage are:

-   `TwoLayerNeural`: Class for the neural network
-   `forward()`: Feedforward function

## Stage 6 - Backpropagation with two layers

We have successfully built a two-layer neural network. Now it is time to implement the backpropagation algorithm for this model. The functions and classes we implement in this stage are:

-   `backprop()`: Backpropagation algorithm

## Stage 7 - Training and visualization with two layers

In the last stage, we put everything together and train our models with 20 epochs, a batch size of 100, and an alpha\* of 0,5. We can use the `plot()` function to visualize the learning process.

\*Alpha is the learning rate.

_If you want to learn more about these functions and their mathematical background, you can find the documentation in the [Equations.md](model/Equations.md) file._
