# Equations

This document describes the equations used in the model.

## Activation Functions

### Scale

The scale function is used to scale the data. It is a good practice to scale the data before training the neural network, since big numbers can cause problems during the training process. The scale function is based on the following equation:

$$
X_{new} = \frac{X}{\max(X)}
$$

### Xavier

The Xavier initialization is used to initialize the weights of the neural network. It is based on the following equation:

$$
 w \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]
$$

### Sigmoid

The sigmoid activation function is used to calculate the output of the neural network. It is based on the following equation:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### Sigmoid Prime

The derivative of the sigmoid function is used to calculate the gradient of the loss function. It is based on the following equation:

$$
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
$$

### Mean squared error

The mean squared error is used to calculate the loss of the neural network. It is based on the following equation:

$$
{MSE} = \frac{1}{n}\sum\limits_{i=1}^n\left(y_{i_{pred}} - y_{i_{true}}\right)^2
$$

### Mean squared error prime

The derivative of the mean squared error is used to calculate the gradient of the loss function. It is based on the following equation:

$$
{MSE}'_i = 2 \cdot \left(y_{i_{pred}} - y_{i_{true}}\right)
$$

## Training

For each epoch, the neural network is trained with a batch of data. The training is based on the following equations:

$$
\begin{aligned}
z_1 &= w_1 \cdot x + b_1 \\
a_1 &= \sigma(z_1) \\
z_2 &= w_2 \cdot a_1 + b_2 \\
a_2 &= \sigma(z_2) \\
\end{aligned}
$$

_(Have a look at the code to see how the equations are implemented.)_

## Accuracy

The accuracy is used to calculate the accuracy of the neural network. It is based on the following equation:

$$
\text{Accuracy} = \frac{\text{True answered items}}{\text{Total number of items}}
$$
