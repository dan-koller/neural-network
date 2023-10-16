# Snippets

This document contains snippets of the code to explain the model and implementation.

I will cover the methods in the same order as they are defined in the [code](../../src/main.py) to make it easier to follow.

## one_hot()

The one_hot() method is used to convert the labels to one-hot encoding. This method was already provided and I do not take credit for it.

## plot()

The plot() method is used to visualize the learning process of the neural network. This method was also already provided and I do not take credit for it.

## scale()

The scale() method is used to rescale the data to turn big numbers into small numbers. This is a good practice to do before training the neural network, since big numbers can cause problems during the training process. You can take a look on the underlying equation [here](../model/Equations.md#scale).

## xavier()

The xavier() method is used to initialize the weights of the neural network. You can take a look on the underlying equation [here](../model/Equations.md#xavier).

## sigmoid()

The sigmoid() method is used to calculate the output of the neural network. You can take a look on the underlying equation [here](../model/Equations.md#sigmoid).

## sigmoid_prime()

The sigmoid_prime() method is used to calculate the gradient of the loss function. You can take a look on the underlying equation [here](../model/Equations.md#sigmoid-prime).

## mse()

This method is used to calculate the loss of the neural network. You can take a look on the underlying equation [here](../model/Equations.md#mean-squared-error).

## mse_prime()

This method is used to calculate the gradient of the loss function. You can take a look on the underlying equation [here](../model/Equations.md#mean-squared-error-prime).

## class OneLayerNeural

This class represents the first layer of the neural network. It contains the following methods:

### \_\_init\_\_()

The \_\_init\_\_() method is used to initialize the weights of the neural network. It is based on the xavier() method.

### forward()

The forward() method is used to calculate the output of the neural network using the sigmoid() method.

### backprop()

The backprop() method is used to calculate the gradient of the loss function using the mse_prime() and sigmoid_prime() methods. After that, the gradient is calculated as delta_W for the weights and delta_b for the bias. The weights and bias are updated using the learning rate (alpha).

## class TwoLayerNeural

This class represents the second layer of the network. It seems quite similar to the first layer at first glance, but differs in its inner workings. It contains the following methods:

### \_\_init\_\_()

The \_\_init\_\_() method is used to initialize the weights of the neural network. It is based on the xavier() method and uses 64 neurons for the hidden layer. You can refresh the concept of neural net layers [here](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1&t=215s).

### forward()

The forward() method is used to calculate the output of the neural network using the sigmoid() method.

```python
    z = X
    for i in range(2):
        z = sigmoid(np.dot(z, self.W[i]) + self.b[i])
    return z
```

It is a bit different from the first layer, since it uses a loop to calculate the output of the hidden layer and the output layer.

### backprop()

The backpropagation process in this layer is a bit more complex than the first layer. It is based on the backprop() method from the first layer, but requires some additional calculations.

1. Get the size of the data (n) and create a vector of ones for the bias calculation (biases)

    ```python
        n = X.shape[0]  # Number of trained samples
        biases = np.ones((1, n))  # Vector of ones for bias calculation
    ```

2. Calculate the output of the network and the gradient of the loss function for both layers

    ```python
        # Calculate the output of the network
        yp = self.forward(X)

        # Calculate the gradient of the loss function with respect to the bias of the output layer
        loss_grad_1 = 2 * alpha / n * ((yp - y) * yp * (1 - yp))

        # Calculate the output of the first layer
        f1_out = sigmoid(np.dot(X, self.W[0]) + self.b[0])

        # Calculate the gradient of the loss function with respect to the bias of the first layer
        loss_grad_0 = np.dot(loss_grad_1, self.W[1].T) * f1_out * (1 - f1_out)
    ```

3. Update the weights and biases using the learning rate (alpha)

    ```python
        # Update weights and biases
        self.W[0] -= np.dot(X.T, loss_grad_0)
        self.W[1] -= np.dot(f1_out.T, loss_grad_1)

        self.b[0] -= np.dot(biases, loss_grad_0)
        self.b[1] -= np.dot(biases, loss_grad_1)
    ```

## train()

The train() method is used to train the neural network.

```python
    n = X.shape[0]
    for i in range(0, n, batch_size):
        model.backprop(X[i:i + batch_size], y[i:i + batch_size], alpha)
```

At first we get the size of the data (n). Then we iterate over the data in batches of size batch_size. For each batch, we call the backprop() method to calculate the gradient of the loss function. After that, the weights and bias are updated using the learning rate (alpha).

## accuracy()

The accuracy() method is used to calculate the accuracy of the neural network using the equation from the [documentation](../model/Equations.md#accuracy).

## main()

The main() method is used to train the neural network and visualize the learning process.

The **first part** of the method provides automatic data download and preparation.

-   In the **first step** the data is rescaled.

    ```python
        X_train, X_test = scale(X_train, X_test)
        n_features = X_train.shape[1]
        n_classes = y_train.shape[1]
    ```

-   After that, we create our model

    ```python
        model = TwoLayerNeural(n_features, n_classes)
    ```

-   Finally, we train the model with 20 epochs\*

    ```python
    r1 = []
    for i in range(20):
        train(model, X_train, y_train, alpha=0.5)
        r1.append(accuracy(model, X_train, y_train))
    ```

    \*You can change the number of epochs to see how the accuracy changes. Please make sure you have necessary computing resources or the training process will take a lot of time.

-   Now we can print the results

    ```python
        print(r1)
    ```
