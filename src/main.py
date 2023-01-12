import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):
    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


def scale(X_train, X_test):
    # Rescale the data since neural networks don't like big numbers
    X_max = np.max(X_train)
    X_train = X_train / X_max
    X_test = X_test / X_max
    return X_train, X_test


def xavier(n_in, n_out):
    # Xavier initialization
    low = -np.sqrt(6 / (n_in + n_out))
    high = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(low, high, (n_in, n_out))


def sigmoid(x):
    # Calculate the sigmoid function
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    # Calculate the derivative of the sigmoid function
    return sigmoid(x) * (1 - sigmoid(x))


def mse(y_pred, y_true):
    # Calculate the mean squared error
    return np.mean((y_pred - y_true) ** 2)


def mse_prime(y_pred, y_true):
    # Calculate the derivative of the mean squared error
    return 2 * (y_pred - y_true)


class OneLayerNeural:
    def __init__(self, n_features, n_classes):
        # Initiate weights and biases using Xavier initialization
        self.W = xavier(n_features, n_classes)
        self.b = xavier(1, n_classes)

    def forward(self, X):
        # Perform a forward step
        return sigmoid(np.dot(X, self.W) + self.b)

    def backprop(self, X, y, alpha):
        # Perform a backward step
        # Calculate the error
        error = (mse_prime(self.forward(X), y) *
                 sigmoid_prime(np.dot(X, self.W) + self.b))

        # Calculate the gradient
        delta_W = (np.dot(X.T, error)) / X.shape[0]
        delta_b = np.mean(error, axis=0)

        # Update weight and bias
        self.W -= alpha * delta_W
        self.b -= alpha * delta_b


def train(model, X, y, alpha, batch_size=100):
    # Perform a single epoch of training
    n = X.shape[0]
    for i in range(0, n, batch_size):
        model.backprop(X[i:i + batch_size], y[i:i + batch_size], alpha)


def accuracy(model, X, y):
    # Calculate the accuracy of the model
    y_pred = np.argmax(model.forward(X), axis=1)
    y_true = np.argmax(y, axis=1)
    return np.mean(y_pred == y_true)


class TwoLayerNeural:
    def __init__(self, n_features, n_classes):
        # Size of the hidden layer (64 neurons)
        hidden_size = 64

        # Initiate weights and biases using Xavier initialization
        self.W = [xavier(n_features, hidden_size),
                  xavier(hidden_size, n_classes)]
        self.b = [xavier(1, hidden_size), xavier(1, n_classes)]

    def forward(self, X):
        # Calculate feedforward step
        z = X
        for i in range(2):
            z = sigmoid(np.dot(z, self.W[i]) + self.b[i])
        return z

    def backprop(self, X, y, alpha):
        n = X.shape[0]  # Number of trained samples
        biases = np.ones((1, n))  # Vector of ones for bias calculation

        # Calculate the output of the network
        yp = self.forward(X)

        # Calculate the gradient of the loss function with respect to the bias of the output layer
        loss_grad_1 = 2 * alpha / n * ((yp - y) * yp * (1 - yp))

        # Calculate the output of the first layer
        f1_out = sigmoid(np.dot(X, self.W[0]) + self.b[0])

        # Calculate the gradient of the loss function with respect to the bias of the first layer
        loss_grad_0 = np.dot(loss_grad_1, self.W[1].T) * f1_out * (1 - f1_out)

        # Update weights and biases
        self.W[0] -= np.dot(X.T, loss_grad_0)
        self.W[1] -= np.dot(f1_out.T, loss_grad_1)

        self.b[0] -= np.dot(biases, loss_grad_0)
        self.b[1] -= np.dot(biases, loss_grad_1)


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    # First step: Rescale the data
    X_train, X_test = scale(X_train, X_test)
    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]

    # Create a class instance and train it
    model = TwoLayerNeural(n_features, n_classes)

    # Train the model (20 epochs)
    r1 = []
    for i in range(20):
        train(model, X_train, y_train, alpha=0.5)
        r1.append(accuracy(model, X_train, y_train))

    # Print the results
    print(r1)
