# Python Neural Network

This is a simple but fully connected neural network with two layers built from scratch. The dataset for this project is 'Fashion-MNIST'. The script will feedforward networks and implement backpropagation and weight initialization. The is also a training loop and visualization of the model's training process.

I built this project in order you get a solid knowledge of how neural networks work. I hope you enjoy it! The idea is based on a [JetBrains Academy](https://hyperskill.org/projects/250) project.

## Requirements

-   Python 3
-   Packages from `requirements.txt`

## Installation

1.  Clone the repository

```bash
git clone https://github.com/dan-koller/Python-Neural-Network
```

2. Create a virtual environment\*

```bash
python3 -m venv venv
```

3. Activate the virtual environment

```bash
source venv/bin/activate
```

4. Install the requirements\*

```bash
pip3 install -r requirements.txt
```

5. Run the app\*

```
python3 main.py
```

_\*) You might need to use `python` and `pip` instead of `python3` and `pip3` depending on your system._

## Usage

In its current and final stage, the app will train a neural network on the Fashion-MNIST\* dataset with 20 epochs, a batch size of 100 and an alpha of 0,5. The final list of accuracies after each epoch is displayed to the standard output.

The dataset will be downloaded automatically if it is not present in the `data` folder. However, if that is inconvenient, feel free to download the [training](https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1) and [test](https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1) datasets on your own.

\*Fashion-MNIST is a dataset of Zalando's article images - consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We use it here as a drop-in replacement for the classic MNIST dataset - often used as the "Hello, World" of machine learning programs for computer vision. You can learn more about Fashion-MNIST [here](https://en.wikipedia.org/wiki/Fashion_MNIST).

If you want to visualize and plot the training process, you need to uncomment the last statement in the `main()` function in `main.py`. This will plot the training process and save the plot as `plot.png`.

## Documentation

You can find the main documentation in the [docs](docs) folder. Here, you can find documentation on how the neural network works and how the training process is implemented.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
