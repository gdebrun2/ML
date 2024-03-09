"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C.
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last.
    The outputs of the last fully-connected layer are passed through
    a sigmoid.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        optim: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        return np.tensordot(W, X, axes=(0, 1)).T + b.reshape(1, -1)

    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        return W.T

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        mask = np.where(X <= 0)
        X[mask] = 0
        return X

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """

        X[X <= 0] = 0
        return X

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        N = y.shape[0]
        predictions = np.argmax(p, axis=1)
        err = y - p
        return np.sum(err.T @ err) / N

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C)
        """
        self.outputs = {}
        # implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.

        N = X.shape[0]
        outputs = X.copy()

        self.outputs['input'] = outputs

        for layer in range(self.num_layers):
            # print("layer number", layer)
            # print("input shape", outputs.shape)

            weights = self.params["W" + str(layer + 1)]
            # print("weight shape", weights.shape)
            bias = self.params["b" + str(layer + 1)]

            linear_output = self.linear(weights, outputs, bias)
            self.outputs["z" + str(layer + 1)] = linear_output

            if layer == self.num_layers - 1:
                outputs = self.sigmoid(linear_output)
                self.outputs["a" + str(layer + 1)] = outputs
                # print("final", outputs.shape)

            else:
                outputs = self.relu(linear_output)
                self.outputs["a" + str(layer + 1)] = outputs

        return outputs

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        loss = self.mse(y, self.outputs["a" + str(self.num_layers)])

        dk = self.outputs["a" + str(self.num_layers)] - y

        for i in range(self.num_layers - 1, 0, -1):

            #print(i)
            # np.tensordot(W, X, axes=(0, 1)).T
            #print(np.tensordot(self.outputs["a" + str(i)], dk, axes=(0, 0)).shape)
            ai = self.outputs["a" + str(i)]
            print(dk.shape, ai.shape)
            self.gradients["W" + str(i + 1)] = dk @ ai #np.tensordot(self.outputs["a" + str(i)], dk, axes=(0, 0))
            self.gradients["b" + str(i + 1)] = dk

            wi = self.params["W" + str(i + 1)]
            dk = np.inner(wi, dk).T * self.relu_grad(self.outputs["z" + str(i)])
        
        self.gradients['W1'] = self.outputs['input']
        self.gradients['b1'] = 89

        return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        pass