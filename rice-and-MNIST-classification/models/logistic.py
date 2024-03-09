"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float, lr_decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.lr_decay = lr_decay

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        output = 1 / (1 + np.exp(-z))
        return output

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        x_train = X_train.copy()
        x_train = np.concatenate([np.ones((x_train.shape[0],1)), x_train], axis=1) # add columns of 1s
        N, D = x_train.shape

        np.random.seed(42)

        self.w = np.random.randn(D - 1)
        self.w = np.concatenate([[1], self.w])

        for epoch in range(self.epochs):

            for i in range(N):
                
                yi = y_train[i]
                xi = x_train[i]
                self.w = self.w + self.lr * self.sigmoid(-yi*np.inner(self.w, xi))*yi*xi

            self.lr *= self.lr_decay


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        x_test = X_test.copy()
        x_test = np.concatenate([np.ones((x_test.shape[0],1)), x_test], axis=1) # add columns of 1s
        N, D = x_test.shape
        predictions = np.zeros(N, dtype = int)

        for idx, x in enumerate(x_test):

            if np.inner(self.w, x) > self.threshold:

                predictions[idx] = 1

            else:

                predictions[idx] = -1

        return predictions
