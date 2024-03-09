"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float, lr_decay: float, batchsize: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.batch_size = batchsize

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
    
        x_train = X_train.copy()
        x_train = np.concatenate([np.ones((x_train.shape[0],1)), x_train], axis=1) # add column of 1s
        N, D = x_train.shape
        np.random.seed(42)

        if self.n_class == 2:

            self.w = np.random.randn(D-1)
            self.w = np.concatenate([[1] , self.w]) # add bias column
            
            for epoch in range(self.epochs):

                sample = np.random.choice(N, self.batch_size)

                for idx in sample:
                    
                    yi = y_train[idx]
                    xi = x_train[idx]

                    if yi * np.inner(self.w, xi) < 0:

                        self.w = self.w + self.lr * yi * xi
                    
                    self.w[1:] = self.w[1:] - self.weight_decay * self.w[1:] # avoid regularizing bias

                self.lr *= self.lr_decay

        
        else:

            self.w = np.random.normal(0, 1, (self.n_class, D - 1))
            self.w = np.concatenate([np.ones((self.n_class, 1)), self.w], axis=1) # add bias column

            for epoch in range(self.epochs):

                sample = np.random.choice(N, self.batch_size)

                for idx in sample:

                    yi = y_train[idx]
                    xi = x_train[idx]
                    preds = np.inner(self.w, xi)

                    if preds[yi] != np.max(preds):

                        mask = np.where(preds > preds[yi])

                        self.w[yi] = self.w[yi] + self.lr*xi

                        self.w[mask] = self.w[mask] - self.lr*xi

                    self.w[:, 1:] = self.w[:, 1:] - self.lr * self.weight_decay * self.w[:, 1:] # avoid regularizing bias

                self.lr *= self.lr_decay

        return None

                       

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
        x_test = np.concatenate([np.ones((x_test.shape[0],1)), x_test], axis=1)
        N, D = x_test.shape
        predictions = np.zeros(N, dtype = int)

        if self.n_class == 2:

            for idx, x in enumerate(x_test):

                if np.inner(self.w, x) > 0:

                    predictions[idx] = 1

                else:

                    predictions[idx] = -1

        else:

            for idx, x in enumerate(x_test):

                predictions[idx] = np.argmax(np.inner(self.w, x))


        return predictions
