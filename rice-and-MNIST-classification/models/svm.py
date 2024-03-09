"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, lr_decay: float, batchsize: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.lr_decay = lr_decay
        self.batchsize = batchsize

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """

        N, D = X_train.shape

        if self.n_class == 2:

            grad = np.zeros(D)

            for xi, yi in zip(X_train, y_train):

                grad[1:] = grad[1:] - self.reg_const * self.w[1:] # avoid regularizing bias

                if yi * np.inner(self.w, xi) < 1:

                    grad = grad - yi * xi

        else:

            grad = np.zeros((self.n_class, D))

            for xi, yi in zip(X_train, y_train):

                preds = np.inner(self.w, xi)
                preds += 1
                preds[yi] -= 1

                grad[1:] = grad[1:] + self.reg_const * self.w[1:] # avoid regularizing bias

                if preds[yi] != np.max(preds):

                    mask = np.where(preds > preds[yi])

                    grad[yi] = grad[yi] - xi
                    grad[mask] = grad[mask] + xi

        return grad/N


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        X_train = X_train.copy()
        X_train = np.concatenate([np.ones((X_train.shape[0],1)), X_train], axis=1) # add columns of 1s
        N, D = X_train.shape

        np.random.seed(42)

        if self.n_class == 2:
       
            self.w = np.random.randn(D - 1)
            self.w = np.concatenate([[1], self.w]) # add bias column

            for epoch in range(self.epochs):

                sample = np.random.choice(N, self.batchsize)
                x_batch = X_train[sample]
                y_batch = y_train[sample]

                grad = self.calc_gradient(x_batch, y_batch)

                self.w -= grad * self.lr

                if epoch in [20, 60, 80, 90, 95]:

                    self.lr *= self.lr_decay

        else:

            self.w = np.random.normal(0, 1, (self.n_class, D - 1))
            self.w = np.concatenate([np.ones((self.n_class, 1)), self.w], axis=1) # add bias column

            for epoch in range(self.epochs):

                sample = np.random.choice(N, self.batchsize)
                x_batch = X_train[sample]
                y_batch = y_train[sample]

                grad = self.calc_gradient(x_batch, y_batch)

                self.w -= grad * self.lr

                if epoch in [20, 60, 80, 90, 95]:

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
        x_test = np.concatenate([np.ones((x_test.shape[0],1)), x_test], axis=1) # add columns of 1s
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
