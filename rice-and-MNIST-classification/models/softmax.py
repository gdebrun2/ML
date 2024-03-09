"""Softmax model."""

import numpy as np


class Softmax:
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

    def calc_gradient(self, x_batch: np.ndarray, y_batch: np.ndarray, p_batch: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """

        grad = np.zeros(self.w.shape)

        for i in range(self.batchsize):
                

            xi = x_batch[i]
            yi = y_batch[i]
            pi = p_batch[:,i]

            grad[yi] = grad[yi] + (pi[yi] - 1) * xi
            grad[yi][1:] = grad[yi][1:] + self.reg_const * self.w[yi][1:]

            for c in range(pi.shape[0]):

                if c != yi:

                    grad[c] = grad[c] +  pi[c] * xi
                    grad[c][1:] = grad[c][1:] + self.reg_const * self.w[c][1:]
                    

        return grad / self.batchsize
    

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        x_train = X_train.copy()
        x_train = np.concatenate([np.ones((x_train.shape[0],1)), x_train], axis=1) # add columns of 1s
        N, D = x_train.shape

        np.random.seed(42)

        if self.n_class == 2:

            self.w = np.random.randn(D - 1)
            self.w = np.concatenate([[1], self.w])

            for epoch in range(self.epochs):

                for i in range(N):
                    
                    yi = y_train[i]
                    xi = x_train[i]
                    sigmoid = 1 / (1 + np.exp(yi * np.inner(self.w, xi)))
                    self.w = self.w + self.lr * sigmoid *yi*xi

                self.lr *= self.lr_decay

        else:


            self.w = np.random.normal(0, 1, (self.n_class, D - 1))
            self.w = np.concatenate([np.ones((self.n_class, 1)), self.w], axis=1) # add bias column

            
            for epoch in range(self.epochs):

                linear = np.inner(self.w, x_train)
                maxes = np.max(linear, axis = 0).reshape(1, -1)
                diffs = (linear - maxes)
                numerator = np.exp(diffs)
                div = np.sum(np.exp(diffs), axis = 0)
                p = numerator / div
        
                sample = np.random.choice(N, self.batchsize)
                x_batch = x_train[sample]
                y_batch = y_train[sample]
                p_batch = p[:, sample]

                grad = self.calc_gradient(x_batch, y_batch, p_batch)

                self.w -= grad * self.lr
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
                
                if np.inner(self.w, x) > .5:

                    predictions[idx] = 1

                else:

                    predictions[idx] = -1

        else:

            linear = np.inner(self.w, x_test)
            maxes = np.max(linear, axis = 0).reshape(1, -1)
            diffs = (linear - maxes)
            numerator = np.exp(diffs)
            div = np.sum(np.exp(diffs), axis = 0)
            p = numerator / div

            for i in range(N):
                
                pi = p[:, i]
                predictions[i] = np.argmax(pi)

        return predictions
    




