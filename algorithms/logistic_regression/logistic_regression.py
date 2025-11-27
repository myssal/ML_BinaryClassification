import numpy as np

class LogisticRegression:
    """
    logistic regression classifier for binary classification.

    parameters:
        learning_rate (float): step size for gradient descent updates.
        n_iters (int): number of iterations for training.
    """
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        """
        sigmoid activation function to normalize any value to the range 0-1.

        parameters:
            x (float or np.array): input value(s)

        returns:
            float or np.array: sigmoid of input
        """
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """
        train the logistic regression model using gradient descent.

        parameters:
            X (np.array): training features of shape (n_samples, n_features)
            y (np.array): target labels of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y = y.ravel()  # ensure y is a 1d array

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        predict binary labels for input features X.

        parameters:
            X (np.array): input features of shape (n_samples, n_features)

        returns:
            np.array: predicted class labels (0 or 1)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
