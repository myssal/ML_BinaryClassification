import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000, verbose_log=False):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.verbose_log = verbose_log
        self.weights = None
        self.bias = None

    def _log(self, message):
        if self.verbose_log:
            print(f"[LogisticRegression] {message}")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        self._log(f"Starting training for {self.n_iters} iterations with learning rate {self.lr:.4f}.")
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y = y.ravel()

        for i in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0:
                self._log(f"Iteration {i}: weights={np.array2string(self.weights, precision=4)}, bias={self.bias:.4f}")
        
        self._log("Training complete.")

    def predict(self, X):

        self._log(f"Starting prediction for {X.shape[0]} samples.")
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        self._log(f"Predicted probabilities (first 5): {np.array2string(y_predicted[:5], precision=4)}")
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        self._log("Prediction complete.")
        return np.array(y_predicted_cls)
