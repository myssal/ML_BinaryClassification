import numpy as np
from collections import Counter

class KNN:
    """
    k-nearest neighbors classifier for binary or multi-class classification.

    parameters:
        k (int): number of nearest neighbors to consider for prediction.
        verbose_log (bool): whether to print detailed logs during prediction
    """
    def __init__(self, k, verbose_log=False):
        self.k = k
        self.verbose_log = verbose_log
        self.X_train = None
        self.y_train = None

    def _log(self, message):
        """helper method to print log messages if verbose_log is enabled"""
        if self.verbose_log:
            print(f"[KNN] {message}")

    def fit(self, X, y):
        """
        store the training data for later use during prediction.

        parameters:
            X (np.array): training features (n_samples, n_features)
            y (np.array): target labels (n_samples,) or (n_samples,1)
        """
        self.X_train = X
        # flatten y and convert to plain python integers
        self.y_train = y.ravel().astype(int)
        self._log(f"training data of shape {X.shape} stored.")
        self._log(f"training labels: {list(self.y_train[:10])} ...")  # preview first 10 labels

    def predict(self, X):
        """
        predict labels for input samples X.

        parameters:
            X (np.array): input features (n_samples, n_features)

        returns:
            np.array: predicted class labels
        """
        self._log(f"starting prediction for {X.shape[0]} samples.")
        predictions = [self._predict_one(x) for x in X]
        self._log("prediction complete.")
        return np.array(predictions)

    def _predict_one(self, x):
        """
        predict the label for a single sample using majority vote among k nearest neighbors.

        parameters:
            x (np.array): a single input sample (n_features,)

        returns:
            int: predicted class for the input sample
        """
        self._log(f"predicting for single sample: {np.array2string(x[:5], precision=4)}...")  # show first 5 features
        # compute euclidean distances from x to all training samples
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]
        self._log(f"computed distances to {len(distances)} training samples.")

        # get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        self._log(f"identified {self.k} nearest neighbors.")

        # get labels of k nearest neighbors as plain integers
        k_nearest_labels = [int(self.y_train[i]) for i in k_indices]
        self._log(f"labels of {self.k}-nearest neighbors: {k_nearest_labels}")

        # majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        predicted_label = int(most_common[0][0])
        self._log(f"majority vote result: {predicted_label}")

        return predicted_label
