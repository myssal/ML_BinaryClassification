import numpy as np
from collections import Counter

class KNN:
    """
    k-nearest neighbors classifier for binary or multi-class classification.

    parameters:
        k (int): number of nearest neighbors to consider for prediction.
    """
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        store the training data for later use during prediction.

        parameters:
            X (np.array): training features of shape (n_samples, n_features)
            y (np.array): target labels of shape (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        predict labels for input samples X.

        parameters:
            X (np.array): input features of shape (n_samples, n_features)

        returns:
            np.array: predicted class labels
        """
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)

    def _predict_one(self, x):
        """
        predict the label for a single sample x using majority vote among k nearest neighbors.

        parameters:
            x (np.array): a single input sample of shape (n_features,)

        returns:
            int or label: predicted class for the input sample
        """
        # compute euclidean distances from x to all training samples
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]

        # get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # get labels of k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # handle case when labels are nested in iterable (like [[1], [0]])
        if hasattr(k_nearest_labels[0], "__iter__"):
            k_nearest_labels = [label[0] for label in k_nearest_labels]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
