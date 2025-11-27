import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)

    def _predict_one(self, x):
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if hasattr(k_nearest_labels[0], "__iter__"):
            k_nearest_labels = [label[0] for label in k_nearest_labels]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]