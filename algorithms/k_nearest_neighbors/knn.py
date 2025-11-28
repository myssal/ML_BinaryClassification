import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k, verbose_log=False):
        self.k = k
        self.verbose_log = verbose_log
        self.X_train = None
        self.y_train = None

    def _log(self, message):
        if self.verbose_log:
            print(f"[KNN] {message}")

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.ravel().astype(int)
        self._log(f"training data of shape {X.shape} stored.")
        self._log(f"training labels: {list(self.y_train[:10])} ...")

    def predict(self, X):
        self._log(f"starting prediction for {X.shape[0]} samples.")
        predictions = [self._predict_one(x) for x in X]
        self._log("prediction complete.")
        return np.array(predictions)

    def _predict_one(self, x):
        self._log(f"predicting for single sample: {np.array2string(x[:5], precision=4)}...")
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]
        self._log(f"computed distances to {len(distances)} training samples.")

        k_indices = np.argsort(distances)[:self.k]
        self._log(f"identified {self.k} nearest neighbors.")

        k_nearest_labels = [int(self.y_train[i]) for i in k_indices]
        self._log(f"labels of {self.k}-nearest neighbors: {k_nearest_labels}")

        most_common = Counter(k_nearest_labels).most_common(1)
        predicted_label = int(most_common[0][0])
        self._log(f"majority vote result: {predicted_label}")

        return predicted_label
