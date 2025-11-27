from .node import Node
import numpy as np

class DecisionTree:
    """
    a decision tree classifier for binary classification problems.
    """

    def __init__(self, min_samples=2, max_depth=2, verbose_log=False):
        """
        constructor for decisiontree class.

        parameters:
            min_samples (int): minimum number of samples required to split an internal node.
            max_depth (int): maximum depth of the decision tree.
            verbose_log (bool): whether to print detailed logs during operation.
        """
        self.root = None
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.verbose_log = verbose_log
        self._log(f"DecisionTree initialized with min_samples={self.min_samples}, max_depth={self.max_depth}.")

    def _log(self, message):
        """
        helper method to print log messages if verbose_log is enabled.
        """
        if self.verbose_log:
            print(f"[DecisionTree] {message}")

    def split_data(self, dataset, feature, threshold):
        """
        splits the given dataset into two datasets based on the given feature and threshold.

        parameters:
            dataset (ndarray): input dataset.
            feature (int): index of the feature to be split on.
            threshold (float): threshold value to split the feature on.

        returns:
            left_dataset (ndarray): subset of the dataset with values less than or equal to the threshold.
            right_dataset (ndarray): subset of the dataset with values greater than the threshold.
        """
        left_dataset = []
        right_dataset = []

        # loop over each row in the dataset and split based on the given feature and threshold
        for row in dataset:
            if row[feature] <= threshold:
                left_dataset.append(row)
            else:
                right_dataset.append(row)

        # convert lists to numpy arrays
        left_dataset = np.array(left_dataset)
        right_dataset = np.array(right_dataset)
        return left_dataset, right_dataset

    def entropy(self, y):
        """
        computes the entropy of the given label values.

        parameters:
            y (ndarray): input label values.

        returns:
            entropy (float): entropy of the given label values.
        """
        entropy = 0
        labels = np.unique(y)
        for label in labels:
            label_examples = y[y == label]
            pl = len(label_examples) / len(y)  # probability of the label
            entropy += -pl * np.log2(pl)  # accumulate entropy
        self._log(f"  Calculated entropy: {entropy:.4f}.")
        return entropy

    def information_gain(self, parent, left, right):
        """
        computes the information gain from splitting the parent dataset into two datasets.

        parameters:
            parent (ndarray): input parent dataset.
            left (ndarray): subset of the parent dataset after split on a feature.
            right (ndarray): subset of the parent dataset after split on a feature.

        returns:
            information_gain (float): information gain of the split.
        """
        parent_entropy = self.entropy(parent)
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        entropy_left, entropy_right = self.entropy(left), self.entropy(right)
        weighted_entropy = weight_left * entropy_left + weight_right * entropy_right
        gain = parent_entropy - weighted_entropy
        self._log(f"  Calculated information gain: {gain:.4f}.")
        return gain

    def best_split(self, dataset, num_samples, num_features):
        """
        finds the best split for the given dataset.

        parameters:
            dataset (ndarray): the dataset to split.
            num_samples (int): number of samples in the dataset.
            num_features (int): number of features in the dataset.

        returns:
            dict: dictionary with best split feature, threshold, gain, left and right datasets.
        """
        best_split = {'gain': -1, 'feature': None, 'threshold': None}
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                left_dataset, right_dataset = self.split_data(dataset, feature_index, threshold)
                if len(left_dataset) and len(right_dataset):
                    y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    gain = self.information_gain(y, left_y, right_y)
                    if gain > best_split["gain"]:
                        best_split.update({
                            "feature": feature_index,
                            "threshold": threshold,
                            "left_dataset": left_dataset,
                            "right_dataset": right_dataset,
                            "gain": gain
                        })
                        self._log(f"  Found better split: feature={feature_index}, threshold={threshold:.4f}, gain={gain:.4f}.")
        return best_split

    def calculate_leaf_value(self, y):
        """
        calculates the most occurring value in y.

        parameters:
            y (list): the list of y values.

        returns:
            most_occuring_value: the most frequent label in y.
        """
        y = list(y)
        leaf_value = max(y, key=y.count)
        self._log(f"  Calculated leaf value: {leaf_value}.")
        return leaf_value

    def build_tree(self, dataset, current_depth=0):
        """
        recursively builds a decision tree from the given dataset.

        parameters:
            dataset (ndarray): dataset to build the tree from.
            current_depth (int): current depth of the tree.

        returns:
            Node: root node of the built tree.
        """
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape
        self._log(f"Building tree at depth {current_depth} with {n_samples} samples.")

        # stopping conditions
        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            best = self.best_split(dataset, n_samples, n_features)
            if best["gain"] > 0: # Ensure gain is positive before splitting
                self._log(f"  Split found at depth {current_depth}: feature={best['feature']}, threshold={best['threshold']:.4f}, gain={best['gain']:.4f}.")
                left_node = self.build_tree(best["left_dataset"], current_depth + 1)
                right_node = self.build_tree(best["right_dataset"], current_depth + 1)
                return Node(best["feature"], best["threshold"], left_node, right_node, best["gain"])
            else:
                self._log(f"  No further split provides positive gain at depth {current_depth}. Creating leaf node.")

        leaf_value = self.calculate_leaf_value(y)
        self._log(f"  Creating leaf node at depth {current_depth} with value {leaf_value}.")
        return Node(value=leaf_value)

    def fit(self, X, y):
        """
        builds and fits the decision tree to X and y.

        parameters:
            X (ndarray): feature matrix.
            y (ndarray): target values.
        """
        self._log(f"Starting fit with X_shape={X.shape}, y_shape={y.shape}.")
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(dataset)
        self._log("Fit complete. Decision tree built.")

    def predict(self, X):
        """
        predicts class labels for each instance in X.

        parameters:
            X (ndarray): feature matrix to predict.

        returns:
            list: predicted class labels.
        """
        self._log(f"Starting prediction for {X.shape[0]} samples.")
        predictions = [self.make_prediction(x, self.root) for x in X]
        self._log("Prediction complete.")
        return np.array(predictions)

    def make_prediction(self, x, node):
        """
        traverses the tree to predict the target value for a single feature vector.

        parameters:
            x (ndarray): feature vector.
            node (Node): current node.

        returns:
            predicted label for the feature vector.
        """
        if node.value is not None:  # leaf node
            self._log(f"    Reached leaf node with value {node.value}.")
            return node.value
        
        self._log(f"    Traversing node: feature={node.feature}, threshold={node.threshold:.4f}.")
        feature = x[node.feature]
        if feature <= node.threshold:
            self._log(f"    Feature value {feature:.4f} <= threshold {node.threshold:.4f}. Going left.")
            return self.make_prediction(x, node.left)
        else:
            self._log(f"    Feature value {feature:.4f} > threshold {node.threshold:.4f}. Going right.")
            return self.make_prediction(x, node.right)
