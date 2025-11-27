from .node import Node
import numpy as np

class DecisionTree:
    """
    a decision tree classifier for binary classification problems.
    """

    def __init__(self, min_samples=2, max_depth=2):
        """
        constructor for decisiontree class.

        parameters:
            min_samples (int): minimum number of samples required to split an internal node.
            max_depth (int): maximum depth of the decision tree.
        """
        self.root = None
        self.min_samples = min_samples
        self.max_depth = max_depth

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
        return parent_entropy - weighted_entropy

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
        return max(y, key=y.count)

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

        # stopping conditions
        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            best = self.best_split(dataset, n_samples, n_features)
            if best["gain"]:
                left_node = self.build_tree(best["left_dataset"], current_depth + 1)
                right_node = self.build_tree(best["right_dataset"], current_depth + 1)
                return Node(best["feature"], best["threshold"], left_node, right_node, best["gain"])

        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)

    def fit(self, X, y):
        """
        builds and fits the decision tree to X and y.

        parameters:
            X (ndarray): feature matrix.
            y (ndarray): target values.
        """
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        """
        predicts class labels for each instance in X.

        parameters:
            X (ndarray): feature matrix to predict.

        returns:
            list: predicted class labels.
        """
        predictions = [self.make_prediction(x, self.root) for x in X]
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
            return node.value
        feature = x[node.feature]
        if feature <= node.threshold:
            return self.make_prediction(x, node.left)
        else:
            return self.make_prediction(x, node.right)
