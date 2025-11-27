class Node:
    """
    represent a node in decision tree
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        """
        initializes a new instance of the node class.

        args:
            feature: the feature used for splitting at this node. defaults to None.
            threshold: the threshold used for splitting at this node. defaults to None.
            left: the left child node. defaults to None.
            right: the right child node. defaults to None.
            gain: the gain of the split. defaults to None.
            value: if this node is a leaf node, this attribute represents the predicted value
                for the target variable. defaults to None.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value
