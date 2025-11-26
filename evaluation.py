import numpy as np
import pickle
import os
from utils.log import ConsoleLogger as cl

def train_test_split(X, y, random_state=41, test_size=0.2):
    """
    split the dataset into training and testing sets.

    parameters:
    -----------
    X : numpy.ndarray
        feature array.
    y : numpy.ndarray
        target array.
    random_state : int
        seed for random shuffling.
    test_size : float
        proportion of samples to use as test set.

    returns:
    --------
    X_train, X_test, y_train, y_test : numpy.ndarray
        splitted training and testing features and labels.
    """
    n_samples = X.shape[0]
    np.random.seed(random_state)

    # shuffle indices randomly
    shuffled_indices = np.random.permutation(np.arange(n_samples))

    # calculate train and test value set
    test_size = int(n_samples * test_size)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def accuracy(y_true, y_pred):
    """
    compute the accuracy of predictions.

    parameters:
    -----------
    y_true : numpy.ndarray
        true labels.
    y_pred : numpy.ndarray
        predicted labels.

    returns:
    --------
    float : accuracy score (correct predictions / total samples)
    """

    y_true = y_true.flatten()
    total_samples = len(y_true)

    correct_predictions = np.sum(y_true == y_pred)

    return correct_predictions / total_samples


def balanced_accuracy(y_true, y_pred):
    """
    compute the balanced accuracy for multi-class classification.

    parameters:
    -----------
    y_true : numpy.ndarray
        true labels.
    y_pred : numpy.ndarray
        predicted labels.

    returns:
    --------
    float : balanced accuracy (average of sensitivity and specificity per class)
    """
    y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    n_classes = len(np.unique(y_true))
    sen, spec = [], []

    for i in range(n_classes):
        mask_true = y_true == i
        mask_pred = y_pred == i

        # calculate true positives, true negatives, false positives, false negatives
        TP = np.sum(mask_true & mask_pred)
        TN = np.sum((mask_true != True) & (mask_pred != True))
        FP = np.sum((mask_true != True) & mask_pred)
        FN = np.sum(mask_true & (mask_pred != True))

        # calculate sensitivity (recall) and specificity for class i
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        sen.append(sensitivity)
        spec.append(specificity)

    # average sensitivity and specifity across classes
    average_sen = np.mean(sen)
    average_spec = np.mean(spec)

    balanced_acc = (average_sen + average_spec) / n_classes # final balanced accuracy

    return balanced_acc


def save_model(model, file_path="./data/result/trained_data.pkl"):
    """
    save a python object (e.g., trained model) to a pickle file.

    parameters:
    -----------
    model : object
        the object to be saved.
    file_path : str
        path where the pickle file will be stored.
    """

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    cl.info(f"Trained model saved to: {file_path}")


def load_model(file_path="./data/result/trained_data.pkl"):
    """
    load a python object (e.g., trained model) from a pickle file.

    parameters:
    -----------
    file_path : str
        path to the pickle file.

    returns:
    --------
    object : the loaded object
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    cl.info(f"Trained model loaded from: {file_path}")
    return model
