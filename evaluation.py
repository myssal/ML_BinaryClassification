import numpy as np
import pickle
import os

from config import Settings
from utils.log import ConsoleLogger as cl

settings = Settings()

def train_test_split(X, y, random_state=41, test_size=0.2):
    n_samples = X.shape[0]
    np.random.seed(random_state)

    shuffled_indices = np.random.permutation(np.arange(n_samples))

    test_size = int(n_samples * test_size)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def accuracy(y_true, y_pred):

    y_true = y_true.flatten()
    total_samples = len(y_true)

    correct_predictions = np.sum(y_true == y_pred)

    return correct_predictions / total_samples


def precision(y_true, y_pred, target_class):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tp = np.sum((y_true == target_class) & (y_pred == target_class))

    fp = np.sum((y_true != target_class) & (y_pred == target_class))

    denominator = tp + fp

    if denominator == 0:
        return 0.0

    return tp / denominator


def recall(y_true, y_pred, target_class):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tp = np.sum((y_true == target_class) & (y_pred == target_class))

    fn = np.sum((y_true == target_class) & (y_pred != target_class))

    denominator = tp + fn

    if denominator == 0:
        return 0.0

    return tp / denominator

def balanced_accuracy(y_true, y_pred):

    y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    n_classes = len(np.unique(y_true))
    sen, spec = [], []

    for i in range(n_classes):
        mask_true = y_true == i
        mask_pred = y_pred == i

        TP = np.sum(mask_true & mask_pred)
        TN = np.sum((mask_true != True) & (mask_pred != True))
        FP = np.sum((mask_true != True) & mask_pred)
        FN = np.sum(mask_true & (mask_pred != True))

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        sen.append(sensitivity)
        spec.append(specificity)

    average_sen = np.mean(sen)
    average_spec = np.mean(spec)

    balanced_acc = (average_sen + average_spec) / n_classes

    return balanced_acc


def save_model(model, file_path=settings.DECISION_TREE_MODEL):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    cl.info(f"Trained model saved to: {file_path}")


def load_model(file_path=settings.DECISION_TREE_MODEL):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    cl.info(f"Trained model loaded from: {file_path}")
    return model
