from evaluation import train_test_split, accuracy, balanced_accuracy, save_model, load_model


def train_and_save_pipeline(X, y, model_class, model_params=None, test_size=0.2, random_state=41,
                            save_path="./data/result/trained_data.pkl"):
    """
    split dataset, train the model on training set, and save the trained model.

    parameters:
    -----------
    X : numpy.ndarray
        feature array.
    y : numpy.ndarray
        target array.
    model_class : class
        the model class to instantiate (e.g., DecisionTree).
    model_params : dict
        parameters to initialize the model.
    test_size : float
        proportion of dataset to reserve as test set.
    random_state : int
        random seed for reproducibility.
    save_path : str
        file path to save the trained model.

    returns:
    --------
    X_test, y_test : numpy.ndarray
        test set features and labels for later evaluation.
    """
    if model_params is None:
        model_params = {}

    # split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # initialize the model with provided parameters
    model = model_class(**model_params)

    # train the model on the training set
    model.fit(X_train, y_train)

    # save the trained model to disk
    save_model(model, file_path=save_path)

    return X_test, y_test  # return test set for evaluation


def load_and_evaluate_pipeline(X_test, y_test, load_path="./data/result/trained_data.pkl"):
    """
    load a trained model from file, predict on test set, and evaluate performance.

    parameters:
    -----------
    X_test : numpy.ndarray
        feature array of the test set.
    y_test : numpy.ndarray
        true labels for the test set.
    load_path : str
        path to the saved trained model (.pkl file).

    returns:
    --------
    dict :
        dictionary containing:
        - 'accuracy': float
        - 'balanced_accuracy': float
        - 'predictions': numpy.ndarray of predicted labels
    """
    # load the trained model from file
    model = load_model(file_path=load_path)

    # predict labels for the test set
    predictions = model.predict(X_test)

    # calculate evaluation metrics
    acc = accuracy(y_test, predictions)
    bal_acc = balanced_accuracy(y_test, predictions)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "predictions": predictions
    }
