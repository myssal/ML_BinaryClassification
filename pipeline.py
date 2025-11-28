from config import Settings
from evaluation import train_test_split, accuracy, balanced_accuracy, precision, recall, save_model, load_model

settings = Settings()

def train_and_save_pipeline(X, y, model_class, model_params=None, test_size=0.2, random_state=41,
                            save_path=settings.DECISION_TREE_MODEL):
    if model_params is None:
        model_params = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = model_class(**model_params)

    model.fit(X_train, y_train)

    save_model(model, file_path=save_path)

    return X_test, y_test


def load_and_evaluate_pipeline(X_test, y_test, load_path=settings.DECISION_TREE_MODEL):
    model = load_model(file_path=load_path)

    predictions = model.predict(X_test)

    acc = accuracy(y_test, predictions)
    bal_acc = balanced_accuracy(y_test, predictions)

    # prec_maligant
    prec_1 = precision(y_test, predictions, target_class=1)
    rec_1 = recall(y_test, predictions, target_class=1)

    # prec_benign
    prec_0 = precision(y_test, predictions, target_class=0)
    rec_0 = recall(y_test, predictions, target_class=0)
    # ---------------------

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_positive": prec_1,
        "recall_positive": rec_1,
        "precision_negative": prec_0,
        "recall_negative": rec_0,
        "predictions": predictions
    }
