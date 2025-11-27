from config import Settings
from test import ModelTester
from train_model import Train

if __name__ == "__main__":

    settings = Settings()

    train = Train()
    train.run_knn()
    train.run_decision_tree()
    train.run_logistic_regression()

    # Đường dẫn file JSON
    # tester = ModelTester(settings.TEST_FILE, settings.SCALER_PARAMS)
    #
    # # Test Decision Tree
    # tester.run_test(settings.DECISION_TREE_MODEL, settings.DECISION_TREE)
    #
    # # Test KNN
    # tester.run_test(settings.KNN_MODEL, settings.KNN)
    #
    # # Test Logistic Regression
    # tester.run_test(settings.LOGISTIC_REGRESSION_MODEL, settings.LOGISTIC_REGRESSION)