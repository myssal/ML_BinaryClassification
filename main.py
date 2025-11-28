from config import Settings
from test import ModelTester
from train_model import Train
from utils.log import ConsoleLogger as cl
if __name__ == "__main__":

    settings = Settings()

    train = Train(verbose_log=True)

    #train.run_knn()
    train.run_decision_tree()
    #train.run_logistic_regression()


    # tester = ModelTester(settings.TEST_FILE, settings.SCALER_PARAMS)
    # tester.run_test(settings.DECISION_TREE_MODEL, settings.DECISION_TREE)
    # tester.run_test(settings.KNN_MODEL, settings.KNN)
    # tester.run_test(settings.LOGISTIC_REGRESSION_MODEL, settings.LOGISTIC_REGRESSION)