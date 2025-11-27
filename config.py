from functools import lru_cache

class Settings:
    TEST_FILE = r"data/test/test.json"
    DATASET_FILE = r"data/breast-cancer.csv"
    DATASET_CLEAN_FILE = r"data/clean_data/breast_cancer_clean.csv"

    DECISION_TREE = "Decision Tree"
    KNN = "K-Nearest Neighbors"
    LOGISTIC_REGRESSION = "Logistic Regression"

    DECISION_TREE_MODEL = r"./data/result/decisiontree_model.pkl"
    KNN_MODEL = r"./data/result/knn_model.pkl"
    LOGISTIC_REGRESSION_MODEL = r"./data/result/logistic_model.pkl"

    SCALER_PARAMS = r"./data/result/scaler_params.json"
    FEATURE_CONFIG = r"./data/result/feature_config.json"


    TARGET_LABEL = "diagnosis"

@lru_cache
def get_setting() -> Settings:
    return Settings()