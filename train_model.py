import csv
import os
from datetime import datetime

from algorithms.logistic_regression.logistic_regression import LogisticRegression
from config import Settings
from algorithms.k_nearest_neighbors.knn import KNN
from preprocessing.feature_selection import FeatureSelection
from pipeline import train_and_save_pipeline, load_and_evaluate_pipeline
from algorithms.decisiontree.decisiontree import DecisionTree
from utils.log import ConsoleLogger as cl

settings = Settings()

class Train:
    """
    class for training different machine learning models on a dataset.

    parameters:
        input_csv (str): path to the input csv file.
        corr_threshold (float): correlation threshold for feature selection.
    """
    def __init__(self, input_csv=settings.DATASET_FILE, corr_threshold=0.25):
        self.input_csv = input_csv
        self.corr_threshold = corr_threshold

        # variables to store preprocessed data for caching
        # avoids reloading or reprocessing when switching models
        self.X = None
        self.y = None
        self.fs = None

    def _prepare_data_once(self):
        """
        internal function: check if data is already loaded.
        if not, load and preprocess it. otherwise, reuse cached data.
        """
        if self.X is None or self.y is None:
            cl.info(">>> preparing input data (run once)...")
            self.fs = FeatureSelection(self.input_csv)
            self.X, self.y = self.fs.prepare_data(corr_threshold=self.corr_threshold)

            self.fs.save_processed_dataset(output_path=settings.DATASET_CLEAN_FILE)

            # optionally save feature configuration
            self.fs.save_features_config(settings.FEATURE_CONFIG)

            # save scaler parameters
            self.fs.save_scaler_params(settings.SCALER_PARAMS)
        else:
            cl.info(">>> using cached data already in memory.")

    def _run_generic_pipeline(self, model_name, model_class, params, save_path):
        """
        core function to train any model with a standard pipeline.
        """
        # ensure data is ready
        self._prepare_data_once()

        cl.info(f"\n>>> starting training: {model_name.upper()}")

        # train model and save
        X_test, y_test = train_and_save_pipeline(
            X=self.X,
            y=self.y,
            model_class=model_class,
            model_params=params,
            test_size=0.2,
            random_state=41,
            save_path=save_path
        )

        # load model and evaluate
        results = load_and_evaluate_pipeline(
            X_test=X_test,
            y_test=y_test,
            load_path=save_path
        )

        # display results
        self._display_results(model_name, params, results)

        self._log_to_csv(model_name, params, results)

    def _display_results(self, model_name, params, results):
        """
        helper to display training results in a readable format
        """
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        cl.info(f"=== results: {model_name} ({param_str}) ===")

        cl.info(f"accuracy (overall):     {results['accuracy']:.4f}")
        cl.info(f"balanced accuracy:      {results['balanced_accuracy']:.4f}")

        cl.info("\n--- class details ---")
        cl.info(
            f"class Malignant (1) -> precision: {results['precision_positive']:.4f} | recall: {results['recall_positive']:.4f}")
        cl.info(
            f"class Benign (0) -> precision: {results['precision_negative']:.4f} | recall: {results['recall_negative']:.4f}")
        cl.info("-" * 40)

    def _log_to_csv(self, model_name, params, results, filepath = settings.MODEL_COMPARISON):
        file_exists = os.path.isfile(filepath)

        with open(filepath, mode = 'a', newline='', encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                header = [
                    "Timestamp", "Parameters",
                    "Accuracy", "Balanced Acc",
                    "Precision (M)", "Recall (M)",
                    "Precision (B)", "Recall (B)"
                ]
                writer.writerow(header)
            else: writer.writerow([model_name])

            param_str = str(params).replace(',', ';')
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                param_str,
                f"{results['accuracy']:.4f}",
                f"{results['balanced_accuracy']:.4f}",
                f"{results['precision_positive']:.4f}",
                f"{results['recall_positive']:.4f}",
                f"{results['precision_negative']:.4f}",
                f"{results['recall_negative']:.4f}"
            ]

            writer.writerow(row)
            cl.info(f"Đã lưu kết quả vào file: {filepath}")

    def run_decision_tree(self):
        """
        run the decision tree model with predefined parameters
        """
        self._run_generic_pipeline(
            model_name=settings.DECISION_TREE,
            model_class=DecisionTree,
            params={"min_samples": 2, "max_depth": 2},
            save_path=settings.DECISION_TREE_MODEL
        )

    def run_knn(self):
        """
        run the k-nearest neighbors model with predefined parameters
        """
        self._run_generic_pipeline(
            model_name=settings.KNN,
            model_class=KNN,
            params={"k": 5},
            save_path=settings.KNN_MODEL
        )

    def run_logistic_regression(self):
        """
        run the logistic regression model with predefined parameters
        """
        self._run_generic_pipeline(
            model_name=settings.LOGISTIC_REGRESSION,
            model_class=LogisticRegression,
            params={"learning_rate": 0.01, "n_iters": 2000},
            save_path=settings.LOGISTIC_REGRESSION_MODEL
        )
