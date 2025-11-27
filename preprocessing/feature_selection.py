import os

import pandas as pd
import numpy as np
import json

from config import Settings
from utils.log import ConsoleLogger as cl

settings = Settings()

class FeatureSelection:
    """
    Prepare dataset for decision tree
    """

    def __init__(self, csv_input):
        self.csv_input = csv_input
        self.data_frame = self._get_dataframe()
        self.corr = self._get_correlation() # compute correlation matrix
        self.selected_features = None # list of selected feature names
        self.X = None # feature array
        self.y = None # target array

        self.mean = None
        self.std = None

    def _get_dataframe(self):
        """
        load CSV file into a pandas DataFrame.
        """
        try:
            df = pd.read_csv(self.csv_input)
            return df
        except FileNotFoundError:
            cl.error("File not found.")
        except Exception as e:
            cl.error_generic(e)
        return None

    def _get_correlation(self):
        """
        encode target column and compute correlation matrix.
        """
        try:
            self.data_frame[settings.TARGET_LABEL] = (self.data_frame[settings.TARGET_LABEL] == 'M').astype(int)
            corr = self.data_frame.corr() # correlation between features
            return corr
        except Exception as e:
            cl.error_generic(e)
        return None

    def feature_correlated_selection(self, corr_threshold=0.25):
        """
        select features with correlation above the threshold with target.
        """
        cor_target = abs(self.corr[settings.TARGET_LABEL]) # absolute correlation with target
        relevant = cor_target[cor_target > corr_threshold]

        names = []
        values = []

        for index, value in relevant.items():
            if index != settings.TARGET_LABEL:
                names.append(index)
                values.append(value)

        #for n, v in zip(names, values):
        #    cl.info(f"Feature: {n}, Correlation: {v}")

        self.selected_features = names
        return names

    def assign_training_data(self):
        """
        assign selected features to X and target to y.
        """
        if not self.selected_features:
            cl.error("No features selected.")
            return None
        try:
            self.X = self.data_frame[self.selected_features].values
            self.y = self.data_frame[settings.TARGET_LABEL].values.reshape(-1, 1)
            cl.info("Training data assigned.")
        except Exception as e:
            cl.error_generic(e)

    def scale_features(self):
        if self.X is None:
            cl.error("Training data not assigned.")
            return None
        self.mean = np.mean(self.X, axis=0) # mean per feature
        self.std = np.std(self.X, axis=0) # std per feature

        self.X = (self.X - self.mean) / self.std # standardize features
        cl.info("Features scaled.")
        return self.X

    def prepare_data(self, corr_threshold=0.25):
        """
        Preprocess pipeline: select features, assign training data, and scale features.

        Parameters:
        -----------
        corr_threshold : float
            Threshold to select features based on correlation with target (Default by 0.25).

        Returns:
        --------
        X : numpy.ndarray
            Scaled feature array
        y : numpy.ndarray
            Target array
        """
        self.feature_correlated_selection(corr_threshold=corr_threshold)
        self.assign_training_data()
        self.scale_features()
        return self.X, self.y

    def save_features_config(self, output_path=settings.FEATURE_CONFIG):
        if self.selected_features is None:
            cl.warn("Chưa có features nào được chọn để lưu!")
            return

        data_to_save = {
            "n_features": len(self.selected_features),
            "selected_columns": list(self.selected_features)
        }

        try:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False)

            cl.info(f"Đã lưu cấu hình features vào: {output_path}")

        except Exception as e:
            cl.error_generic(e)

    def save_processed_dataset(self, output_path = settings.DATASET_CLEAN_FILE):
        if self.selected_features is None:
            cl.warn("Chưa c features nào được chọn. Hãy chạt prepare_data!")
            return

        try:
            columns_to_save = list(self.selected_features)

            if settings.TARGET_LABEL not in columns_to_save:
                columns_to_save.append(settings.TARGET_LABEL)

            df_clean = self.data_frame[columns_to_save]

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            df_clean.to_csv(output_path, index = False)

            cl.info(f"Đã lưu dataset clean")
        except KeyError as e:
            cl.error(f"Lỗi: không tìm thấy cột {e} trong dữ liệu gốc.")
        except Exception as e:
            cl.error_generic(e)

    def save_scaler_params(self, output_path = settings.SCALER_PARAMS):
        if not hasattr(self, 'mean') or not hasattr(self, 'std'):
            cl.warn("Chưa scale dữ liệu, không có tham số để lưu")
            return
        data = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": list(self.selected_features)
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        cl.info(f"Đã lưu tham số Scaler vào: {output_path}")
