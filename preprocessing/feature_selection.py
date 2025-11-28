import os

import pandas as pd
import numpy as np
import json

from config import Settings
from utils.log import ConsoleLogger as cl

settings = Settings()

class FeatureSelection:

    def __init__(self, csv_input):
        self.csv_input = csv_input
        self.data_frame = self._get_dataframe()
        self.corr = self._get_correlation()
        self.selected_features = None
        self.X = None
        self.y = None

        self.mean = None
        self.std = None

    def _get_dataframe(self):
        try:
            df = pd.read_csv(self.csv_input)
            return df
        except FileNotFoundError:
            cl.error("File not found.")
        except Exception as e:
            cl.error_generic(e)
        return None

    def _get_correlation(self):
        try:
            self.data_frame[settings.TARGET_LABEL] = (self.data_frame[settings.TARGET_LABEL] == 'M').astype(int)
            corr = self.data_frame.corr()
            return corr
        except Exception as e:
            cl.error_generic(e)
        return None

    def feature_correlated_selection(self, corr_threshold=settings.CORRELATION_THRESHOLD):
        cor_target = abs(self.corr[settings.TARGET_LABEL])
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
        self.mean = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis=0)

        self.X = (self.X - self.mean) / self.std
        cl.info("Features scaled.")
        return self.X

    def prepare_data(self, corr_threshold=settings.CORRELATION_THRESHOLD):
        self.feature_correlated_selection(corr_threshold=corr_threshold)
        self.assign_training_data()
        self.scale_features()
        return self.X, self.y

    def save_features_config(self, output_path=settings.FEATURE_CONFIG):
        if self.selected_features is None:
            cl.warn("No feature selected to save!")
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

            cl.info(f"Features config saved to: {output_path}")

        except Exception as e:
            cl.error_generic(e)

    def save_processed_dataset(self, output_path = settings.DATASET_CLEAN_FILE):
        if self.selected_features is None:
            cl.warn("No features selected, please run feature_data()")
            return

        try:
            columns_to_save = list(self.selected_features)

            if settings.TARGET_LABEL not in columns_to_save:
                columns_to_save.append(settings.TARGET_LABEL)

            df_clean = self.data_frame[columns_to_save]

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            df_clean.to_csv(output_path, index = False)

            cl.info(f"Saved cleaned dataset.")
        except KeyError as e:
            cl.error(f"Column {e} not found in original dataset.")
        except Exception as e:
            cl.error_generic(e)

    def save_scaler_params(self, output_path = settings.SCALER_PARAMS):
        if not hasattr(self, 'mean') or not hasattr(self, 'std'):
            cl.warn("Data hasn't been scaled, no parameters to save")
            return
        data = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": list(self.selected_features)
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        cl.info(f"Scaler params saved to: {output_path}")
