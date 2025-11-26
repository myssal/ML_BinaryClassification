import pandas as pd
import numpy as np
from logger.log import ConsoleLogger as cl

class FeatureSelection:
    """
    Prepare dataset for decision tree
    """

    def __init__(self, csv_input):
        self.csv_input = csv_input
        self.data_frame = self.get_dataframe()
        self.corr = self.get_correlation()
        self.selected_features = None
        self.X = None
        self.y = None

    def get_dataframe(self):
        try:
            df = pd.read_csv(self.csv_input)
            return df
        except FileNotFoundError:
            cl.error("File not found.")
        except Exception as e:
            cl.error_generic(e)
        return None

    def get_correlation(self):
        try:
            self.data_frame['diagnosis'] = (self.data_frame['diagnosis'] == 'M').astype(int)
            corr = self.data_frame.corr()
            return corr
        except Exception as e:
            cl.error_generic(e)
        return None

    def feature_correlated_selection(self, corr_threshold=0.25):
        cor_target = abs(self.corr["diagnosis"])
        relevant = cor_target[cor_target > corr_threshold]

        names = []
        values = []

        for index, value in relevant.items():
            if index != "diagnosis":
                names.append(index)
                values.append(value)

        for n, v in zip(names, values):
            cl.info(f"Feature: {n}, Correlation: {v}")

        self.selected_features = names
        return names

    def assign_training_data(self):
        if not self.selected_features:
            cl.error("No features selected.")
            return None
        try:
            self.X = self.data_frame[self.selected_features].values
            self.y = self.data_frame['diagnosis'].values.reshape(-1, 1)
            cl.info("Training data assigned.")
        except Exception as e:
            cl.error_generic(e)

    def scale_features(self):
        if self.X is None:
            cl.error("Training data not assigned.")
            return None
        mean = np.mean(self.X, axis=0)
        std = np.std(self.X, axis=0)
        self.X = (self.X - mean) / std
        cl.info("Features scaled.")
        return self.X
