import pandas as pd
import numpy as np
from utils.log import ConsoleLogger as cl


class FeatureSelection:
    """
    Prepare dataset for decision tree
    """

    def __init__(self, csv_input):
        self.csv_input = csv_input
        self.data_frame = self.get_dataframe()
        self.corr = self.get_correlation() # compute correlation matrix
        self.selected_features = None # list of selected feature names
        self.X = None # feature array
        self.y = None # target array

    def get_dataframe(self):
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

    def get_correlation(self):
        """
        encode target column and compute correlation matrix.
        """
        try:
            self.data_frame['diagnosis'] = (self.data_frame['diagnosis'] == 'M').astype(int)
            corr = self.data_frame.corr() # correlation between features
            return corr
        except Exception as e:
            cl.error_generic(e)
        return None

    def feature_correlated_selection(self, corr_threshold=0.25):
        """
        select features with correlation above the threshold with target.
        """
        cor_target = abs(self.corr["diagnosis"]) # absolute correlation with target
        relevant = cor_target[cor_target > corr_threshold]

        names = []
        values = []

        for index, value in relevant.items():
            if index != "diagnosis":
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
            self.y = self.data_frame['diagnosis'].values.reshape(-1, 1)
            cl.info("Training data assigned.")
        except Exception as e:
            cl.error_generic(e)

    def scale_features(self):
        if self.X is None:
            cl.error("Training data not assigned.")
            return None
        mean = np.mean(self.X, axis=0) # mean per feature
        std = np.std(self.X, axis=0) # std per feature
        self.X = (self.X - mean) / std # standardize features
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