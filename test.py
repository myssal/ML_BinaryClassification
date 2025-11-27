import json
import os
import sys
import pickle

import numpy as np

from train_model import settings
from utils.log import ConsoleLogger as cl

class ModelTester:
    def __init__(self, json_path = settings.TEST_FILE, scaler_path = settings.SCALER_PARAMS):
        self.json_path = json_path
        self.scaler_path = scaler_path
        self.test_data = self._load_json(self.json_path)
        self.scaler_params = self._load_json(self.scaler_path)

    def _load_json(self, path):
        if not os.path.exists(path):
            cl.error(f"File {path} not found.")
            sys.exit(1)

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _convert_data_to_array(self, data_dict, features_order):
        data_list = []
        try:
            for feature in features_order:
                val = data_dict.get(feature)
                if val is None:
                    cl.warn(f"Feature '{feature}' not found in test dataset!")
                    val = 0.0
                data_list.append(val)
        except Exception as e:
            cl.error_generic(e)
            return None

        return np.array(data_list).reshape(1, -1)

    def _preprocess_input(self, raw_data_array):
        """
            Chuần hoá dữ liệu đầu vào: (X - mean) /std
        """
        if self.scaler_params is None:
            cl.warn("No scaler info, result maybe wrong.")
            return raw_data_array

        mean = np.array(self.scaler_params["mean"])
        std = np.array(self.scaler_params["std"])

        scaled_data = (raw_data_array - mean) / std
        return scaled_data


    def run_test(self, model_path, model_name):
        cl.info(f"\n{'='*20} TESTING {model_name.upper()} {'='*20}")

        if not os.path.exists(model_path):
            cl.error(f"No model found in: {model_path}")
            return

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        features_order = self.test_data["features_order"]
        test_cases = self.test_data["test_cases"]

        correct_count = 0

        for case in test_cases:
            cl.info(f"\n>>> Case ID: {case['id']}")
            cl.info(f"    Description: {case['note']}")

            X_raw = self._convert_data_to_array(case["data"], features_order)

            X_input = self._preprocess_input(X_raw)

            try:
                # Predict
                prediction = model.predict(X_input)
                pred_val = prediction[0]

                # Compare value
                expected = case['expected_value']
                is_correct = (pred_val == expected)

                result_str = "Correct" if is_correct else "Wrong"
                if is_correct: correct_count += 1

                label_map = {1: "Malignant (M)", 0: "Benign (B)"}
                cl.info(f"    Prediction: {pred_val} -> {label_map.get(pred_val, 'Unknown')}")
                cl.info(f"    Actual: {expected} -> {label_map.get(expected, 'Unknown')}")
                cl.info(f"    Result: {result_str}")

            except Exception as e:
                cl.error(f"    Error predicting: {e}")
                cl.error(f"    Features input number: {X_input.shape[1]}")
