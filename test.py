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
            cl.error(f"Lỗi: không tìm thấy file {path}")
            sys.exit(1)

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _convert_data_to_array(self, data_dict, features_order):
        data_list = []
        try:
            for feature in features_order:
                val = data_dict.get(feature)
                if val is None:
                    cl.warn(f"Cảnh báo: Thiếu thuộc tính '{feature}' trong dữ liệu test!")
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
            cl.warn("Cảnh báo: Không có thông tin Scaler. Dữ liệu có thể sai lệch.")
            return raw_data_array

        mean = np.array(self.scaler_params["mean"])
        std = np.array(self.scaler_params["std"])

        scaled_data = (raw_data_array - mean) / std
        return scaled_data


    def run_test(self, model_path, model_name):
        cl.info(f"\n{'='*20} TESTING {model_name.upper()} {'='*20}")

        if not os.path.exists(model_path):
            cl.error(f"Không tìm thấy file model: {model_path}")
            return

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        features_order = self.test_data["features_order"]
        test_cases = self.test_data["test_cases"]

        correct_count = 0

        for case in test_cases:
            cl.info(f"\n>>> Case ID: {case['id']}")
            cl.info(f"    Mô tả: {case['note']}")

            X_raw = self._convert_data_to_array(case["data"], features_order)

            X_input = self._preprocess_input(X_raw)

            try:
                # Dự đoán
                prediction = model.predict(X_input)
                pred_val = prediction[0]

                # So sánh kết quả
                expected = case['expected_value']
                is_correct = (pred_val == expected)

                result_str = "CHÍNH XÁC" if is_correct else "SAI"
                if is_correct: correct_count += 1

                label_map = {1: "Ác tính (M)", 0: "Lành tính (B)"}
                cl.info(f"    Dự đoán: {pred_val} -> {label_map.get(pred_val, 'Unknown')}")
                cl.info(f"    Thực tế: {expected} -> {label_map.get(expected, 'Unknown')}")
                cl.info(f"    Kết luận: {result_str}")

            except Exception as e:
                cl.error(f"    Lỗi khi dự đoán: {e}")
                cl.error(f"    Số feature đầu vào: {X_input.shape[1]}")
