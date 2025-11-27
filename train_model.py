from algorithms.logistic_regression.logistic_regression import LogisticRegression
from config import Settings
from algorithms.k_nearest_neighbors.knn import KNN
from preprocessing.feature_selection import FeatureSelection
from pipeline import train_and_save_pipeline, load_and_evaluate_pipeline
from algorithms.decisiontree.decisiontree import DecisionTree
from utils.log import ConsoleLogger as cl

settings = Settings()

class Train:
    def __init__(self, input_csv=settings.DATASET_FILE, corr_threshold=0.25):
        """
        Khởi tạo class Train.
        :param input_csv: Đường dẫn file csv.
        :param corr_threshold: Ngưỡng tương quan để lọc feature.
        """
        self.input_csv = input_csv
        self.corr_threshold = corr_threshold

        # Biến lưu trữ dữ liệu đã xử lý để dùng chung (Caching)
        # Giúp không phải đọc lại file mỗi khi đổi thuật toán
        self.X = None
        self.y = None
        self.fs = None

    def _prepare_data_once(self):
        """
        Hàm nội bộ: Kiểm tra xem dữ liệu đã được load chưa.
        Nếu chưa thì load và xử lý. Nếu rồi thì bỏ qua.
        """
        if self.X is None or self.y is None:
            cl.info(">>> Đang chuẩn bị dữ liệu đầu vào (Chạy 1 lần)...")
            self.fs = FeatureSelection(self.input_csv)
            self.X, self.y = self.fs.prepare_data(corr_threshold=self.corr_threshold)

            self.fs.save_processed_dataset(output_path=settings.DATASET_CLEAN_FILE)

            # (Tùy chọn) Lưu config feature
            self.fs.save_features_config(settings.FEATURE_CONFIG)

            self.fs.save_scaler_params(settings.SCALER_PARAMS)
        else:
            cl.info(">>> Sử dụng lại dữ liệu đã cache trong bộ nhớ.")

    def _run_generic_pipeline(self, model_name, model_class, params, save_path):
        """
        Hàm cốt lõi (Core Logic): Chạy quy trình train cho BẤT KỲ model nào.
        """
        # 1. Đảm bảo dữ liệu đã có
        self._prepare_data_once()

        cl.info(f"\n>>> BẮT ĐẦU HUẤN LUYỆN: {model_name.upper()}")


        # 2. Train và Save
        X_test, y_test = train_and_save_pipeline(
            X=self.X,
            y=self.y,
            model_class=model_class,
            model_params=params,
            test_size=0.2,
            random_state=41,
            save_path=save_path
        )

        # 3. Load và Evaluate
        results = load_and_evaluate_pipeline(
            X_test=X_test,
            y_test=y_test,
            load_path=save_path
        )

        # 4. In kết quả
        self._display_results(model_name, params, results)

    def _display_results(self, model_name, params, results):
        """Hàm nội bộ để in kết quả đẹp mắt"""
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        cl.info(f"=== KẾT QUẢ: {model_name} ({param_str}) ===")

        cl.info(f"Accuracy (Tổng):     {results['accuracy']:.4f}")
        cl.info(f"Balanced Accuracy:   {results['balanced_accuracy']:.4f}")

        cl.info("\n--- Chi tiết từng lớp ---")
        cl.info(
            f"Lớp ÁC TÍNH (1) -> Precision: {results['precision_positive']:.4f} | Recall: {results['recall_positive']:.4f}")
        cl.info(
            f"Lớp LÀNH TÍNH (0) -> Precision: {results['precision_negative']:.4f} | Recall: {results['recall_negative']:.4f}")
        cl.info("-" * 40)

    def run_decision_tree(self):
        self._run_generic_pipeline(
            model_name=settings.DECISION_TREE,
            model_class=DecisionTree,
            params={"min_samples": 2, "max_depth": 2},
            save_path=settings.DECISION_TREE_MODEL
        )

    def run_knn(self):
        self._run_generic_pipeline(
            model_name=settings.KNN,
            model_class=KNN,
            params={"k": 5},
            save_path = settings.KNN_MODEL
        )

    def run_logistic_regression(self):
        self._run_generic_pipeline(
            model_name = settings.LOGISTIC_REGRESSION,
            model_class = LogisticRegression,
            params = {"learning_rate": 0.01, "n_iters": 2000},
            save_path = settings.LOGISTIC_REGRESSION_MODEL
        )