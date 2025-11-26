from preprocessing.feature_selection import FeatureSelection
from pipeline import train_and_save_pipeline, load_and_evaluate_pipeline
from decisiontree.decisiontree import DecisionTree
from utils.log import ConsoleLogger as cl
import utils.pickle_helper as ph

# step 0: prepare your dataset
input_csv = r"./data/breast-cancer.csv"

# initialize FeatureSelection and prepare X, y
fs = FeatureSelection(input_csv)
X, y = fs.prepare_data(corr_threshold=0.25)  # select features, assign training data, scale features


# step 1: train and save model from train set (20% of original dataset)
# split the dataset, train and save the trained model
X_test, y_test = train_and_save_pipeline(
    X=X,
    y=y,
    model_class=DecisionTree,
    model_params={"min_samples": 2, "max_depth": 2},  # model hyperparameters
    test_size=0.2,        # train set size
    random_state=41,
    save_path="./data/result/trained_data.pkl"
)

# ph.show_decision_tree_pkl(r"./data/result/trained_data.pkl")


# step 2: load trained model and evaluate
# load the trained model from disk, predict on the test set, and calculate metrics
results = load_and_evaluate_pipeline(
    X_test=X_test,
    y_test=y_test,
    load_path="./data/result/trained_data.pkl"
)


# step 3: display results
cl.info(f"Accuracy on test set: {results['accuracy']}")
cl.info(f"Balanced Accuracy on test set: {results['balanced_accuracy']}")
