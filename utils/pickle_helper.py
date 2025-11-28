import pickle
import pprint


def show_decision_tree_pkl(file_path):
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

    def print_node(node, depth=0):
        if node is None:
            return
        indent = "  " * depth
        feature = getattr(node, "feature", None)
        threshold = getattr(node, "threshold", None)
        value = getattr(node, "value", None)
        print(f"{indent}Feature: {feature}, Threshold: {threshold}, Value: {value}")
        print_node(getattr(node, "left", None), depth + 1)
        print_node(getattr(node, "right", None), depth + 1)

    if hasattr(model, "root"):
        print(f"DecisionTree structure from '{file_path}':")
        print_node(model.root)
    else:
        print("The loaded object does not have a 'root' attribute. Cannot display tree structure.")