# Decision Tree

Implement the decision tree algorithm for a binary classification problem ([breast cancer prediction](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data)).

## Project structure
```aiignore
│   config.py # project configuration
│   evaluation.py # measure the well-trained of model when performing on unseen data
│   main.py # main entry of project
│   pipeline.py  # pipeline for specific task (e.g train model, load model and evaluate...)
│   train_model.py # central class for running algorithm in /algorithms
│
├───algorithms
│   ├───decisiontree
│   │   │   decisiontree.py 
│   │   │   node.py # helper class for decision tree
│   │
│   ├───k_nearest_neighbors
│   │   │   knn.py
│   │
│   └───logistic_regression
│       │   logistic_regression.py
│
├───data
│   │   breast-cancer.csv # base dataset
│   │
│   ├───clean_data # cleaned dataset after preprocessing
│   │
│   ├───result # trained models, comparision file etc..
│   │
│   └───test
│           test.json # test data for model
│
├───preprocessing
│   │   feature_selection.py # preprocess dataset for algorithms
│
└───utils
    │   log.py # custom console log with tag
    │   pickle_helper.py # .pkl file parser
```
## Installation:
```aiignore
git clone https://github.com/myssal/DecisionTree.git
cd DecisionTree

poetry install
poetry shell
```
