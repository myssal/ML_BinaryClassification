# Decision Tree

Implement the decision tree algorithm for a binary classification problem ([breast cancer prediction](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data)).

## Project structure
```aiignore
│   evaluation.py # measure the well-trained of model when performing on unseen data
│   main.py # main entry of project
│   pipeline.py # pipeline for specific task (e.g train model, load model and evaluate...)
│
├───data
│   │   breast-cancer.csv # dataset
│ 
├───decisiontree
│   │   decisiontree.py # decision tree implementation from scratch 
│   │   node.py # helper node class
│
├───preprocessing
│   │   feature_selection.py # preprocess dataset for algorithm 
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
