"""
===================================================================
ALGORITHM 6: DECISION TREE
Group Member: Member 6
Hyper-parameters tuned (sensitivity analysis): max_depth, criterion, min_samples_split
===================================================================
All experiment logic (CV tuning, plots, metrics) lives in
experiment_utils.py so that every member is evaluated identically.
Only the model and its hyper-parameter grid are defined here.

Run alone :  python algorithms/algo6_decision_tree.py
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline import SEED
from experiment_utils import run_experiment
from sklearn.tree import DecisionTreeClassifier

def run_algo6():
    return run_experiment(
        member="Member 6",
        name="Decision Tree",
        short="algo6_dt",
        estimator=DecisionTreeClassifier(random_state=SEED),
        param_grid={
            'max_depth': [1, 2, 3, 4, 5, 6, 8, 10, None],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 5, 10, 20],
        },
    )


if __name__ == "__main__":
    run_algo6()
