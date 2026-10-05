"""
===================================================================
ALGORITHM 3: RANDOM FOREST
Group Member: Member 3
Hyper-parameters tuned (sensitivity analysis): n_estimators, max_depth, max_features
===================================================================
All experiment logic (CV tuning, plots, metrics) lives in
experiment_utils.py so that every member is evaluated identically.
Only the model and its hyper-parameter grid are defined here.

Run alone :  python algorithms/algo3_random_forest.py
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline import SEED
from experiment_utils import run_experiment
from sklearn.ensemble import RandomForestClassifier

def run_algo3():
    return run_experiment(
        member="Member 3",
        name="Random Forest",
        short="algo3_rf",
        estimator=RandomForestClassifier(random_state=SEED),
        param_grid={
            'n_estimators': [10, 50, 100, 200, 300],
            'max_depth': [3, 5, 8, None],
            'max_features': ['sqrt', 'log2'],
        },
    )


if __name__ == "__main__":
    run_algo3()
