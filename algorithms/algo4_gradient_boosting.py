"""
===================================================================
ALGORITHM 4: GRADIENT BOOSTING
Group Member: Member 4
Hyper-parameters tuned (sensitivity analysis): n_estimators, learning_rate, max_depth
===================================================================
All experiment logic (CV tuning, plots, metrics) lives in
experiment_utils.py so that every member is evaluated identically.
Only the model and its hyper-parameter grid are defined here.

Run alone :  python algorithms/algo4_gradient_boosting.py
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline import SEED
from experiment_utils import run_experiment
from sklearn.ensemble import GradientBoostingClassifier

def run_algo4():
    return run_experiment(
        member="Member 4",
        name="Gradient Boosting",
        short="algo4_gb",
        estimator=GradientBoostingClassifier(random_state=SEED),
        param_grid={
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
            'max_depth': [2, 3, 4],
        },
    )


if __name__ == "__main__":
    run_algo4()
