"""
===================================================================
ALGORITHM 5: k-NEAREST NEIGHBORS (k-NN)
Group Member: Member 5
Hyper-parameters tuned (sensitivity analysis): n_neighbors (k), weights, metric
===================================================================
All experiment logic (CV tuning, plots, metrics) lives in
experiment_utils.py so that every member is evaluated identically.
Only the model and its hyper-parameter grid are defined here.

Run alone :  python algorithms/algo5_knn.py
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline import SEED
from experiment_utils import run_experiment
from sklearn.neighbors import KNeighborsClassifier

def run_algo5():
    return run_experiment(
        member="Member 5",
        name="k-Nearest Neighbors (k-NN)",
        short="algo5_knn",
        estimator=KNeighborsClassifier(),
        param_grid={
            'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21, 25],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
        },
    )


if __name__ == "__main__":
    run_algo5()
