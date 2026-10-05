"""
===================================================================
ALGORITHM 2: SUPPORT VECTOR MACHINE (SVM)
Group Member: Member 2
Hyper-parameters tuned (sensitivity analysis): kernel, C, gamma (rbf/poly), degree (poly)
===================================================================
All experiment logic (CV tuning, plots, metrics) lives in
experiment_utils.py so that every member is evaluated identically.
Only the model and its hyper-parameter grid are defined here.

Run alone :  python algorithms/algo2_svm.py
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline import SEED
from experiment_utils import run_experiment
from sklearn.svm import SVC

def run_algo2():
    return run_experiment(
        member="Member 2",
        name="Support Vector Machine (SVM)",
        short="algo2_svm",
        estimator=SVC(random_state=SEED),
        param_grid=[
            {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 5, 10, 50, 100]},
            {'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 5, 10, 50, 100],
             'gamma': ['scale', 0.001, 0.01, 0.1]},
            {'kernel': ['poly'], 'C': [0.01, 0.1, 1, 5, 10, 50, 100],
             'gamma': ['scale'], 'degree': [2, 3]},
        ],
    )


if __name__ == "__main__":
    run_algo2()
