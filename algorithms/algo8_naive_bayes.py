"""
===================================================================
ALGORITHM 8: GAUSSIAN NAIVE BAYES
Group Member: Member 8
Hyper-parameters tuned (sensitivity analysis): var_smoothing (the only meaningful hyper-parameter of GaussianNB)
===================================================================
All experiment logic (CV tuning, plots, metrics) lives in
experiment_utils.py so that every member is evaluated identically.
Only the model and its hyper-parameter grid are defined here.

Run alone :  python algorithms/algo8_naive_bayes.py
"""
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline import SEED
from experiment_utils import run_experiment
from sklearn.naive_bayes import GaussianNB

def run_algo8():
    return run_experiment(
        member="Member 8",
        name="Gaussian Naive Bayes",
        short="algo8_gnb",
        estimator=GaussianNB(),
        param_grid={
            'var_smoothing': [float(v) for v in np.logspace(-12, 1, 14)],
        },
    )


if __name__ == "__main__":
    run_algo8()
