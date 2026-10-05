"""
===================================================================
ALGORITHM 1: NEURAL NETWORK (MLP)
Group Member: Member 1
Hyper-parameters tuned (sensitivity analysis): hidden_layer_sizes (layers/neurons), activation, alpha, learning_rate_init
===================================================================
All experiment logic (CV tuning, plots, metrics) lives in
experiment_utils.py so that every member is evaluated identically.
Only the model and its hyper-parameter grid are defined here.

Run alone :  python algorithms/algo1_neural_network.py
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline import SEED
from experiment_utils import run_experiment
from sklearn.neural_network import MLPClassifier

def run_algo1():
    return run_experiment(
        member="Member 1 (You)",
        name="Neural Network (MLP)",
        short="algo1_mlp",
        estimator=MLPClassifier(max_iter=2000, random_state=SEED),
        param_grid={
            'hidden_layer_sizes': [(10,), (50,), (100,), (50, 25), (100, 50)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.01, 1.0],
            'learning_rate_init': [0.001, 0.01],
        },
    )


if __name__ == "__main__":
    run_algo1()
