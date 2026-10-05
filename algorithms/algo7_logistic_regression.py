"""
===================================================================
ALGORITHM 7: LOGISTIC REGRESSION
Group Member: Member 7
Hyper-parameters tuned (sensitivity analysis): C, penalty (l1/l2)
===================================================================
All experiment logic (CV tuning, plots, metrics) lives in
experiment_utils.py so that every member is evaluated identically.
Only the model and its hyper-parameter grid are defined here.

Run alone :  python algorithms/algo7_logistic_regression.py
"""
import os
import sys
import warnings

# scikit-learn >= 1.8 deprecates the `penalty` argument (it still works: we verified that
# L1 zeroes coefficients and L2 does not). Silence ONLY these two harmless messages, in this
# process and in the worker processes, so that any real warning stays visible.
_HARMLESS = ["'penalty' was deprecated", "Inconsistent values"]
for _m in _HARMLESS:
    warnings.filterwarnings("ignore", message=_m)
os.environ["PYTHONWARNINGS"] = ",".join(f"ignore:{_m}" for _m in _HARMLESS)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_pipeline import SEED
from experiment_utils import run_experiment
from sklearn.linear_model import LogisticRegression

def run_algo7():
    return run_experiment(
        member="Member 7",
        name="Logistic Regression",
        short="algo7_lr",
        estimator=LogisticRegression(solver='liblinear', max_iter=5000, random_state=SEED),
        param_grid={
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
        },
    )


if __name__ == "__main__":
    run_algo7()
