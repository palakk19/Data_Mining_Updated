"""
===================================================================
DATA MINING GROUP PROJECT - SHARED DATA PIPELINE
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI Repository
===================================================================
This module does ONE job: create the shared, reproducible train/test split.

  * Stratified 80% train / 20% test split, random_state = 42
    (455 training patients, 114 test patients - identical for every member)
  * The files saved in data/ are RAW (unscaled) features.

Z-score standardisation is NOT done here any more. It is done inside each
model's scikit-learn Pipeline (see experiment_utils.py), so that during
cross-validation the scaler is re-fitted on the training folds only. Scaling
the whole training set first would let a little information from the
validation fold leak into the scaler, which makes CV scores slightly too
optimistic. The test set is never used to fit anything.

Labels follow scikit-learn:  0 = Malignant,  1 = Benign
"""
import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.20
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CLASS_NAMES = ['Malignant', 'Benign']  # index = label

_FILES = {
    'X_train': 'X_train_raw.csv',
    'X_test': 'X_test_raw.csv',
    'y_train': 'y_train.csv',
    'y_test': 'y_test.csv',
}


def setup_pipeline(verbose=True):
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=['target'])
    y = df['target']  # 1 = Benign, 0 = Malignant

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(DATA_DIR, _FILES['X_train']), index=False)
    X_test.to_csv(os.path.join(DATA_DIR, _FILES['X_test']), index=False)
    y_train.to_csv(os.path.join(DATA_DIR, _FILES['y_train']), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, _FILES['y_test']), index=False)

    if verbose:
        print("=" * 60)
        print("SHARED DATASET: Breast Cancer Wisconsin (Diagnostic), UCI")
        print("=" * 60)
        print(f"  Samples        : {len(df)}   Features: {X.shape[1]}")
        print(f"  Missing values : {int(df.isna().sum().sum())}")
        print(f"  Class split    : Benign={int((y == 1).sum())}, Malignant={int((y == 0).sum())}")
        print(f"  Train / Test   : {len(X_train)} / {len(X_test)}  (stratified, seed={SEED})")
        print(f"  Saved raw split to: {DATA_DIR}")


def load_shared_data():
    """Return X_train, X_test, y_train, y_test (RAW features, 1-D label arrays)."""
    if not os.path.exists(os.path.join(DATA_DIR, _FILES['X_train'])):
        setup_pipeline(verbose=False)
    X_train = pd.read_csv(os.path.join(DATA_DIR, _FILES['X_train']))
    X_test = pd.read_csv(os.path.join(DATA_DIR, _FILES['X_test']))
    y_train = pd.read_csv(os.path.join(DATA_DIR, _FILES['y_train'])).values.ravel()
    y_test = pd.read_csv(os.path.join(DATA_DIR, _FILES['y_test'])).values.ravel()
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    setup_pipeline()
