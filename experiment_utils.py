"""
===================================================================
EXPERIMENT ENGINE (shared by all 8 algorithm files)
===================================================================
Every algorithm follows exactly the same simple steps:

  1. Load the shared train/test split          (data_pipeline.py)
  2. Z-score scaling + the model in a Pipeline (scaler is fitted on the
                                                training folds only)
  3. SENSITIVITY ANALYSIS / TUNING:
       try every combination of the hyperparameter values in the grid,
       score each with 5-fold cross-validation on the TRAINING set
  4. Pick the combination with the best CV accuracy
  5. Train it on the full training set and test it ONCE on the test set
  6. Report accuracy, precision, recall, F1 (malignant class) and ROC-AUC
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline import load_shared_data, SEED

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_cv():
    """5-fold stratified cross-validation (same folds for every algorithm)."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def build_pipeline(estimator):
    return Pipeline([('scaler', StandardScaler()), ('clf', estimator)])


def _grid_list(param_grid):
    return [param_grid] if isinstance(param_grid, dict) else list(param_grid)


def _prefix(param_grid):
    return [{f"clf__{k}": v for k, v in g.items()} for g in _grid_list(param_grid)]


def _ordered_values(param_grid):
    """{parameter: [values in the order given in the grid]}"""
    out = {}
    for g in _grid_list(param_grid):
        for k, vals in g.items():
            lst = out.setdefault(k, [])
            for v in vals:
                if str(v) not in [str(x) for x in lst]:
                    lst.append(v)
    return out


def _fmt_params(d):
    return ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in d.items())


def get_score_benign(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, score_benign):
    """Precision / Recall / F1 are for the MALIGNANT class (label 0): missing a
    cancer is worse than a false alarm. Labels: 0 = malignant, 1 = benign."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # rows = true, cols = predicted
    try:
        auc = float(roc_auc_score(y_true, score_benign))
    except ValueError:
        auc = float('nan')
    return {
        'Test Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'F1-Score': float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'ROC-AUC': auc,
        'Missed Malignant (FN)': int(cm[0, 1]),
        'False Alarms (FP)': int(cm[1, 0]),
    }


# ----------------------------------------------------------------------
# Sensitivity analysis table + plot
# ----------------------------------------------------------------------
def _sensitivity_table(cv_results, param_grid):
    vals = _ordered_values(param_grid)
    keys = list(vals.keys())
    rows = []
    for i, d in enumerate(cv_results['params']):
        row = {k: (str(d[f"clf__{k}"]) if f"clf__{k}" in d else 'n/a') for k in keys}
        row['cv_accuracy'] = float(cv_results['mean_test_score'][i])
        row['train_accuracy'] = float(cv_results['mean_train_score'][i])
        row['rank'] = int(cv_results['rank_test_score'][i])
        rows.append(row)
    return pd.DataFrame(rows), vals


def plot_sensitivity(table, vals, name, path, best_params):
    """One chart per hyperparameter, showing ONLY the mean 5-fold CV accuracy.
    The value that GridSearchCV actually chose (best_params) is marked with a
    gold star / gold bar and labelled."""
    GOLD = '#f2a900'
    keys = list(vals.keys())
    cols = min(len(keys), 3)
    rows_n = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(max(4.6 * cols, 9.0), 3.8 * rows_n), squeeze=False)
    axes = axes.ravel()

    def _nice(l):
        try:
            return f"{float(l):g}"
        except ValueError:
            return l

    for ax, p in zip(axes, keys):
        labels = [str(v) for v in vals[p]]
        cv_m = []
        for lab in labels:
            sub = table[table[p] == lab]
            cv_m.append(sub['cv_accuracy'].max())
        # value chosen by GridSearchCV (None if the winning model doesn't use this parameter)
        key = f"clf__{p}"
        chosen = str(best_params[key]) if key in best_params else None
        ci = labels.index(chosen) if chosen in labels else None
        x = np.arange(len(labels))
        is_bar = all(isinstance(v, str) for v in vals[p])
        if is_bar:
            colors = [GOLD if i == ci else 'tab:blue' for i in range(len(labels))]
            ax.bar(x, cv_m, color=colors)
            ax.set_ylim(max(0, min(cv_m) - 0.03), min(1.0, max(cv_m) + 0.012))
        else:
            ax.plot(x, cv_m, marker='o', color='tab:blue', zorder=2)
            lo, hi = min(cv_m), max(cv_m)
            ax.set_ylim(lo - max(0.004, (hi - lo) * 0.2), hi + max(0.008, (hi - lo) * 0.5))
        if ci is not None:
            if not is_bar:
                ax.plot(x[ci], cv_m[ci], marker='*', markersize=18, color=GOLD,
                        markeredgecolor='black', markeredgewidth=0.8, zorder=3, linestyle='None')
            # keep the label inside the panel when the chosen value is at an edge
            if is_bar:
                ha, dx = 'center', 0
            elif ci == 0 and len(labels) > 1:
                ha, dx = 'left', -4
            elif ci == len(labels) - 1 and len(labels) > 1:
                ha, dx = 'right', 4
            else:
                ha, dx = 'center', 0
            ax.annotate(f"chosen: {_nice(labels[ci])}\n{cv_m[ci]:.2%}", (x[ci], cv_m[ci]),
                        textcoords='offset points', xytext=(dx, 11), ha=ha,
                        fontsize=8, fontweight='bold')
        else:
            ax.text(0.5, 0.04, "not used by the chosen model", transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=8, style='italic', color='dimgray')
        shown = [_nice(l) for l in labels]
        long = max(len(l) for l in shown) > 5
        ax.set_xticks(x)
        ax.set_xticklabels(shown, rotation=45 if long else 0, ha='right' if long else 'center', fontsize=8)
        ax.set_xlabel(p)
        ax.set_ylabel("Mean 5-fold CV accuracy")
        ax.margins(x=0.08)
    for ax in axes[len(keys):]:
        ax.axis('off')
    fig.suptitle(f"{name}: effect of each hyperparameter (mean 5-fold CV accuracy)\n"
                 f"Chosen (gold) = {_fmt_params(best_params)}", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------
def run_experiment(member, name, short, estimator, param_grid):
    print("=" * 70)
    print(f"{member.upper()}: {name.upper()}")
    print("=" * 70)
    X_train, X_test, y_train, y_test = load_shared_data()

    # Steps 3-4: sensitivity analysis / tuning on the TRAINING set only
    grid = GridSearchCV(build_pipeline(clone(estimator)), _prefix(param_grid), cv=make_cv(),
                        scoring='accuracy', n_jobs=-1, refit=True, return_train_score=True)
    grid.fit(X_train, y_train)

    table, vals = _sensitivity_table(grid.cv_results_, param_grid)
    table.sort_values('rank').to_csv(os.path.join(RESULTS_DIR, f'sensitivity_{short}.csv'), index=False)
    plot_sensitivity(table, vals, name, os.path.join(RESULTS_DIR, f'sensitivity_{short}.png'),
                 grid.best_params_)
    # Step 5: the best model (already refit on the full training set) is tested ONCE
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    score = get_score_benign(best_model, X_test)
    m = compute_metrics(y_test, y_pred, score)
    pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'score_benign': score}).to_csv(
        os.path.join(RESULTS_DIR, f'predictions_{short}.csv'), index=False)

    result = {
        'Member': member,
        'Algorithm': name,
        'Best Parameter': _fmt_params(grid.best_params_),
        'CV Train Accuracy': float(grid.best_score_),
        **m,
    }
    with open(os.path.join(RESULTS_DIR, f'result_{short}.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  Combinations tried : {len(table)}  (each scored with 5-fold CV)")
    print(f"  Best parameters    : {result['Best Parameter']}")
    print(f"  Best CV accuracy   : {result['CV Train Accuracy']:.4f}")
    print(f"  TEST accuracy      : {m['Test Accuracy']:.4f}")
    print(f"  Malignant P/R/F1   : {m['Precision']:.4f} / {m['Recall']:.4f} / {m['F1-Score']:.4f}")
    print(f"  Missed malignant   : {m['Missed Malignant (FN)']}   False alarms: {m['False Alarms (FP)']}")
    return result


def run_baseline():
    """Baseline: always predict the majority class (benign)."""
    X_train, X_test, y_train, y_test = load_shared_data()
    pipe = build_pipeline(DummyClassifier(strategy='most_frequent'))
    cv_acc = float(cross_val_score(pipe, X_train, y_train, cv=make_cv(), scoring='accuracy').mean())
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    m = compute_metrics(y_test, y_pred, get_score_benign(pipe, X_test))
    result = {'Member': 'Baseline', 'Algorithm': 'Majority-class baseline',
              'Best Parameter': 'always predict benign', 'CV Train Accuracy': cv_acc, **m}
    with open(os.path.join(RESULTS_DIR, 'result_baseline.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result
