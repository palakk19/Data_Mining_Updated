"""
===================================================================
MASTER SCRIPT - runs all 8 algorithms and builds the final comparison
===================================================================
  python evaluate_all_results.py               # run everything
  python evaluate_all_results.py --plots-only  # rebuild table/plots from the
                                               # saved results/result_*.json

Outputs (in results/):
  final_8_algorithm_comparison.csv     main results table
  all_8_algorithms_comparison.png      CV vs test accuracy
  comparison_f1_score.png              F1 (malignant class)
  comparison_precision_recall.png      precision and recall (malignant class)
  comparison_roc_curves.png            ROC curves
  final_summary.md                     short written summary
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'algorithms'))

from data_pipeline import SEED, setup_pipeline
import experiment_utils as eu

RESULTS_DIR = eu.RESULTS_DIR
SHORTS = ['algo1_mlp', 'algo2_svm', 'algo3_rf', 'algo4_gb',
          'algo5_knn', 'algo6_dt', 'algo7_lr', 'algo8_gnb']
FRONT_COLS = ['Member', 'Algorithm', 'Best Parameter', 'CV Train Accuracy', 'Test Accuracy',
              'Precision', 'Recall', 'F1-Score', 'ROC-AUC']


def collect_results(plots_only):
    if plots_only:
        results = []
        for s in SHORTS:
            p = os.path.join(RESULTS_DIR, f'result_{s}.json')
            if not os.path.exists(p):
                sys.exit(f"Missing {p}. Run that algorithm first "
                         f"(python algorithms/<file>.py) or run this script without --plots-only.")
            results.append(json.load(open(p)))
        bp = os.path.join(RESULTS_DIR, 'result_baseline.json')
        base = json.load(open(bp)) if os.path.exists(bp) else eu.run_baseline()
        return results, base

    setup_pipeline(verbose=True)
    import algo1_neural_network as a1, algo2_svm as a2, algo3_random_forest as a3
    import algo4_gradient_boosting as a4, algo5_knn as a5, algo6_decision_tree as a6
    import algo7_logistic_regression as a7, algo8_naive_bayes as a8
    runners = [a1.run_algo1, a2.run_algo2, a3.run_algo3, a4.run_algo4,
               a5.run_algo5, a6.run_algo6, a7.run_algo7, a8.run_algo8]
    return [r() for r in runners], eu.run_baseline()


def short_label(name):
    return (name.replace('Support Vector Machine (SVM)', 'SVM')
                .replace('Neural Network (MLP)', 'Neural Net (MLP)')
                .replace('k-Nearest Neighbors (k-NN)', 'k-NN')
                .replace('Gaussian Naive Bayes', 'Gaussian NB')
                .replace('Logistic Regression', 'Logistic Reg.')
                .replace('Gradient Boosting', 'Gradient Boost'))


# ----------------------------------------------------------------------
# Plots (kept deliberately simple)
# ----------------------------------------------------------------------
def _bars(df, cols, labels, title, ylabel, fname):
    x = np.arange(len(df))
    w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (c, lab) in enumerate(zip(cols, labels)):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, df[c], w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(a) for a in df['Algorithm']], rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.8, 1.0)
    ax.set_title(title)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(cols), frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
    plt.close()


def plot_accuracy(df, base):
    _bars(df, ['CV Train Accuracy', 'Test Accuracy'], ['Cross-validation accuracy', 'Test accuracy'],
          f"Accuracy of the 8 algorithms (y-axis starts at 80%; majority-class baseline = {base['Test Accuracy']:.0%})",
          "Accuracy", 'all_8_algorithms_comparison.png')


def plot_f1(df):
    _bars(df, ['F1-Score'], ['F1 (malignant class)'],
          "F1 score of the 8 algorithms (y-axis starts at 80%)", "F1 score", 'comparison_f1_score.png')


def plot_precision_recall(df):
    _bars(df, ['Precision', 'Recall'], ['Precision (malignant)', 'Recall (malignant)'],
          "Precision and recall for malignant cases (y-axis starts at 80%)", "Score",
          'comparison_precision_recall.png')


def plot_roc(df):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, row in df.iterrows():
        p = pd.read_csv(os.path.join(RESULTS_DIR, f'predictions_{SHORTS[i]}.csv'))
        fpr, tpr, _ = roc_curve(p['y_true'] == 0, -p['score_benign'])  # malignant = positive
        auc = roc_auc_score(p['y_true'] == 0, -p['score_benign'])
        ax.plot(fpr, tpr, label=f"{short_label(row['Algorithm'])} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (malignant = positive class)")
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparison_roc_curves.png'), dpi=150)
    plt.close()


def write_summary(df, base):
    best = df.loc[df['Test Accuracy'].idxmax()]
    lines = ["# Results summary (auto-generated)\n",
             "Dataset: Breast Cancer Wisconsin (Diagnostic), UCI. 569 patients, 30 features, no missing values. "
             f"Stratified 80/20 split (455 train / 114 test, seed {SEED}).\n",
             "Method: Z-score scaling, then every hyperparameter combination was scored with 5-fold "
             "cross-validation on the training set; the best one was tested once on the test set.\n",
             "| Algorithm | Best parameters | CV accuracy | Test accuracy | Precision | Recall | F1 | Missed malignant |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['Algorithm']} | {r['Best Parameter']} | {r['CV Train Accuracy']:.2%} | "
                     f"{r['Test Accuracy']:.2%} | {r['Precision']:.2%} | {r['Recall']:.2%} | "
                     f"{r['F1-Score']:.2%} | {int(r['Missed Malignant (FN)'])} |")
    lines.append(f"| *Baseline: always predict benign* | - | {base['CV Train Accuracy']:.2%} | "
                 f"{base['Test Accuracy']:.2%} | {base['Precision']:.2%} | {base['Recall']:.2%} | "
                 f"{base['F1-Score']:.2%} | {int(base['Missed Malignant (FN)'])} |")
    lines += ["\nNotes:\n",
              f"* Highest test accuracy: {best['Algorithm']} ({best['Test Accuracy']:.2%}).",
              "* Published reference: Street, Wolberg & Mangasarian (1993) reported about 97% accuracy "
              "with 10-fold cross-validation on this dataset.",
              "* Precision, Recall and F1 are for the MALIGNANT class. 'Missed malignant' is the number "
              "of cancers predicted benign (false negatives).",
              "* The test set has only 114 patients, so one patient is about 0.9 percentage points. "
              "Small differences between algorithms are not meaningful."]
    open(os.path.join(RESULTS_DIR, 'final_summary.md'), 'w').write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plots-only', action='store_true',
                    help="skip tuning; rebuild table/plots from saved results/result_*.json")
    args = ap.parse_args()

    results, base = collect_results(args.plots_only)
    df = pd.DataFrame(results)
    df = df[FRONT_COLS + [c for c in df.columns if c not in FRONT_COLS]]
    df.to_csv(os.path.join(RESULTS_DIR, 'final_8_algorithm_comparison.csv'), index=False)

    plot_accuracy(df, base)
    plot_f1(df)
    plot_precision_recall(df)
    plot_roc(df)
    write_summary(df, base)

    print("\n" + "=" * 90 + "\nFINAL COMPARISON\n" + "=" * 90)
    print(df[['Algorithm', 'CV Train Accuracy', 'Test Accuracy', 'Precision', 'Recall',
              'F1-Score', 'Missed Malignant (FN)']].round(4).to_string(index=False))
    print(f"\nBaseline (always 'benign'): test accuracy {base['Test Accuracy']:.4f}")
    print("\nAll tables and plots saved in results/")


if __name__ == "__main__":
    main()
