# Breast Cancer Diagnosis: 8-Algorithm Comparison (Data Mining Group Project)

**Task:** classify a breast mass as malignant or benign from 30 cell-nucleus measurements.
**Dataset:** Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
(569 patients, 30 numeric features, no missing values; 212 malignant / 357 benign).
Labels follow scikit-learn: 0 = malignant, 1 = benign.
**Published reference:** Street, Wolberg & Mangasarian (1993), "Nuclear feature extraction for breast tumor
diagnosis", IS&T/SPIE vol. 1905 (about 97% accuracy, 10-fold CV).

## Method (the same for all 8 algorithms)
1. **Split:** stratified 80/20, `random_state=42` -> 455 train / 114 test.
2. **Scaling:** Z-score (`StandardScaler`) inside a scikit-learn `Pipeline`, so it is fitted on the training folds only.
3. **Sensitivity analysis / tuning:** every combination of the hyperparameter values below is scored with
   **5-fold cross-validation on the training set** (`GridSearchCV`, metric = accuracy).
4. **Best parameters:** the combination with the highest CV accuracy.
5. **Final test:** train the best setting on the full training set and test it **once** on the test set.
6. **Comparison:** accuracy, precision, recall, F1 (malignant class), ROC-AUC, against a majority-class baseline
   (always predict benign) and the published result.

## Hyperparameters tuned
| Algorithm | Parameters |
|---|---|
| Neural Network (MLP) | hidden_layer_sizes, activation, alpha, learning_rate_init |
| SVM | kernel, C, gamma (rbf), degree (poly) |
| Random Forest | n_estimators, max_depth, max_features |
| Gradient Boosting | n_estimators, learning_rate, max_depth |
| k-NN | n_neighbors, weights, metric |
| Decision Tree | max_depth, criterion, min_samples_split |
| Logistic Regression | C, penalty |
| Gaussian Naive Bayes | var_smoothing (its only meaningful parameter) |

## How to run
```
pip install -r requirements.txt
python evaluate_all_results.py                 # everything: all 8 algorithms + table + plots
python algorithms/algo5_knn.py                 # or just ONE algorithm (each member runs their own)
python evaluate_all_results.py --plots-only    # rebuild the comparison from saved results (seconds)
```

## Files
| File | Purpose |
|---|---|
| `data_pipeline.py` | shared stratified split |
| `experiment_utils.py` | the shared steps: tuning with 5-fold CV, sensitivity plot, final test, metrics |
| `algorithms/algo1..8_*.py` | one file per member: only the model and its hyperparameter grid |
| `evaluate_all_results.py` | runs everything and builds the comparison table and plots |

## Outputs in `results/`
Per algorithm: `sensitivity_<algo>.png` (effect of each hyperparameter), `sensitivity_<algo>.csv` (every
combination tried), `result_<algo>.json`, `predictions_<algo>.csv`.
Comparison: `final_8_algorithm_comparison.csv`, `all_8_algorithms_comparison.png`, `comparison_f1_score.png`,
`comparison_precision_recall.png`, `comparison_roc_curves.png`, `final_summary.md`.

## Limitations
- The test set has only 114 patients (one patient = 0.9 percentage points), so small differences between
  algorithms are not meaningful.
- "Best" means best among the values in the grid, not a global optimum.
- The published 97% used 10-fold CV on all 569 samples, so it is a qualitative reference.
