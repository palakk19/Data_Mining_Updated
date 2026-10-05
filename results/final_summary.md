# Results summary (auto-generated)

Dataset: Breast Cancer Wisconsin (Diagnostic), UCI. 569 patients, 30 features, no missing values. Stratified 80/20 split (455 train / 114 test, seed 42).

Method: Z-score scaling, then every hyperparameter combination was scored with 5-fold cross-validation on the training set; the best one was tested once on the test set.

| Algorithm | Best parameters | CV accuracy | Test accuracy | Precision | Recall | F1 | Missed malignant |
|---|---|---|---|---|---|---|---|
| Neural Network (MLP) | activation=logistic, alpha=1.0, hidden_layer_sizes=(50, 25), learning_rate_init=0.001 | 98.46% | 96.49% | 97.50% | 92.86% | 95.12% | 3 |
| Support Vector Machine (SVM) | C=0.1, kernel=linear | 97.58% | 98.25% | 97.62% | 97.62% | 97.62% | 1 |
| Random Forest | max_depth=8, max_features=sqrt, n_estimators=100 | 96.48% | 95.61% | 95.12% | 92.86% | 93.98% | 3 |
| Gradient Boosting | learning_rate=0.2, max_depth=3, n_estimators=200 | 97.36% | 95.61% | 97.44% | 90.48% | 93.83% | 4 |
| k-Nearest Neighbors (k-NN) | metric=manhattan, n_neighbors=3, weights=uniform | 97.14% | 96.49% | 97.50% | 92.86% | 95.12% | 3 |
| Decision Tree | criterion=entropy, max_depth=4, min_samples_split=10 | 93.41% | 92.98% | 90.48% | 90.48% | 90.48% | 4 |
| Logistic Regression | C=0.1, penalty=l2 | 98.24% | 98.25% | 97.62% | 97.62% | 97.62% | 1 |
| Gaussian Naive Bayes | var_smoothing=0.1 | 94.07% | 94.74% | 95.00% | 90.48% | 92.68% | 4 |
| *Baseline: always predict benign* | - | 62.64% | 63.16% | 0.00% | 0.00% | 0.00% | 42 |

Notes:

* Highest test accuracy: Support Vector Machine (SVM) (98.25%).
* Published reference: Street, Wolberg & Mangasarian (1993) reported about 97% accuracy with 10-fold cross-validation on this dataset.
* Precision, Recall and F1 are for the MALIGNANT class. 'Missed malignant' is the number of cancers predicted benign (false negatives).
* The test set has only 114 patients, so one patient is about 0.9 percentage points. Small differences between algorithms are not meaningful.
