# 10-fold cross-validation on the training split, grouped by patient

Computed on the MIMIC-IV cohort. Only aggregate values are reported. F1, sensitivity and specificity use the Youden threshold of the out-of-fold predictions.

| model              | AUC mean (SD) over folds   |    f1 |   sensitivity |   specificity |   threshold |
|:-------------------|:---------------------------|------:|--------------:|--------------:|------------:|
| TabPFN-3.5         | 0.813 (0.046)              | 0.693 |         0.65  |         0.856 |       0.472 |
| SVM                | 0.805 (0.054)              | 0.707 |         0.703 |         0.819 |       0.423 |
| LogisticRegression | 0.800 (0.055)              | 0.694 |         0.687 |         0.813 |       0.415 |
| AdaBoost           | 0.792 (0.046)              | 0.671 |         0.721 |         0.728 |       0.362 |
| GaussianNB         | 0.785 (0.052)              | 0.672 |         0.715 |         0.738 |       0.09  |
| XGBoost            | 0.782 (0.038)              | 0.657 |         0.601 |         0.854 |       0.703 |
| MLP                | 0.775 (0.050)              | 0.665 |         0.718 |         0.719 |       0.324 |
| ComplementNB       | 0.773 (0.078)              | 0.671 |         0.74  |         0.705 |       0.472 |
| LightGBM           | 0.768 (0.039)              | 0.641 |         0.598 |         0.831 |       0.501 |
