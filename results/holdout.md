# Patient-level holdout (139 stays, no patient shared with training)

Computed on the MIMIC-IV cohort. Only aggregate values are reported. AUC intervals come from 1000 bootstrap resamples. F1, sensitivity and specificity use each model's threshold from cross-validation.

| model              | AUC (95% CI)        |    f1 |   sensitivity |   specificity |
|:-------------------|:--------------------|------:|--------------:|--------------:|
| MLP                | 0.838 (0.768-0.901) | 0.714 |         0.833 |         0.682 |
| LogisticRegression | 0.823 (0.751-0.890) | 0.703 |         0.722 |         0.788 |
| SVM                | 0.813 (0.736-0.880) | 0.654 |         0.648 |         0.788 |
| AdaBoost           | 0.801 (0.728-0.868) | 0.673 |         0.704 |         0.753 |
| XGBoost            | 0.801 (0.728-0.871) | 0.588 |         0.556 |         0.788 |
| ComplementNB       | 0.800 (0.723-0.869) | 0.672 |         0.778 |         0.659 |
| GaussianNB         | 0.796 (0.722-0.868) | 0.649 |         0.685 |         0.729 |
| LightGBM           | 0.754 (0.677-0.834) | 0.54  |         0.5   |         0.776 |
