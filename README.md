# AKIRiskPredictor

[![CI](https://github.com/doanan293/AKIRiskPredictor/actions/workflows/ci.yml/badge.svg)](https://github.com/doanan293/AKIRiskPredictor/actions/workflows/ci.yml)

Predicts acute kidney injury (AKI) within seven days of ICU admission for adults with diabetic ketoacidosis, from first-day MIMIC-IV data, and compares nine classifiers, including the TabPFN-3.5 foundation model, under patient-level validation.

## Results

TabPFN-3.5 is the strongest model in this study. It has the best cross-validated AUC of the nine classifiers (0.813) and the highest specificity, and it needed no hyperparameter tuning.

The cohort has 970 labelled ICU stays from 714 patients, 38.9 % with AKI. Models are compared with 10-fold cross-validation on the training split, grouped by patient so that no patient is in two folds ([results/cv.md](results/cv.md)). Each model except TabPFN was first tuned with Optuna's quick budget:

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

A holdout of 139 stays from other patients was then scored once, to check that the cross-validation results carry over ([results/holdout.md](results/holdout.md)):

| model              | AUC (95% CI)        |    f1 |   sensitivity |   specificity |
|:-------------------|:--------------------|------:|--------------:|--------------:|
| MLP                | 0.838 (0.768-0.901) | 0.714 |         0.833 |         0.682 |
| LogisticRegression | 0.823 (0.751-0.890) | 0.703 |         0.722 |         0.788 |
| TabPFN-3.5         | 0.819 (0.749-0.886) | 0.667 |         0.648 |         0.812 |
| SVM                | 0.813 (0.736-0.880) | 0.654 |         0.648 |         0.788 |
| AdaBoost           | 0.801 (0.728-0.868) | 0.673 |         0.704 |         0.753 |
| XGBoost            | 0.801 (0.728-0.871) | 0.588 |         0.556 |         0.788 |
| ComplementNB       | 0.800 (0.723-0.869) | 0.672 |         0.778 |         0.659 |
| GaussianNB         | 0.796 (0.722-0.868) | 0.649 |         0.685 |         0.729 |
| LightGBM           | 0.754 (0.677-0.834) | 0.54  |         0.5   |         0.776 |

On the holdout TabPFN-3.5 scores 0.819, within 0.02 of the best model, and every confidence interval overlaps. A set of 139 stays is too small to separate the top models, so the ranking above relies on the 10-fold estimate, which uses all 831 training stays. Logistic regression is close behind in both tables, is easy to explain and has no license restriction, so it stays the default for `aki predict`. Set `predict_model: TabPFN-3.5` in the config to predict with TabPFN.

![ROC curves on the holdout](results/roc_holdout.png)

Also generated: [baseline characteristics](results/table_1.md), [feature ranking](results/selected_features.md), [tuned hyperparameters](results/tuned_params.json), [calibration](results/calibration_holdout.png) and [decision curves](results/decision_curve_holdout.png).

### Compared with the earlier version

An earlier version of this project reported a holdout AUC of 0.859 for TabPFN and 0.839 for logistic regression. Those numbers came from a split by ICU stay, which let stays of the same patient land on both sides, and from imputation, feature selection and tuning fitted on all training data before cross-validation. This version fixes both, so its numbers are lower and more honest. The TabPFN row now uses TabPFN-3.5 (September 2026) instead of the earlier TabPFN-3 checkpoint.

## How it works

```mermaid
flowchart LR
    json[ICU stays JSON] --> load[first value of each<br/>lab and vital sign]
    load --> split[holdout split<br/>by patient]
    split --> cv[10-fold CV<br/>grouped by patient]
    subgraph fold[fitted inside every training fold]
        drop[drop columns<br/>over 20% missing] --> enc[one-hot race] --> scale[scale] --> knn[KNN impute] --> shap[top 20 features<br/>by mean SHAP] --> model[classifier]
    end
    cv --> fold
    fold --> report[tables and figures<br/>in results/]
```

- `data.py` reduces each stay to one row: the earliest value of every lab and vital sign, severity scores, eGFR, comorbidity flags (absent means no), gender, liver disease grade and race.
- `validation.py` splits by patient with `StratifiedGroupKFold` and computes bootstrap confidence intervals.
- `pipeline.py` is a scikit-learn `Pipeline`, so dropping sparse columns, imputation, scaling and SHAP feature selection are refitted on each training fold and never see validation rows. A test checks that a feature leaking the label only in validation rows is never selected.
- `models.py` holds the classifiers, `tuning.py` the Optuna search spaces, and `reporting.py` the tables and plots. Counts below 11 are suppressed in Table 1, following the PhysioNet small-cell rule.

## Quickstart

The repository ships a synthetic cohort, so everything runs without access to MIMIC-IV:

```bash
uv sync
uv run aki report          # tune, train and evaluate on data/sample (about 5 minutes on CPU)
uv run aki predict         # one probability per stay in data/sample/test.json
```

Running `report` on the sample overwrites `results/` with numbers from synthetic data, which mean nothing clinically. Use `git checkout results` to get the real results back.

TabPFN-3.5 is optional. Its weights are released by Prior Labs for non-commercial use only, so they are not in this repository. To include it:

```bash
uv sync --extra tabpfn
hf download Prior-Labs/tabpfn_3_5 tabpfn-v3.5-20260909.safetensors --local-dir local-data/tabpfn
```

Without the extra, `aki` skips TabPFN and runs the other eight models. With the extra but no local weights, the `tabpfn` package downloads them itself. `TABPFN_MODEL_PATH` points to weights stored elsewhere.

Each step also has its own command: `aki prepare`, `aki select-features`, `aki tune`, `aki train`, `aki evaluate`, `aki predict` and `aki synthesize`. Settings live in [configs/default.yaml](configs/default.yaml).

## Data

The real cohort comes from [MIMIC-IV](https://physionet.org/content/mimiciv/). Access needs PhysioNet credentialing and a signed data use agreement, which forbids sharing the data, so it is not in this repository and never was in its history. With access, put the extracted `train.json` and `test.json` in `local-data/` (ignored by git) and the CLI picks them up automatically. Only aggregate tables and figures are committed under `results/`. See [data/README.md](data/README.md) for the schema and the synthetic generator.

## Limitations

- One hospital system and 139 holdout stays, so the intervals are wide and there is no external validation.
- Only the first value of each measurement is used. Trends over the first day are ignored.
- The quick tuning budget runs one to three Optuna trials per model. A larger search may change the ranking.
- This is a research project. It is not a validated clinical tool.

## Development

```bash
uv run ruff check && uv run ruff format --check && uv run pyrefly check && uv run pytest
```

## License

MIT, see [LICENSE](LICENSE). MIMIC-IV data is subject to the PhysioNet credentialed health data license and is not covered by this license. TabPFN-3.5 weights are licensed by Prior Labs GmbH under the TabPFN-3.5 Non-Commercial License and are not covered by this license either.
