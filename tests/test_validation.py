from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from aki_risk_predictor.data import LABEL, load_stays
from aki_risk_predictor.validation import bootstrap_ci, grouped_cv, patient_holdout

SAMPLE = Path(__file__).parents[1] / "data" / "sample"


@pytest.fixture(scope="module")
def stays() -> pd.DataFrame:
    return load_stays(SAMPLE / "train.json")


def test_holdout_keeps_each_patient_on_one_side(stays: pd.DataFrame) -> None:
    train, test = patient_holdout(stays, test_size=0.15, seed=42)
    assert len(train) + len(test) == len(stays)
    assert set(stays.loc[train, "subjectId"]).isdisjoint(stays.loc[test, "subjectId"])
    assert 0.1 < len(test) / len(stays) < 0.2


def test_holdout_is_stratified(stays: pd.DataFrame) -> None:
    train, test = patient_holdout(stays)
    assert abs(stays.loc[test, LABEL].mean() - stays.loc[train, LABEL].mean()) < 0.08


def test_every_cv_fold_is_disjoint_by_patient(stays: pd.DataFrame) -> None:
    folds = grouped_cv(n_splits=5, seed=0).split(stays, stays[LABEL], groups=stays["subjectId"])
    for train_rows, val_rows in folds:
        train_ids = set(stays["subjectId"].iloc[train_rows])
        assert train_ids.isdisjoint(stays["subjectId"].iloc[val_rows])


def test_bootstrap_interval_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, size=300), 0, 1)
    point, low, high = bootstrap_ci(y, p, roc_auc_score, n=200)
    assert low <= point <= high
    assert 0.5 < point <= 1.0
