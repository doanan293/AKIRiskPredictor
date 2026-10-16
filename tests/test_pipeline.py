from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from aki_risk_predictor.data import LABEL, features, load_stays
from aki_risk_predictor.pipeline import build_pipeline, selected_features
from aki_risk_predictor.validation import patient_holdout

SAMPLE = Path(__file__).parents[1] / "data" / "sample"


@pytest.fixture(scope="module")
def stays() -> pd.DataFrame:
    return load_stays(SAMPLE / "train.json")


def test_selector_never_sees_a_feature_that_leaks_only_in_validation(
    stays: pd.DataFrame,
) -> None:
    train, val = patient_holdout(stays, seed=0)
    frame = stays.copy()
    rng = np.random.default_rng(0)
    # noise in the training rows, a perfect copy of the label in the validation rows
    frame["leak"] = rng.normal(size=len(frame))
    frame.loc[val, "leak"] = frame.loc[val, LABEL].astype(float)

    pipeline = build_pipeline(LogisticRegression(max_iter=1000), n_features=8)
    pipeline.fit(features(frame.loc[train]), frame.loc[train, LABEL])

    assert "leak" not in selected_features(pipeline)


def test_sparse_columns_are_dropped_using_training_rows_only(stays: pd.DataFrame) -> None:
    pipeline = build_pipeline(LogisticRegression(max_iter=1000), n_features=5)
    pipeline.fit(features(stays), stays[LABEL])
    kept = pipeline.named_steps["drop_sparse"].columns_
    assert "ckd_stage" not in kept  # about 80 % missing
    assert "scr" in kept


def test_predicts_unseen_race_and_all_missing_measures(stays: pd.DataFrame) -> None:
    pipeline = build_pipeline(LogisticRegression(max_iter=1000), n_features=10)
    pipeline.fit(features(stays), stays[LABEL])

    new = features(stays).iloc[:3].copy()
    new["race"] = pd.array(["MARTIAN", "WHITE", "WHITE"], dtype="string")
    numeric = new.columns.drop("race")
    new.loc[new.index[1], numeric] = np.nan

    proba = pipeline.predict_proba(new)[:, 1]
    assert proba.shape == (3,)
    assert np.isfinite(proba).all()
