import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aki_risk_predictor.data import LABEL, Cohort, features, load_stays
from aki_risk_predictor.models import MODEL_NAMES, get_model
from aki_risk_predictor.pipeline import build_pipeline
from aki_risk_predictor.tuning import BUDGETS, SEARCH_SPACES, CVSettings, tune

SAMPLE = Path(__file__).parents[1] / "data" / "sample"


@pytest.fixture(scope="module")
def stays() -> pd.DataFrame:
    return load_stays(SAMPLE / "train.json")


@pytest.mark.parametrize("name", [n for n in MODEL_NAMES if n != "TabPFN-3-Plus"])
def test_every_model_fits_and_predicts(stays: pd.DataFrame, name: str) -> None:
    rows = stays.iloc[:100]
    pipeline = build_pipeline(get_model(name), n_features=8)
    pipeline.fit(features(rows), rows[LABEL])
    proba = pipeline.predict_proba(features(rows))[:, 1]
    assert proba.shape == (100,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unknown_model_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        get_model("RandomForest")


def test_every_tuned_model_has_a_search_space() -> None:
    for budget in BUDGETS.values():
        assert {n for n, trials in budget.items() if trials} <= set(SEARCH_SPACES)


def test_quick_tuning_returns_params_within_a_minute(stays: pd.DataFrame) -> None:
    start = time.perf_counter()
    params = tune(
        "LogisticRegression",
        Cohort.from_stays(stays),
        budget="quick",
        cv=CVSettings(n_splits=3, n_features=8),
    )
    assert time.perf_counter() - start < 60
    assert set(params) == {"C", "class_weight"}
    assert np.isfinite(params["C"])  # type: ignore[arg-type]
