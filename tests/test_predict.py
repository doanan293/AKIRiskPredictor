from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from aki_risk_predictor.data import LABEL, features, load_stays
from aki_risk_predictor.models import get_model
from aki_risk_predictor.pipeline import build_pipeline
from aki_risk_predictor.predict import predict

SAMPLE = Path(__file__).parents[1] / "data" / "sample"


@pytest.fixture(scope="module")
def fitted() -> Pipeline:
    train = load_stays(SAMPLE / "train.json")
    pipeline = build_pipeline(get_model("LogisticRegression"), n_features=10)
    return pipeline.fit(features(train), train[LABEL])


def test_one_output_row_per_input_stay(fitted: Pipeline) -> None:
    test = load_stays(SAMPLE / "test.json")
    assert test["subjectId"].duplicated().any()  # some patients have several stays
    out = predict(fitted, test)
    assert len(out) == len(test) == 150
    assert list(out.columns) == ["stayId", "probability", "prediction"]
    assert out["stayId"].tolist() == test.index.tolist()


def test_stay_with_every_measure_missing_still_gets_a_probability(fitted: Pipeline) -> None:
    test = load_stays(SAMPLE / "test.json").iloc[:2].copy()
    numeric = [c for c in test.columns if c not in {"subjectId", "race"}]
    test.loc[test.index[0], numeric] = np.nan
    test["race"] = pd.array(["NEVER SEEN", "WHITE"], dtype="string")
    out = predict(fitted, test)
    assert np.isfinite(out["probability"]).all()
    assert set(out["prediction"]) <= {0, 1}
