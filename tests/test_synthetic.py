import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from aki_risk_predictor.synthetic import ID_LOW, Record, generate_cohort


@pytest.fixture(scope="module")
def large_cohort() -> list[Record]:
    return generate_cohort(2000, seed=1)


def _first(series: object) -> float:
    assert isinstance(series, dict)
    return float(series[min(series)])


def test_same_seed_gives_identical_output() -> None:
    assert generate_cohort(50, seed=3) == generate_cohort(50, seed=3)
    assert generate_cohort(50, seed=3) != generate_cohort(50, seed=4)


def test_records_have_the_original_top_level_keys() -> None:
    labelled = generate_cohort(20, seed=0)
    unlabelled = generate_cohort(20, seed=0, labelled=False)
    assert all(
        set(r) == {"subjectId", "hadmId", "stayId", "akdPositive", "measures"} for r in labelled
    )
    assert all(set(r) == {"subjectId", "hadmId", "stayId", "measures"} for r in unlabelled)


def test_positive_rate_matches_target(large_cohort: list[Record]) -> None:
    rate = np.mean([r["akdPositive"] is True for r in large_cohort])
    assert abs(rate - 0.389) < 0.05


def test_label_has_signal_in_kidney_and_severity_features(large_cohort: list[Record]) -> None:
    rows = []
    for r in large_cohort:
        m = r["measures"]
        assert isinstance(m, dict)
        rows.append(
            [
                m["egfr"],
                _first(m["scr"]) if "scr" in m else 1.1,
                m["age"],
                _first(m["weight"]) if "weight" in m else 74.0,
                m["sofa"],
            ]
        )
    X = np.array(rows, dtype=float)
    y = np.array([r["akdPositive"] is True for r in large_cohort], dtype=int)
    proba = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5, method="predict_proba")
    assert roc_auc_score(y, proba[:, 1]) > 0.7


def test_ids_cannot_collide_with_real_mimic_ids() -> None:
    real_looking = {18888888, 12345678, 20000001, 30000123, 25000000, 15555555, 11111111}
    ids = {
        r[key] for r in generate_cohort(500, seed=5) for key in ("subjectId", "hadmId", "stayId")
    }
    assert all(isinstance(i, int) for i in ids)
    ids = {i for i in ids if isinstance(i, int)}
    assert min(ids) >= ID_LOW
    assert not ids & real_looking


def test_several_stays_share_a_subject() -> None:
    records = generate_cohort(400, seed=6)
    subjects = {r["subjectId"] for r in records}
    assert 1.2 < len(records) / len(subjects) < 1.5
