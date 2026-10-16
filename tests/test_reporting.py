import numpy as np
import pandas as pd

from aki_risk_predictor.data import LABEL
from aki_risk_predictor.reporting import net_benefit, table_1, threshold_metrics, youden_threshold


def test_table_1_suppresses_small_cells() -> None:
    n = 60
    frame = pd.DataFrame(
        {
            LABEL: [1] * 20 + [0] * 40,
            "age": np.linspace(20, 80, n),
            "hypertension": [1] * 30 + [0] * 30,
            "uti": [1] * 5 + [0] * 55,  # 5 cases overall: below the cell size
        }
    )
    table = table_1(frame, min_cell=11).set_index("characteristic")
    assert table.loc["uti, n (%)", "overall"] == "<11"
    assert table.loc["uti, n (%)", "AKI"] == "<11"
    assert table.loc["uti, n (%)", "no AKI"] == "0 (0.0%)"
    assert table.loc["hypertension, n (%)", "overall"] == "30 (50.0%)"
    assert table.loc["stays, n", "AKI"] == "20 (33.3%)"


def test_threshold_metrics_on_known_predictions() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    metrics = threshold_metrics(y, p, 0.5)
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5


def test_youden_threshold_separates_perfect_scores() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = youden_threshold(y, p)
    assert 0.2 < threshold <= 0.8


def test_net_benefit_of_perfect_model_is_prevalence_at_low_threshold() -> None:
    y = np.array([1, 0, 0, 0])
    p = np.array([1.0, 0.0, 0.0, 0.0])
    assert net_benefit(y, p, np.array([0.01]))[0] == 0.25
