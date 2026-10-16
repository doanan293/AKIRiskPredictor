"""Patient-level splits and bootstrap confidence intervals.

One patient can have several ICU stays. Every split here keeps all stays of a patient on the same
side, so a model is never scored on a patient it has already seen.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from aki_risk_predictor.data import LABEL

Metric = Callable[[np.ndarray, np.ndarray], float]


def grouped_cv(n_splits: int = 10, seed: int = 42) -> StratifiedGroupKFold:
    """Stratified by label, grouped by patient."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def patient_holdout(
    frame: pd.DataFrame, *, test_size: float = 0.15, seed: int = 42
) -> tuple[pd.Index, pd.Index]:
    """Split stay ids into train and holdout with no patient on both sides."""
    n_splits = max(2, round(1 / test_size))
    folds = grouped_cv(n_splits, seed).split(frame, frame[LABEL], groups=frame["subjectId"])
    train_rows, test_rows = next(iter(folds))
    return frame.index[train_rows], frame.index[test_rows]


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, metric: Metric, n: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Point estimate and 95 % percentile interval from `n` bootstrap resamples."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    scores: list[float] = []
    for _ in range(n):
        rows = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[rows])) < 2:  # noqa: PLR2004 - a metric like AUC needs both classes
            continue
        scores.append(metric(y[rows], p[rows]))
    low, high = np.percentile(scores, [2.5, 97.5])
    return metric(y, p), float(low), float(high)
