"""Score new ICU stays with a fitted pipeline."""

import pandas as pd
from sklearn.pipeline import Pipeline

from aki_risk_predictor.data import features


def predict(pipeline: Pipeline, stays: pd.DataFrame, *, threshold: float = 0.5) -> pd.DataFrame:
    """One row per input stay with columns `stayId`, `probability` and `prediction`.

    Stays are never deduplicated, so two stays of the same patient get two rows.
    """
    probability = pipeline.predict_proba(features(stays))[:, 1]
    return pd.DataFrame(
        {
            "stayId": stays.index.to_numpy(),
            "probability": probability,
            "prediction": (probability >= threshold).astype(int),
        }
    )
