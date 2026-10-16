"""Preprocessing and feature selection that are fitted inside each training fold."""

from typing import Self, cast

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _text_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def _number_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


class DropSparse(TransformerMixin, BaseEstimator):
    """Drop columns whose share of missing values in the training data exceeds `threshold`."""

    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:  # noqa: ARG002
        missing = X.isna().mean()
        self.columns_ = list(missing[missing <= self.threshold].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns_]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:  # noqa: ARG002
        return np.asarray(self.columns_, dtype=object)


class ShapSelector(TransformerMixin, BaseEstimator):
    """Keep the `n_features` columns with the largest mean |SHAP value| of an XGBoost model.

    The XGBoost model is fitted on the data passed to `fit`, so inside a pipeline it only ever
    sees the training fold. SHAP values come from XGBoost's exact TreeSHAP (`pred_contribs`).
    """

    def __init__(self, n_features: int = 20, seed: int = 42) -> None:
        self.n_features = n_features
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=self.seed, n_jobs=1
        )
        model.fit(X, y)
        contributions = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        importance = np.abs(contributions[:, :-1]).mean(axis=0)  # last column is the bias
        self.importance_ = pd.Series(importance, index=X.columns).sort_values(ascending=False)
        self.columns_ = list(self.importance_.index[: self.n_features])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns_]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:  # noqa: ARG002
        return np.asarray(self.columns_, dtype=object)


def build_pipeline(
    model: ClassifierMixin, *, n_features: int = 20, missing_threshold: float = 0.2, seed: int = 42
) -> Pipeline:
    """drop sparse columns -> one-hot race -> scale -> KNN impute -> SHAP top-k -> model.

    Scaling comes before the KNN imputer so that neighbour distances are not dominated by
    features with large units, such as platelet counts.
    """
    encode = ColumnTransformer(
        [
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                _text_columns,
            ),
            ("numbers", "passthrough", _number_columns),
        ],
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        [
            ("drop_sparse", DropSparse(missing_threshold)),
            ("encode", encode),
            ("scale", StandardScaler()),
            ("impute", KNNImputer(n_neighbors=5)),
            ("select", ShapSelector(n_features, seed)),
            ("model", model),
        ]
    )
    return cast("Pipeline", pipeline.set_output(transform="pandas"))


def selected_features(pipeline: Pipeline) -> list[str]:
    selector = cast("ShapSelector", pipeline.named_steps["select"])
    return list(selector.columns_)
