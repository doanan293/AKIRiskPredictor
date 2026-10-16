"""The nine classifiers compared in the study."""

import importlib
from collections.abc import Callable
from typing import cast

from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

MODEL_NAMES = (
    "TabPFN-3-Plus",
    "XGBoost",
    "LightGBM",
    "LogisticRegression",
    "AdaBoost",
    "GaussianNB",
    "ComplementNB",
    "MLP",
    "SVM",
)
HIDDEN_LAYERS = {"16": (16,), "32": (32,), "64": (64,), "32_16": (32, 16), "64_32": (64, 32)}
SEED = 42

Params = dict[str, object]


def _tabpfn(params: Params) -> ClassifierMixin:
    try:
        module = importlib.import_module("tabpfn")
    except ModuleNotFoundError as error:
        msg = "TabPFN is optional. Install it with `uv sync --extra tabpfn`."
        raise ModuleNotFoundError(msg) from error
    return cast("ClassifierMixin", module.TabPFNClassifier(random_state=SEED, **params))


def _adaboost(params: Params) -> ClassifierMixin:
    params = dict(params)
    depth = cast("int", params.pop("estimator_max_depth", 1))
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth, random_state=SEED),
        random_state=SEED,
        **params,  # type: ignore[arg-type]
    )


def _mlp(params: Params) -> ClassifierMixin:
    params = dict(params)
    layers = HIDDEN_LAYERS[cast("str", params.pop("hidden_layer_size", "32"))]
    return MLPClassifier(
        hidden_layer_sizes=layers,
        max_iter=1000,
        random_state=SEED,
        **params,  # type: ignore[arg-type]
    )


def _complement_nb(params: Params) -> ClassifierMixin:
    # ComplementNB needs non-negative inputs, and the shared pipeline standardizes features
    return cast(
        "ClassifierMixin",
        make_pipeline(MinMaxScaler(clip=True), ComplementNB(**params)),  # type: ignore[arg-type]
    )


def _svm(params: Params) -> ClassifierMixin:
    # Platt scaling on held-out folds, the replacement for the deprecated SVC(probability=True)
    svc = SVC(random_state=SEED, max_iter=10_000, **params)  # type: ignore[arg-type]
    return CalibratedClassifierCV(svc, ensemble=False, cv=5)


def _xgboost(params: Params) -> ClassifierMixin:
    return XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1, **params)


def _lightgbm(params: Params) -> ClassifierMixin:
    return LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1, **params)  # type: ignore[arg-type]


def _logistic(params: Params) -> ClassifierMixin:
    return LogisticRegression(max_iter=1000, random_state=SEED, **params)  # type: ignore[arg-type]


def _gaussian_nb(params: Params) -> ClassifierMixin:
    return GaussianNB(**params)  # type: ignore[arg-type]


def get_model(name: str, params: Params | None = None) -> ClassifierMixin:
    """Build a classifier by name, optionally with tuned hyperparameters."""
    params = params or {}
    builders: dict[str, Callable[[Params], ClassifierMixin]] = {
        "TabPFN-3-Plus": _tabpfn,
        "XGBoost": _xgboost,
        "LightGBM": _lightgbm,
        "LogisticRegression": _logistic,
        "AdaBoost": _adaboost,
        "GaussianNB": _gaussian_nb,
        "ComplementNB": _complement_nb,
        "MLP": _mlp,
        "SVM": _svm,
    }
    if name not in builders:
        msg = f"unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}"
        raise ValueError(msg)
    return cast("ClassifierMixin", builders[name](params))
