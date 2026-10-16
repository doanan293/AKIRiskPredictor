"""Optuna search over each model's hyperparameters, scored by patient-grouped CV AUC."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score

from aki_risk_predictor.data import Cohort
from aki_risk_predictor.models import HIDDEN_LAYERS, Params, get_model
from aki_risk_predictor.pipeline import build_pipeline
from aki_risk_predictor.validation import grouped_cv

Budget = Literal["quick", "deep"]

# number of Optuna trials per model, as in the original study
BUDGETS: dict[Budget, dict[str, int]] = {
    "quick": {
        "XGBoost": 3,
        "LightGBM": 2,
        "SVM": 2,
        "LogisticRegression": 2,
        "AdaBoost": 2,
        "GaussianNB": 1,
        "ComplementNB": 1,
        "MLP": 2,
        "TabPFN-3.5": 0,
    },
    "deep": {
        "XGBoost": 200,
        "LightGBM": 20,
        "SVM": 5,
        "LogisticRegression": 5,
        "AdaBoost": 5,
        "GaussianNB": 2,
        "ComplementNB": 2,
        "MLP": 5,
        "TabPFN-3.5": 0,
    },
}


def _xgboost(trial: optuna.Trial, y: np.ndarray) -> Params:
    ratio = float((y == 0).sum() / max(1, (y == 1).sum()))
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", ratio * 0.5, ratio * 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
    }


def _lightgbm(trial: optuna.Trial, _y: np.ndarray) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }


def _logistic(trial: optuna.Trial, _y: np.ndarray) -> Params:
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _svm(trial: optuna.Trial, _y: np.ndarray) -> Params:
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    params: Params = {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "kernel": kernel,
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }
    if kernel != "linear":
        params["gamma"] = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
    return params


def _adaboost(trial: optuna.Trial, _y: np.ndarray) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
        "estimator_max_depth": trial.suggest_int("estimator_max_depth", 1, 5),
    }


def _gaussian_nb(trial: optuna.Trial, _y: np.ndarray) -> Params:
    return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)}


def _complement_nb(trial: optuna.Trial, _y: np.ndarray) -> Params:
    return {"alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True)}


def _mlp(trial: optuna.Trial, _y: np.ndarray) -> Params:
    return {
        "hidden_layer_size": trial.suggest_categorical("hidden_layer_size", list(HIDDEN_LAYERS)),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-1, log=True),
    }


SEARCH_SPACES: dict[str, Callable[[optuna.Trial, np.ndarray], Params]] = {
    "XGBoost": _xgboost,
    "LightGBM": _lightgbm,
    "LogisticRegression": _logistic,
    "SVM": _svm,
    "AdaBoost": _adaboost,
    "GaussianNB": _gaussian_nb,
    "ComplementNB": _complement_nb,
    "MLP": _mlp,
}


@dataclass(frozen=True)
class CVSettings:
    n_splits: int = 5
    n_features: int = 20
    seed: int = 42


def grouped_cv_auc(name: str, params: Params, cohort: Cohort, cv: CVSettings) -> float:
    """Mean AUC over patient-grouped folds, with all preprocessing fitted inside each fold."""
    pipeline = build_pipeline(get_model(name, params), n_features=cv.n_features, seed=cv.seed)
    scores = cross_val_score(
        pipeline,
        cohort.X,
        cohort.y,
        groups=cohort.groups,
        cv=grouped_cv(cv.n_splits, cv.seed),
        scoring="roc_auc",
    )
    return float(np.mean(scores))


def tune(
    name: str,
    cohort: Cohort,
    *,
    budget: Budget = "quick",
    cv: CVSettings | None = None,
) -> Params:
    """Best hyperparameters found by TPE within the budget, or {} when the model is not tuned."""
    cv = cv or CVSettings()
    n_trials = BUDGETS[budget].get(name, 0)
    if n_trials == 0 or name not in SEARCH_SPACES:
        return {}
    space = SEARCH_SPACES[name]
    labels = cohort.y.to_numpy()

    def objective(trial: optuna.Trial) -> float:
        return grouped_cv_auc(name, space(trial, labels), cohort, cv)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cv.seed)
    )
    study.optimize(objective, n_trials=n_trials)
    # replaying the best trial turns Optuna's flat choices back into model parameters
    best = cast("optuna.Trial", optuna.trial.FixedTrial(study.best_params))
    return space(best, labels)
