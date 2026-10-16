"""The study steps: split, select features, tune, train, evaluate, predict."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from aki_risk_predictor.config import AkiConfig
from aki_risk_predictor.data import LABEL, Cohort, features, load_stays
from aki_risk_predictor.models import Params, get_model
from aki_risk_predictor.pipeline import ShapSelector, build_pipeline
from aki_risk_predictor.predict import predict
from aki_risk_predictor.reporting import (
    holdout_row,
    plot_calibration,
    plot_decision_curve,
    plot_roc,
    results_table,
    table_1,
    threshold_metrics,
    write_markdown,
    youden_threshold,
)
from aki_risk_predictor.tuning import CVSettings, tune
from aki_risk_predictor.validation import grouped_cv, patient_holdout

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FittedModel:
    pipeline: Pipeline
    threshold: float


def _split_path(cfg: AkiConfig) -> Path:
    return cfg.models_dir / "split.json"


def _params_path(cfg: AkiConfig) -> Path:
    return cfg.output_dir / "tuned_params.json"


def _model_path(cfg: AkiConfig, name: str) -> Path:
    return cfg.models_dir / f"{name}.joblib"


def _data_note(cfg: AkiConfig) -> str:
    if cfg.uses_sample_data:
        return (
            "Computed on the synthetic sample in data/sample. The numbers mean nothing clinically."
        )
    return "Computed on the MIMIC-IV cohort. Only aggregate values are reported."


def available_models(cfg: AkiConfig) -> list[str]:
    names: list[str] = []
    for name in cfg.models:
        try:
            get_model(name)
        except ModuleNotFoundError as error:
            log.warning("skipping %s: %s", name, error)
            continue
        names.append(name)
    return names


def prepare(cfg: AkiConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split labelled stays into train and holdout by patient and write Table 1."""
    stays = load_stays(cfg.train_path)
    train_ids, holdout_ids = patient_holdout(stays, test_size=cfg.holdout_size, seed=cfg.seed)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    split = {"train": [int(i) for i in train_ids], "holdout": [int(i) for i in holdout_ids]}
    _split_path(cfg).write_text(json.dumps(split), encoding="utf-8")
    write_markdown(
        table_1(stays, cfg.min_cell),
        cfg.output_dir / "table_1.md",
        "Baseline characteristics",
        _data_note(cfg) + f" Counts below {cfg.min_cell} are suppressed.",
    )
    return stays.loc[train_ids], stays.loc[holdout_ids]


def load_split(cfg: AkiConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not _split_path(cfg).exists():
        return prepare(cfg)
    split = cast("dict[str, list[int]]", json.loads(_split_path(cfg).read_text(encoding="utf-8")))
    stays = load_stays(cfg.train_path)
    return stays.loc[split["train"]], stays.loc[split["holdout"]]


def select_features(cfg: AkiConfig) -> pd.Series:
    """Feature ranking on the whole training split, for reporting only.

    Models never use this ranking. Each one re-selects features inside every training fold.
    """
    train, _ = load_split(cfg)
    pipeline = build_pipeline(
        get_model("LogisticRegression"), n_features=cfg.n_features, seed=cfg.seed
    )
    pipeline.fit(features(train), train[LABEL])
    selector = cast("ShapSelector", pipeline.named_steps["select"])
    ranking = selector.importance_.head(cfg.n_features)
    table = pd.DataFrame({"feature": ranking.index, "mean |SHAP|": ranking.round(4).to_numpy()})
    write_markdown(
        table,
        cfg.output_dir / "selected_features.md",
        f"Top {cfg.n_features} features by mean |SHAP| (XGBoost, training split)",
        _data_note(cfg),
    )
    return ranking


def tune_all(cfg: AkiConfig) -> dict[str, Params]:
    train, _ = load_split(cfg)
    cohort = Cohort.from_stays(train)
    settings = CVSettings(cfg.tune_cv_splits, cfg.n_features, cfg.seed)
    best: dict[str, Params] = {}
    for name in available_models(cfg):
        log.info("tuning %s (%s budget)", name, cfg.budget)
        best[name] = tune(name, cohort, budget=cfg.budget, cv=settings)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _params_path(cfg).write_text(json.dumps(best, indent=2, default=str) + "\n", encoding="utf-8")
    return best


def _tuned_params(cfg: AkiConfig) -> dict[str, Params]:
    if not _params_path(cfg).exists():
        return tune_all(cfg)
    return cast("dict[str, Params]", json.loads(_params_path(cfg).read_text(encoding="utf-8")))


def _out_of_fold(
    pipeline: Pipeline, cohort: Cohort, cfg: AkiConfig
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold probabilities and per-fold AUCs from patient-grouped CV."""
    oof = np.zeros(len(cohort.y))
    aucs: list[float] = []
    folds = grouped_cv(cfg.cv_splits, cfg.seed).split(cohort.X, cohort.y, groups=cohort.groups)
    for train_rows, val_rows in folds:
        model = cast("Pipeline", clone(pipeline))
        model.fit(cohort.X.iloc[train_rows], cohort.y.iloc[train_rows])
        oof[val_rows] = model.predict_proba(cohort.X.iloc[val_rows])[:, 1]
        aucs.append(float(roc_auc_score(cohort.y.iloc[val_rows], oof[val_rows])))
    return oof, aucs


def train_all(cfg: AkiConfig) -> pd.DataFrame:
    """Cross-validate every model on the training split, then refit it on the whole split."""
    train, _ = load_split(cfg)
    cohort = Cohort.from_stays(train)
    params = _tuned_params(cfg)
    rows: dict[str, dict[str, object]] = {}
    for name in available_models(cfg):
        log.info("training %s", name)
        pipeline = build_pipeline(
            get_model(name, params.get(name)),
            n_features=cfg.n_features,
            missing_threshold=cfg.missing_threshold,
            seed=cfg.seed,
        )
        oof, aucs = _out_of_fold(pipeline, cohort, cfg)
        y = cohort.y.to_numpy()
        threshold = youden_threshold(y, oof)
        rows[name] = {
            "AUC mean (SD) over folds": f"{np.mean(aucs):.3f} ({np.std(aucs, ddof=1):.3f})",
            "_auc": float(np.mean(aucs)),
            **{k: round(v, 3) for k, v in threshold_metrics(y, oof, threshold).items()},
            "threshold": round(threshold, 3),
        }
        pipeline.fit(cohort.X, cohort.y)
        joblib.dump(FittedModel(pipeline, threshold), _model_path(cfg, name))
    table = results_table(rows)
    write_markdown(
        table,
        cfg.output_dir / "cv.md",
        f"{cfg.cv_splits}-fold cross-validation on the training split, grouped by patient",
        _data_note(cfg)
        + " F1, sensitivity and specificity use the Youden threshold of the out-of-fold"
        " predictions.",
    )
    return table


def load_model(cfg: AkiConfig, name: str) -> FittedModel:
    return cast("FittedModel", joblib.load(_model_path(cfg, name)))


def evaluate_all(cfg: AkiConfig) -> pd.DataFrame:
    """Score every trained model once on the patient-level holdout."""
    _, holdout = load_split(cfg)
    y = holdout[LABEL].to_numpy()
    rows: dict[str, dict[str, object]] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in available_models(cfg):
        if not _model_path(cfg, name).exists():
            continue
        fitted = load_model(cfg, name)
        p = fitted.pipeline.predict_proba(features(holdout))[:, 1]
        curves[name] = (y, p)
        rows[name] = holdout_row(y, p, fitted.threshold, n_bootstrap=cfg.bootstrap, seed=cfg.seed)
    table = results_table(rows)
    write_markdown(
        table,
        cfg.output_dir / "holdout.md",
        f"Patient-level holdout ({len(holdout)} stays, no patient shared with training)",
        _data_note(cfg)
        + f" AUC intervals come from {cfg.bootstrap} bootstrap resamples. F1, sensitivity and"
        " specificity use each model's threshold from cross-validation.",
    )
    plot_roc(curves, cfg.output_dir / "roc_holdout.png")
    plot_calibration(curves, cfg.output_dir / "calibration_holdout.png")
    plot_decision_curve(curves, cfg.output_dir / "decision_curve_holdout.png")
    return table


def predict_file(cfg: AkiConfig, out: Path) -> pd.DataFrame:
    fitted = load_model(cfg, cfg.predict_model)
    stays = load_stays(cfg.test_path)
    predictions = predict(fitted.pipeline, stays, threshold=fitted.threshold)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    return predictions


def run_all(cfg: AkiConfig) -> pd.DataFrame:
    prepare(cfg)
    select_features(cfg)
    tune_all(cfg)
    train_all(cfg)
    return evaluate_all(cfg)
