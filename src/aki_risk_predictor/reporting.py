"""Aggregate tables and figures. Nothing written here identifies a patient."""

from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from aki_risk_predictor.data import FLAGS, LABEL
from aki_risk_predictor.validation import bootstrap_ci

CONTINUOUS = ("age", "weight", "egfr", "scr", "bun", "bg", "bicarbonate", "sofa", "saps2", "oasis")
BINARY = ("gender_male", *FLAGS)


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Probability cut-off that maximizes sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y, p)
    best = int(np.argmax(tpr - fpr))
    return float(min(1.0, thresholds[best]))


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (np.asarray(p) >= threshold).astype(int)
    y = np.asarray(y)
    positives = max(1, int((y == 1).sum()))
    negatives = max(1, int((y == 0).sum()))
    return {
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "sensitivity": float(((predicted == 1) & (y == 1)).sum() / positives),
        "specificity": float(((predicted == 0) & (y == 0)).sum() / negatives),
    }


def holdout_row(
    y: np.ndarray, p: np.ndarray, threshold: float, *, n_bootstrap: int, seed: int
) -> dict[str, object]:
    auc, low, high = bootstrap_ci(y, p, roc_auc_score, n=n_bootstrap, seed=seed)
    row: dict[str, object] = {"AUC (95% CI)": f"{auc:.3f} ({low:.3f}-{high:.3f})", "_auc": auc}
    row.update({k: round(v, 3) for k, v in threshold_metrics(y, p, threshold).items()})
    return row


def results_table(rows: Mapping[str, dict[str, object]]) -> pd.DataFrame:
    """One row per model, best AUC first."""
    table = pd.DataFrame.from_dict(dict(rows), orient="index")
    table = table.sort_values("_auc", ascending=False).drop(columns="_auc")
    table.index.name = "model"
    return table.reset_index()


def _count(n: int, total: int, min_cell: int) -> str:
    if 0 < n < min_cell:
        return f"<{min_cell}"
    return f"{n} ({100 * n / max(1, total):.1f}%)"


def table_1(frame: pd.DataFrame, min_cell: int = 11) -> pd.DataFrame:
    """Baseline characteristics by outcome. Counts below `min_cell` are shown as "<11"."""
    groups = {
        "overall": frame,
        "AKI": frame[frame[LABEL] == 1],
        "no AKI": frame[frame[LABEL] == 0],
    }
    rows: dict[str, dict[str, str]] = {
        "stays, n": {name: _count(len(g), len(frame), min_cell) for name, g in groups.items()}
    }
    for column in CONTINUOUS:
        if column in frame.columns:
            rows[f"{column}, median [IQR]"] = {
                name: (
                    f"{g[column].median():.1f} "
                    f"[{g[column].quantile(0.25):.1f}, {g[column].quantile(0.75):.1f}]"
                )
                for name, g in groups.items()
            }
    for column in BINARY:
        if column in frame.columns:
            rows[f"{column}, n (%)"] = {
                name: _count(int(g[column].sum()), len(g), min_cell) for name, g in groups.items()
            }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "characteristic"
    return table.reset_index()


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y)
    benefit = []
    for t in thresholds:
        predicted = p >= t
        tp = float((predicted & (y == 1)).sum())
        fp = float((predicted & (y == 0)).sum())
        benefit.append(tp / n - fp / n * t / (1 - t))
    return np.array(benefit)


def plot_roc(curves: Mapping[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.set_title("ROC on the patient-level holdout")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_calibration(curves: Mapping[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (y, p) in curves.items():
        observed, predicted = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed AKI rate")
    ax.set_title("Calibration on the holdout")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_decision_curve(curves: Mapping[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    thresholds = np.linspace(0.01, 0.8, 80)
    fig, ax = plt.subplots(figsize=(7, 5))
    y_any = next(iter(curves.values()))[0]
    prevalence = float(np.mean(y_any))
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    for name, (y, p) in curves.items():
        ax.plot(thresholds, net_benefit(y, p, thresholds), label=name)
    ax.plot(thresholds, treat_all, linestyle="--", color="grey", label="treat all")
    ax.axhline(0, color="black", linewidth=0.8, label="treat none")
    ax.set_ylim(-0.05, max(0.05, prevalence + 0.05))
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.set_title("Decision curve on the holdout")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_markdown(table: pd.DataFrame, path: Path, title: str, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"# {title}\n\n" + (f"{note}\n\n" if note else "") + table.to_markdown(index=False)
    path.write_text(body + "\n", encoding="utf-8")
