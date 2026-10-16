"""Command line entry point: `aki <command>`."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from aki_risk_predictor import experiment
from aki_risk_predictor.config import AkiConfig, load_config
from aki_risk_predictor.synthetic import generate_cohort, write_json

app = typer.Typer(help="AKI risk prediction for ICU patients with diabetic ketoacidosis.")
ConfigOption = Annotated[Path, typer.Option("--config", help="YAML config")]
DEFAULT_CONFIG = Path("configs/default.yaml")


def _config(path: Path) -> AkiConfig:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(path) if path.exists() else AkiConfig()
    print(f"data: {cfg.train_path}" + (" (synthetic sample)" if cfg.uses_sample_data else ""))
    return cfg


@app.command()
def synthesize(
    out: Annotated[Path, typer.Option(help="output JSON file")],
    n: Annotated[int, typer.Option(help="number of ICU stays")] = 600,
    seed: Annotated[int, typer.Option(help="random seed")] = 0,
    unlabelled: Annotated[bool, typer.Option(help="leave out akdPositive")] = False,
) -> None:
    """Write a synthetic cohort in the MIMIC-IV extract schema."""
    write_json(generate_cohort(n, seed=seed, labelled=not unlabelled), out)
    print(f"wrote {n} stays to {out}")


@app.command()
def prepare(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Split labelled stays into train and holdout by patient, write Table 1."""
    train, holdout = experiment.prepare(_config(config))
    print(f"train: {len(train)} stays, holdout: {len(holdout)} stays")


@app.command("select-features")
def select_features(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Rank features by mean |SHAP| on the training split."""
    print(experiment.select_features(_config(config)).to_string())


@app.command()
def tune(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Search hyperparameters with Optuna and patient-grouped CV."""
    for name, params in experiment.tune_all(_config(config)).items():
        print(name, params)


@app.command()
def train(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Cross-validate and fit every model on the training split."""
    print(experiment.train_all(_config(config)).to_string(index=False))


@app.command()
def evaluate(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Score trained models on the holdout and draw ROC, calibration and decision curves."""
    print(experiment.evaluate_all(_config(config)).to_string(index=False))


@app.command()
def predict(
    config: ConfigOption = DEFAULT_CONFIG,
    out: Annotated[Path, typer.Option(help="CSV with one row per stay")] = Path(
        "results/models/predictions.csv"
    ),
) -> None:
    """Predict AKI risk for the unlabelled stays in test_path."""
    predictions = experiment.predict_file(_config(config), out)
    print(f"wrote {len(predictions)} predictions to {out}")


@app.command()
def report(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Run prepare, select-features, tune, train and evaluate in one go."""
    print(experiment.run_all(_config(config)).to_string(index=False))
