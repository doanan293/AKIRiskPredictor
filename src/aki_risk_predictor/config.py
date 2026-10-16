"""Typed experiment settings read from YAML."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from aki_risk_predictor.models import MODEL_NAMES

LOCAL_DATA = Path("local-data")
SAMPLE_DATA = Path("data/sample")


def data_file(name: str, local_dir: Path = LOCAL_DATA, sample_dir: Path = SAMPLE_DATA) -> Path:
    """The real extract when it is on this machine, otherwise the committed synthetic sample."""
    local = local_dir / name
    return local if local.exists() else sample_dir / name


class AkiConfig(BaseModel):
    train_path: Path = Field(default_factory=lambda: data_file("train.json"))
    test_path: Path = Field(default_factory=lambda: data_file("test.json"))
    output_dir: Path = Path("results")
    models_dir: Path = Path("results/models")  # fitted models stay out of git
    models: list[str] = Field(default_factory=lambda: list(MODEL_NAMES))
    predict_model: str = "LogisticRegression"
    seed: int = 42
    n_features: int = 20
    missing_threshold: float = 0.2
    holdout_size: float = 0.15
    cv_splits: int = 10
    tune_cv_splits: int = 5
    budget: Literal["quick", "deep"] = "quick"
    bootstrap: int = 1000
    min_cell: int = 11

    @property
    def uses_sample_data(self) -> bool:
        return SAMPLE_DATA in self.train_path.parents


def load_config(path: Path) -> AkiConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AkiConfig.model_validate(raw)
