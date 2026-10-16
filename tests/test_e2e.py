import shutil
from pathlib import Path

from typer.testing import CliRunner

from aki_risk_predictor.cli import app

ROOT = Path(__file__).parents[1]
RUNNER = CliRunner()


def test_cli_lists_every_command() -> None:
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("synthesize", "prepare", "select-features", "tune", "train", "evaluate",
                    "predict", "report"):  # fmt: skip
        assert command in result.output


def test_end_to_end_on_synthetic_sample(tmp_path: Path) -> None:
    sample = tmp_path / "sample"
    shutil.copytree(ROOT / "data" / "sample", sample)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"train_path: {sample / 'train.json'}\n"
        f"test_path: {sample / 'test.json'}\n"
        f"output_dir: {tmp_path / 'results'}\n"
        f"models_dir: {tmp_path / 'models'}\n"
        "models: [LogisticRegression, XGBoost]\n"
        "cv_splits: 3\ntune_cv_splits: 3\nbootstrap: 100\nn_features: 10\n"
    )
    for command in ("prepare", "train", "evaluate"):
        result = RUNNER.invoke(app, [command, "--config", str(config)])
        assert result.exit_code == 0, result.output

    results = tmp_path / "results"
    for name in ("table_1.md", "cv.md", "holdout.md", "roc_holdout.png"):
        assert (results / name).exists(), name
    assert "LogisticRegression" in (results / "holdout.md").read_text()

    out = tmp_path / "predictions.csv"
    result = RUNNER.invoke(app, ["predict", "--config", str(config), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert len(out.read_text().strip().splitlines()) == 151


def test_synthesize_writes_requested_stays(tmp_path: Path) -> None:
    out = tmp_path / "cohort.json"
    result = RUNNER.invoke(app, ["synthesize", "--n", "12", "--seed", "3", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_text().count('"stayId"') == 12
