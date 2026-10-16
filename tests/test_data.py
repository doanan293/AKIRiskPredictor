import json
from pathlib import Path

import pytest

from aki_risk_predictor.data import FLAGS, LABEL, TIME_SERIES, features, load_stays

SAMPLE = Path(__file__).parents[1] / "data" / "sample"


def _write(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "stays.json"
    path.write_text(json.dumps(records))
    return path


def _record(stay_id: int, measures: dict[str, object]) -> dict[str, object]:
    return {
        "subjectId": 1,
        "hadmId": 2,
        "stayId": stay_id,
        "akdPositive": True,
        "measures": measures,
    }


def test_one_row_per_record() -> None:
    raw = json.loads((SAMPLE / "train.json").read_text())
    frame = load_stays(SAMPLE / "train.json")
    assert len(frame) == len(raw)
    assert frame.index.name == "stayId"
    assert set(frame[LABEL].unique()) <= {0, 1}
    assert set(TIME_SERIES) <= set(frame.columns)


def test_unlabelled_file_has_no_label_column() -> None:
    assert LABEL not in load_stays(SAMPLE / "test.json").columns


def test_absent_flag_means_zero(tmp_path: Path) -> None:
    frame = load_stays(_write(tmp_path, [_record(1, {"hypertension": True}), _record(2, {})]))
    assert frame.loc[1, "hypertension"] == 1
    assert frame.loc[2, "hypertension"] == 0
    assert all(frame.loc[2, flag] == 0 for flag in FLAGS)


@pytest.mark.parametrize(("grade", "code"), [("NONE", 0), ("MILD", 1), ("SEVERE", 2)])
def test_liver_disease_is_ordinal(tmp_path: Path, grade: str, code: int) -> None:
    frame = load_stays(_write(tmp_path, [_record(1, {"liver_disease": grade})]))
    assert frame.loc[1, "liver_disease"] == code


def test_first_value_uses_earliest_timestamp_not_dict_order(tmp_path: Path) -> None:
    scr = {"2150-01-01T10:00:00": 3.0, "2150-01-01T02:00:00": 1.5, "2150-01-01T06:00:00": 2.0}
    frame = load_stays(_write(tmp_path, [_record(1, {"scr": scr})]))
    assert frame.loc[1, "scr"] == 1.5


def test_missing_series_is_nan_and_features_drop_ids(tmp_path: Path) -> None:
    frame = load_stays(_write(tmp_path, [_record(1, {"gender": "M", "race": "WHITE"})]))
    assert frame["bg"].isna().all()
    assert frame.loc[1, "gender_male"] == 1
    assert {"subjectId", LABEL}.isdisjoint(features(frame).columns)
