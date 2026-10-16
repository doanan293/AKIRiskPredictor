"""Turn the JSON extract into one feature row per ICU stay."""

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

TIME_SERIES = (
    "ag", "bg", "bicarbonate", "bun", "calcium", "dbp", "gcs", "gcs_unable", "hb", "hr",
    "phosphate", "plt", "rr", "sbp", "scr", "wbc", "weight",
)  # fmt: skip
NUMERIC = ("age", "ckd_stage", "dka_type", "egfr", "oasis", "preiculos", "saps2", "sofa")
FLAGS = (
    "chronic_pulmonary_disease", "congestive_heart_failure", "history_aci", "history_ami",
    "hypertension", "macroangiopathy", "malignant_cancer", "microangiopathy", "uti",
)  # fmt: skip
LIVER_DISEASE = {"NONE": 0, "MILD": 1, "SEVERE": 2}
LABEL = "akd_positive"
CATEGORICAL = ("race",)


def first_value(series: dict[str, float]) -> float:
    """Value at the earliest timestamp (ISO-8601 strings sort chronologically)."""
    if not series:
        return float("nan")
    return float(series[min(series)])


def _row(record: dict[str, object]) -> dict[str, object]:
    measures = cast("dict[str, object]", record.get("measures", {}))
    row: dict[str, object] = {"stayId": record["stayId"], "subjectId": record["subjectId"]}
    if "akdPositive" in record:
        row[LABEL] = int(bool(record["akdPositive"]))
    for name in TIME_SERIES:
        value = measures.get(name)
        row[name] = first_value(cast("dict[str, float]", value)) if value else np.nan
    for name in NUMERIC:
        value = measures.get(name)
        row[name] = float(cast("float", value)) if value is not None else np.nan
    for name in FLAGS:
        # the extract leaves a comorbidity out when the patient does not have it
        row[name] = int(bool(measures.get(name, False)))
    gender = measures.get("gender")
    row["gender_male"] = np.nan if gender is None else int(gender == "M")
    liver = measures.get("liver_disease")
    row["liver_disease"] = LIVER_DISEASE.get(str(liver), np.nan) if liver is not None else np.nan
    race = measures.get("race")
    row["race"] = str(race) if race is not None else "UNKNOWN"
    return row


def load_stays(path: Path) -> pd.DataFrame:
    """One row per ICU stay, indexed by `stayId`.

    Columns: `subjectId`, the label `akd_positive` when the file has it, the first value of every
    time series, numeric scores, comorbidity flags as 0/1, `gender_male`, `liver_disease`
    (0 none, 1 mild, 2 severe) and `race` as a string.
    """
    records = cast("list[dict[str, object]]", json.loads(path.read_text(encoding="utf-8")))
    frame = pd.DataFrame([_row(record) for record in records]).set_index("stayId")
    frame["race"] = frame["race"].astype("string")
    return frame


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """The model inputs: everything except identifiers and the label."""
    return frame.drop(columns=[c for c in ("subjectId", LABEL) if c in frame.columns])
