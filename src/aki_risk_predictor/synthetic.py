"""Synthetic ICU stays that follow the MIMIC-IV extract schema but contain no real patient data.

Distributions are rough matches to aggregate statistics of the real cohort (medians, ranges,
missingness, how often each comorbidity flag appears). The label depends on eGFR, creatinine,
age, weight and SOFA so that models have a real signal to find.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ID_LOW = 90_000_000
ID_HIGH = 100_000_000
STAYS_PER_SUBJECT = 1.36
SCR_MISSING = 0.004
WEIGHT_MISSING = 0.015
CKD_STAGE_PRESENT = 0.2
# The real extract has about 35 vital-sign readings per stay. Fewer keep the sample files small.
VITAL_POINTS_LOW = 6
VITAL_POINTS_HIGH = 13

# name: (median of first value, spread as log-normal sigma, points in the first 24 h, missing rate)
LABS: dict[str, tuple[float, float, int, float]] = {
    "ag": (19.0, 0.35, 4, 0.004),
    "bg": (262.0, 0.55, 4, 0.004),
    "bicarbonate": (17.0, 0.35, 4, 0.004),
    "bun": (22.0, 0.6, 4, 0.004),
    "calcium": (8.3, 0.1, 4, 0.008),
    "phosphate": (2.6, 0.45, 4, 0.007),
    "hb": (11.2, 0.18, 2, 0.056),
    "plt": (238.0, 0.35, 2, 0.054),
    "wbc": (11.4, 0.45, 2, 0.056),
}
VITALS: dict[str, tuple[float, float]] = {
    "dbp": (71.0, 0.2),
    "sbp": (129.0, 0.15),
    "hr": (101.0, 0.17),
    "rr": (19.0, 0.25),
}
ALWAYS_PRESENT_FLAGS: dict[str, float] = {
    "chronic_pulmonary_disease": 0.16,
    "congestive_heart_failure": 0.14,
    "malignant_cancer": 0.034,
}
# These flags only appear in the extract when they are true.
SPARSE_FLAGS: dict[str, float] = {
    "history_aci": 0.071,
    "history_ami": 0.147,
    "hypertension": 0.388,
    "macroangiopathy": 0.233,
    "microangiopathy": 0.355,
    "uti": 0.113,
}
LIVER_DISEASE = (("NONE", 0.906), ("MILD", 0.073), ("SEVERE", 0.021))
RACES = (
    ("WHITE", 0.55),
    ("BLACK/AFRICAN AMERICAN", 0.26),
    ("UNKNOWN", 0.045),
    ("OTHER", 0.035),
    ("HISPANIC OR LATINO", 0.02),
    ("HISPANIC/LATINO - PUERTO RICAN", 0.02),
    ("WHITE - OTHER EUROPEAN", 0.015),
    ("HISPANIC/LATINO - DOMINICAN", 0.015),
    ("ASIAN", 0.02),
)
_LABEL_WEIGHTS = {"scr": 1.0, "egfr": -0.7, "age": 0.4, "weight": 0.3, "sofa": 0.6}

Record = dict[str, object]


def _pick(rng: np.random.Generator, options: tuple[tuple[str, float], ...]) -> str:
    names = [name for name, _ in options]
    weights = np.array([weight for _, weight in options])
    return str(rng.choice(names, p=weights / weights.sum()))


def _series(
    rng: np.random.Generator, start: np.datetime64, first: float, n_points: int, sigma: float
) -> dict[str, float]:
    """Values drifting around `first`, at increasing timestamps within 24 hours."""
    seconds = np.sort(rng.uniform(0, 24 * 3600, size=n_points))
    seconds[0] = min(seconds[0], rng.uniform(0, 7200))
    stamps = np.datetime_as_string(start + seconds.astype("timedelta64[s]"), unit="s")
    values = first * np.exp(np.cumsum(rng.normal(0, sigma / 4, size=n_points)))
    values[0] = first
    return dict(zip(stamps.tolist(), values.round(1).tolist(), strict=True))


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() or 1.0)


def _labels(
    rng: np.random.Generator, drivers: dict[str, np.ndarray], positive_rate: float
) -> np.ndarray:
    n = len(next(iter(drivers.values())))
    score = rng.normal(0, 1.0, size=n)
    for name, weight in _LABEL_WEIGHTS.items():
        score += weight * _standardize(drivers[name])
    threshold = float(np.quantile(score, 1 - positive_rate))
    return score > threshold


def _unique_ids(rng: np.random.Generator, n: int) -> list[int]:
    ids: dict[int, None] = {}
    while len(ids) < n:
        for value in rng.integers(ID_LOW, ID_HIGH, size=n - len(ids)):
            ids[int(value)] = None
    return list(ids)


@dataclass(frozen=True)
class _Drivers:
    """Per-stay values that the label depends on."""

    age: np.ndarray
    weight: np.ndarray
    sofa: np.ndarray
    scr: np.ndarray
    egfr: np.ndarray


def _draw_drivers(rng: np.random.Generator, n: int) -> _Drivers:
    age = np.clip(rng.normal(49, 18, size=n), 18, 91).round()
    scr = np.clip(rng.lognormal(np.log(1.1), 0.6, size=n), 0.2, 15).round(1)
    egfr = 110 * scr**-0.9 * 0.993 ** (age - 40) * rng.lognormal(0, 0.1, size=n)
    return _Drivers(
        age=age,
        weight=np.clip(rng.lognormal(np.log(74), 0.25, size=n), 35, 200).round(1),
        sofa=np.clip(rng.poisson(2.8, size=n), 0, 18),
        scr=scr,
        egfr=np.clip(egfr, 3, 170),
    )


def _time_series(
    rng: np.random.Generator, start: np.datetime64, scr: float, weight: float
) -> dict[str, object]:
    measures: dict[str, object] = {}
    for name, (median, sigma, points, missing) in LABS.items():
        if rng.random() >= missing:
            first = float(rng.lognormal(np.log(median), sigma))
            measures[name] = _series(rng, start, first, max(1, int(rng.poisson(points))), sigma)
    if rng.random() >= SCR_MISSING:
        measures["scr"] = _series(rng, start, scr, max(1, int(rng.poisson(4))), 0.3)
    for name, (median, sigma) in VITALS.items():
        first = float(rng.lognormal(np.log(median), sigma))
        measures[name] = _series(
            rng, start, first, int(rng.integers(VITAL_POINTS_LOW, VITAL_POINTS_HIGH)), sigma
        )
    gcs = _series(rng, start, float(rng.choice([13, 14, 15, 15, 15])), int(rng.integers(3, 8)), 0)
    measures["gcs"] = gcs
    measures["gcs_unable"] = dict.fromkeys(gcs, 0)
    if rng.random() >= WEIGHT_MISSING:
        measures["weight"] = _series(rng, start, weight, 1, 0)
    return measures


def _static(rng: np.random.Generator, drivers: _Drivers, i: int) -> dict[str, object]:
    measures: dict[str, object] = {
        "age": int(drivers.age[i]),
        "egfr": float(drivers.egfr[i]),
        "sofa": int(drivers.sofa[i]),
        "saps2": int(np.clip(rng.normal(28 + 2 * drivers.sofa[i], 10), 5, 90)),
        "oasis": float(round(float(np.clip(rng.normal(27, 8), 5, 60)))),
        "preiculos": round(float(rng.exponential(150)), 2),
        "dka_type": int(rng.choice([1, 2], p=[0.8, 0.2])),
    }
    if rng.random() < CKD_STAGE_PRESENT:
        measures["ckd_stage"] = int(rng.choice(6, p=[0.5, 0.05, 0.1, 0.15, 0.1, 0.1]))
    for name, rate in ALWAYS_PRESENT_FLAGS.items():
        measures[name] = bool(rng.random() < rate)
    for name, rate in SPARSE_FLAGS.items():
        if rng.random() < rate:
            measures[name] = True
    measures["gender"] = str(rng.choice(["F", "M"], p=[0.53, 0.47]))
    measures["liver_disease"] = _pick(rng, LIVER_DISEASE)
    measures["race"] = _pick(rng, RACES)
    return measures


def generate_cohort(
    n_stays: int, *, seed: int, labelled: bool = True, positive_rate: float = 0.389
) -> list[Record]:
    """Generate `n_stays` synthetic ICU stays in the original JSON schema."""
    rng = np.random.default_rng(seed)
    n_subjects = max(1, round(n_stays / STAYS_PER_SUBJECT))
    subject_ids = _unique_ids(rng, n_subjects)
    stay_subjects = [subject_ids[i % n_subjects] for i in range(n_stays)]
    rng.shuffle(stay_subjects)
    admission_ids = _unique_ids(rng, n_stays)
    stay_ids = _unique_ids(rng, n_stays)
    drivers = _draw_drivers(rng, n_stays)
    labels = _labels(
        rng,
        {
            "scr": np.log(drivers.scr),
            "egfr": drivers.egfr,
            "age": drivers.age,
            "weight": drivers.weight,
            "sofa": drivers.sofa,
        },
        positive_rate,
    )
    # dates are shifted far into the future, as in MIMIC-IV
    starts = np.datetime64("2110-01-01T00:00:00") + rng.integers(
        0, 90 * 365 * 24 * 60, size=n_stays
    ).astype("timedelta64[m]")

    records: list[Record] = []
    for i in range(n_stays):
        measures = _time_series(rng, starts[i], float(drivers.scr[i]), float(drivers.weight[i]))
        measures.update(_static(rng, drivers, i))
        record: Record = {
            "subjectId": stay_subjects[i],
            "hadmId": admission_ids[i],
            "stayId": stay_ids[i],
        }
        if labelled:
            record["akdPositive"] = bool(labels[i])
        record["measures"] = dict(sorted(measures.items()))
        records.append(record)
    return records


def write_json(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, separators=(",", ":")) + "\n", encoding="utf-8")
