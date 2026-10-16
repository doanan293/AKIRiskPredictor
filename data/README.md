# Data

## data/sample (committed)

`train.json` (600 labelled stays) and `test.json` (150 unlabelled stays) are synthetic. They were generated with

```bash
uv run aki synthesize --n 600 --seed 0 --out data/sample/train.json
uv run aki synthesize --n 150 --seed 1 --unlabelled --out data/sample/test.json
```

and contain no real patient data. Identifiers are drawn from 90,000,000 to 99,999,999, a range MIMIC-IV does not use. Each record has the same schema as the real extract:

| field | meaning |
|---|---|
| `subjectId`, `hadmId`, `stayId` | patient, hospital admission and ICU stay identifiers (about 1.36 stays per patient) |
| `akdPositive` | label: acute kidney injury within seven days of ICU admission (training file only) |
| `measures` | time series as `{timestamp: value}` (labs, vital signs, GCS, weight), scores (SOFA, SAPS II, OASIS), eGFR, age, comorbidity flags, gender, liver disease grade, race |

Distributions roughly follow aggregate statistics of the real cohort (medians, missingness, how often each comorbidity appears, 38.9 % positive). The label is driven by creatinine, eGFR, age, weight and SOFA plus noise, so the models have a signal to find. Results on synthetic data say nothing about clinical performance.

## Real data (not committed)

The real cohort (970 labelled and 243 unlabelled ICU stays of adults with diabetic ketoacidosis) comes from [MIMIC-IV](https://physionet.org/content/mimiciv/) on PhysioNet. Access requires credentialing, CITI training and signing the PhysioNet data use agreement, which forbids sharing the data. With access, put the extracted `train.json` and `test.json` in `local-data/`, which is ignored by git. The CLI uses `local-data/` when it exists and falls back to `data/sample/`.
