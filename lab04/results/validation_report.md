# Validation report

## Ingested schema

The source frame has 48,842 rows and 15 columns. `class` is the source name for `income`.
`?` and parser-produced nulls are missing values, not invalid categories. No rows are dropped by validation.

| Field | Expected | Observed dtype | Missing | Literal ? | Type errors | Range errors | Category errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| age | int | int64 | 0 | 0 | 0 | 0 | 0 |
| workclass | category | category | 2799 | 0 | 0 | 0 | 0 |
| fnlwgt | int | int64 | 0 | 0 | 0 | 0 | 0 |
| education | category | category | 0 | 0 | 0 | 0 | 0 |
| education-num | int | int64 | 0 | 0 | 0 | 0 | 0 |
| marital-status | category | category | 0 | 0 | 0 | 0 | 0 |
| occupation | category | category | 2809 | 0 | 0 | 0 | 0 |
| relationship | category | category | 0 | 0 | 0 | 0 | 0 |
| race | category | category | 0 | 0 | 0 | 0 | 0 |
| sex | category | category | 0 | 0 | 0 | 0 | 0 |
| capital-gain | int | int64 | 0 | 0 | 0 | 0 | 0 |
| capital-loss | int | int64 | 0 | 0 | 0 | 0 | 0 |
| hours-per-week | int | int64 | 0 | 0 | 0 | 0 | 0 |
| native-country | category | category | 857 | 0 | 0 | 0 | 0 |
| income | category | category | 0 | 0 | 0 | 0 | 0 |

Total violations: **0**. Row-level findings are in `validation_issues.csv`.

Required ranges: age 17–90; hours-per-week 1–99; capital-gain 0–99,999; capital-loss 0–4,356.
All categorical vocabularies, including `?`, are defined in `../config.yaml`.

## Split and duplicate checks

- Training rows before cleaning: 34189.
- Exact duplicate training rows removed: 24.
- Training rows retained: 34165.
- Test rows retained: 14653; no test rows or labels were removed or changed.
- Exact duplicates in the complete source frame: 52.
- Test rows with features also present in retained training data: 27.

Exact training duplicates are detected on all 15 original columns before imputation or clipping.
Cross-partition matches are reported; the required train-only deduplication does not remove them.

## Label-quality checks

- Identical-feature groups with conflicting income labels: 5.
- Rows in those groups: 10; see `label_conflicts.csv`.
- Seeded sample: 50 rows; dictionary review recorded for 50 rows.
- Sampled records and review findings: `label_sample.csv`; saved review input: `../label_review.csv`.
- Review checks values against the dictionary; it cannot establish the true income of an individual.

## Training-only fitted statistics

| Numerical feature | Median | 1st percentile | 99th percentile |
| --- | --- | --- | --- |
| age | 37 | 17 | 74 |
| fnlwgt | 178272 | 27035.2 | 509629 |
| education-num | 10 | 3 | 16 |
| capital-gain | 0 | 0 | 15024 |
| capital-loss | 0 | 0 | 1980 |
| hours-per-week | 40 | 8 | 80 |

| Categorical feature | Training mode |
| --- | --- |
| workclass | Private |
| education | HS-grad |
| marital-status | Married-civ-spouse |
| occupation | Prof-specialty |
| relationship | Husband |
| race | White |
| sex | Male |
| native-country | United-States |

Medians, modes, clipping thresholds, scaling and one-hot categories are fitted on retained training rows only.
Test data are transformed with these fixed statistics. Unknown categories are encoded as all-zero blocks.
The saved preprocessor is reloaded and required to reproduce test transformations exactly.
