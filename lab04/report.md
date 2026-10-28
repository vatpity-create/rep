# Lab 4: Data pipeline and quality

## Run

From the repository root:

```powershell
python main.py
```

The command fetches Adult version 2, validates the original frame, splits it,
fits preprocessing and both LogisticRegression baselines, and writes the results.
The problem definition is in [problem_definition.md](problem_definition.md).
The schema and settings are in [config.yaml](config.yaml).

## Method

The 48,842 rows are split 70/30 with income stratification and random_state=42
before cleaning. Exact duplicates are removed from training only:
34189 training rows become 34165; all
14653 test rows are retained.

Numerical missing values use training medians and categorical missing values use
training modes. All six numerical features are clipped to training 1st/99th
percentiles. Clipped numerical columns are standardised and categorical columns
are one-hot encoded; both operations are fitted on training data only.
Unknown test categories are ignored by the encoder. No income labels are imputed.

Derived features are calculated after numerical imputation and clipping:

```text
capital_net = capital-gain - capital-loss
hours_per_age = hours-per-week / (age - 16)
```

Both baselines use the same rows, split and LogisticRegression settings:
lbfgs, C=1, max_iter=2000, random_state=42, one BLAS thread.
The positive class for binary F1 is `>50K`.

## Test results

| Features | Encoded columns | Accuracy | F1 |
| --- | --- | --- | --- |
| without_derived | 105 | 0.853955 | 0.660102 |
| with_derived | 107 | 0.857777 | 0.669626 |

`capital_net` is a linear combination of existing inputs; it adds no independent
information, although regularisation can change the fitted coefficients.
`hours_per_age` introduces a nonlinear ratio. The table records their measured
combined effect without selecting settings on the test set.

## Validation and label quality

The [validation report](results/validation_report.md) lists schema, range,
missing-value and duplicate checks. The duplicate-conflict audit found
5 identical-feature groups with different income
labels, covering 10 rows. These labels are reported
without correction. The sampled rows were re-checked against the data
dictionary; reviewed count:
**50/50**. Review findings are in `results/label_sample.csv`.

There are 27 test rows whose feature values
also occur in retained training data. Train-only preprocessing prevents fitted
statistics from leaking across partitions. The prescribed duplicate policy does
not establish person-disjoint or duplicate-disjoint evaluation.

## Reproducibility

`results/preprocessor.joblib` contains the fitted transformer with both derived
features. `results/processed_dataset.npz` stores processed train/test features,
labels, feature names and original split indices, including removed duplicate IDs.
The downloaded source is cached under the ignored `data/openml/` directory.
`results/configuration.yaml` records settings, versions, counts and this SHA256:

```text
446bc9034f92443b731ea1bb13d8bf3005aba8fb065737f4541b4c5f7dd42d24
```

`results/processed_dataset.sha256` hashes the exact NPZ file bytes. Fixed data
version, seeds, column order and package versions allow a second one-command run
to reproduce the same processed dataset. The preprocessor reload and preservation
of raw data and the held-out partition are checked during every run.

Source, licence, bias and privacy details are in [data_card.md](data_card.md).
