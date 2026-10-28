from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lab04.src.preprocessing import AdultPreprocessor, CATEGORICAL, FEATURES, NUMERIC

LAB = ROOT / "lab04"
RESULTS = LAB / "results"


def save_csv(frame, path, index=False):
    frame.to_csv(path, index=index, na_rep="?", lineterminator="\n", float_format="%.17g")


def frame_hash(frame):
    payload = frame.to_csv(index=True, na_rep="?", lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate(raw, schema):
    rows, issues = [], []
    expected = [rule.get("source", name) for name, rule in schema.items()]
    if list(raw.columns) != expected:
        raise ValueError("The input columns do not match the configured schema.")
    for name, rule in schema.items():
        source = rule.get("source", name)
        values = raw[source]
        missing = values.isna() | values.eq("?")
        if rule["type"] == "int":
            numeric = pd.to_numeric(values, errors="coerce")
            invalid_type = ~missing & (~np.isfinite(numeric) | numeric.mod(1).ne(0))
            dtype_ok = pd.api.types.is_numeric_dtype(values)
            invalid_category = pd.Series(False, index=raw.index)
            bounds = rule.get("range")
            invalid_range = ~missing & ~invalid_type & ~numeric.between(*bounds) if bounds else pd.Series(False, index=raw.index)
        else:
            invalid_type = pd.Series(False, index=raw.index)
            dtype_ok = isinstance(values.dtype, pd.CategoricalDtype)
            invalid_category = ~missing & ~values.isin(rule["categories"])
            invalid_range = pd.Series(False, index=raw.index)
        if not dtype_ok:
            issues.append(dict(source_row="", column=name, value=str(values.dtype), issue="column_dtype"))
        if name == "income":
            for row_id in raw.index[missing]:
                issues.append(dict(source_row=int(row_id), column=name, value="?", issue="missing_target"))
        for kind, mask in (("type", invalid_type), ("range", invalid_range), ("category", invalid_category)):
            for row_id in raw.index[mask]:
                issues.append(dict(source_row=int(row_id), column=name, value=str(values.loc[row_id]), issue=kind))
        rows.append(dict(column=name, expected=rule["type"], dtype=str(values.dtype),
                         missing=int(missing.sum()), question_marks=int(values.eq("?").sum()),
                         type_violations=int(invalid_type.sum()) + int(not dtype_ok),
                         range_violations=int(invalid_range.sum()), category_violations=int(invalid_category.sum())))
    return rows, issues


def audit_labels(raw, config):
    grouped = raw.groupby(FEATURES, observed=True, dropna=False, sort=True)
    label_counts = grouped["class"].nunique()
    conflict_mask = grouped["class"].transform("nunique").gt(1)
    conflicts = raw.loc[conflict_mask].copy()
    conflicts.insert(0, "conflict_group", grouped.ngroup().loc[conflict_mask].to_numpy())
    conflicts.insert(0, "source_row", conflicts.index)
    save_csv(conflicts, RESULTS / "label_conflicts.csv")
    sample = raw.sample(n=config["sample_size"], random_state=config["random_state"]).copy()
    sample.insert(0, "source_row", sample.index)
    sample["record_sha256"] = [frame_hash(raw.loc[[row_id]]) for row_id in sample.index]
    review_path = LAB / "label_review.csv"
    reviewed = False
    if review_path.exists():
        review = pd.read_csv(review_path, keep_default_na=False)
        if set(review.source_row) != set(sample.source_row) or len(review) != len(sample):
            raise ValueError("The saved label review does not match the seeded 50-row sample.")
        sample = sample.merge(review, on=["source_row", "record_sha256"], how="left", validate="one_to_one")
        if sample["finding"].isna().any():
            raise ValueError("The reviewed records have changed.")
        reviewed = True
    else:
        sample["finding"] = "Pending dictionary review"
        save_csv(sample, RESULTS / "label_sample.csv")
        raise FileNotFoundError("The 50-row dictionary review is required in lab04/label_review.csv.")
    save_csv(sample, RESULTS / "label_sample.csv")
    return dict(conflicting_groups=int(label_counts.gt(1).sum()), conflicting_rows=int(conflict_mask.sum()),
                reviewed_rows=len(sample) if reviewed else 0, sample_rows=len(sample))


def fit_baselines(train, test, config):
    y_train = train["class"].eq(config["dataset"]["positive_label"]).to_numpy(dtype=np.int64)
    y_test = test["class"].eq(config["dataset"]["positive_label"]).to_numpy(dtype=np.int64)
    metrics = []
    processed = None
    preprocessor = None
    with threadpool_limits(limits=1):
        for engineered in (False, True):
            transformer = AdultPreprocessor(
                add_features=engineered,
                lower_quantile=config["preprocessing"]["lower_quantile"],
                upper_quantile=config["preprocessing"]["upper_quantile"],
            )
            x_train = transformer.fit_transform(train[FEATURES])
            x_test = transformer.transform(test[FEATURES])
            if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
                raise ValueError("Processed features contain non-finite values.")
            model = LogisticRegression(**{key: value for key, value in config["model"].items() if key != "name"})
            model.fit(x_train, y_train)
            if np.any(model.n_iter_ >= config["model"]["max_iter"]):
                raise RuntimeError("LogisticRegression did not converge.")
            predictions = model.predict(x_test)
            metrics.append(dict(features="with_derived" if engineered else "without_derived",
                                train_rows=len(train), test_rows=len(test), encoded_features=x_train.shape[1],
                                accuracy=accuracy_score(y_test, predictions), f1=f1_score(y_test, predictions)))
            print(f"{metrics[-1]['features']}: accuracy={metrics[-1]['accuracy']:.6f}, F1={metrics[-1]['f1']:.6f}", flush=True)
            if engineered:
                preprocessor = transformer
                processed = dict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                 train_indices=train.index.to_numpy(), test_indices=test.index.to_numpy(),
                                 feature_names=transformer.get_feature_names_out())
    joblib.dump(preprocessor, RESULTS / "preprocessor.joblib", compress=3)
    restored = joblib.load(RESULTS / "preprocessor.joblib")
    np.testing.assert_array_equal(restored.transform(test[FEATURES]), processed["x_test"])
    save_csv(pd.DataFrame(metrics), RESULTS / "metrics.csv")
    statistics = pd.DataFrame(dict(
        feature=NUMERIC, median=preprocessor.numeric_imputer_.statistics_,
        percentile_01=preprocessor.lower_bounds_, percentile_99=preprocessor.upper_bounds_,
    ))
    save_csv(statistics, RESULTS / "preprocessing_statistics.csv")
    return metrics, processed, preprocessor


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(str(value) for value in row) + " |" for row in rows])


def write_validation(validation, issues, counts, labels, transformer):
    lines = ["# Validation report", "", "## Ingested schema", "",
             "The source frame has 48,842 rows and 15 columns. `class` is the source name for `income`.",
             "`?` and parser-produced nulls are missing values, not invalid categories. No rows are dropped by validation.", "",
             table(["Field", "Expected", "Observed dtype", "Missing", "Literal ?", "Type errors", "Range errors", "Category errors"],
                   [[row[key] for key in ["column", "expected", "dtype", "missing", "question_marks",
                                          "type_violations", "range_violations", "category_violations"]] for row in validation]),
             "", f"Total violations: **{len(issues)}**. Row-level findings are in `validation_issues.csv`.",
             "", "Required ranges: age 17–90; hours-per-week 1–99; capital-gain 0–99,999; capital-loss 0–4,356.",
             "All categorical vocabularies, including `?`, are defined in `../config.yaml`.",
             "", "## Split and duplicate checks", "",
             f"- Training rows before cleaning: {counts['train_before']}.",
             f"- Exact duplicate training rows removed: {counts['removed_train_duplicates']}.",
             f"- Training rows retained: {counts['train_after']}.",
             f"- Test rows retained: {counts['test']}; no test rows or labels were removed or changed.",
             f"- Exact duplicates in the complete source frame: {counts['raw_exact_duplicates']}.",
             f"- Test rows with features also present in retained training data: {counts['overlapping_test_feature_rows']}.",
             "", "Exact training duplicates are detected on all 15 original columns before imputation or clipping.",
             "Cross-partition matches are reported; the required train-only deduplication does not remove them.",
             "", "## Label-quality checks", "",
             f"- Identical-feature groups with conflicting income labels: {labels['conflicting_groups']}.",
             f"- Rows in those groups: {labels['conflicting_rows']}; see `label_conflicts.csv`.",
             f"- Seeded sample: {labels['sample_rows']} rows; dictionary review recorded for {labels['reviewed_rows']} rows.",
             "- Sampled records and review findings: `label_sample.csv`; saved review input: `../label_review.csv`.",
             "- Review checks values against the dictionary; it cannot establish the true income of an individual."]
    if transformer is not None:
        lines += ["", "## Training-only fitted statistics", "",
                  table(["Numerical feature", "Median", "1st percentile", "99th percentile"],
                        [[name, f"{median:.10g}", f"{lower:.10g}", f"{upper:.10g}"] for name, median, lower, upper in
                         zip(NUMERIC, transformer.numeric_imputer_.statistics_, transformer.lower_bounds_, transformer.upper_bounds_)]),
                  "", table(["Categorical feature", "Training mode"],
                             zip(CATEGORICAL, transformer.categorical_imputer_.statistics_)),
                  "", "Medians, modes, clipping thresholds, scaling and one-hot categories are fitted on retained training rows only.",
                  "Test data are transformed with these fixed statistics. Unknown categories are encoded as all-zero blocks.",
                  "The saved preprocessor is reloaded and required to reproduce test transformations exactly."]
    (RESULTS / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(metrics, counts, labels, digest):
    content = f"""# Lab 4: Data pipeline and quality

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
{counts['train_before']} training rows become {counts['train_after']}; all
{counts['test']} test rows are retained.

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

{table(['Features', 'Encoded columns', 'Accuracy', 'F1'], [[row['features'], row['encoded_features'], f"{row['accuracy']:.6f}", f"{row['f1']:.6f}"] for row in metrics])}

`capital_net` is a linear combination of existing inputs; it adds no independent
information, although regularisation can change the fitted coefficients.
`hours_per_age` introduces a nonlinear ratio. The table records their measured
combined effect without selecting settings on the test set.

## Validation and label quality

The [validation report](results/validation_report.md) lists schema, range,
missing-value and duplicate checks. The duplicate-conflict audit found
{labels['conflicting_groups']} identical-feature groups with different income
labels, covering {labels['conflicting_rows']} rows. These labels are reported
without correction. The sampled rows were re-checked against the data
dictionary; reviewed count:
**{labels['reviewed_rows']}/50**. Review findings are in `results/label_sample.csv`.

There are {counts['overlapping_test_feature_rows']} test rows whose feature values
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
{digest}
```

`results/processed_dataset.sha256` hashes the exact NPZ file bytes. Fixed data
version, seeds, column order and package versions allow a second one-command run
to reproduce the same processed dataset. The preprocessor reload and preservation
of raw data and the held-out partition are checked during every run.

Source, licence, bias and privacy details are in [data_card.md](data_card.md).
"""
    (LAB / "report.md").write_text(content, encoding="utf-8")


def write_data_card(raw, counts, labels, digest):
    positive = int(raw["class"].eq(">50K").sum())
    female = int(raw["sex"].eq("Female").sum())
    white = int(raw["race"].eq("White").sum())
    content = f"""# Adult data card

## Source and licence

Adult was extracted by Barry Becker from the 1994 US Census database and
published by Barry Becker and Ronny Kohavi. It is accessed through OpenML
dataset 1590, version 2, using scikit-learn `fetch_openml`.
The [UCI source and data dictionary](https://archive.ics.uci.edu/dataset/2/adult)
identify the original collection and list its licence as
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
OpenML's older metadata uses the licence description `Public`.
Citation: Becker, B. and Kohavi, R. (1996), Adult, UCI Machine Learning
Repository, [doi:10.24432/C5XW20](https://doi.org/10.24432/C5XW20).

## Purpose and contents

This copy is used for an educational binary income classifier and a comparison
of preprocessing choices. Each row describes an adult; the label indicates
annual income `<=50K` or `>50K`. There are **48,842 rows**, **14 features** and
**one target**. OpenML calls the target `class`; the lab schema calls it `income`.
Six features are numerical and eight are categorical. The full dictionary,
allowed categories and required numerical ranges are recorded in `config.yaml`.
`fnlwgt` is the census final weight and is used as an input here, not a sample weight.

The positive class contains {positive:,} rows ({positive / len(raw):.2%});
the remaining {len(raw) - positive:,} rows have income `<=50K`.
The sample contains {female:,} Female records ({female / len(raw):.2%}) and
{white:,} White records ({white / len(raw):.2%}), using the original category names.
These are sample descriptions, not population estimates.

## Collection, coverage and limitations

UCI describes extraction conditions including age above 16, positive adjusted
gross income, final weight above 1 and positive working hours. The data represent
a selected historical US census sample. They do not describe the current labour
market or worldwide populations. The USD 50,000 threshold is historical and has
not been adjusted for inflation. These collection constraints limit generalisation.

## Quality and processing

The original frame is retained in memory without cleaning during ingestion.
Missing markers `?` arrive as nulls in the OpenML parser. Missing counts are
2,799 for workclass, 2,809 for occupation and 857 for native-country; the other
fields have no missing values. Missingness is reported before imputation.
The validation report lists all type, range and category violations; none is
silently dropped or repaired.

A stratified 70/30 split with seed 42 produces {counts['train_before']:,} raw
training rows and {counts['test']:,} test rows. Removing
{counts['removed_train_duplicates']} exact training duplicates leaves
{counts['train_after']:,} training rows. Test membership and labels are unchanged.
Imputation medians/modes, 1st/99th percentile limits, standardisation and one-hot
encoding are learned only from retained training rows. Test features use those
fixed transformations. Derived inputs are `capital_net` and `hours_per_age`,
calculated after imputation and clipping. The target is never an input.

The complete frame contains {counts['raw_exact_duplicates']} exact duplicate rows.
The conflict audit found {labels['conflicting_groups']} feature-identical groups
with different income labels ({labels['conflicting_rows']} rows). Labels remain
unchanged because the true income cannot be recovered from these features.
The seeded 50-row sample has {labels['reviewed_rows']} saved dictionary reviews;
findings are in `results/label_sample.csv`. Plausibility checks cannot independently
verify the income labels. There are {counts['overlapping_test_feature_rows']} test
rows with features present in training, so the required split is not a guarantee
of distinct people or duplicate-free evaluation.

## Known bias and appropriate use

Income labels can reflect historical employment and social inequalities.
Sex, race, native-country and related proxy features can reproduce group
differences. Unequal representation, class imbalance, missingness and coarse
categories can affect errors. The source sex field offers only Female/Male;
it does not represent the full range of identities. Accuracy and F1 are aggregate
metrics and do not establish fairness. This baseline is intended for this lab,
not for employment, credit or other decisions about individuals.

## Privacy

The published feature set has no names, street addresses or direct identifiers.
Combinations of age, occupation, education, country and household information
may still be identifying when linked with external information. Income and
demographic attributes are sensitive. No re-identification or external linkage
is part of this pipeline.

## Reproduction and storage

Run `python main.py` from the repository root. `config.yaml` contains the source
version, seed, schema, transformations and model settings. Package versions and
the complete run configuration are saved in `results/configuration.yaml`.
The fitted transformer is `results/preprocessor.joblib`; processed features,
labels and row indices are in `results/processed_dataset.npz`.
The downloaded source cache is in `data/openml/` and is excluded from Git.

Processed NPZ SHA256:

```text
{digest}
```
"""
    (LAB / "data_card.md").write_text(content, encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((LAB / "config.yaml").read_text(encoding="utf-8"))
    dataset = fetch_openml(name=config["dataset"]["name"], version=config["dataset"]["version"],
                           as_frame=config["dataset"]["as_frame"], data_home=ROOT / "data" / "openml")
    raw = dataset.frame
    raw_hash = frame_hash(raw)
    if raw.shape != (config["dataset"]["expected_rows"], config["dataset"]["expected_columns"]):
        raise ValueError(f"Unexpected ingestion shape: {raw.shape}")
    ingestion = dict(rows=len(raw), columns=len(raw.columns), column_names=list(raw.columns),
                     openml_id=int(dataset.details["id"]), version=int(dataset.details["version"]),
                     ingested_frame_sha256=raw_hash)
    (RESULTS / "ingestion.json").write_text(json.dumps(ingestion, indent=2) + "\n", encoding="utf-8")
    print(f"Ingestion: {len(raw)} rows, {len(raw.columns)} columns", flush=True)
    validation, issues = validate(raw, config["schema"])
    save_csv(pd.DataFrame(issues, columns=["source_row", "column", "value", "issue"]), RESULTS / "validation_issues.csv")
    train_ids, test_ids = train_test_split(
        raw.index.to_numpy(), test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"], stratify=raw["class"],
    )
    train_raw, test = raw.loc[train_ids].copy(), raw.loc[test_ids].copy()
    test_hash = frame_hash(test)
    duplicate_mask = train_raw.duplicated(keep="first")
    removed_ids = train_raw.index[duplicate_mask].to_numpy()
    train = train_raw.loc[~duplicate_mask].copy()
    labels = audit_labels(raw, config["audit"])
    overlap = test[FEATURES].merge(train[FEATURES].drop_duplicates(), how="left", on=FEATURES, indicator=True)
    counts = dict(train_before=len(train_raw), train_after=len(train), test=len(test),
                  removed_train_duplicates=len(removed_ids), raw_exact_duplicates=int(raw.duplicated().sum()),
                  overlapping_test_feature_rows=int(overlap["_merge"].eq("both").sum()))
    if issues or raw["class"].isna().any() or not raw["class"].isin(["<=50K", ">50K"]).all():
        write_validation(validation, issues, counts, labels, None)
        raise ValueError("Input validation failed; see results/validation_report.md.")
    metrics, processed, transformer = fit_baselines(train, test, config)
    processed["train_indices_before_deduplication"] = train_ids
    processed["removed_train_indices"] = removed_ids
    np.savez_compressed(RESULTS / "processed_dataset.npz", **processed)
    digest = hashlib.sha256((RESULTS / "processed_dataset.npz").read_bytes()).hexdigest()
    (RESULTS / "processed_dataset.sha256").write_text(f"{digest}  processed_dataset.npz\n", encoding="utf-8")
    if frame_hash(raw) != raw_hash or frame_hash(test) != test_hash:
        raise AssertionError("Raw data or the held-out partition was modified.")
    if set(train.index) & set(test.index) or len(test) != len(test_ids):
        raise AssertionError("Train/test partition integrity failed.")
    versions = {name: version(name) for name in ["pandas", "numpy", "scikit-learn", "joblib", "PyYAML", "matplotlib"]}
    manifest = dict(config=config, dataset_sha256=digest, ingested_frame_sha256=raw_hash,
                    python=platform.python_version(), packages=versions, rows=counts,
                    label_audit=labels, blas_threads=1)
    (RESULTS / "configuration.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    write_validation(validation, issues, counts, labels, transformer)
    write_report(metrics, counts, labels, digest)
    write_data_card(raw, counts, labels, digest)
    print(f"SHA256: {digest}", flush=True)
    print(f"Dictionary review: {labels['reviewed_rows']}/50 rows", flush=True)


if __name__ == "__main__":
    main()
