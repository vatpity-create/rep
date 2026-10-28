# Adult data card

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

The positive class contains 11,687 rows (23.93%);
the remaining 37,155 rows have income `<=50K`.
The sample contains 16,192 Female records (33.15%) and
41,762 White records (85.50%), using the original category names.
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

A stratified 70/30 split with seed 42 produces 34,189 raw
training rows and 14,653 test rows. Removing
24 exact training duplicates leaves
34,165 training rows. Test membership and labels are unchanged.
Imputation medians/modes, 1st/99th percentile limits, standardisation and one-hot
encoding are learned only from retained training rows. Test features use those
fixed transformations. Derived inputs are `capital_net` and `hours_per_age`,
calculated after imputation and clipping. The target is never an input.

The complete frame contains 52 exact duplicate rows.
The conflict audit found 5 feature-identical groups
with different income labels (10 rows). Labels remain
unchanged because the true income cannot be recovered from these features.
The seeded 50-row sample has 50 saved dictionary reviews;
findings are in `results/label_sample.csv`. Plausibility checks cannot independently
verify the income labels. There are 27 test
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
446bc9034f92443b731ea1bb13d8bf3005aba8fb065737f4541b4c5f7dd42d24
```
