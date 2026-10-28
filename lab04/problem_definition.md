# Problem definition

The task is to predict whether a person's annual income exceeds USD 50,000
using the UCI Adult dataset. This is a binary classification problem based on
records extracted from the 1994 United States Census database. The dataset is
loaded through `fetch_openml(name='adult', version=2, as_frame=True)` and contains
48,842 observations with 14 input features and one income label.

The inputs are age, workclass, fnlwgt, education, education-num, marital-status,
occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week,
and native-country. Six are numerical and eight are categorical. The output is
1 for income `>50K` and 0 for income `<=50K`. The OpenML target column is named
`class`; it corresponds to `income` in the validation schema.

Success is measured using test accuracy and binary F1, with `>50K` as the
positive class. F1 is necessary because the income classes are imbalanced.
A LogisticRegression baseline is evaluated both with the original inputs and
with `capital_net` and `hours_per_age`. Both versions use the same split and
training rows, so their metrics can be compared directly.

The main constraints are preventing data leakage and making the pipeline
reproducible. A stratified 70/30 train/test split is created before cleaning.
Only exact duplicates inside the training partition are removed. Imputation,
clipping thresholds, scaling, encoding and model fitting use training data
only. Test rows and labels remain in their original partition; evaluation
uses transformations already fitted on training data. Schema violations and
conflicting labels are reported rather than silently removed or corrected.

Reproducibility requires a fixed dataset version, fixed seeds, recorded package
versions, saved preprocessing, configuration YAML and a SHA256 hash of the
processed dataset. The intended use is an educational comparison of data
preparation choices. Historical income patterns and sensitive attributes limit
generalisation; these predictions are not a basis for decisions about individuals.
