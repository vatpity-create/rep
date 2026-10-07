# Lab 01 - Environment and first system measurements

## Goal

Create a reproducible Python 3.11.8 environment; train the required logistic
regression and random-forest baselines on Breast Cancer Wisconsin (Diagnostic);
and measure their accuracy, training cost, single-sample latency, serialized
size, and peak process memory.

## Method

The dataset was loaded with `load_breast_cancer(return_X_y=True)` and split into
70% training and 30% test data with stratification and `random_state=42`.
`LogisticRegression(max_iter=1000, random_state=42)` and
`RandomForestClassifier(n_estimators=100, random_state=42)` were trained.

All prescribed random seeds were set to 42. Each training measurement used one
unrecorded warm-up followed by five fresh-model fits; the reported time is the
median. Each model performed one unrecorded inference warm-up and 100 separate
single-sample predictions; the reported latency is the median. RSS was sampled
with `psutil` every 1 ms during the measured sections. Models were serialized
with `joblib` without compression. The exact interpreter and library versions
are recorded in `results/versions.txt`.

The prescribed unscaled logistic-regression configuration reached its
`max_iter=1000` limit and emitted a convergence warning. It was retained
unchanged because it is the baseline required by the assignment.

## Results

| Model | Test accuracy | Median training time (ms) | Median one-sample latency (us) | Model size (KiB) | Peak inference RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.9415 | 1111.232 | 363.900 | 1.030 | 230.062 |
| RandomForestClassifier | 0.9357 | 456.543 | 8108.550 | 284.071 | 231.285 |

The detailed, machine-readable evidence is in:

- `results/baseline_accuracy.csv`
- `results/system_measurements.csv`
- `results/deployment_fit.csv`
- `results/run_metadata.json`

### Resource-budget decision

| Model | Cloud (<=1 GiB, <=100 ms, <=500 MiB) | Edge (<=1 GiB, <=50 ms, <=50 MiB) | Mobile (<=256 MiB, <=20 ms, <=10 MiB) | TinyML (<=256 KiB, <=10 ms, <=100 KiB) |
|---|---|---|---|---|
| LogisticRegression | yes | yes | yes | no |
| RandomForestClassifier | yes | yes | yes | no |

The deployment comparison uses peak RSS during inference, median one-sample
latency, and serialized artifact size. The Cloud and Edge rows use the stated
upper allocation of 1 GiB; a model below the lower end of a platform class
still fits that platform's maximum budget.

## Conclusions

1. On this fixed 70/30 split, the logistic-regression baseline generalized a
   little better: 0.9415 test accuracy versus 0.9357 for the random forest.
   This shows that a more complex ensemble is not automatically the most
   accurate choice for this small tabular dataset.
2. The forest trained faster on this machine, but its single-example inference
   was about 22 times slower (8108.550 us versus 363.900 us) and its serialized
   artifact was about 276 times larger. For latency-sensitive serving, the
   linear model is therefore the stronger trade-off here.
3. Both artifacts and their measured Python-process inference RSS fit the
   stated Cloud, Edge, and Mobile upper budgets. Neither is a TinyML candidate:
   the Python runtime alone is about 230 MiB, far above 256 KiB; additionally,
   the forest artifact exceeds the 100 KiB TinyML size limit.
