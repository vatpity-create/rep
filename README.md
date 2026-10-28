# Machine Learning Systems Labs

This repository follows the laboratory layout and reproducibility rules from the
course manual. Each lab keeps its source code, generated evidence, and report in
its own directory. Downloadable data belongs in `data/`, which is intentionally
ignored by Git.

## Required environment

The required interpreter is **Python 3.11.8**. On Windows, create and activate
the environment with:

```powershell
.\scripts\setup_environment.ps1
.\.venv\Scripts\Activate.ps1
```

The manual pins `mlflow==2.14.1` (which declares `pyarrow<16`) and
`pyarrow==16.1.0` simultaneously. Those two pins make a direct
`pip install -r requirements.txt` unsatisfiable. The setup script retains every
required top-level version, installs MLflow's dependencies first, then applies
the course-mandated PyArrow pin. `requirements.txt` remains the authoritative
list of required versions.

## Run Lab 04

Run `python main.py` to reproduce the Adult data pipeline, validation report,
LogisticRegression comparison, processed-data SHA256 and data card.
See `lab04/report.md` and `lab04/config.yaml` for the method and settings.

## Run Lab 03

Run `python lab03/src/run_lab.py` in the activated environment to train and compare
MLP, CNN, and patch-attention models on Fashion-MNIST.
Each model trains for five epochs with Adam, lr 0.001,
batch size 128, using the shared 55k/5k/10k train/validation/test split.

The manual's exact layer sizes do not meet its 100k–150k parameter target.
The default run therefore includes both the exact `reference` architectures
and a separate `matched` comparison with approximately 119k parameters each.
Use `python lab03/src/run_lab.py --suite reference` for only the exact manual models, or
`python lab03/src/run_lab.py --suite matched` for only the size-matched models.

Results, checkpoints, raw latency samples, torchinfo summaries, and PNG/SVG
plots are saved in `lab03/results/`. The report in `lab03/report.md` is generated
from the actual results. Run `python lab03/src/run_lab.py --report-only` to rebuild plots
and the report without training. CPU timing defaults to two intra-op threads;
use `--threads` and `--device cpu` / `--device cuda` to select the environment.

Run the focused architecture and FLOPs checks with:

```powershell
.\.venv\Scripts\python.exe lab03/src/check_lab.py
```

## Run Lab 02

Run `python lab02/src/run_lab.py` to train all six NumPy network configurations and
regenerate the evidence in `lab02/results/`. See `lab02/report.md` for
the report and `lab02/src/run_lab.py` for the implementation.

## Run Lab 01

From the repository root, run:

```powershell
python lab01/src/run_lab.py
```

The launcher first writes the interpreter and pinned package versions to
`lab01/results/versions.txt`, then trains and benchmarks both required models.
It regenerates the CSV files, model artifacts, and deployment-fit table in
`lab01/results/` deterministically.

To run the individual steps instead:

```powershell
python lab01/src/print_versions.py
python lab01/src/run_benchmarks.py
```

## Repository layout

```text
repo/
├── data/                 # reproducibly downloaded data; ignored by Git
├── lab01/ ... lab11/     # one folder per laboratory
└── project/              # final project source, results, and cards
```

Before submitting a lab, inspect the regenerated files, update the report if
needed, and make a commit from your own Git identity.
