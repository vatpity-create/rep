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

## Run Lab 02

Run `python main.py` to train all six NumPy network configurations and
regenerate the evidence in `lab02/results/`. See `lab02/report.md` for
the Ukrainian report and `lab02/src/run_lab.py` for the implementation.

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
