from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_run(output, name):
    directory = output / "runs" / name
    return (json.loads((directory / "summary.json").read_text(encoding="utf-8")),
            pd.read_csv(directory / "history.csv"))


def experiment_names(config):
    return (["profile"] + [f"workers_{n}" for n in config["studies"]["num_workers"]]
            + [f"batch_{n}" for n in config["studies"]["batch_sizes"]]
            + [f"schedule_{name}" for name in config["schedulers"]]
            + ["optimizer_SGD", "accumulation", "recovery"])


def verify_results(config, root):
    import mlflow
    import torch
    from mlflow.tracking import MlflowClient

    output = root / config["results_dir"]
    names = experiment_names(config)
    from run_lab import experiment_config

    mlflow.set_tracking_uri((root / config["mlflow"]["directory"]).as_uri())
    client = MlflowClient()
    lab03 = json.loads((root / "lab03/results/configuration.json").read_text(encoding="utf-8"))
    checked_runs = []
    for name in names:
        summary, history = read_run(output, name)
        expected = experiment_config(config, name)
        assert len(history) == expected["training"]["epochs"], name
        assert history["epoch"].tolist() == list(range(1, len(history) + 1)), name
        assert (history.train_samples == 55_000).all(), name
        assert np.isfinite(history.select_dtypes("number").to_numpy()).all(), name
        assert ((history.validation_accuracy >= 0) & (history.validation_accuracy <= 1)).all(), name
        assert (history.training_seconds > 0).all() and (history.peak_rss_mib > 0).all(), name
        np.testing.assert_allclose(history.throughput_samples_s,
                                   history.train_samples / history.training_seconds, rtol=1e-12)
        updates = math.ceil(55_000 / summary["effective_batch_size"])
        assert (history.optimizer_updates == updates).all(), name
        assert summary["test_samples"] == 10_000, name
        metadata = json.loads((output / "runs" / name / "environment.json").read_text(encoding="utf-8"))
        assert metadata["parameters"] == 118_634, name
        for key in ["train_indices_sha256", "validation_indices_sha256"]:
            assert metadata["split"][key] == lab03[key], (name, key)
        run = client.get_run(summary["run_id"])
        assert run.info.status == "FINISHED", name
        assert len(client.get_metric_history(run.info.run_id, "train_loss")) == len(history), name
        artifacts = {artifact.path for artifact in client.list_artifacts(run.info.run_id)}
        assert {"configuration.yaml", "history.csv", "environment.json", "summary.json"} <= artifacts, name
        checked_runs.append({"experiment": name, "run_id": summary["run_id"],
                             "epochs": len(history), "status": run.info.status})
    profile = pd.read_csv(output / "profile_breakdown.csv")
    assert set(profile.phase) == {"data_loading", "forward", "backward", "optimizer_step"}
    assert (profile.cpu_seconds > 0).all() and abs(profile.share_percent.sum() - 100) < 1e-8
    profiled_batches = math.ceil(55_000 / config["training"]["batch_size"])
    phase_calls = profile.set_index("phase").calls.to_dict()
    assert phase_calls["forward"] == phase_calls["backward"] == profiled_batches
    assert phase_calls["data_loading"] == profiled_batches + 1
    assert phase_calls["optimizer_step"] == 2 * profiled_batches
    _, direct = read_run(output, "batch_256")
    _, accumulated = read_run(output, "accumulation")
    accuracy_gap = float(abs(direct.validation_accuracy - accumulated.validation_accuracy).max())
    loss_gap = float(abs(direct.train_loss - accumulated.train_loss).max())
    assert accuracy_gap <= 0.03 and loss_gap <= 0.10, (accuracy_gap, loss_gap)
    _, uninterrupted = read_run(output, f"batch_{config['training']['batch_size']}")
    _, recovered = read_run(output, "recovery")
    curve_columns = ["epoch", "learning_rate", "train_loss", "train_accuracy",
                     "train_samples", "optimizer_updates", "validation_loss", "validation_accuracy"]
    np.testing.assert_array_equal(uninterrupted[curve_columns], recovered[curve_columns])
    baseline = torch.load(output / "runs" / f"batch_{config['training']['batch_size']}" / "checkpoint.pt",
                          map_location="cpu", weights_only=False)
    resumed = torch.load(output / "runs/recovery/checkpoint.pt", map_location="cpu", weights_only=False)
    max_weight_difference = max((baseline["model"][key] - value).abs().max().item()
                                for key, value in resumed["model"].items())
    assert max_weight_difference == 0.0, max_weight_difference
    interrupted = torch.load(output / "runs/recovery/epoch03.pt", map_location="cpu", weights_only=False)
    assert interrupted["epoch"] == config["studies"]["interrupt_after_epoch"]
    assert interrupted["mlflow_run_id"] == resumed["mlflow_run_id"]
    pd.DataFrame(checked_runs).to_csv(output / "mlflow_runs.csv", index=False)
    result = {
        "all_required_runs_finished": True, "runs": len(checked_runs),
        "all_runs_use_lab03_split": True, "all_runs_use_118634_parameter_cnn": True,
        "all_training_samples_per_epoch": 55_000, "all_test_samples": 10_000,
        "accumulation_max_validation_accuracy_gap": accuracy_gap,
        "accumulation_max_train_loss_gap": loss_gap,
        "accumulation_same_optimizer_update_counts": True,
        "recovery_curves_identical": True, "recovery_max_parameter_difference": max_weight_difference,
        "recovery_interrupted_epoch": interrupted["epoch"], "recovery_fresh_process": True,
        "mlflow_parameters_metrics_artifacts_present": True,
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def markdown_table(frame, columns, headers, formats):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(formatter(row[column])
                                      for column, formatter in zip(columns, formats)) + " |")
    return "\n".join(lines)


def generate_proposal(config, root):
    import shutil

    project = root / "project"
    project.mkdir(exist_ok=True)
    shutil.copyfile(root / config["project"]["source_pdf"], project / "proposal.pdf")


def generate_outputs(config, root):
    import mlflow
    from mlflow.tracking import MlflowClient

    output = root / config["results_dir"]
    generate_proposal(config, root)
    summaries = {name: read_run(output, name)[0] for name in experiment_names(config)}
    histories = {name: read_run(output, name)[1] for name in experiment_names(config)}
    checks = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    profile = pd.read_csv(output / "profile_breakdown.csv")
    worker_rows = []
    for workers in config["studies"]["num_workers"]:
        summary = summaries[f"workers_{workers}"]
        row = histories[f"workers_{workers}"].iloc[0]
        worker_rows.append({"num_workers": workers,
                            "throughput_samples_s": summary["throughput_samples_s"],
                            "startup_excluded_samples_s": row.steady_throughput_samples_s,
                            "loader_startup_seconds": row.loader_startup_seconds,
                            "data_loading_percent": 100 * row.data_loading_seconds / row.training_seconds,
                            "startup_excluded_loading_percent":
                            100 * max(0, row.data_loading_seconds - row.loader_startup_seconds)
                            / (row.training_seconds - row.loader_startup_seconds),
                            "peak_rss_mib": row.peak_rss_mib})
    workers = pd.DataFrame(worker_rows)
    workers.to_csv(output / "worker_study.csv", index=False)
    batches = pd.DataFrame([summaries[f"batch_{batch}"] for batch in config["studies"]["batch_sizes"]])
    batches.to_csv(output / "batch_study.csv", index=False)
    schedules = pd.DataFrame([summaries[f"schedule_{name}"] for name in config["schedulers"]])
    schedules.to_csv(output / "scheduler_study.csv", index=False)
    optimizers = pd.DataFrame([summaries["schedule_CosineAnnealingLR"], summaries["optimizer_SGD"]])
    optimizers.to_csv(output / "optimizer_study.csv", index=False)
    pairs = [
        ("accumulation", "batch_256", "accumulation"),
        ("recovery", f"batch_{config['training']['batch_size']}", "recovery"),
    ]
    comparisons = {}
    for key, first, second in pairs:
        left, right = histories[first], histories[second]
        comparison = pd.DataFrame({
            "epoch": left.epoch,
            "reference_train_loss": left.train_loss, "comparison_train_loss": right.train_loss,
            "reference_validation_accuracy": left.validation_accuracy,
            "comparison_validation_accuracy": right.validation_accuracy,
            "reference_updates": left.optimizer_updates, "comparison_updates": right.optimizer_updates,
            "train_loss_difference": abs(left.train_loss - right.train_loss),
            "validation_accuracy_difference": abs(left.validation_accuracy - right.validation_accuracy),
        })
        comparison.to_csv(output / f"{key}_comparison.csv", index=False)
        comparisons[key] = comparison
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(workers.num_workers, workers.throughput_samples_s, "o-", label="Whole epoch")
    ax.plot(workers.num_workers, workers.startup_excluded_samples_s, "s--", label="Initial loader wait excluded")
    ax.set(xlabel="DataLoader workers", ylabel="Training throughput (samples/s)",
           xticks=config["studies"]["num_workers"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "worker_throughput.png", dpi=160)
    plt.close(fig)
    recovery = comparisons["recovery"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(recovery.epoch, recovery.reference_train_loss, "o-", label="Uninterrupted")
    axes[0].plot(recovery.epoch, recovery.comparison_train_loss, "x--", label="Resumed")
    axes[1].plot(recovery.epoch, 100 * recovery.reference_validation_accuracy, "o-", label="Uninterrupted")
    axes[1].plot(recovery.epoch, 100 * recovery.comparison_validation_accuracy, "x--", label="Resumed")
    for ax in axes:
        ax.axvline(config["studies"]["interrupt_after_epoch"] + 0.5, color="grey", linestyle=":")
        ax.set(xlabel="Epoch", xticks=recovery.epoch)
        ax.legend()
    axes[0].set_ylabel("Training cross-entropy")
    axes[1].set_ylabel("Validation accuracy (%)")
    fig.tight_layout()
    fig.savefig(output / "recovery_curves.png", dpi=160)
    plt.close(fig)

    integer = lambda value: str(int(value))
    decimal = lambda value: f"{value:.2f}"
    percent = lambda value: f"{100 * value:.2f}%"
    profile_table = markdown_table(profile, ["phase", "cpu_seconds", "share_percent"],
                                   ["Phase", "CPU range time (s)", "Share"],
                                   [str, decimal, lambda value: f"{value:.2f}%"])
    worker_table = markdown_table(workers, ["num_workers", "throughput_samples_s", "startup_excluded_samples_s",
                                           "loader_startup_seconds", "data_loading_percent", "peak_rss_mib"],
                                  ["Workers", "Samples/s", "Without initial wait", "Initial wait (s)", "Loading share", "Peak RSS (MiB)"],
                                  [integer, decimal, decimal, decimal, lambda value: f"{value:.2f}%", decimal])
    batch_table = markdown_table(batches, ["batch_size", "epochs", "throughput_samples_s", "peak_rss_mib", "test_accuracy"],
                                 ["Batch", "Epochs", "Samples/s", "Peak RSS (MiB)", "Final test accuracy"],
                                 [integer, integer, decimal, decimal, percent])
    schedule_table = markdown_table(schedules, ["scheduler", "epochs", "throughput_samples_s", "validation_accuracy", "test_accuracy"],
                                    ["Adam schedule", "Epochs", "Samples/s", "Validation accuracy", "Test accuracy"],
                                    [str, integer, decimal, percent, percent])
    optimizer_table = markdown_table(optimizers, ["optimizer", "scheduler", "epochs", "throughput_samples_s", "test_accuracy"],
                                     ["Optimizer", "Schedule", "Epochs", "Samples/s", "Test accuracy"],
                                     [str, str, integer, decimal, percent])
    accumulation_table = markdown_table(comparisons["accumulation"],
                                        ["epoch", "reference_train_loss", "comparison_train_loss", "reference_validation_accuracy",
                                         "comparison_validation_accuracy", "comparison_updates"],
                                        ["Epoch", "Batch 256 loss", "64 x 4 loss", "Batch 256 validation", "64 x 4 validation", "Updates"],
                                        [integer, lambda v: f"{v:.6f}", lambda v: f"{v:.6f}", percent, percent, integer])
    best_worker = workers.loc[workers.throughput_samples_s.idxmax()]
    best_steady_worker = workers.loc[workers.startup_excluded_samples_s.idxmax()]
    baseline_worker = workers.loc[workers.num_workers == 0].iloc[0]
    worker_conclusion = (
        "Model computation is the main limit in the zero-worker run: loading after the initial wait "
        f"accounts for {baseline_worker['startup_excluded_loading_percent']:.2f}% of its remaining epoch. "
        "More loader processes cannot eliminate the forward and backward computation."
        if baseline_worker["startup_excluded_loading_percent"] < 50 else
        "Data preparation or waiting is the main limit in the zero-worker run: loading after the initial wait "
        f"accounts for {baseline_worker['startup_excluded_loading_percent']:.2f}% of its remaining epoch. "
        "Worker parallelism can overlap this work with model computation."
    )
    best_batch = batches.loc[batches.throughput_samples_s.idxmax()]
    dominant_phase = profile.loc[profile.cpu_seconds.idxmax(), "phase"]
    profile_epoch_seconds = float(histories["profile"].training_seconds.sum())
    profile_range_seconds = float(profile.cpu_seconds.sum())
    best_schedule = schedules.loc[schedules.validation_accuracy.idxmax()]
    environment = json.loads((output / "runs/batch_128/environment.json").read_text(encoding="utf-8"))
    report = f"""# Lab 06: Training pipelines and project proposal

## Reproduction

```powershell
python main.py
```

The single configuration object is loaded from [config.yaml](config.yaml).
Each experiment runs in a fresh Python process. This prevents a previous batch
size's CPU allocator cache from determining the next run's peak-memory result.
Regenerate the tables, figures, and report without training with
`python main.py --report-only`. Open the recorded experiments with
`mlflow ui --backend-store-uri lab06/results/tracking` in the activated environment.

## Model, data, and controls

The size-matched CNN from Lab 3 had the highest validation accuracy, 92.04%,
among the six recorded models. Its architecture is unchanged: Conv(1,48,3), ReLU,
max-pool, Conv(48,128,3), ReLU, max-pool, and Linear(128 x 7 x 7,10), with
118,634 parameters. All studies start from the same fresh seeded weights;
the Lab 3 test scores are not used to select a model.

Fashion-MNIST uses the original 55,000/5,000/10,000 train/validation/test split,
`ToTensor()` and normalization with mean 0.2860 and standard deviation 0.3530.
Split hashes match Lab 3. Every epoch processes all 55,000 training samples,
including the final partial batch. Validation is evaluated each epoch; the
test set is evaluated once at the final epoch of each run. ReduceLROnPlateau uses
validation loss; the other schedules use the epoch number. Schedulers do not
receive test labels or test scores.

The measurements use {environment['device']}, float32, {config['threads']} intra-op
threads, one inter-op thread, seed {config['seed']}, and deterministic algorithms.
The default is Adam, learning rate {config['training']['learning_rate']}, no weight
decay, batch {config['training']['batch_size']}, and {config['training']['num_workers']}
loader worker for the batch, schedule, optimizer, accumulation, and recovery
comparisons, selected from the worker study. The profile uses zero workers.
Each epoch has a reproducible permutation generated from `seed + epoch_index`.
Package versions and split hashes are recorded in every run's `environment.json`.

Throughput is total training samples divided by training-loop wall time. It
includes loading, forward, backward, optimizer updates, and metric bookkeeping;
it excludes validation, test evaluation, checkpoint writing, and MLflow logging.
Each configuration is measured once. Timing differences include normal machine
load and are observations from this CPU, rather than hardware-independent claims.
CPU peak memory is sampled every {config['memory_sample_seconds'] * 1000:.0f} ms as
the sum of the main process and worker RSS during training. Shared pages can be
counted more than once across workers, and peaks between samples can be missed.
The CPU batch-size study uses one worker throughout. GPU allocated memory is zero because
CUDA is unavailable on this installation.

## One-epoch profiling

`torch.profiler` records one complete training epoch, with non-overlapping user
ranges around loading, forward/loss, backward, and optimizer update/gradient reset.
The percentages below normalize the four inclusive range durations to 100%.
They do not sum nested operator times. Loading includes iterator creation,
waiting for batches, transforms, and input transfer. Bookkeeping and profiler
overhead outside these ranges are excluded from the phase denominator.

{profile_table}

The profiled training loop took {profile_epoch_seconds:.2f} s. The four ranges
total {profile_range_seconds:.2f} s; the remaining
{profile_epoch_seconds - profile_range_seconds:.2f} s includes profiler event
collection/aggregation and host bookkeeping outside those ranges.

The largest measured phase is **{dominant_phase}**. The operator breakdown is
saved in [profiler_operators.txt](results/runs/profile/profiler_operators.txt),
with the complete operator totals in
[profiler_operators.csv](results/runs/profile/profiler_operators.csv).
All batches are profiled in windows of {config['profiling']['active_steps_per_window']}
steps, and their event totals are added. This bounds profiler memory while
covering the complete epoch, including the final partial window.
Profiling adds overhead, so the study tables use unprofiled training runs.

## Data-loader workers

One full epoch is trained at each worker count, with all other settings fixed.
The initial wait includes Windows worker spawning and the first batch fetch.
The additional throughput column divides the same sample count by training
time with this initial loading wait removed. It helps separate startup cost
from the remaining epoch, but is not a separate warm-epoch experiment.

{worker_table}

![Training throughput by worker count](results/worker_throughput.png)

The highest whole-epoch throughput was {best_worker['throughput_samples_s']:.2f}
samples/s with num_workers={int(best_worker['num_workers'])}. Compare loading wait
with compute time: extra workers can overlap preprocessing, but Windows process
startup, inter-process transfer, RAM pressure, and CPU contention also consume
resources. Fashion-MNIST is already held in memory after ingestion, so this
study measures transformations and batching rather than repeated disk reads.
The range breakdown and measured loading percentages identify whether fetching
or model computation dominates; increasing workers alone cannot remove the
convolution/backpropagation cost.

{worker_conclusion} Excluding only the initial loading wait, the fastest worker
count was {int(best_steady_worker['num_workers'])}; the full measurements are
saved in [worker_study.csv](results/worker_study.csv).

## Batch-size study

Each batch size is trained for five epochs using Adam and a fixed learning rate.
All configurations start from the same weights and see the same epoch order.

{batch_table}

Batch {int(best_batch['batch_size'])} had the highest measured throughput.
Larger batches require larger activations and convolution workspaces, while
performing fewer optimizer updates per epoch. The table therefore reports final
accuracy as well as speed and measured RAM use; equal epochs do not imply equal
numbers of parameter updates.

## Schedules and optimizers

The scheduler comparison fixes Adam and gives each schedule ten epochs.
StepLR multiplies the learning rate by {config['schedulers']['StepLR']['gamma']}
every {config['schedulers']['StepLR']['step_size']} epochs. CosineAnnealingLR uses
T_max=10 and eta_min={config['schedulers']['CosineAnnealingLR']['eta_min']}.
ReduceLROnPlateau monitors validation loss with factor
{config['schedulers']['ReduceLROnPlateau']['factor']}, patience
{config['schedulers']['ReduceLROnPlateau']['patience']}, and relative threshold
{config['schedulers']['ReduceLROnPlateau']['threshold']}. The learning rate actually
used for each epoch is stored in its history and MLflow metrics.

{schedule_table}

{best_schedule['scheduler']} had the highest final validation accuracy in this
single-seed schedule comparison. The plateau scheduler reduced its learning
rate when validation loss stopped improving according to the configured condition.

The optimizer comparison fixes CosineAnnealingLR and a ten-epoch budget for
both methods. Adam uses lr={config['training']['learning_rate']}; SGD uses
lr={config['studies']['sgd_learning_rate']} and momentum={config['training']['momentum']}.
These optimizer-specific learning rates are fixed in advance. The comparison
is between these two configured training recipes, so differences cannot be
attributed to the optimizer independently of learning rate. The Adam/Cosine run
is shared between the two comparisons.

{optimizer_table}

## Gradient accumulation

Batch 64 with four accumulated microbatches is compared with true batch 256,
using the same sample order, initialization, optimizer, and five-epoch budget.
Cross-entropy is summed per microbatch and divided by the actual number of
samples in its update group. The final group has 216 samples; dividing every
microbatch mean by four would give the final 24-sample microbatch too much weight.
Each epoch performs 215 optimizer updates in both runs. Gradients are reset only
at the start of an update group, and the optimizer steps after the group ends.
The CNN has no batch normalization or dropout that would change microbatch semantics.

{accumulation_table}

The maximum validation-accuracy gap over the five epochs is
{100 * checks['accumulation_max_validation_accuracy_gap']:.4f} percentage points;
the maximum training-loss difference is {checks['accumulation_max_train_loss_gap']:.6f}.
The configured check requires gaps no larger than 3 percentage points and 0.10
loss. Minor differences can arise from floating-point reduction order. Accumulation
emulates the update batch size while retaining microbatch activation memory;
it requires four forward/backward passes per update and can cost throughput.

## Checkpoint recovery

Training stops after epoch 3 and the process exits. A second process restores
the checkpoint and completes epochs 4 and 5 in the same MLflow run. The saved
state contains model weights, optimizer state, scheduler state when present,
epoch/history, configuration, Python/NumPy/PyTorch/CUDA RNG state, and the loader
generator state. The epoch-based sampler reproduces subsequent data orders.
Checkpoints are written through a temporary file and atomically replaced.

![Uninterrupted and resumed training curves](results/recovery_curves.png)

Compared with the uninterrupted batch-128 run, all loss and accuracy values are
identical, and the final maximum absolute parameter difference is
{checks['recovery_max_parameter_difference']:.1f}. The epoch-3 checkpoint is retained
as `results/runs/recovery/epoch03.pt`. This verifies epoch-boundary recovery;
resuming partway through a minibatch is outside this experiment.

## Tracking and project proposal

The {checks['runs']} completed study runs log the resolved configuration, dataset split,
per-epoch losses, accuracy, learning rate, throughput, memory, and final test
metrics. Each run also logs its configuration, environment, history, and summary
as artifacts. Study tables and plots are attached to the relevant runs.
The actual MLflow file store is in `results/tracking/`, and
[mlflow_runs.csv](results/mlflow_runs.csv) lists the run IDs.
[verification.json](results/verification.json) records the completed checks.

The one-page individual proposal is [proposal.pdf](../project/proposal.pdf), with
editable text in [proposal.md](../project/proposal.md), transcribed from the supplied
Task 8 PDF. Its planned 54,000/6,000 project split is separate from the unchanged
55,000/5,000 split used in the Lab 6 experiments. Student ID is
{config['project']['student_id'] or 'not provided'}. Instructor approval is pending.

Reading: Reddi, Chapter 8 and §8.7, Chapter 5 §5.5, and Chapter 13 §13.3.
API references: [PyTorch profiler](https://docs.pytorch.org/docs/2.3/profiler.html),
[DataLoader](https://docs.pytorch.org/docs/2.3/data.html), and
[MLflow tracking](https://mlflow.org/docs/2.14.1/tracking.html).
"""
    (root / "lab06/report.md").write_text(report, encoding="utf-8")
    mlflow.set_tracking_uri((root / config["mlflow"]["directory"]).as_uri())
    client = MlflowClient()
    artifact_groups = {
        "workers_0": ["worker_study.csv", "worker_throughput.png"],
        "batch_128": ["batch_study.csv"],
        "schedule_StepLR": ["scheduler_study.csv"],
        "optimizer_SGD": ["optimizer_study.csv"],
        "accumulation": ["accumulation_comparison.csv"],
        "recovery": ["recovery_comparison.csv", "recovery_curves.png", "verification.json",
                     "runs/recovery/checkpoint.pt", "runs/recovery/epoch03.pt"],
    }
    for name, files in artifact_groups.items():
        for filename in files:
            client.log_artifact(summaries[name]["run_id"], str(output / filename))
    print(f"Verified {checks['runs']} MLflow runs. Report: {root / 'lab06/report.md'}", flush=True)
