"""Reproducibly train and benchmark the two required Lab 01 baselines."""

from __future__ import annotations

import csv
import gc
import json
import os
import platform
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import psutil
import sklearn
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


SEED = 42
TRAINING_REPEATS = 5
INFERENCE_REPEATS = 100
BYTES_PER_MIB = 1024**2
BYTES_PER_KIB = 1024
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "lab01" / "results"


@dataclass(frozen=True)
class ModelSpec:
    """A reproducible model definition and its on-disk artifact name."""

    display_name: str
    factory: Callable[[], object]
    artifact_name: str


@dataclass(frozen=True)
class TrainingMeasurement:
    """Training-time and process-memory evidence for one fitted model."""

    model: object
    median_time_ms: float
    peak_rss_bytes: int
    peak_rss_delta_bytes: int


@dataclass(frozen=True)
class InferenceMeasurement:
    """Single-example latency and process-memory evidence."""

    median_latency_ms: float
    peak_rss_bytes: int
    peak_rss_delta_bytes: int


class PeakRSSMonitor:
    """Sample this Python process and retain its highest observed RSS value."""

    def __init__(self, interval_seconds: float = 0.001) -> None:
        self._process = psutil.Process(os.getpid())
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_rss_bytes = 0
        self.peak_rss_bytes = 0

    def _sample(self) -> None:
        rss_bytes = self._process.memory_info().rss
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            self._sample()

    def __enter__(self) -> "PeakRSSMonitor":
        self.start_rss_bytes = self._process.memory_info().rss
        self.peak_rss_bytes = self.start_rss_bytes
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._sample()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()

    @property
    def peak_rss_delta_bytes(self) -> int:
        return max(0, self.peak_rss_bytes - self.start_rss_bytes)


def set_global_seed() -> None:
    """Set every seed prescribed by the laboratory manual."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def model_specs() -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(
            display_name="LogisticRegression",
            factory=lambda: LogisticRegression(max_iter=1000, random_state=SEED),
            artifact_name="logistic_regression.joblib",
        ),
        ModelSpec(
            display_name="RandomForestClassifier",
            factory=lambda: RandomForestClassifier(n_estimators=100, random_state=SEED),
            artifact_name="random_forest.joblib",
        ),
    )


def train_and_measure(model_factory: Callable[[], object], X_train: np.ndarray, y_train: np.ndarray) -> TrainingMeasurement:
    """Warm up once, then report the median of five fresh-model training runs."""
    warmup_model = model_factory()
    warmup_model.fit(X_train, y_train)
    del warmup_model
    gc.collect()

    elapsed_ms: list[float] = []
    peak_rss_values: list[int] = []
    peak_rss_deltas: list[int] = []
    fitted_model: object | None = None

    for _ in range(TRAINING_REPEATS):
        model = model_factory()
        with PeakRSSMonitor() as monitor:
            start_ns = time.perf_counter_ns()
            model.fit(X_train, y_train)
            stop_ns = time.perf_counter_ns()
        elapsed_ms.append((stop_ns - start_ns) / 1_000_000)
        peak_rss_values.append(monitor.peak_rss_bytes)
        peak_rss_deltas.append(monitor.peak_rss_delta_bytes)
        fitted_model = model

    assert fitted_model is not None
    return TrainingMeasurement(
        model=fitted_model,
        median_time_ms=float(np.median(elapsed_ms)),
        peak_rss_bytes=max(peak_rss_values),
        peak_rss_delta_bytes=max(peak_rss_deltas),
    )


def infer_and_measure(model: object, sample: np.ndarray) -> InferenceMeasurement:
    """Warm up once, then time 100 individual one-sample predictions."""
    model.predict(sample)

    elapsed_ms: list[float] = []
    with PeakRSSMonitor() as monitor:
        for _ in range(INFERENCE_REPEATS):
            start_ns = time.perf_counter_ns()
            model.predict(sample)
            stop_ns = time.perf_counter_ns()
            elapsed_ms.append((stop_ns - start_ns) / 1_000_000)

    return InferenceMeasurement(
        median_latency_ms=float(np.median(elapsed_ms)),
        peak_rss_bytes=monitor.peak_rss_bytes,
        peak_rss_delta_bytes=monitor.peak_rss_delta_bytes,
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write deterministic UTF-8 CSV evidence."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_for_budget(
    peak_inference_rss_bytes: int,
    median_latency_ms: float,
    model_size_bytes: int,
    memory_limit_bytes: int,
    latency_limit_ms: float,
    size_limit_bytes: int,
) -> bool:
    """Check the three limits using peak inference RSS, latency, and artifact size."""
    return (
        peak_inference_rss_bytes <= memory_limit_bytes
        and median_latency_ms <= latency_limit_ms
        and model_size_bytes <= size_limit_bytes
    )


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    """Create all required Lab 01 baseline and system-measurement artifacts."""
    set_global_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )

    accuracy_rows: list[dict[str, str]] = []
    measurement_rows: list[dict[str, str]] = []
    deployment_rows: list[dict[str, str]] = []

    for spec in model_specs():
        set_global_seed()
        training = train_and_measure(spec.factory, X_train, y_train)
        accuracy = float(training.model.score(X_test, y_test))
        inference = infer_and_measure(training.model, X_test[:1])

        artifact_path = RESULTS_DIR / spec.artifact_name
        joblib.dump(training.model, artifact_path)
        model_size_bytes = artifact_path.stat().st_size
        model_size_kib = model_size_bytes / BYTES_PER_KIB
        model_size_mib = model_size_bytes / BYTES_PER_MIB

        accuracy_rows.append(
            {
                "model": spec.display_name,
                "test_accuracy": f"{accuracy:.4f}",
            }
        )
        measurement_rows.append(
            {
                "model": spec.display_name,
                "training_time_median_ms": f"{training.median_time_ms:.6f}",
                "single_sample_inference_median_ms": f"{inference.median_latency_ms:.6f}",
                "single_sample_inference_median_us": f"{inference.median_latency_ms * 1000:.3f}",
                "model_artifact": artifact_path.relative_to(ROOT).as_posix(),
                "model_size_bytes": str(model_size_bytes),
                "model_size_kib": f"{model_size_kib:.3f}",
                "model_size_mib": f"{model_size_mib:.6f}",
                "training_peak_rss_bytes": str(training.peak_rss_bytes),
                "training_peak_rss_mib": f"{training.peak_rss_bytes / BYTES_PER_MIB:.3f}",
                "training_peak_rss_delta_mib": f"{training.peak_rss_delta_bytes / BYTES_PER_MIB:.3f}",
                "inference_peak_rss_bytes": str(inference.peak_rss_bytes),
                "inference_peak_rss_mib": f"{inference.peak_rss_bytes / BYTES_PER_MIB:.3f}",
                "inference_peak_rss_delta_mib": f"{inference.peak_rss_delta_bytes / BYTES_PER_MIB:.3f}",
            }
        )

        cloud_fit = fit_for_budget(
            inference.peak_rss_bytes,
            inference.median_latency_ms,
            model_size_bytes,
            memory_limit_bytes=1024 * BYTES_PER_MIB,
            latency_limit_ms=100,
            size_limit_bytes=500 * BYTES_PER_MIB,
        )
        edge_fit = fit_for_budget(
            inference.peak_rss_bytes,
            inference.median_latency_ms,
            model_size_bytes,
            memory_limit_bytes=1024 * BYTES_PER_MIB,
            latency_limit_ms=50,
            size_limit_bytes=50 * BYTES_PER_MIB,
        )
        mobile_fit = fit_for_budget(
            inference.peak_rss_bytes,
            inference.median_latency_ms,
            model_size_bytes,
            memory_limit_bytes=256 * BYTES_PER_MIB,
            latency_limit_ms=20,
            size_limit_bytes=10 * BYTES_PER_MIB,
        )
        tinyml_fit = fit_for_budget(
            inference.peak_rss_bytes,
            inference.median_latency_ms,
            model_size_bytes,
            memory_limit_bytes=256 * BYTES_PER_KIB,
            latency_limit_ms=10,
            size_limit_bytes=100 * BYTES_PER_KIB,
        )
        deployment_rows.append(
            {
                "model": spec.display_name,
                "cloud_fit": yes_no(cloud_fit),
                "edge_fit": yes_no(edge_fit),
                "mobile_fit": yes_no(mobile_fit),
                "tinyml_fit": yes_no(tinyml_fit),
                "decision_basis": (
                    "Peak inference RSS, median one-sample latency, and serialized model size "
                    "were compared with the stated upper budget for each target."
                ),
            }
        )

    write_csv(
        RESULTS_DIR / "baseline_accuracy.csv",
        ["model", "test_accuracy"],
        accuracy_rows,
    )
    write_csv(
        RESULTS_DIR / "system_measurements.csv",
        [
            "model",
            "training_time_median_ms",
            "single_sample_inference_median_ms",
            "single_sample_inference_median_us",
            "model_artifact",
            "model_size_bytes",
            "model_size_kib",
            "model_size_mib",
            "training_peak_rss_bytes",
            "training_peak_rss_mib",
            "training_peak_rss_delta_mib",
            "inference_peak_rss_bytes",
            "inference_peak_rss_mib",
            "inference_peak_rss_delta_mib",
        ],
        measurement_rows,
    )
    write_csv(
        RESULTS_DIR / "deployment_fit.csv",
        ["model", "cloud_fit", "edge_fit", "mobile_fit", "tinyml_fit", "decision_basis"],
        deployment_rows,
    )

    metadata = {
        "seed": SEED,
        "dataset": "sklearn.datasets.load_breast_cancer(return_X_y=True)",
        "dataset_samples": int(X.shape[0]),
        "dataset_features": int(X.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "split": "70/30 stratified, random_state=42",
        "training_warmup_runs": 1,
        "training_timed_runs": TRAINING_REPEATS,
        "inference_warmup_runs": 1,
        "inference_timed_runs": INFERENCE_REPEATS,
        "timing_clock": "time.perf_counter_ns",
        "memory_measurement": "psutil.Process(os.getpid()).memory_info().rss sampled every 1 ms",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
    }
    with (RESULTS_DIR / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print("Lab 01 benchmark complete.")
    for row in measurement_rows:
        print(
            f"{row['model']}: training median {row['training_time_median_ms']} ms, "
            f"single-sample median {row['single_sample_inference_median_us']} us"
        )


if __name__ == "__main__":
    main()
