"""Train, profile, benchmark, and report Lab 03. Run via python main.py."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter, perf_counter_ns

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from costs import profile_model
from data import load_datasets, make_loaders
from models import SPECS, build_model

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab03" / "results"


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_total, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_total += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        count += labels.numel()
    return dict(loss=loss_total / count, accuracy=correct / count, correct=correct, count=count)


def train_model(model, spec, loaders, device, directory, seed):
    train_loader, validation_loader, test_loader = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    records = []
    for epoch in range(1, 6):
        start = perf_counter()
        model.train()
        loss_total, correct, count = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {spec.key}, epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * labels.numel()
            correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
            count += labels.numel()
        validation = evaluate(model, validation_loader, device)
        records.append(dict(
            suite=spec.suite, model=spec.name, epoch=epoch,
            train_loss=loss_total / count, train_accuracy=correct / count,
            validation_loss=validation["loss"], validation_accuracy=validation["accuracy"],
            train_samples=count, validation_samples=validation["count"],
            epoch_seconds=perf_counter() - start,
        ))
        write_csv(directory / f"{spec.name.lower()}_history.csv", records)
        print(
            f"{spec.key:20s} epoch {epoch}/5  loss={loss_total / count:.4f}  "
            f"train={correct / count:.2%}  val={validation['accuracy']:.2%}  "
            f"{records[-1]['epoch_seconds']:.1f}s", flush=True,
        )
    # The held-out test set is evaluated only at the mandated final epoch.
    test = evaluate(model, test_loader, device)
    torch.save(dict(
        model_state={name: value.detach().cpu() for name, value in model.state_dict().items()},
        model_spec=spec.configuration(), epoch=5, seed=seed,
        optimizer="Adam", learning_rate=1e-3, test_accuracy=test["accuracy"],
    ), directory / f"{spec.name.lower()}.pt")
    return dict(
        test_accuracy=test["accuracy"], test_loss=test["loss"],
        test_correct=test["correct"], test_samples=test["count"],
        validation_accuracy=records[-1]["validation_accuracy"],
        training_seconds=sum(row["epoch_seconds"] for row in records),
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure_latency(model, inputs, device, spec):
    model.eval()
    statistics, raw = {}, []
    for batch_size, runs in ((1, 100), (128, 20)):
        batch = inputs[:batch_size].to(device).contiguous()
        if len(batch) != batch_size:
            raise ValueError("Latency input must contain at least 128 images.")
        for _ in range(20):
            model(batch)
        synchronize(device)
        samples = []
        for run in range(1, runs + 1):
            synchronize(device)
            start = perf_counter_ns()
            model(batch)
            synchronize(device)
            elapsed_ms = (perf_counter_ns() - start) / 1e6
            samples.append(elapsed_ms)
            raw.append(dict(suite=spec.suite, model=spec.name, batch_size=batch_size,
                            run=run, latency_ms=elapsed_ms))
        median = float(np.median(samples))
        statistics.update({
            f"latency_b{batch_size}_ms": median,
            f"latency_b{batch_size}_p25_ms": float(np.percentile(samples, 25)),
            f"latency_b{batch_size}_p75_ms": float(np.percentile(samples, 75)),
            f"latency_b{batch_size}_per_image_ms": median / batch_size,
            f"throughput_b{batch_size}_images_s": batch_size * 1000 / median,
        })
    return statistics, raw


def environment_metadata(device, threads, seed, suites):
    cpu_name = platform.processor()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return dict(
        run_started_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        processor=cpu_name, logical_cpus=os.cpu_count(),
        device=str(device), device_name=(torch.cuda.get_device_name(device) if device.type == "cuda"
                                        else cpu_name),
        torch_threads=threads, torch_interop_threads=torch.get_num_interop_threads(),
        dtype="float32", seed=seed, deterministic_algorithms=True,
        cudnn_benchmark=False, cuda_version=torch.version.cuda,
        packages={name: version(name) for name in
                  ("torch", "torchvision", "torchinfo", "thop", "numpy", "matplotlib")},
        dataset="torchvision.datasets.FashionMNIST", train_samples=55_000,
        validation_samples=5_000, test_samples=10_000,
        transform="Compose([ToTensor(), Normalize((0.2860,), (0.3530,))])",
        split_seed=seed, shuffle_seed=seed, batch_size=128, drop_last=False,
        num_workers=0, epochs=5, optimizer="Adam", learning_rate=1e-3,
        adam_betas=[0.9, 0.999], adam_eps=1e-8, weight_decay=0,
        loss="CrossEntropyLoss", checkpoint_selection="final epoch 5; no test-set tuning",
        suites=suites, models=[spec.configuration() for spec in SPECS if spec.suite in suites],
        latency_warmups=20, latency_runs={"1": 100, "128": 20},
        latency_statistic="median; raw runs and IQR also saved",
        latency_scope="forward only; input already on device; excludes loader/transforms/transfers",
        latency_mode="eval + inference_mode; normal PyTorch eager inference, MHA fastpath enabled",
        cuda_timing="synchronize before and after each timed forward",
        core_flops_convention="2 FLOPs/MAC; matrix products and convolutions only; batch 1",
        profiling_fastpath="MHA fastpath disabled only for torchinfo/THOP; then restored",
        attention="16x7x7 patches; 4 heads; 2 independently initialized encoders; dropout 0.1; "
                  "ReLU; post-LayerNorm; mean pool; no position embedding or class token",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("reference", "matched", "both"), default="both",
                        help="Exact manual models, 100k-150k models, or both (default).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2, help="CPU intra-op threads (default: 2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-only", action="store_true", help="Rebuild report and plots from saved CSVs.")
    args = parser.parse_args()
    if args.report_only:
        from reporting import generate_outputs
        generate_outputs(RESULTS)
        return 0
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment; use --device cpu")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    seed_everything(args.seed)
    suites = ["reference", "matched"] if args.suite == "both" else [args.suite]
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = environment_metadata(device, args.threads, args.seed, suites)
    (RESULTS / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Lab 03 | {device} | {args.threads} threads | seed {args.seed} | suites {suites}", flush=True)
    train, validation, test, train_indices, validation_indices = load_datasets(ROOT / "data", args.seed)
    np.savez_compressed(RESULTS / "split_indices.npz", train=train_indices.numpy(),
                        validation=validation_indices.numpy())
    config["train_indices_sha256"] = hashlib.sha256(train_indices.numpy().tobytes()).hexdigest()
    config["validation_indices_sha256"] = hashlib.sha256(validation_indices.numpy().tobytes()).hexdigest()
    print(f"Fashion-MNIST ready: {len(train)} train / {len(validation)} validation / {len(test)} test.", flush=True)
    for suite in suites:
        directory = RESULTS / suite
        directory.mkdir(parents=True, exist_ok=True)
        metrics, hand_rows, latency_rows = [], [], []
        for spec in (candidate for candidate in SPECS if candidate.suite == suite):
            seed_everything(args.seed)
            model = build_model(spec).to(device)
            loaders = make_loaders(train, validation, test, args.seed, device)
            timing_inputs, _ = next(iter(loaders[2]))
            costs, hand, model_summary = profile_model(model, spec, timing_inputs[:1].to(device))
            (directory / f"{spec.name.lower()}_summary.txt").write_text(model_summary, encoding="utf-8")
            hand_rows.extend(hand)
            write_csv(directory / "hand_calculations.csv", hand_rows)
            print(f"{spec.key}: {costs['parameters']:,} parameters; {costs['hand_flops']:,} core FLOPs/image", flush=True)
            training = train_model(model, spec, loaders, device, directory, args.seed)
            latency, raw = measure_latency(model, timing_inputs, device, spec)
            latency_rows.extend(raw)
            metrics.append(dict(suite=suite, model=spec.name, **costs, **training, **latency))
            write_csv(directory / "metrics.csv", metrics)
            write_csv(directory / "latency_runs.csv", latency_rows)
            print(f"{spec.key}: test={training['test_accuracy']:.2%}; "
                  f"B1={latency['latency_b1_ms']:.4f}ms; B128={latency['latency_b128_ms']:.4f}ms", flush=True)
    config["run_finished_utc"] = datetime.now(timezone.utc).isoformat()
    (RESULTS / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    from reporting import generate_outputs
    generate_outputs(RESULTS)
    print(f"Evidence: {RESULTS}\nReport: {ROOT / 'lab03' / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
