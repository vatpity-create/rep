"""Run the YAML-driven Lab 06 training studies."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def experiment_config(base, name):
    config = copy.deepcopy(base)
    settings = config["training"]
    study = config["studies"]
    if name == "profile":
        settings.update(epochs=1, num_workers=config["profiling"]["num_workers"])
    elif name.startswith("workers_"):
        settings.update(epochs=1, num_workers=int(name.split("_")[1]))
    elif name.startswith("batch_"):
        settings.update(epochs=study["batch_epochs"], batch_size=int(name.split("_")[1]))
    elif name.startswith("schedule_"):
        settings.update(epochs=study["comparison_epochs"], scheduler=name.removeprefix("schedule_"))
    elif name == "optimizer_SGD":
        settings.update(epochs=study["comparison_epochs"], optimizer="SGD",
                        learning_rate=study["sgd_learning_rate"], scheduler="CosineAnnealingLR")
    elif name == "accumulation":
        settings.update(epochs=study["batch_epochs"], batch_size=study["accumulation_batch_size"],
                        accumulation_steps=study["accumulation_steps"])
    elif name != "recovery":
        raise ValueError(f"Unknown experiment: {name}")
    return config


def write_csv(path, rows):
    import csv
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_experiment(config, name, resume=False, stop_after=None):
    import shutil
    import platform
    from datetime import datetime, timezone
    from importlib.metadata import version

    import mlflow
    import torch
    from training import (configure, evaluate, flatten_config, load_checkpoint,
                          make_components, save_checkpoint, train_epoch)

    configure(config)
    output = ROOT / config["results_dir"]
    directory = output / "runs" / name
    directory.mkdir(parents=True, exist_ok=True)
    settings = config["training"]
    mlflow.set_tracking_uri((ROOT / config["mlflow"]["directory"]).as_uri())
    mlflow.set_experiment(config["mlflow"]["experiment"])
    model, optimizer, scheduler, train, validation, test, split = make_components(config, ROOT)
    device = torch.device(config["device"])
    checkpoint_path = directory / "checkpoint.pt"
    history, first_epoch, existing_run_id = [], 1, None
    if resume:
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler, train, config, device)
        history, first_epoch = checkpoint["history"], checkpoint["epoch"] + 1
        existing_run_id = checkpoint["mlflow_run_id"]
    with mlflow.start_run(run_id=existing_run_id, run_name=name) as active:
        run_id = active.info.run_id
        mlflow.set_tags({"study": name.split("_")[0], "lab": "06",
                         "initialization": "fresh seeded Lab 03 architecture"})
        mlflow.log_params({key: value for key, value in flatten_config(config).items()
                           if not key.startswith("project.")})
        mlflow.log_params(split)
        (directory / "configuration.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        if first_epoch == 1:
            metadata = {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "platform": platform.platform(),
                "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                "device": str(device), "threads": torch.get_num_threads(),
                "parameters": sum(p.numel() for p in model.parameters()), "dtype": "float32",
                "packages": {package: version(package) for package in
                             ("torch", "torchvision", "mlflow", "numpy", "pandas", "matplotlib", "PyYAML")},
                "split": split, "mlflow_run_id": run_id,
            }
            (directory / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if resume:
            mlflow.set_tag("resumed_after_epoch", first_epoch - 1)
        end_epoch = min(settings["epochs"], stop_after or settings["epochs"])
        for epoch in range(first_epoch, end_epoch + 1):
            train.sampler.epoch = epoch - 1
            learning_rate = optimizer.param_groups[0]["lr"]
            if name == "profile":
                totals = {}

                def collect_window(profiler):
                    for event in profiler.key_averages():
                        total = totals.setdefault(event.key, {"count": 0, "cpu_us": 0.0, "self_cpu_us": 0.0})
                        total["count"] += event.count
                        total["cpu_us"] += event.cpu_time_total
                        total["self_cpu_us"] += event.self_cpu_time_total

                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    schedule=torch.profiler.schedule(wait=0, warmup=0,
                                                      active=config["profiling"]["active_steps_per_window"]),
                    on_trace_ready=collect_window,
                ) as profiler:
                    measurements = train_epoch(model, optimizer, train, device, config, profiler)
                operator_rows = [{"operator": key, **value} for key, value in totals.items()]
                operator_rows.sort(key=lambda value: value["self_cpu_us"], reverse=True)
                write_csv(directory / "profiler_operators.csv", operator_rows)
                (directory / "profiler_operators.txt").write_text(
                    "Operator | Calls | Self CPU time (ms) | Total CPU time (ms)\n" +
                    "\n".join(f"{row['operator']} | {row['count']} | {row['self_cpu_us'] / 1000:.3f} | "
                              f"{row['cpu_us'] / 1000:.3f}" for row in operator_rows[:30]) + "\n",
                    encoding="utf-8")
                phases = ["data_loading", "forward", "backward", "optimizer_step"]
                durations = {phase: totals[phase]["cpu_us"] / 1e6 for phase in phases}
                total_phase = sum(durations.values())
                write_csv(output / "profile_breakdown.csv", [
                    {"phase": phase, "cpu_seconds": durations[phase],
                     "share_percent": 100 * durations[phase] / total_phase,
                     "calls": totals[phase]["count"]}
                    for phase in phases
                ])
            else:
                measurements = train_epoch(model, optimizer, train, device, config)
            validation_result = evaluate(model, validation, device)
            if scheduler is not None:
                if settings["scheduler"] == "ReduceLROnPlateau":
                    scheduler.step(validation_result["loss"])
                else:
                    scheduler.step()
            row = {"epoch": epoch, "learning_rate": learning_rate, **measurements,
                   "validation_loss": validation_result["loss"],
                   "validation_accuracy": validation_result["accuracy"]}
            history.append(row)
            write_csv(directory / "history.csv", history)
            mlflow.log_metrics({key: value for key, value in row.items() if key != "epoch"}, step=epoch)
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, train, epoch, history, config, run_id)
            if name == "recovery" and epoch == config["studies"]["interrupt_after_epoch"]:
                shutil.copy2(checkpoint_path, directory / "epoch03.pt")
                mlflow.set_tag("interrupted_after_epoch", epoch)
            print(f"{name} | epoch {epoch}/{settings['epochs']} | "
                  f"loss {row['train_loss']:.4f} | validation {row['validation_accuracy']:.2%} | "
                  f"{row['throughput_samples_s']:.0f} samples/s | {row['training_seconds']:.1f}s", flush=True)
        complete = end_epoch == settings["epochs"]
        if complete:
            test_result = evaluate(model, test, device)
            summary = {
                "experiment": name, "run_id": run_id,
                "epochs": end_epoch, "batch_size": settings["batch_size"],
                "effective_batch_size": settings["batch_size"] * settings["accumulation_steps"],
                "num_workers": settings["num_workers"], "optimizer": settings["optimizer"],
                "scheduler": settings["scheduler"],
                "throughput_samples_s": sum(row["train_samples"] for row in history)
                                        / sum(row["training_seconds"] for row in history),
                "peak_rss_mib": max(row["peak_rss_mib"] for row in history),
                "peak_gpu_allocated_mib": max(row["peak_gpu_allocated_mib"] for row in history),
                "validation_accuracy": history[-1]["validation_accuracy"],
                "test_accuracy": test_result["accuracy"], "test_loss": test_result["loss"],
                "test_samples": test_result["samples"],
            }
            (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            mlflow.log_metrics({"final_test_accuracy": test_result["accuracy"],
                                "final_test_loss": test_result["loss"]}, step=end_epoch)
        for filename in ["configuration.yaml", "environment.json", "history.csv", "summary.json",
                         "profiler_operators.txt", "profiler_operators.csv"]:
            if (directory / filename).is_file():
                mlflow.log_artifact(str(directory / filename))
        if name == "profile":
            mlflow.log_artifact(str(output / "profile_breakdown.csv"))
        if not complete:
            mlflow.end_run(status="KILLED")
    return 0


def run_all(config, config_path):
    names = ["profile"]
    names += [f"workers_{value}" for value in config["studies"]["num_workers"]]
    names += [f"batch_{value}" for value in config["studies"]["batch_sizes"]]
    names += [f"schedule_{value}" for value in config["schedulers"]]
    names += ["optimizer_SGD", "accumulation"]
    for name in names:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                        "--run", name], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                    "--run", "recovery", "--stop-after", str(config["studies"]["interrupt_after_epoch"])],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                    "--run", "recovery", "--resume"], cwd=ROOT, check=True)
    from reporting import generate_outputs, verify_results

    verify_results(config, ROOT)
    generate_outputs(config, ROOT)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "lab06" / "config.yaml")
    parser.add_argument("--run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.report_only:
        from reporting import generate_outputs, verify_results
        verify_results(config, ROOT)
        generate_outputs(config, ROOT)
        return 0
    if args.run:
        return run_experiment(experiment_config(config, args.run), args.run, args.resume, args.stop_after)
    return run_all(config, args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
