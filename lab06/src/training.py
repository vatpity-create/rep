from __future__ import annotations

import hashlib
import json
import os
import random
import threading
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from lab03.src.data import load_datasets
from lab03.src.models import CNN


class EpochSampler(Sampler):
    def __init__(self, dataset, seed):
        self.size = len(dataset)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self):
        return self.size


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def configure(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("KINETO_LOG_LEVEL", "5")
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])


def make_components(config, root):
    device = torch.device(config["device"])
    train, validation, test, train_indices, validation_indices = load_datasets(
        root / config["data_root"], config["seed"]
    )
    model = CNN(*config["model"]["channels"]).to(device)
    settings = config["training"]
    options = dict(lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    if settings["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **options)
    else:
        optimizer = torch.optim.SGD(model.parameters(), momentum=settings["momentum"], **options)
    name = settings["scheduler"]
    if name == "none":
        scheduler = None
    elif name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings["epochs"], **config["schedulers"][name]
        )
    else:
        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer, **config["schedulers"][name]
        )
    generator = torch.Generator().manual_seed(config["seed"])
    sampler = EpochSampler(train, config["seed"])
    train_loader = DataLoader(
        train, batch_size=settings["batch_size"], sampler=sampler,
        num_workers=settings["num_workers"], worker_init_fn=seed_worker,
        generator=generator, persistent_workers=settings["num_workers"] > 0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(validation, batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=0)
    split = {
        "train_samples": len(train), "validation_samples": len(validation),
        "test_samples": len(test),
        "train_indices_sha256": hashlib.sha256(train_indices.numpy().tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(validation_indices.numpy().tobytes()).hexdigest(),
    }
    return model, optimizer, scheduler, train_loader, validation_loader, test_loader, split


class PeakMemory:
    def __init__(self, interval, device):
        self.interval = interval
        self.device = device
        self.process = psutil.Process()
        self.peak_bytes = 0
        self.stop = threading.Event()

    def sample(self):
        total = 0
        for process in [self.process, *self.process.children(recursive=True)]:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.peak_bytes = max(self.peak_bytes, total)
        return total

    def monitor(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.initial_bytes = self.sample()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.sample()
        self.stop.set()
        self.thread.join()
        self.gpu_bytes = (torch.cuda.max_memory_allocated(self.device)
                          if self.device.type == "cuda" else 0)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.numel()
    return {"loss": loss / count, "accuracy": correct / count, "samples": count}


def train_epoch(model, optimizer, loader, device, config, profiler=None):
    model.train()
    settings = config["training"]
    effective_batch = settings["batch_size"] * settings["accumulation_steps"]
    total_samples = len(loader.dataset)
    loss_total, correct, count, updates = 0.0, 0, 0, 0
    timings = dict(data_loading=0.0, forward=0.0, backward=0.0, optimizer_step=0.0)
    startup_seconds = first_batch_size = 0

    def scope(name):
        return torch.profiler.record_function(name) if profiler is not None else nullcontext()

    def measure_start():
        synchronize(device)
        return perf_counter()

    with PeakMemory(config["memory_sample_seconds"], device) as memory:
        epoch_start = measure_start()
        data_start = epoch_start
        with scope("data_loading"):
            iterator = iter(loader)
        timings["data_loading"] += perf_counter() - data_start
        for batch_index in range(len(loader)):
            data_start = measure_start()
            with scope("data_loading"):
                images, labels = next(iterator)
                images = images.to(device, non_blocking=device.type == "cuda")
                labels = labels.to(device, non_blocking=device.type == "cuda")
                synchronize(device)
            timings["data_loading"] += perf_counter() - data_start
            if batch_index == 0:
                startup_seconds = perf_counter() - epoch_start
                first_batch_size = len(labels)
            if batch_index % settings["accumulation_steps"] == 0:
                group_samples = min(effective_batch, total_samples - count)
                step_start = measure_start()
                with scope("optimizer_step"):
                    optimizer.zero_grad(set_to_none=True)
                timings["optimizer_step"] += perf_counter() - step_start
            forward_start = measure_start()
            with scope("forward"):
                logits = model(images)
                loss_sum = nn.functional.cross_entropy(logits, labels, reduction="sum")
            synchronize(device)
            timings["forward"] += perf_counter() - forward_start
            if not torch.isfinite(loss_sum):
                raise FloatingPointError("Non-finite training loss.")
            backward_start = measure_start()
            with scope("backward"):
                (loss_sum / group_samples).backward()
            synchronize(device)
            timings["backward"] += perf_counter() - backward_start
            last_batch = batch_index + 1 == len(loader)
            if (batch_index + 1) % settings["accumulation_steps"] == 0 or last_batch:
                step_start = measure_start()
                with scope("optimizer_step"):
                    optimizer.step()
                synchronize(device)
                timings["optimizer_step"] += perf_counter() - step_start
                updates += 1
            loss_total += loss_sum.detach().item()
            correct += (logits.detach().argmax(1) == labels).sum().item()
            count += len(labels)
            if profiler is not None:
                profiler.step()
        synchronize(device)
        seconds = perf_counter() - epoch_start
    return {
        "train_loss": loss_total / count, "train_accuracy": correct / count,
        "train_samples": count, "optimizer_updates": updates,
        "training_seconds": seconds, "throughput_samples_s": count / seconds,
        "loader_startup_seconds": startup_seconds,
        "steady_throughput_samples_s": count / (seconds - startup_seconds),
        "peak_rss_mib": memory.peak_bytes / 1024**2,
        "initial_rss_mib": memory.initial_bytes / 1024**2,
        "peak_gpu_allocated_mib": memory.gpu_bytes / 1024**2,
        **{f"{name}_seconds": value for name, value in timings.items()},
    }


def save_checkpoint(path, model, optimizer, scheduler, loader, epoch, history, config, run_id):
    checkpoint = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch, "history": history, "config": config, "mlflow_run_id": run_id,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(), "loader_rng": loader.generator.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, scheduler, loader, config, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    keys = ("seed", "device", "data_root", "model", "training", "schedulers")
    if any(checkpoint["config"][key] != config[key] for key in keys):
        raise ValueError("The checkpoint configuration differs from this run.")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"].cpu())
    loader.generator.set_state(checkpoint["loader_rng"].cpu())
    if checkpoint["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return checkpoint


def flatten_config(config, prefix=""):
    result = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_config(value, name))
        else:
            result[name] = json.dumps(value) if isinstance(value, list) else str(value)
    return result
