"""Audit the submitted artifacts and independently profile representative layers."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json

import numpy as np
import torch
from thop import profile
from torch import nn

from costs import count_mha_macs, visible_attention_operations
from models import SPECS, build_model


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _require(condition, message):
    if not condition:
        raise ValueError(f"Evidence verification failed: {message}")


def verify_evidence(results, groups, config):
    indices = np.load(results / "split_indices.npz")
    train, validation = indices["train"], indices["validation"]
    _require(len(train) == 55_000 and len(validation) == 5_000, "split lengths")
    _require(np.array_equal(np.sort(np.concatenate([train, validation])), np.arange(60_000)),
             "train/validation must be disjoint and cover the official train partition")
    expected_order = torch.randperm(60_000, generator=torch.Generator().manual_seed(config["seed"])).numpy()
    _require(np.array_equal(np.concatenate([train, validation]), expected_order), "seeded split")
    for name, values in (("train", train), ("validation", validation)):
        _require(hashlib.sha256(values.tobytes()).hexdigest() == config[f"{name}_indices_sha256"],
                 f"{name} split hash")
    verified_models, epochs, latency_count = 0, 0, 0
    for suite, metrics in groups.items():
        directory = results / suite
        hand_by_model, timings = defaultdict(list), defaultdict(list)
        for row in _rows(directory / "hand_calculations.csv"):
            hand_by_model[row["model"]].append(row)
        for row in _rows(directory / "latency_runs.csv"):
            timings[(row["model"], int(row["batch_size"]))].append(row)
        representative = []
        for row in metrics:
            name = row["model"]
            spec = next(spec for spec in SPECS if spec.suite == suite and spec.name == name)
            hand = hand_by_model[name]
            _require(sum(int(item["hand_flops"]) for item in hand) == row["hand_flops"], spec.key + " FLOPs")
            _require(sum(int(item["hand_parameters"]) for item in hand) == row["parameters"], spec.key + " parameters")
            _require(row["hand_flops"] == row["thop_core_flops"], spec.key + " THOP core FLOPs")
            history = _rows(directory / f"{name.lower()}_history.csv")
            _require([int(item["epoch"]) for item in history] == [1, 2, 3, 4, 5], spec.key + " epochs")
            _require(all(int(item["train_samples"]) == 55_000 and int(item["validation_samples"]) == 5_000
                         for item in history), spec.key + " sample counts")
            _require(np.isclose(float(history[-1]["validation_accuracy"]), row["validation_accuracy"]),
                     spec.key + " final validation accuracy")
            _require(row["test_samples"] == 10_000 and 0 <= row["test_correct"] <= 10_000,
                     spec.key + " official test count")
            _require(row["test_correct"] / 10_000 == row["test_accuracy"], spec.key + " test accuracy")
            checkpoint = torch.load(directory / f"{name.lower()}.pt", map_location="cpu", weights_only=True)
            _require(checkpoint["epoch"] == 5 and checkpoint["seed"] == config["seed"], spec.key + " checkpoint protocol")
            _require(checkpoint["test_accuracy"] == row["test_accuracy"], spec.key + " checkpoint accuracy")
            _require(checkpoint["model_spec"] == spec.configuration(), spec.key + " checkpoint architecture")
            model = build_model(spec)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            _require(sum(parameter.numel() for parameter in model.parameters()) == row["parameters"]
                     == row["torchinfo_parameters"] == row["thop_core_parameters"], spec.key + " checkpoint parameters")
            _require(all(torch.isfinite(parameter).all() for parameter in model.parameters()), spec.key + " finite weights")
            for batch, runs in ((1, 100), (128, 20)):
                records = timings[(name, batch)]
                _require([int(item["run"]) for item in records] == list(range(1, runs + 1)),
                         spec.key + f" B={batch} runs")
                samples = np.array([float(item["latency_ms"]) for item in records])
                _require(np.all(np.isfinite(samples)) and np.all(samples > 0), spec.key + " positive latency")
                for key, measured in ((f"latency_b{batch}_ms", np.median(samples)),
                                      (f"latency_b{batch}_p25_ms", np.percentile(samples, 25)),
                                      (f"latency_b{batch}_p75_ms", np.percentile(samples, 75))):
                    _require(np.isclose(row[key], measured, rtol=1e-12, atol=1e-12), spec.key + " latency summary")
                _require(np.isclose(row[f"latency_b{batch}_per_image_ms"], np.median(samples) / batch),
                         spec.key + " per-image latency")
                latency_count += runs
            # This profiles a real submodule from the saved trained checkpoint,
            # independently of the whole-network hand/THOP comparison.
            if name in ("MLP", "CNN"):
                layer = model.layers[1 if name == "MLP" else 0]
                sample = torch.zeros(1, 784) if name == "MLP" else torch.zeros(1, 1, 28, 28)
                inputs = (sample,)
                hand_flops = int(hand[0]["hand_flops"])
                layer_name = "Linear 784 to " + str(spec.widths[0]) if name == "MLP" else "Conv2d 1 to " + str(spec.widths[0])
            else:
                layer = model.encoders[0].self_attn
                sample = torch.zeros(1, 16, spec.widths[0])
                inputs = (sample, sample, sample)
                hand_flops = sum(int(item["hand_flops"]) for item in hand
                                 if item["layer"] in {"encoder1.qkv", "encoder1.scores",
                                                      "encoder1.weighted_values", "encoder1.output_projection"})
                layer_name = f"MultiheadAttention L=16 d={spec.widths[0]} h=4"
            with visible_attention_operations():
                macs, _ = profile(deepcopy(layer), inputs=inputs, verbose=False,
                                  custom_ops={nn.MultiheadAttention: count_mha_macs})
            _require(int(2 * macs) == hand_flops, spec.key + " representative layer FLOPs")
            representative.append(dict(suite=suite, model=name, layer=layer_name,
                                       hand_flops=hand_flops, thop_macs=int(macs),
                                       thop_flops=int(2 * macs), gap_percent=0.0))
            verified_models += 1
            epochs += len(history)
        with (directory / "representative_layers.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(representative[0]))
            writer.writeheader()
            writer.writerows(representative)
    verification = dict(
        status="passed", models=verified_models, completed_epochs=epochs,
        raw_latency_runs=latency_count, train_samples=55_000, validation_samples=5_000,
        test_samples=10_000, checks=[
            "split coverage, disjointness, seed and SHA-256",
            "five full train/validation epochs per model",
            "manual, torchinfo, THOP and checkpoint parameter counts",
            "manual and THOP whole-model core FLOPs",
            "independent representative-layer THOP profiles",
            "strict checkpoint restoration, finite weights and recorded test accuracy",
            "100/20 raw latency runs, medians, quartiles and per-image units",
        ],
    )
    (results / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return verification
