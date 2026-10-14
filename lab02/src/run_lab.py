"""Two-hidden-layer digits classifier using NumPy and manual derivatives."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter_ns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

RESULTS = Path(__file__).resolve().parents[1] / "results"


class NeuralNetwork:
    def __init__(self, width: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, 0.01, (64, width)), "b1": np.zeros(width),
            "W2": rng.normal(0, 0.01, (width, width)), "b2": np.zeros(width),
            "W3": rng.normal(0, 0.01, (width, 10)), "b3": np.zeros(10),
        }

    def forward(self, x):
        p = self.params
        z1 = x @ p["W1"] + p["b1"]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = np.maximum(z2, 0)
        logits = a2 @ p["W3"] + p["b3"]
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        probs = np.exp(log_probs)
        return probs, (x, z1, a1, z2, a2, probs, log_probs)

    @staticmethod
    def loss(cache, targets):
        # Log-softmax avoids clipping and remains stable for extreme logits.
        return float(-np.sum(targets * cache[-1]) / len(targets))

    def backward(self, cache, targets):
        x, z1, a1, z2, a2, probs, _ = cache
        d3 = (probs - targets) / len(targets)
        d2 = (d3 @ self.params["W3"].T) * (z2 > 0)
        d1 = (d2 @ self.params["W2"].T) * (z1 > 0)
        return {"W3": a2.T @ d3, "b3": d3.sum(axis=0),
                "W2": a1.T @ d2, "b2": d2.sum(axis=0),
                "W1": x.T @ d1, "b1": d1.sum(axis=0)}


def gradient_check(model, x, y, seed=123):
    """Check ten distinct, uniformly sampled weight coordinates, without retries."""
    _, cache = model.forward(x)
    analytic = model.backward(cache, y)
    keys = ["W1", "W2", "W3"]
    sizes = [model.params[k].size for k in keys]
    boundaries = np.cumsum(sizes)
    records = []
    # Compute the loss difference with log1p/expm1 to avoid subtracting
    # two nearly equal losses. This also works on Windows where longdouble
    # has the same precision as float64.
    numeric_model = NeuralNetwork(model.params["b1"].size)
    numeric_model.params = {k: v.astype(np.longdouble) for k, v in model.params.items()}
    numeric_x = x.astype(np.longdouble)
    numeric_y = y.astype(np.longdouble)
    for flat in np.random.default_rng(seed).choice(sum(sizes), 10, replace=False):
        layer = int(np.searchsorted(boundaries, flat, side="right"))
        key = keys[layer]
        offset = int(flat - (boundaries[layer - 1] if layer else 0))
        index = np.unravel_index(offset, model.params[key].shape)
        original = numeric_model.params[key][index]
        try:
            numeric_model.params[key][index] = original + np.longdouble("1e-5")
            _, plus_cache = numeric_model.forward(numeric_x)
            plus_logits = plus_cache[4] @ numeric_model.params["W3"] + numeric_model.params["b3"]
            numeric_model.params[key][index] = original - np.longdouble("1e-5")
            minus_probs, minus_cache = numeric_model.forward(numeric_x)
            minus_logits = minus_cache[4] @ numeric_model.params["W3"] + numeric_model.params["b3"]
        finally:
            numeric_model.params[key][index] = original
        delta = plus_logits - minus_logits
        loss_difference = np.mean(np.log1p(np.sum(minus_probs * np.expm1(delta), axis=1))
                                  - np.sum(numeric_y * delta, axis=1))
        numeric = float(loss_difference / np.longdouble("2e-5"))
        exact = float(analytic[key][index])
        relative = abs(exact - numeric) / max(1e-12, abs(exact) + abs(numeric))
        records.append(dict(parameter=key, row=int(index[0]), column=int(index[1]),
                            analytic=exact, numerical=numeric, relative_error=relative,
                            passed=bool(relative < 1e-6)))
    return records


def timing(model, x, y):
    for _ in range(10):
        model.backward(model.forward(x)[1], y)
    _, cache = model.forward(x)
    forward, backward = [], []
    for _ in range(100):
        start = perf_counter_ns()
        model.forward(x)
        forward.append((perf_counter_ns() - start) / 1e6)
        start = perf_counter_ns()
        model.backward(cache, y)
        backward.append((perf_counter_ns() - start) / 1e6)
    return float(np.median(forward)), float(np.median(backward))


def write_csv(name, rows):
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    x = digits.data.astype(np.float64) / 16.0
    y = np.eye(10)[digits.target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=digits.target, random_state=42)
    checks, summaries, history = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    # Fix BLAS to one thread for reproducible small-matrix timing overhead.
    with threadpool_limits(limits=1, user_api="blas"):
        for ax, width in zip(axes, (16, 64)):
            initial = NeuralNetwork(width)
            probs, cache = initial.forward(x_train)
            assert probs.shape == (len(x_train), 10)
            assert np.allclose(probs.sum(axis=1), 1, atol=1e-6, rtol=0)
            initial_loss = initial.loss(cache, y_train)
            assert abs(initial_loss - np.log(10)) < 0.01
            records = gradient_check(initial, x_train[:32], y_train[:32])
            checks.extend(dict(width=width, **row) for row in records)
            passed = sum(row["passed"] for row in records)
            print(f"W={width}: initial loss={initial_loss:.8f}; gradient check {passed}/10", flush=True)
            assert passed >= 9, f"Gradient check failed for width {width}"
            for lr in (0.001, 0.01, 0.1):
                model = NeuralNetwork(width)
                rng = np.random.default_rng(2026)
                losses = []
                for epoch in range(1, 51):
                    order = rng.permutation(len(x_train))
                    for start in range(0, len(order), 32):
                        batch = order[start:start + 32]
                        _, cache = model.forward(x_train[batch])
                        gradients = model.backward(cache, y_train[batch])
                        for key in model.params:
                            model.params[key] -= lr * gradients[key]
                    train_probs, train_cache = model.forward(x_train)
                    test_probs, test_cache = model.forward(x_test)
                    loss = model.loss(train_cache, y_train)
                    test_loss = model.loss(test_cache, y_test)
                    accuracy = float(np.mean(test_probs.argmax(1) == y_test.argmax(1)))
                    losses.append(loss)
                    history.append(dict(width=width, learning_rate=lr, epoch=epoch,
                                        train_loss=loss, test_loss=test_loss, test_accuracy=accuracy))
                    print(f"W={width} lr={lr:g} epoch={epoch:02d}/50 loss={loss:.6f} test_accuracy={accuracy:.4f}", flush=True)
                fwd, bwd = timing(model, x_train[:32], y_train[:32])
                summaries.append(dict(width=width, learning_rate=lr, initial_loss=initial_loss,
                                      final_train_loss=loss, final_test_loss=test_loss,
                                      test_accuracy=accuracy, forward_ms=fwd, backward_ms=bwd))
                ax.plot(range(1, 51), losses, label=f"lr={lr:g}")
            ax.set(title=f"Hidden width W={width}", xlabel="Epoch", ylabel="Training cross-entropy")
            ax.grid(alpha=0.25)
            ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=160)
    plt.close(fig)
    write_csv("gradient_check.csv", checks)
    write_csv("training_history.csv", history)
    write_csv("metrics.csv", summaries)
    metadata = dict(train_samples=len(x_train), test_samples=len(x_test), batch_size=32,
                    epochs=50, dtype="float64", initialization="normal(mean=0, std=0.01)",
                    split_random_state=42, init_seed=42, shuffle_seed=2026,
                    timing_runs=100, timing_warmups=10, timing_batch_size=32, blas_threads=1)
    (RESULTS / "configuration.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nFinal results (timing: median of 100 runs, batch=32):")
    print("Width   LR       Test accuracy   Forward ms   Backward ms")
    for row in summaries:
        print(f"{row['width']:5d}   {row['learning_rate']:<7g}  "
              f"{row['test_accuracy']:12.2%}   {row['forward_ms']:10.5f}   "
              f"{row['backward_ms']:11.5f}")
    print(f"Evidence saved to {RESULTS}")


if __name__ == "__main__":
    main()
