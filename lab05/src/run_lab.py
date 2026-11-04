from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import platform
import sys
from time import perf_counter, perf_counter_ns

import numpy as np
import sklearn
from sklearn.datasets import make_moons
import torch
from torch import nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from models import ScalarModel, initial_parameters, scalar_loss, torch_model
from verification import verify_gradients

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"
SEED = 42
SAMPLES = 128
EPOCHS = 100
LEARNING_RATE = 0.1
TRAINING_RUNS = 5
WARMUPS = 30
TIMING_RUNS = 100


def write_csv(name, rows):
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar_training(features, labels, initial):
    model = ScalarModel(initial)
    parameters = model.parameters()
    initial_loss = scalar_loss(model, features, labels).data
    gc.collect()
    start = perf_counter()
    for _ in range(EPOCHS):
        for parameter in parameters:
            parameter.grad = 0.0
        loss = scalar_loss(model, features, labels)
        loss.backward()
        for parameter in parameters:
            parameter.data -= LEARNING_RATE * parameter.grad
    elapsed = perf_counter() - start
    final_loss = scalar_loss(model, features, labels).data
    predictions = np.array([model(row).data >= 0.5 for row in features])
    return dict(seconds=elapsed, initial_loss=initial_loss, final_loss=final_loss,
                training_accuracy=float(np.mean(predictions == labels))), np.array([p.data for p in parameters])


def torch_training(features, labels, initial):
    model = torch_model(initial)
    inputs = torch.from_numpy(features)
    targets = torch.from_numpy(labels.astype(np.float64)).reshape(-1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    with torch.no_grad():
        initial_loss = nn.functional.binary_cross_entropy(model(inputs), targets).item()
    gc.collect()
    start = perf_counter()
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
    elapsed = perf_counter() - start
    with torch.no_grad():
        probabilities = model(inputs)
        final_loss = nn.functional.binary_cross_entropy(probabilities, targets).item()
        accuracy = float((probabilities.ge(0.5) == targets.bool()).double().mean())
    weights = torch.cat([parameter.detach().flatten() for parameter in model.parameters()]).numpy()
    return dict(seconds=elapsed, initial_loss=initial_loss, final_loss=final_loss,
                training_accuracy=accuracy), weights, model


def compare_training(features, labels):
    rows, parameter_errors = [], []
    initial = initial_parameters(SEED)
    model = None
    for run in range(1, TRAINING_RUNS + 1):
        scalar_result, scalar_weights = scalar_training(features, labels, initial)
        torch_result, torch_weights, model = torch_training(features, labels, initial)
        error = float(np.max(np.abs(scalar_weights - torch_weights)))
        assert error < 1e-6
        assert abs(scalar_result["final_loss"] - torch_result["final_loss"]) < 1e-6
        for engine, record in (("scalar", scalar_result), ("pytorch", torch_result)):
            assert record["final_loss"] < record["initial_loss"]
            rows.append(dict(engine=engine, run=run, **record))
        parameter_errors.append(error)
        write_csv("training_runs.csv", rows)
        print(f"Training run {run}/5: scalar={scalar_result['seconds']:.3f}s, "
              f"PyTorch={torch_result['seconds']:.3f}s, parameter difference={error:.3g}", flush=True)
    summary = []
    for engine in ("scalar", "pytorch"):
        records = [row for row in rows if row["engine"] == engine]
        summary.append(dict(engine=engine, runs=len(records),
                            median_seconds=float(np.median([row["seconds"] for row in records])),
                            initial_loss=records[0]["initial_loss"], final_loss=records[0]["final_loss"],
                            training_accuracy=records[0]["training_accuracy"]))
    write_csv("training_summary.csv", summary)
    return model.eval(), summary, max(parameter_errors)


def benchmark(experiment, functions, calls):
    for function in functions.values():
        for _ in range(WARMUPS):
            function()
    rows = []
    variants = list(functions)
    for run in range(1, TIMING_RUNS + 1):
        for variant in variants if run % 2 else reversed(variants):
            function = functions[variant]
            start = perf_counter_ns()
            for _ in range(calls):
                function()
            elapsed_ms = (perf_counter_ns() - start) / calls / 1e6
            rows.append(dict(experiment=experiment, variant=variant, run=run,
                             calls_per_run=calls, milliseconds_per_call=elapsed_ms))
    summary = []
    for variant in variants:
        samples = [row["milliseconds_per_call"] for row in rows if row["variant"] == variant]
        summary.append(dict(variant=variant, median_ms=float(np.median(samples)),
                            p25_ms=float(np.percentile(samples, 25)), p75_ms=float(np.percentile(samples, 75)),
                            runs=TIMING_RUNS, calls_per_run=calls))
    return summary, rows


@torch.inference_mode()
def compare_execution(model, features):
    inputs = torch.from_numpy(features)
    graph = symbolic_trace(model)
    ShapeProp(graph).propagate(inputs)
    (RESULTS / "model_graph.txt").write_text(str(graph.graph) + "\n", encoding="utf-8")
    nodes = []
    for node in graph.graph.nodes:
        metadata = node.meta.get("tensor_meta")
        nodes.append(dict(node=node.name, operation=node.op, target=str(node.target),
                          shape=str(tuple(metadata.shape)) if metadata else "",
                          dtype=str(metadata.dtype) if metadata else "",
                          stride=str(tuple(metadata.stride)) if metadata else ""))
    write_csv("graph_nodes.csv", nodes)
    failure = ""
    start = perf_counter()
    try:
        optimized = torch.compile(model, fullgraph=True, dynamic=False, backend="inductor")
        optimized(inputs)
        if optimized is model:
            raise RuntimeError("torch.compile returned the eager model.")
        mode = "compiled"
    except (RuntimeError, ImportError, NotImplementedError, AssertionError, OSError) as error:
        failure = f"{type(error).__name__}: {str(error).splitlines()[0]}"
        start = perf_counter()
        optimized = torch.jit.trace(model, inputs, check_inputs=[(inputs.flip(0),)])
        optimized(inputs)
        (RESULTS / "traced_graph.txt").write_text(str(optimized.inlined_graph) + "\n", encoding="utf-8")
        mode = "traced"
    setup_ms = (perf_counter() - start) * 1000
    expected, actual = model(inputs), optimized(inputs)
    torch.testing.assert_close(expected, actual, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(model(inputs.flip(0)), optimized(inputs.flip(0)), atol=1e-10, rtol=1e-10)
    summary, raw = benchmark("execution", {"eager": lambda: model(inputs), mode: lambda: optimized(inputs)}, 100)
    write_csv("execution_times.csv", summary)
    details = dict(mode=mode, backend="inductor" if mode == "compiled" else "torch.jit.trace",
                   fallback_reason=failure, setup_and_first_forward_ms=setup_ms,
                   max_output_difference=float((expected - actual).abs().max()),
                   warmups=WARMUPS, runs=TIMING_RUNS, calls_per_run=100, batch_size=len(inputs))
    print(f"Execution: {mode}; eager={summary[0]['median_ms']:.6f}ms, "
          f"{mode}={summary[1]['median_ms']:.6f}ms", flush=True)
    return summary, raw, details


@torch.inference_mode()
def compare_layout(features):
    generator = torch.Generator().manual_seed(SEED)
    base = torch.randn(512, 512, dtype=torch.float64, generator=generator)
    view = base.t()
    contiguous = view.contiguous()
    assert not view.is_contiguous() and contiguous.is_contiguous()
    torch.testing.assert_close(view, contiguous, rtol=0, atol=0)
    torch.testing.assert_close(view.softmax(dim=1), contiguous.softmax(dim=1), rtol=0, atol=0)
    summary, raw = benchmark("contiguity", {
        "contiguous": lambda: contiguous.softmax(dim=1),
        "non_contiguous": lambda: view.softmax(dim=1),
    }, 10)
    write_csv("layout_times.csv", summary)
    rows = []
    for name, tensor in (("model_input", torch.from_numpy(features)), ("transposed_view", view),
                         ("contiguous_copy", contiguous)):
        rows.append(dict(tensor=name, shape=str(tuple(tensor.shape)), dtype=str(tensor.dtype),
                         stride=str(tuple(tensor.stride())), contiguous=tensor.is_contiguous(),
                         bytes=tensor.numel() * tensor.element_size()))
    write_csv("tensor_metadata.csv", rows)
    print(f"Layout: contiguous={summary[0]['median_ms']:.6f}ms, "
          f"non-contiguous={summary[1]['median_ms']:.6f}ms", flush=True)
    return summary, raw


def write_report(checks, training, execution, layout, configuration):
    details = configuration["execution"]
    scalar_seconds, torch_seconds = [row["median_seconds"] for row in training]
    eager_ms, optimized_ms = [row["median_ms"] for row in execution]
    contiguous_ms, strided_ms = [row["median_ms"] for row in layout]
    training_table = "\n".join(
        f"| {row['engine']} | {row['median_seconds']:.6f} | {row['initial_loss']:.6f} "
        f"| {row['final_loss']:.6f} | {100 * row['training_accuracy']:.2f}% |" for row in training
    )
    execution_table = "\n".join(
        f"| {row['variant']} | {row['median_ms']:.6f} | {row['p25_ms']:.6f} | {row['p75_ms']:.6f} |"
        for row in execution
    )
    layout_table = "\n".join(
        f"| {row['variant']} | {row['median_ms']:.6f} | {row['p25_ms']:.6f} | {row['p75_ms']:.6f} |"
        for row in layout
    )
    if optimized_ms < eager_ms:
        amortization = (f"Setup is amortised after approximately "
                        f"{details['setup_and_first_forward_ms'] / (eager_ms - optimized_ms):.0f} "
                        "calls at these measured per-call times.")
    else:
        amortization = "There is no measured per-call saving to amortise setup for this input."
    fallback = f"Compilation fallback: {details['fallback_reason']}." if details["fallback_reason"] else ""
    content = f"""# Lab 5: Frameworks and automatic differentiation

## Run and settings

From the repository root:

```powershell
python main.py
```

Python {configuration['python']}, PyTorch {configuration['packages']['torch']},
NumPy {configuration['packages']['numpy']} and scikit-learn {configuration['packages']['sklearn']}.
All experiments use CPU, float64, one intra-op thread and one inter-op thread.
Python floats and PyTorch float64 provide comparable precision. Seed: 42.
Full settings are in `results/configuration.json`.

## Scalar engine and gradient verification

`Value` stores a scalar value, gradient, parent nodes, operation name and local
backward rule. Addition, multiplication, constant powers and ReLU construct
the graph during the forward pass. Sigmoid and logarithm support the BCE model.

Backward traversal first builds a topological ordering with each node visited
once. It seeds the output gradient with 1 and visits nodes in reverse order.
Local rules add gradient contributions with `+=`, including when a parameter
has multiple downstream uses. Intermediate gradients are reset on each backward
call; leaf gradients accumulate until the training loop clears them.

The checked expression is `((a*b+c)**2).relu().sum()`, using 16 values in each
of a, b and c. The scalar implementation sums the individual output nodes.
Central differences use epsilon=1e-6.

| Comparison | Maximum absolute gradient difference |
|---|---:|
| Scalar vs PyTorch autograd | {checks['max_torch_absolute_error']:.12g} |
| Scalar vs numerical gradients | {checks['max_numerical_absolute_error']:.12g} |

All {checks['checked_derivatives']} derivatives satisfy the required difference
**<1e-6**. Additional checks cover topological ordering, a shared subgraph,
gradient accumulation across two backward calls, ReLU at negative/zero/positive
inputs, constant powers and sigmoid/BCE derivatives. Detailed comparisons are
in `results/gradient_check.csv` and `results/gradient_check.json`.

## Tiny-model training

Data: `make_moons(n_samples=128, noise=0.1, random_state=42)`.
Architecture: **2 → 8 → ReLU → 1 → sigmoid**, with **33 parameters**.
Loss: mean binary cross-entropy. Both engines use full-batch SGD, lr=0.1,
for 100 epochs. Initial weights come from the same NumPy generator with
normal(0,0.5) weights and zero biases. Every repetition resets the model.
The reported accuracy is training accuracy on these 128 examples.

Timings are the **median of five complete training runs**, including forward
graph construction, loss, gradient clearing, backward and parameter updates.
Data generation, model/optimizer construction and final evaluation are excluded.

| Engine | Median training time, s | Initial BCE | Final BCE | Training accuracy |
|---|---:|---:|---:|---:|
{training_table}

PyTorch is {scalar_seconds / torch_seconds:.2f}× faster in this measurement.
The largest final parameter difference between the two implementations is
{configuration['maximum_trained_parameter_difference']:.12g}. Vectorised tensor
kernels avoid creating and traversing thousands of Python scalar nodes.
The scalar engine also tracks gradients for scalar constants; PyTorch requests
gradients for model parameters. Raw timings are in `results/training_runs.csv`.

## Graph and execution modes

The model's FX graph is saved in [model_graph.txt](results/model_graph.txt).
[graph_nodes.csv](results/graph_nodes.csv) records each node's operation, shape,
dtype and stride. The model input is a contiguous float64 tensor of shape
(128,2), stride (2,1); output shape is (128,1).

The optimized mode is **{details['mode']}**, backend **{details['backend']}**.
{fallback}
Outputs agree with eager execution within 1e-10; the measured maximum difference
is {details['max_output_difference']:.12g}. Both original and reversed inputs
are checked. Inference uses `eval()` and `torch.inference_mode()`.

Each variant receives 30 warmups, then 100 timed blocks of 100 forwards.
The table gives the median time per forward and its interquartile range.
Variant order alternates between blocks. Compilation/tracing and the first
forward are outside steady-state timing; together they took
{details['setup_and_first_forward_ms']:.3f} ms in this run. Existing compiler
caches can affect that setup measurement.

| Mode | Median ms/forward | 25th percentile | 75th percentile |
|---|---:|---:|---:|
{execution_table}

The eager/optimized latency ratio is {eager_ms / optimized_ms:.3f}×.
{amortization}

Compilation helps when a repeatedly used, stable graph saves enough dispatch
or kernel-launch work to repay setup cost. Larger graphs can also benefit from
operator fusion and memory optimisations. Small models may remain dominated by
framework overhead. Shape or control-flow changes can cause recompilation.
Tracing is a fallback for this fixed operator sequence; it records the executed
path and is unsuitable for unrecorded data-dependent branches.

## Tensor representation and contiguity

The layout experiment applies `softmax(dim=1)` to two tensors with identical
values and shape (512,512), dtype float64. The transposed view has strides
(1,512) and is non-contiguous; its contiguous copy has strides (512,1).
Input creation and the explicit `.contiguous()` copy are outside timing.
The softmax outputs are checked for equality.

Each variant receives 30 warmups, then 100 blocks of 10 operations.

| Input layout | Median ms/operation | 25th percentile | 75th percentile |
|---|---:|---:|---:|
{layout_table}

The non-contiguous/contiguous latency ratio is **{strided_ms / contiguous_ms:.3f}×**.
A transpose changes strides without copying data. Row-wise softmax on a strided
view may require a contiguous temporary, adding data movement. A contiguous
copy preserves logical values while changing storage order. This measurement
concerns row-wise softmax; other operators can handle strides differently.
Tensor properties are saved in `results/tensor_metadata.csv`; individual
execution-mode and layout measurements are in `results/timing_runs.csv`.

## References

- Vijay Janapa Reddi, *Introduction to Machine Learning Systems*, Chapter 7,
  §7.3, §7.4 and §7.6–7.8 in the assigned edition.
- [PyTorch autograd](https://docs.pytorch.org/docs/2.3/autograd.html).
- [torch.compile](https://docs.pytorch.org/docs/2.3/generated/torch.compile.html).
- [torch.jit.trace](https://docs.pytorch.org/docs/2.3/generated/torch.jit.trace.html).
"""
    (LAB / "report.md").write_text(content, encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    checks = verify_gradients(RESULTS)
    print(f"Gradient checks: PyTorch error={checks['max_torch_absolute_error']:.3g}, "
          f"numerical error={checks['max_numerical_absolute_error']:.3g}", flush=True)
    features, labels = make_moons(n_samples=SAMPLES, noise=0.1, random_state=SEED)
    model, training, weight_error = compare_training(features, labels)
    gc.collect()
    execution, execution_raw, execution_details = compare_execution(model, features)
    layout, layout_raw = compare_layout(features)
    write_csv("timing_runs.csv", execution_raw + layout_raw)
    configuration = dict(
        seed=SEED, python=platform.python_version(), platform=platform.platform(),
        packages=dict(torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__),
        device="cpu", dtype="float64", torch_threads=1, torch_interop_threads=1,
        deterministic_algorithms=True, dataset="make_moons", samples=SAMPLES, noise=0.1,
        architecture="2 -> 8 -> ReLU -> 1 -> sigmoid", parameters=33,
        loss="binary cross-entropy", optimizer="SGD", learning_rate=LEARNING_RATE,
        epochs=EPOCHS, batch_size=SAMPLES, training_runs=TRAINING_RUNS,
        training_timing="training loop only; excludes data, model and optimizer construction and final evaluation",
        shared_initialization="NumPy default_rng(42), normal(0,0.5) weights and zero biases",
        maximum_trained_parameter_difference=weight_error,
        execution=execution_details, layout_operation="softmax(dim=1)", layout_shape=[512, 512],
        layout_warmups=WARMUPS, layout_runs=TIMING_RUNS, layout_calls_per_run=10,
    )
    (RESULTS / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    write_report(checks, training, execution, layout, configuration)
    print(f"Results: {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
