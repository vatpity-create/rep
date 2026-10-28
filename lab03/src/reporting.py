"""Generate Lab 03 figures and report from saved results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"MLP": "#2864b7", "CNN": "#18834b", "Attention": "#b55716"}
SUITE_LABELS = {"reference": "Exact manual architectures", "matched": "Models with similar parameter counts"}


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    for record in records:
        for key, value in record.items():
            try:
                record[key] = float(value)
            except ValueError:
                pass
    return records


def save_figure(figure, results, name):
    figure.tight_layout()
    figure.savefig(results / f"{name}.png", dpi=180, bbox_inches="tight")
    svg_path = results / f"{name}.svg"
    figure.savefig(svg_path, bbox_inches="tight")
    # Matplotlib leaves spaces at the ends of SVG path lines. Normalize them so
    # regenerated, versioned figures also pass git diff --check.
    svg_path.write_text("\n".join(line.rstrip() for line in
                                   svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8")
    plt.close(figure)


def make_plots(results, groups):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    count = len(groups)
    figure, axes = plt.subplots(1, count, figsize=(6.5 * count, 4.7), squeeze=False)
    for axis, (suite, records) in zip(axes[0], groups.items()):
        for row in records:
            x, y = row["parameters"] / 1000, 100 * row["test_accuracy"]
            axis.scatter(x, y, s=95, color=COLORS[row["model"]], zorder=3)
            axis.annotate(f"{row['model']}\n{int(row['parameters']):,}; {y:.2f}%", (x, y),
                          xytext=(6, 9), textcoords="offset points", fontsize=9)
        if suite == "reference":
            axis.axvspan(100, 150, color="#7790a8", alpha=0.12, label="100k-150k target")
            axis.set_xlim(0, max(row["parameters"] for row in records) / 1000 * 1.3)
            axis.legend(loc="lower right")
        else:
            axis.set_xlim(117, 122)
        accuracy = [100 * row["test_accuracy"] for row in records]
        axis.set_ylim(min(accuracy) - 2, max(accuracy) + 4)
        axis.set(title=SUITE_LABELS[suite], xlabel="Trainable parameters (thousands)", ylabel="Test accuracy (%)")
        axis.grid(alpha=0.22)
    save_figure(figure, results, "accuracy_vs_parameters")

    figure, axes = plt.subplots(count, 2, figsize=(12.8, 4.6 * count), squeeze=False)
    for pair, (suite, records) in zip(axes, groups.items()):
        for axis, batch in zip(pair, (1, 128)):
            for row in records:
                x, y = row[f"latency_b{batch}_ms"], 100 * row["test_accuracy"]
                axis.errorbar(x, y, xerr=[[x - row[f"latency_b{batch}_p25_ms"]],
                                          [row[f"latency_b{batch}_p75_ms"] - x]],
                              fmt="o", markersize=8, capsize=4, color=COLORS[row["model"]])
                axis.annotate(f"{row['model']} ({x:.3f} ms)", (x, y), xytext=(5, 10),
                              textcoords="offset points", fontsize=9)
            max_latency = max(row[f"latency_b{batch}_p75_ms"] for row in records)
            accuracy = [100 * row["test_accuracy"] for row in records]
            axis.set(xlim=(0, max_latency * 1.45), ylim=(min(accuracy) - 2, max(accuracy) + 4),
                     title=f"{SUITE_LABELS[suite]} | batch {batch}",
                     xlabel=f"Median forward latency per batch (ms), B={batch}; bars: IQR",
                     ylabel="Test accuracy (%)")
            axis.grid(alpha=0.22)
    save_figure(figure, results, "accuracy_vs_latency")

    figure, axes = plt.subplots(count, 2, figsize=(12.8, 4.3 * count), squeeze=False)
    for pair, (suite, records) in zip(axes, groups.items()):
        for row in records:
            history = read_csv(results / suite / f"{row['model'].lower()}_history.csv")
            epochs = [record["epoch"] for record in history]
            pair[0].plot(epochs, [record["train_loss"] for record in history], "-o",
                         label=row["model"], color=COLORS[row["model"]])
            pair[1].plot(epochs, [100 * record["validation_accuracy"] for record in history], "-o",
                         label=row["model"], color=COLORS[row["model"]])
        for axis in pair:
            axis.set_xticks(range(1, 6))
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.22)
            axis.legend()
        pair[0].set(title=SUITE_LABELS[suite], ylabel="Online training cross-entropy")
        pair[1].set(title=SUITE_LABELS[suite], ylabel="Validation accuracy (%)")
    save_figure(figure, results, "training_curves")

    figure, axes = plt.subplots(1, count, figsize=(6.5 * count, 4.7), squeeze=False)
    for axis, (suite, records) in zip(axes[0], groups.items()):
        positions = np.arange(len(records))
        for shift, key, label, color in (
            (-0.24, "hand_flops", "Hand: core FLOPs", "#2864b7"),
            (0, "thop_raw_ops", "2 x raw THOP output (incomplete)", "#b55716"),
            (0.24, "thop_core_flops", "THOP: aligned core FLOPs", "#18834b"),
        ):
            axis.bar(positions + shift, [row[key] * (2 if key == "thop_raw_ops" else 1) / 1e6
                                         for row in records], width=0.23, label=label, color=color)
        axis.set_xticks(positions, [row["model"] for row in records])
        axis.set(title=SUITE_LABELS[suite], ylabel="Million operations per image, B=1")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.22)
    save_figure(figure, results, "flops_comparison")


def metrics_table(records):
    lines = [
        "| Model | Parameters | Core FLOPs | FP32 weights, KiB | Test accuracy | B=1, ms | B=128, ms/batch | B=128, ms/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(f"| {row['model']} | {int(row['parameters']):,} | {int(row['hand_flops']):,} "
                     f"| {row['weights_kib']:.2f} | {100 * row['test_accuracy']:.2f}% "
                     f"| {row['latency_b1_ms']:.5f} | {row['latency_b128_ms']:.5f} "
                     f"| {row['latency_b128_per_image_ms']:.5f} |")
    return "\n".join(lines)


def make_report(results, groups, config):
    sections = [f"""# Lab 3: Architectures and their costs

## Method and reproduction

MLP, CNN and patch-attention classifiers are compared on Fashion-MNIST using
accuracy, parameter count, core FLOPs, inference latency and memory access patterns.
Run from the repository root in the activated environment:

```powershell
python lab03/src/run_lab.py
```

The default run trains both the exact manual architectures (`reference`) and
models with similar parameter counts (`matched`). To select one experiment or
rebuild the report and plots from saved results:

```powershell
python lab03/src/run_lab.py --suite reference
python lab03/src/run_lab.py --suite matched
python lab03/src/run_lab.py --report-only
python lab03/src/check_lab.py
```

Fashion-MNIST's official 60,000 training images are split into **55,000 training
and 5,000 validation** examples. The official **10,000 test** images remain separate.
Transforms are `ToTensor()` and `Normalize((0.2860,), (0.3530,))`, without augmentation.
All models share the split and shuffle seeds and use batches of 128 with
`drop_last=False`. Split indices are saved in `results/split_indices.npz`.

Each model trains for **5 epochs** with Adam, lr=0.001, betas=(0.9,0.999),
eps=1e-8, weight_decay=0 and cross-entropy on logits. Validation is evaluated
after each epoch; test accuracy uses the final epoch, without test-based selection.
Python, NumPy, PyTorch, splitting and shuffling use seed **{config['seed']}**.
Deterministic algorithms are enabled.

Measurements use **{config['device_name']}**, `{config['device']}`, FP32,
{config['torch_threads']} intra-op threads and {config['torch_interop_threads']} inter-op thread.
Python {config['python'].split()[0]}, PyTorch {config['packages']['torch']},
torchvision {config['packages']['torchvision']}, torchinfo {config['packages']['torchinfo']},
THOP {config['packages']['thop']}, NumPy {config['packages']['numpy']} and
Matplotlib {config['packages']['matplotlib']}. Full settings are in `results/configuration.json`.

## Architectures

The exact layer sizes in the manual do not satisfy its 100k–150k parameter target.
Both the specified architectures and a separate comparison near 119k parameters
are therefore included.

| Experiment | MLP hidden widths | CNN channels | Attention d / FF width | Parameters: MLP / CNN / Attention |
|---|---|---|---|---|
| reference | 256, 128 | 16, 32 | 64 / 128 | 235,146 / 20,490 / 70,794 |
| matched | 128, 128 | 48, 128 | 84 / 168 | 118,282 / 118,634 / 119,794 |

MLP flattens 784 pixels, applies two Linear/ReLU hidden layers and produces
10 logits. CNN uses two 3×3 padded convolutions, each followed by ReLU and
2×2 max pooling, then flattens the 7×7 feature maps and applies a Linear classifier.

Attention splits each image into 16 non-overlapping 7×7 patches, projects each
49-value patch, applies two encoder layers, mean-pools the tokens and produces
10 logits. Both experiments use four heads, ReLU, dropout 0.1, post-LayerNorm
and residual connections. Encoder layers are initialised independently.
The manual specifies no position embedding or class token, so neither is added.
Mean pooling therefore makes predictions invariant to patch order, while
pixel positions within a patch are retained.

Only widths change in the matched experiment. Its parameter counts differ by
about 1.28%, but the operation counts can still differ substantially.

## Parameter counts

Biases are included. Linear uses `in*out + out`; Conv2d uses
`K*K*Cin*Cout + Cout`; each LayerNorm(d) has `2*d` trainable coefficients.
For the reference models:

```text
MLP:
  (784*256+256) + (256*128+128) + (128*10+10)
  = 200960 + 32896 + 1290 = 235146

CNN:
  (3*3*1*16+16) + (3*3*16*32+32) + (1568*10+10)
  = 160 + 4640 + 15690 = 20490

Attention:
  Patch projection: 49*64+64 = 3200
  One encoder: QKV 12480 + output projection 4160
               + FF1 8320 + FF2 8256 + two LayerNorms 256 = 33472
  Total: 3200 + 2*33472 + (64*10+10) = 70794
```

Manual totals agree with `torchinfo.summary` and the parameter tensors.
Layer-by-layer calculations are in `results/<suite>/hand_calculations.csv`;
summary tables and tensor shapes are in `*_summary.txt`.

## FLOPs by hand and with THOP

One multiply plus one addition counts as **2 FLOPs**. Core FLOPs cover matrix
products and convolutions for one image. Bias additions, activations, max-pool
comparisons, softmax, LayerNorm, residual additions and mean pooling are excluded.
Convolution counts include operations on zero padding. These counts describe
arithmetic, rather than measured processor instructions.

```text
MLP Linear(784,256): 2 * 784 * 256 = 401408 FLOPs.
Full MLP: 401408 + 65536 + 2560 = 469504.

CNN Conv1, output 16x28x28:
  2 * 3 * 3 * 1 * 16 * 28 * 28 = 225792 FLOPs.
Conv2, output 32x14x14:
  2 * 3 * 3 * 16 * 32 * 14 * 14 = 1806336 FLOPs.
Classifier: 2 * 1568 * 10 = 31360.
Full CNN: 225792 + 1806336 + 31360 = 2063488.

One attention encoder: L=16, d=64, h=4, head_dim=16, FF=128.
  QKV:                2 * 16 * 64 * (3 * 64) = 393216
  QK^T:               2 * h * L^2 * (d/h)    = 32768
  Attention @ V:      2 * h * L^2 * (d/h)    = 32768
  Output projection:  2 * 16 * 64 * 64       = 131072
  FF1 + FF2:          2 * (2 * 16 * 64 * 128) = 524288
  Encoder total: 1114112 FLOPs.
Full Attention: patch projection 100352 + 2*1114112 + classifier 1280
  = 2329856 FLOPs.
```

Summing over heads gives dimension d; no extra factor h is applied on top of d.
The full attention model's formula is
`2*L*49*d + 2*(8*L*d^2 + 4*L^2*d + 4*L*d*FF) + 2*d*10`.

THOP reports MACs for Linear and Conv, so those counts are doubled. Its default
rules omit MultiheadAttention operations and count LayerNorm in different units.
The raw result is saved for comparison. A separate profile adds a complete MHA
hook for QKV, scores, weighted values and output projection, and excludes LayerNorm
to match the manual core convention. The resulting core counts agree exactly.

PyTorch 2.3's fused encoder can bypass module hooks. Its MHA fastpath is disabled
only while counting and then restored for inference timing. Profiling uses model
copies so temporary THOP buffers do not enter the saved checkpoints.
"""]
    for suite, records in groups.items():
        sections.append(f"## {suite.capitalize()} results\n\n" + metrics_table(records))
        lines = ["| Model | Hand core FLOPs | Raw THOP ops | 2×raw, diagnostic | THOP core FLOPs | Core gap | Raw THOP params / torchinfo |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for row in records:
            lines.append(f"| {row['model']} | {int(row['hand_flops']):,} | {int(row['thop_raw_ops']):,} "
                         f"| {int(2 * row['thop_raw_ops']):,} | {int(row['thop_core_flops']):,} "
                         f"| {row['core_gap_percent']:.2f}% | {int(row['thop_raw_parameters']):,} / {int(row['torchinfo_parameters']):,} |")
        sections.append("\n".join(lines))
        layers = read_csv(results / suite / "representative_layers.csv")
        lines = ["Independent profiles of layers restored from the final checkpoints:",
                 "\n| Layer | Hand FLOPs | THOP MACs | THOP FLOPs |", "|---|---:|---:|---:|"]
        for layer in layers:
            lines.append(f"| {layer['layer']} | {int(layer['hand_flops']):,} "
                         f"| {int(layer['thop_macs']):,} | {int(layer['thop_flops']):,} |")
        sections.append("\n".join(lines))
        attention = next(row for row in records if row["model"] == "Attention")
        d = 64 if suite == "reference" else 84
        missing = 2 * (4 * 16 * d * d + 2 * 16 * 16 * d)
        norms = 4 * 4 * 16 * d
        sections.append(f"For {suite} Attention, raw THOP {int(attention['thop_raw_ops']):,} = "
                        f"core MACs {int(attention['thop_core_macs']):,} − omitted MHA MACs {missing:,} "
                        f"+ LayerNorm operations {norms:,}. The raw parameter count also omits MHA weights.")
    sections.append("""## Inference timing

Each trained model uses `eval()` and `torch.inference_mode()`. After 20 warmups
for each batch size, latency is the **median of 100 runs for B=1** and the
**median of 20 runs for B=128**, measured with `perf_counter_ns()`.
The same first 128 test images are prepared on the device before timing.
The timer covers the full forward pass, including patch extraction and mean
pooling, but excludes loading, transforms and device transfers. CUDA is
synchronised before and after each measured forward pass.

B=128 latency is the time for the whole batch; dividing by 128 gives ms/image.
Every measurement is in `latency_runs.csv`. Horizontal error bars show the
25th–75th percentile range of latency, not accuracy confidence intervals.

## Figures

![Accuracy against parameters](results/accuracy_vs_parameters.png)

![Accuracy against latency for batches 1 and 128](results/accuracy_vs_latency.png)

![Manual and profiled FLOPs](results/flops_comparison.png)

![Training loss and validation accuracy](results/training_curves.png)

SVG versions are also saved in `results/`.

## Architecture trade-offs and memory access

**MLP:** dense layers use regular matrix operations. At B=1, each weight serves
one image; larger batches reuse weights across images and improve arithmetic
intensity. The first reference layer has 802,816 bytes of FP32 weights and
401,408 FLOPs: about 0.5 FLOP/byte of weight traffic for B=1, or 64 FLOP/byte
with complete weight reuse across B=128. These estimates exclude cache effects
and other traffic. MLP has few operators but spends many parameters on the
first layer and has no built-in spatial locality.

**CNN:** sliding windows reuse filters at many image positions. This explains
how the reference CNN performs 2,063,488 core FLOPs with only 20,490 parameters.
Pooling reduces later activation sizes. Activations still require memory:
the first convolution produces 16*28*28 values, or 49 KiB per image, in addition
to the model's 80.04 KiB of weights. Increasing channel widths raises both
arithmetic and activation traffic, even when total parameters match an MLP.

**Attention:** QKV and feed-forward layers are dense matrix products, while
attention adds global token interactions, softmax and reductions. At L=16 and
h=4, a logical score matrix contains h*L*L=1,024 values: 4 KiB per image per
encoder, or 512 KiB at B=128. QKV contains 3*L*d values. Scores scale quadratically
with sequence length; for these short sequences, projections and feed-forward
layers account for most core FLOPs. Normalisation, residual operations and
operator overhead also affect latency. Fused kernels can avoid materialising
some tensors, so logical tensor sizes are not measured peak memory.
Dense attention uses regular tiled matrix operations; content-dependent weights
do not imply random hardware gathers. Omitting positions also loses the spatial
order between patches, which limits this particular classifier.

Chapter 4.6–4.7 relates these differences to weight sharing, locality, broadcast,
reductions and data movement. Parameter count and FLOPs alone do not determine
latency. Establishing a memory or compute bottleneck would require hardware
counters or a roofline analysis; the discussion here describes access patterns.
""")
    sections.append("## Conclusions")
    for suite, records in groups.items():
        best = max(records, key=lambda row: row["test_accuracy"])
        fastest = min(records, key=lambda row: row["latency_b1_ms"])
        smallest = min(records, key=lambda row: row["parameters"])
        cnn = next(row for row in records if row["model"] == "CNN")
        mlp = next(row for row in records if row["model"] == "MLP")
        sections.append(
            f"**{suite.capitalize()}:** {best['model']} has the highest accuracy "
            f"({100 * best['test_accuracy']:.2f}%); {fastest['model']} has the lowest B=1 latency "
            f"({fastest['latency_b1_ms']:.5f} ms). The smallest model is {smallest['model']} "
            f"({int(smallest['parameters']):,} parameters). CNN improves on MLP by "
            f"{100 * (cnn['test_accuracy'] - mlp['test_accuracy']):.2f} percentage points, "
            f"using {cnn['hand_flops'] / mlp['hand_flops']:.2f}× the core FLOPs and "
            f"{cnn['latency_b1_ms'] / mlp['latency_b1_ms']:.2f}× the B=1 latency."
        )
        pareto = [row["model"] for row in records if not any(
            other["test_accuracy"] >= row["test_accuracy"] and
            other["parameters"] <= row["parameters"] and
            other["latency_b1_ms"] <= row["latency_b1_ms"] and (
                other["test_accuracy"] > row["test_accuracy"] or
                other["parameters"] < row["parameters"] or
                other["latency_b1_ms"] < row["latency_b1_ms"])
            for other in records)]
        sections.append(f"The Pareto choices for accuracy, parameter count and B=1 latency are "
                        f"**{', '.join(pareto)}**. Choose {best['model']} when accuracy is the priority "
                        f"and {fastest['model']} when single-image response time is the priority.")
        lines = ["| Model | B=128 speedup per image over B=1 | B=128 images/s |", "|---|---:|---:|"]
        for row in records:
            speedup = row["latency_b1_ms"] / row["latency_b128_per_image_ms"]
            lines.append(f"| {row['model']} | {speedup:.2f}× | {row['throughput_b128_images_s']:.0f} |")
        sections.append("\n".join(lines))
    sections.append("""CNN's spatial locality and weight sharing suit these small images. MLP is a
reasonable choice under a tight latency budget. Attention does not improve
accuracy in either experiment under the specified five-epoch training protocol.

These results use one seed and one device. Background load, temperature and cache
state affect timing. FP32 weight size is not total process memory, and the results
should not be generalised to all models in these architecture families.

## Saved results

- `configuration.json` and `split_indices.npz`: environment, settings and split.
- `<suite>/metrics.csv`: accuracy, parameters, FLOPs, latency and throughput.
- `<suite>/hand_calculations.csv` and `representative_layers.csv`: cost calculations and layer profiles.
- `<suite>/*_summary.txt`: tensor shapes and torchinfo parameter counts.
- `<suite>/*_history.csv`: five training and validation epochs.
- `<suite>/latency_runs.csv`: all 100+20 latency measurements per model.
- `<suite>/*.pt`: final weights, model settings, epoch, seed and test accuracy.
- `verification.json`: split, checkpoint, count and latency consistency checks.

`check_lab.py` checks tensor shapes, gradients, patch coverage, independent encoder
initialisation, batch independence, patch-order invariance and profiler counts.
The report generator also verifies saved checkpoints and the raw timing summaries.

## References

1. Vijay Janapa Reddi, *Introduction to Machine Learning Systems*, Chapter 4,
   §4.2, §4.3, §4.5–4.7 in the assigned edition.
   [DNN architectures](https://mlsysbook.ai/vol1/nn_architectures/nn_architectures.html).
   Section numbering may differ in the current online edition.
2. [torchvision 0.18: FashionMNIST](https://docs.pytorch.org/vision/0.18/generated/torchvision.datasets.FashionMNIST.html).
3. [THOP: MAC counting and custom hooks](https://github.com/Lyken17/pytorch-OpCounter).
4. [PyTorch: TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html).
5. [torchinfo](https://github.com/TylerYep/torchinfo).
""")
    (results.parent / "report.md").write_text("\n\n".join(sections).strip() + "\n", encoding="utf-8")


def generate_outputs(results: Path):
    config = json.loads((results / "configuration.json").read_text(encoding="utf-8"))
    groups = {suite: read_csv(results / suite / "metrics.csv") for suite in config["suites"]}
    for suite, records in groups.items():
        if {row["model"] for row in records} != {"MLP", "CNN", "Attention"} or len(records) != 3:
            raise ValueError(f"Incomplete results for {suite}; finish training before generating the report.")
        for row in records:
            history = read_csv(results / suite / f"{row['model'].lower()}_history.csv")
            if [record["epoch"] for record in history] != list(range(1, 6)):
                raise ValueError(f"Expected five completed epochs for {suite}/{row['model']}.")
    from validation import verify_evidence
    verify_evidence(results, groups, config)
    make_plots(results, groups)
    make_report(results, groups, config)
