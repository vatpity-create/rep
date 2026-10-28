# Lab 3: Architectures and their costs

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
Python, NumPy, PyTorch, splitting and shuffling use seed **42**.
Deterministic algorithms are enabled.

Measurements use **AMD Ryzen 5 5500U with Radeon Graphics**, `cpu`, FP32,
2 intra-op threads and 1 inter-op thread.
Python 3.11.8, PyTorch 2.3.1,
torchvision 0.18.1, torchinfo 1.8.0,
THOP 0.1.1-2209072238, NumPy 1.26.4 and
Matplotlib 3.9.0. Full settings are in `results/configuration.json`.

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


## Reference results

| Model | Parameters | Core FLOPs | FP32 weights, KiB | Test accuracy | B=1, ms | B=128, ms/batch | B=128, ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 235,146 | 469,504 | 918.54 | 87.76% | 0.08030 | 0.69075 | 0.00540 |
| CNN | 20,490 | 2,063,488 | 80.04 | 89.18% | 0.37565 | 19.59360 | 0.15308 |
| Attention | 70,794 | 2,329,856 | 276.54 | 83.88% | 0.78665 | 12.06595 | 0.09427 |

| Model | Hand core FLOPs | Raw THOP ops | 2×raw, diagnostic | THOP core FLOPs | Core gap | Raw THOP params / torchinfo |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 469,504 | 234,752 | 469,504 | 469,504 | 0.00% | 235,146 / 235,146 |
| CNN | 2,063,488 | 1,031,744 | 2,063,488 | 2,063,488 | 0.00% | 20,490 / 20,490 |
| Attention | 2,329,856 | 591,488 | 1,182,976 | 2,329,856 | 0.00% | 37,514 / 70,794 |

Independent profiles of layers restored from the final checkpoints:

| Layer | Hand FLOPs | THOP MACs | THOP FLOPs |
|---|---:|---:|---:|
| Linear 784 to 256 | 401,408 | 200,704 | 401,408 |
| Conv2d 1 to 16 | 225,792 | 112,896 | 225,792 |
| MultiheadAttention L=16 d=64 h=4 | 589,824 | 294,912 | 589,824 |

For reference Attention, raw THOP 591,488 = core MACs 1,164,928 − omitted MHA MACs 589,824 + LayerNorm operations 16,384. The raw parameter count also omits MHA weights.

## Matched results

| Model | Parameters | Core FLOPs | FP32 weights, KiB | Test accuracy | B=1, ms | B=128, ms/batch | B=128, ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 118,282 | 236,032 | 462.04 | 87.48% | 0.09180 | 0.35890 | 0.00280 |
| CNN | 118,634 | 22,478,848 | 463.41 | 91.98% | 0.87260 | 78.91010 | 0.61649 |
| Attention | 119,794 | 3,918,096 | 467.95 | 84.63% | 0.61585 | 16.50580 | 0.12895 |

| Model | Hand core FLOPs | Raw THOP ops | 2×raw, diagnostic | THOP core FLOPs | Core gap | Raw THOP params / torchinfo |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 236,032 | 118,016 | 236,032 | 236,032 | 0.00% | 118,282 / 118,282 |
| CNN | 22,478,848 | 11,239,424 | 22,478,848 | 22,478,848 | 0.00% | 118,634 / 118,634 |
| Attention | 3,918,096 | 991,368 | 1,982,736 | 3,918,096 | 0.00% | 62,674 / 119,794 |

Independent profiles of layers restored from the final checkpoints:

| Layer | Hand FLOPs | THOP MACs | THOP FLOPs |
|---|---:|---:|---:|
| Linear 784 to 128 | 200,704 | 100,352 | 200,704 |
| Conv2d 1 to 48 | 677,376 | 338,688 | 677,376 |
| MultiheadAttention L=16 d=84 h=4 | 989,184 | 494,592 | 989,184 |

For matched Attention, raw THOP 991,368 = core MACs 1,959,048 − omitted MHA MACs 989,184 + LayerNorm operations 21,504. The raw parameter count also omits MHA weights.

## Inference timing

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


## Conclusions

**Reference:** CNN has the highest accuracy (89.18%); MLP has the lowest B=1 latency (0.08030 ms). The smallest model is CNN (20,490 parameters). CNN improves on MLP by 1.42 percentage points, using 4.40× the core FLOPs and 4.68× the B=1 latency.

The Pareto choices for accuracy, parameter count and B=1 latency are **MLP, CNN**. Choose CNN when accuracy is the priority and MLP when single-image response time is the priority.

| Model | B=128 speedup per image over B=1 | B=128 images/s |
|---|---:|---:|
| MLP | 14.88× | 185306 |
| CNN | 2.45× | 6533 |
| Attention | 8.35× | 10608 |

**Matched:** CNN has the highest accuracy (91.98%); MLP has the lowest B=1 latency (0.09180 ms). The smallest model is MLP (118,282 parameters). CNN improves on MLP by 4.50 percentage points, using 95.24× the core FLOPs and 9.51× the B=1 latency.

The Pareto choices for accuracy, parameter count and B=1 latency are **MLP, CNN**. Choose CNN when accuracy is the priority and MLP when single-image response time is the priority.

| Model | B=128 speedup per image over B=1 | B=128 images/s |
|---|---:|---:|
| MLP | 32.74× | 356645 |
| CNN | 1.42× | 1622 |
| Attention | 4.78× | 7755 |

CNN's spatial locality and weight sharing suit these small images. MLP is a
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
