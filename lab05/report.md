# Lab 5: Frameworks and automatic differentiation

## Run and settings

From the repository root:

```powershell
python main.py
```

Python 3.11.8, PyTorch 2.3.1+cpu,
NumPy 1.26.4 and scikit-learn 1.5.0.
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
| Scalar vs PyTorch autograd | 0 |
| Scalar vs numerical gradients | 1.17441500969e-09 |

All 48 derivatives satisfy the required difference
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
| scalar | 14.337918 | 0.767776 | 0.331618 | 86.72% |
| pytorch | 0.111375 | 0.767776 | 0.331618 | 86.72% |

PyTorch is 128.73× faster in this measurement.
The largest final parameter difference between the two implementations is
1.11022302463e-16. Vectorised tensor
kernels avoid creating and traversing thousands of Python scalar nodes.
The scalar engine also tracks gradients for scalar constants; PyTorch requests
gradients for model parameters. Raw timings are in `results/training_runs.csv`.

## Graph and execution modes

The model's FX graph is saved in [model_graph.txt](results/model_graph.txt).
[graph_nodes.csv](results/graph_nodes.csv) records each node's operation, shape,
dtype and stride. The model input is a contiguous float64 tensor of shape
(128,2), stride (2,1); output shape is (128,1).

The optimized mode is **compiled**, backend **inductor**.

Outputs agree with eager execution within 1e-10; the measured maximum difference
is 0. Both original and reversed inputs
are checked. Inference uses `eval()` and `torch.inference_mode()`.

Each variant receives 30 warmups, then 100 timed blocks of 100 forwards.
The table gives the median time per forward and its interquartile range.
Variant order alternates between blocks. Compilation/tracing and the first
forward are outside steady-state timing; together they took
11.560 ms in this run. Existing compiler
caches can affect that setup measurement.

| Mode | Median ms/forward | 25th percentile | 75th percentile |
|---|---:|---:|---:|
| eager | 0.048646 | 0.042732 | 0.059397 |
| compiled | 0.060582 | 0.053810 | 0.074779 |

The eager/optimized latency ratio is 0.803×.
There is no measured per-call saving to amortise setup for this input.

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
| contiguous | 3.559225 | 3.385455 | 3.808555 |
| non_contiguous | 4.718130 | 4.455418 | 5.021680 |

The non-contiguous/contiguous latency ratio is **1.326×**.
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
