"""Analytic core FLOPs and THOP with explicit, comparable counting rules."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import torch
from thop import profile
from torch import nn
from torchinfo import summary

from models import ModelSpec


@contextmanager
def visible_attention_operations():
    # PyTorch 2.3's fused encoder bypasses module hooks. Disable only for counts;
    # restore the normal inference implementation before training and timing.
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def hand_costs(spec: ModelSpec) -> list[dict]:
    """Batch=1; one multiply plus one addition = two FLOPs.

    Count matrix products/convolutions only. Bias, activation, comparisons,
    normalization, softmax, residual additions, and mean pool are excluded.
    Parameters DO include all trainable biases and LayerNorm coefficients.
    """
    rows = []

    def add(layer: str, formula: str, flops: int, parameters: int):
        rows.append(dict(suite=spec.suite, model=spec.name, layer=layer,
                         formula=formula, hand_flops=flops, hand_parameters=parameters))

    a, b = spec.widths
    if spec.name == "MLP":
        for index, (inputs, outputs) in enumerate(((784, a), (a, b), (b, 10)), 1):
            add(f"linear{index}", f"2*{inputs}*{outputs}",
                2 * inputs * outputs, (inputs + 1) * outputs)
    elif spec.name == "CNN":
        for index, (cin, cout, side) in enumerate(((1, a, 28), (a, b, 14)), 1):
            add(f"conv{index}", f"2*3*3*{cin}*{cout}*{side}*{side}",
                2 * 9 * cin * cout * side * side, (9 * cin + 1) * cout)
        add("classifier", f"2*{b * 49}*10", 2 * b * 49 * 10, (b * 49 + 1) * 10)
    else:
        seq, d, ff = 16, a, b
        add("patch_projection", f"2*16*49*{d}", 2 * seq * 49 * d, 50 * d)
        for index in range(1, 3):
            prefix = f"encoder{index}."
            add(prefix + "qkv", f"2*16*{d}*3*{d}", 2 * seq * d * 3 * d, 3 * d * d + 3 * d)
            add(prefix + "scores", f"2*16*16*{d}", 2 * seq * seq * d, 0)
            add(prefix + "weighted_values", f"2*16*16*{d}", 2 * seq * seq * d, 0)
            add(prefix + "output_projection", f"2*16*{d}*{d}", 2 * seq * d * d, d * d + d)
            add(prefix + "ff1", f"2*16*{d}*{ff}", 2 * seq * d * ff, (d + 1) * ff)
            add(prefix + "ff2", f"2*16*{ff}*{d}", 2 * seq * ff * d, (ff + 1) * d)
            add(prefix + "two_layernorms", "excluded from core FLOPs", 0, 4 * d)
        add("classifier", f"2*{d}*10", 2 * d * 10, (d + 1) * 10)
    return rows


def count_mha_macs(module: nn.MultiheadAttention, inputs, output):
    """Hook for dense self-attention: QKV, QK^T, AV, and output projection.

    Sequence dimensions come from the runtime input. THOP treats this module
    as a complete subtree, so out_proj is not counted a second time.
    """
    query = inputs[0]
    batch, seq, d = query.shape
    macs = batch * (3 * seq * d * d + 2 * seq * seq * d + seq * d * d)
    module.total_ops += macs


def zero_ops(module, inputs, output):
    """Exclude normalization for the same core-only convention as the manual."""
    module.total_ops += 0


def profile_model(model: nn.Module, spec: ModelSpec, sample: torch.Tensor):
    hand = hand_costs(spec)
    hand_flops = sum(row["hand_flops"] for row in hand)
    hand_parameters = sum(row["hand_parameters"] for row in hand)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    # Copies prevent THOP 0.1.x's temporary buffers on unsupported modules from
    # contaminating checkpoints, optimizer state, or subsequent latency runs.
    with visible_attention_operations(), torch.no_grad():
        info = summary(deepcopy(model), input_data=sample, mode="eval", depth=5,
                       col_names=("input_size", "output_size", "num_params"), verbose=0)
        raw_ops, raw_params = profile(deepcopy(model), inputs=(sample,), verbose=False)
        core_macs, core_params = profile(
            deepcopy(model), inputs=(sample,), verbose=False,
            custom_ops={nn.MultiheadAttention: count_mha_macs, nn.LayerNorm: zero_ops},
        )
    if hand_parameters != actual_parameters or info.total_params != actual_parameters:
        raise AssertionError(f"Parameter count mismatch for {spec.key}")
    if int(core_params) != actual_parameters or int(2 * core_macs) != hand_flops:
        raise AssertionError(f"Analytic/THOP core count mismatch for {spec.key}")
    return dict(
        parameters=actual_parameters, torchinfo_parameters=info.total_params,
        hand_parameters=hand_parameters, hand_flops=hand_flops,
        thop_raw_ops=int(raw_ops), thop_raw_parameters=int(raw_params),
        thop_core_macs=int(core_macs), thop_core_flops=int(2 * core_macs),
        thop_core_parameters=int(core_params),
        core_gap_percent=100 * (2 * core_macs - hand_flops) / hand_flops,
        weights_kib=actual_parameters * sample.element_size() / 1024,
    ), hand, str(info)
