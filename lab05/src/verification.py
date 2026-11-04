from __future__ import annotations

import csv
import json

import numpy as np
import torch

from autodiff import Value


def verify_gradients(results):
    inputs = np.random.default_rng(42).uniform(-1.2, 1.2, (3, 16))
    variables = [[Value(value) for value in row] for row in inputs]
    output = sum(((a * b + c) ** 2).relu() for a, b, c in zip(*variables))
    order = output.topological_order()
    positions = {node: index for index, node in enumerate(order)}
    assert len(order) == len(set(order))
    assert all(positions[parent] < positions[node] for node in order for parent in node.parents)
    output.backward()

    tensors = [torch.tensor(row, dtype=torch.float64, requires_grad=True) for row in inputs]
    reference = ((tensors[0] * tensors[1] + tensors[2]) ** 2).relu().sum()
    reference.backward()
    assert abs(output.data - reference.item()) < 1e-12

    def function(values):
        return float(np.maximum((values[0] * values[1] + values[2]) ** 2, 0).sum())

    rows, epsilon = [], 1e-6
    for group, name in enumerate(("a", "b", "c")):
        for index, node in enumerate(variables[group]):
            plus, minus = inputs.copy(), inputs.copy()
            plus[group, index] += epsilon
            minus[group, index] -= epsilon
            numerical = (function(plus) - function(minus)) / (2 * epsilon)
            expected = tensors[group].grad[index].item()
            rows.append(dict(variable=name, index=index, scalar_gradient=node.grad,
                             torch_gradient=expected, numerical_gradient=numerical,
                             torch_absolute_error=abs(node.grad - expected),
                             numerical_absolute_error=abs(node.grad - numerical)))
    max_torch = max(row["torch_absolute_error"] for row in rows)
    max_numerical = max(row["numerical_absolute_error"] for row in rows)
    assert max_torch < 1e-6 and max_numerical < 1e-6

    shared_input = Value(3)
    shared = shared_input * shared_input
    combined = shared + shared
    combined.backward()
    assert shared_input.grad == 12
    combined.backward()
    assert shared_input.grad == 24
    for value, expected in ((-2, 0), (0, 0), (2, 1)):
        node = Value(value)
        node.relu().backward()
        assert node.grad == expected
    for value, power, expected in ((0, 0, 0), (2, -1, -0.25), (4, 0.5, 0.25)):
        node = Value(value)
        (node ** power).backward()
        assert node.grad == expected
    logits = [Value(value) for value in (-2, 0.5, 2)]
    labels = (0, 1, 1)
    losses = []
    for node, label in zip(logits, labels):
        probability = node.sigmoid()
        losses.append(-(label * probability.log() + (1 - label) * (1 - probability).log()))
    sum(losses).backward()
    torch_logits = torch.tensor([-2, 0.5, 2], dtype=torch.float64, requires_grad=True)
    loss = torch.nn.functional.binary_cross_entropy(
        torch_logits.sigmoid(), torch.tensor(labels, dtype=torch.float64), reduction="sum"
    )
    loss.backward()
    np.testing.assert_allclose([node.grad for node in logits], torch_logits.grad.numpy(), rtol=0, atol=1e-12)

    with (results / "gradient_check.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(expression="((a*b+c)**2).relu().sum()", checked_derivatives=len(rows),
                   epsilon=epsilon, tolerance=1e-6, max_torch_absolute_error=max_torch,
                   max_numerical_absolute_error=max_numerical,
                   topological_order=True, shared_subgraph_accumulation=True,
                   repeated_backward_accumulation=True, relu_boundaries=True,
                   constant_powers=True, sigmoid_bce_gradients=True)
    (results / "gradient_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
