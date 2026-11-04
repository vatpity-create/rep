from __future__ import annotations

import numpy as np
import torch
from torch import nn

from autodiff import Value


def initial_parameters(seed=42):
    generator = np.random.default_rng(seed)
    return (
        generator.normal(0, 0.5, (8, 2)), np.zeros(8),
        generator.normal(0, 0.5, (1, 8)), np.zeros(1),
    )


class ScalarModel:
    def __init__(self, initial):
        first_weights, first_biases, second_weights, second_bias = initial
        self.first_weights = [[Value(weight) for weight in row] for row in first_weights]
        self.first_biases = [Value(bias) for bias in first_biases]
        self.second_weights = [Value(weight) for weight in second_weights[0]]
        self.second_bias = Value(second_bias[0])

    def parameters(self):
        return ([weight for row in self.first_weights for weight in row]
                + self.first_biases + self.second_weights + [self.second_bias])

    def __call__(self, features):
        inputs = [Value(feature) for feature in features]
        hidden = [sum((weight * feature for weight, feature in zip(weights, inputs)), bias).relu()
                  for weights, bias in zip(self.first_weights, self.first_biases)]
        return sum((weight * value for weight, value in zip(self.second_weights, hidden)),
                   self.second_bias).sigmoid()


def torch_model(initial):
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()).double()
    with torch.no_grad():
        for parameter, values in zip(model.parameters(), initial):
            parameter.copy_(torch.from_numpy(values))
    return model


def scalar_loss(model, features, labels):
    losses = []
    for inputs, label in zip(features, labels):
        probability = model(inputs)
        losses.append(-(label * probability.log() + (1 - label) * (1 - probability).log()))
    return sum(losses) * (1 / len(losses))
