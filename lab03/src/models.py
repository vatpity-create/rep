"""The manual's exact architectures and a separate size-matched experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    suite: str
    name: str
    widths: tuple[int, int]

    @property
    def key(self) -> str:
        return f"{self.suite}_{self.name.lower()}"

    def configuration(self) -> dict:
        return asdict(self)


SPECS = (
    ModelSpec("reference", "MLP", (256, 128)),
    ModelSpec("reference", "CNN", (16, 32)),
    ModelSpec("reference", "Attention", (64, 128)),
    ModelSpec("matched", "MLP", (128, 128)),
    ModelSpec("matched", "CNN", (48, 128)),
    ModelSpec("matched", "Attention", (84, 168)),
)


class MLP(nn.Module):
    def __init__(self, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(784, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Linear(hidden2, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class CNN(nn.Module):
    def __init__(self, channels1: int = 16, channels2: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, channels1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels1, channels2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(channels2 * 7 * 7, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class PatchAttention(nn.Module):
    """16 non-overlapping 7x7 patches, two encoders, mean pool, classifier.

    No extra class token or position embedding: neither is in the lab recipe.
    Consequently mean pooling makes the classifier invariant to patch order.
    Encoder layers are constructed independently, rather than identically cloned.
    """

    def __init__(self, d_model: int = 64, feedforward: int = 128):
        super().__init__()
        self.projection = nn.Linear(49, d_model)
        self.encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=feedforward,
                dropout=0.1, activation="relu", batch_first=True,
                norm_first=False,
            )
            for _ in range(2)
        ])
        self.classifier = nn.Linear(d_model, 10)

    @staticmethod
    def patchify(images: torch.Tensor) -> torch.Tensor:
        # [B,1,4,7,4,7] -> [B,4,4,1,7,7] -> [B,16,49], row-major.
        batch = images.shape[0]
        return images.reshape(batch, 1, 4, 7, 4, 7).permute(
            0, 2, 4, 1, 3, 5
        ).reshape(batch, 16, 49)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(self.patchify(images))
        for encoder in self.encoders:
            tokens = encoder(tokens)
        return self.classifier(tokens.mean(dim=1))


def build_model(spec: ModelSpec) -> nn.Module:
    constructors = {"MLP": MLP, "CNN": CNN, "Attention": PatchAttention}
    return constructors[spec.name](*spec.widths)
