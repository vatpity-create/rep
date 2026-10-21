"""Official Fashion-MNIST, the mandated transform, and a shared seeded split."""
from __future__ import annotations

from pathlib import Path
from urllib.error import URLError

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.datasets.utils import download_and_extract_archive


def load_datasets(root: Path, seed: int):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,)),
    ])
    try:
        full_train = FashionMNIST(root=str(root), train=True, download=True, transform=transform)
    except (RuntimeError, URLError) as error:
        # Some networks block the historical HTTP host. Keep using the authors'
        # original files and torchvision's original MD5s on their HTTPS mirror.
        print(f"Original download unavailable ({error}); using the authors' HTTPS mirror.", flush=True)
        for filename, checksum in FashionMNIST.resources:
            download_and_extract_archive(
                "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
                f"master/data/fashion/{filename}",
                download_root=str(root / "FashionMNIST" / "raw"),
                filename=filename, md5=checksum,
            )
        full_train = FashionMNIST(root=str(root), train=True, download=False, transform=transform)
    test = FashionMNIST(root=str(root), train=False, download=False, transform=transform)
    if len(full_train) != 60_000 or len(test) != 10_000:
        raise ValueError("Expected the official 60,000/10,000 Fashion-MNIST partitions.")
    order = torch.randperm(60_000, generator=torch.Generator().manual_seed(seed))
    train_indices, validation_indices = order[:55_000], order[55_000:]
    return (
        Subset(full_train, train_indices.tolist()),
        Subset(full_train, validation_indices.tolist()),
        test, train_indices, validation_indices,
    )


def make_loaders(train, validation, test, seed: int, device: torch.device):
    # A fresh, identically seeded generator per model gives the same epoch orders.
    options = dict(batch_size=128, num_workers=0, pin_memory=device.type == "cuda")
    return (
        DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(seed), **options),
        DataLoader(validation, shuffle=False, **options),
        DataLoader(test, shuffle=False, **options),
    )
