"""Print the Python and pinned-library versions required for Lab 01."""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys
from typing import TextIO

from packaging.version import InvalidVersion, Version


REQUIRED_VERSIONS: tuple[tuple[str, str, str], ...] = (
    ("numpy", "numpy", "1.26.4"),
    ("pandas", "pandas", "2.2.2"),
    ("scikit-learn", "scikit-learn", "1.5.0"),
    ("scipy", "scipy", "1.13.1"),
    ("matplotlib", "matplotlib", "3.9.0"),
    ("seaborn", "seaborn", "0.13.2"),
    ("torch", "torch", "2.3.1"),
    ("torchvision", "torchvision", "0.18.1"),
    ("torchinfo", "torchinfo", "1.8.0"),
    ("thop", "thop", "0.1.1.post2209072238"),
    ("onnx", "onnx", "1.16.1"),
    ("onnxruntime", "onnxruntime", "1.18.0"),
    ("mlflow", "mlflow", "2.14.1"),
    ("memory-profiler", "memory-profiler", "0.61.0"),
    ("psutil", "psutil", "5.9.8"),
    ("codecarbon", "codecarbon", "2.4.2"),
    ("fastapi", "fastapi", "0.111.0"),
    ("uvicorn", "uvicorn", "0.30.1"),
    ("pytest", "pytest", "8.2.2"),
    ("httpx", "httpx", "0.27.0"),
    ("locust", "locust", "2.28.0"),
    ("requests", "requests", "2.32.3"),
    ("pyarrow", "pyarrow", "16.1.0"),
    ("joblib", "joblib", "1.4.2"),
    ("tqdm", "tqdm", "4.66.4"),
)


def installed_version(distribution_name: str) -> str:
    """Return an installed distribution version without importing the package."""
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def versions_match(actual_version: str, expected_version: str) -> bool:
    """Compare package versions using PEP 440 normalization when possible."""
    try:
        return Version(actual_version) == Version(expected_version)
    except InvalidVersion:
        return actual_version == expected_version


def print_versions(stream: TextIO = sys.stdout) -> bool:
    """Write all required versions and return whether they match the pins."""
    python_version = platform.python_version()
    python_matches = python_version == "3.11.8"

    print(f"Python: {python_version}", file=stream)
    print(f"Python executable: {sys.executable}", file=stream)
    print(f"Platform: {platform.platform()}", file=stream)
    print("", file=stream)

    packages_match = True
    for display_name, distribution_name, expected_version in REQUIRED_VERSIONS:
        actual_version = installed_version(distribution_name)
        matches = versions_match(actual_version, expected_version)
        packages_match = packages_match and matches
        marker = "OK" if matches else f"EXPECTED {expected_version}"
        print(f"{display_name}: {actual_version} [{marker}]", file=stream)

    all_match = python_matches and packages_match
    status = "OK" if all_match else "FAILED"
    print("", file=stream)
    print(f"Environment validation: {status}", file=stream)
    if not python_matches:
        print("Expected Python: 3.11.8", file=stream)
    return all_match


def main() -> int:
    return 0 if print_versions() else 1


if __name__ == "__main__":
    raise SystemExit(main())
