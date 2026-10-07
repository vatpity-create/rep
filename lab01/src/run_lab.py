"""Run the complete Lab 01 reproducibility workflow."""

from __future__ import annotations

from pathlib import Path

from print_versions import print_versions
from run_benchmarks import main as run_benchmarks


def main() -> int:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    versions_path = results_dir / "versions.txt"

    with versions_path.open("w", encoding="utf-8") as output:
        environment_is_valid = print_versions(output)
    if not environment_is_valid:
        print(f"Environment validation failed; see {versions_path}")
        return 1

    print(f"Saved environment versions to {versions_path}")
    run_benchmarks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
