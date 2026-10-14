"""Run Lab 02 and regenerate its numerical and graphical evidence."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repository_root = Path(__file__).resolve().parent
    runner = repository_root / "lab02" / "src" / "run_lab.py"
    completed = subprocess.run([sys.executable, str(runner)], cwd=repository_root)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
