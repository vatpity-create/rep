"""Convenience entry point for the first laboratory exercise.

The laboratory implementation deliberately lives in lab01/src, as required by
the course repository layout.  Running this file regenerates all Lab 01
evidence from the pinned environment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repository_root = Path(__file__).resolve().parent
    runner = repository_root / "lab01" / "src" / "run_lab.py"
    completed = subprocess.run([sys.executable, str(runner)], cwd=repository_root)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
