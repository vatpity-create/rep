"""Run Lab 03 and regenerate its numerical and graphical evidence."""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


def main() -> int:
    repository_root = Path(__file__).resolve().parent
    runner = repository_root / "lab03" / "src" / "run_lab.py"
    interpreter = Path(sys.executable)
    if find_spec("torch") is None:
        candidates = (
            repository_root / ".venv" / "Scripts" / "python.exe",
            repository_root / ".venv" / "bin" / "python",
        )
        interpreter = next((path for path in candidates if path.is_file()), interpreter)
        if interpreter != Path(sys.executable):
            print(f"Using project environment: {interpreter}", flush=True)
    completed = subprocess.run(
        [str(interpreter), str(runner), *sys.argv[1:]], cwd=repository_root
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
