from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CLIResult:
    """Small test helper for CLI command execution."""

    return_code: int
    stdout: str
    stderr: str


def run_cli(args: list[str], *, cwd: Path | None = None) -> CLIResult:
    """Run Velvet CLI in a separate process and capture output.

    This keeps command tests robust in environments without optional Typer/Rich
    dependencies while still validating real command paths.
    """

    repo_root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(repo_root / "CLI.py"), *args]
    process = subprocess.run(
        command,
        cwd=str(cwd or repo_root),
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
    )

    return CLIResult(process.returncode, process.stdout, process.stderr)
