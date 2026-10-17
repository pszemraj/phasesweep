"""Docstring sampling can be repeated across independent CLI invocations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sample_docstrings.py"


@pytest.mark.integration
def test_seed_repeats_the_same_sample_across_processes(tmp_path: Path) -> None:
    source = "\n".join(f'def function_{i}():\n    """Docstring {i}."""\n' for i in range(20))
    (tmp_path / "sample.py").write_text(source, encoding="utf-8")

    outputs = [
        subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(tmp_path), "--count", "4", "--seed", seed],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        for seed in ("42", "42", "43")
    ]

    assert outputs[0] == outputs[1]
    assert outputs[0] != outputs[2]
    assert outputs[0].count("--- ") == 4
