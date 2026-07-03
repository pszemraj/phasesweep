#!/usr/bin/env python3
"""Run the packaged toy trainer from a source checkout."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path


def _load_main() -> Callable[[], None]:
    """Load the package implementation, preferring local ``src`` in a checkout."""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))

    from phasesweep.examples.fake_train import main

    return main


if __name__ == "__main__":
    _load_main()()
