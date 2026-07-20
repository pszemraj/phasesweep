"""Loading for immutable per-run experiment snapshots."""

from __future__ import annotations

import hashlib
from pathlib import Path

from phasesweep.config import Experiment
from phasesweep.config.io import load_config_bytes


def load_experiment_snapshot(path: Path, expected_sha256: str, *, source: str) -> Experiment:
    """Read, verify, and parse one immutable experiment snapshot.

    :param Path path: Snapshot file to read.
    :param str expected_sha256: Digest recorded with the run handle.
    :param str source: Human-readable parser source label.
    :raises OSError: If the snapshot cannot be read.
    :raises ValueError: If its digest, syntax, or config kind is invalid.
    :return Experiment: Verified experiment configuration.
    """
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("run snapshot hash mismatch")
    config = load_config_bytes(data, source=source)
    if not isinstance(config, Experiment):
        raise ValueError("config snapshot is not a single experiment")
    return config
