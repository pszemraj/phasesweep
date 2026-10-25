"""Cross-process locking for experiments."""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from phasesweep.config.models import Experiment
from phasesweep.engine.errors import (
    ExperimentLockBusyError,
)
from phasesweep.engine.paths import _artifact_root_identity
from phasesweep.runtime.files import (
    canonical_storage_identity,
    try_lock_file,
    unlock_file,
)
from phasesweep.runtime.files import (
    lock_dir as _lock_dir,
)


def _lock_digest(material: dict[str, Any]) -> str:
    """Hash a lock-material dict into a 24-char hex digest.

    :param dict[str, Any] material: Lock identity material, represented as a
        JSON-serializable mapping.
    :return str: First 24 hex characters of the canonical JSON's SHA-256.
    """
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _lock_path_from_material(name: str, material: dict[str, str], label: str) -> Path:
    """Return a host lock path derived from a namespace identity.

    The filename includes ``name`` as well as the digest; both must derive from the
    same canonical namespace so alternate spellings contend on one lock.

    :param str name: Human-readable prefix derived from the lock identity.
    :param dict[str, str] material: Identity material hashed into the filename.
    :param str label: Short lock role such as ``"output"`` or ``"storage"``.
    :return Path: Absolute lock path; the lock file is created lazily.
    """
    return _lock_dir() / f"{name}__{label}__{_lock_digest(material)}.lock"


def _storage_run_lock_material(experiment: Experiment) -> dict[str, str] | None:
    """Identify the persistent storage study namespace, or return ``None`` for memory.

    :param Experiment experiment: Experiment supplying storage and study namespace.
    :return dict[str, str] | None: Canonical persistent-storage lock material,
        or ``None`` for in-memory storage.
    """
    storage_identity = canonical_storage_identity(experiment.resolved_storage)
    if storage_identity is None:
        return None
    return {
        "kind": "persistent_storage",
        "storage": storage_identity,
        "experiment": experiment.experiment,
    }


def _run_lock_paths(experiment: Experiment) -> list[Path]:
    """Return sorted same-host output and persistent-storage lock paths.

    In-memory runs need only the output lock. Acquisition is nonblocking.

    :param Experiment experiment: Experiment whose lock set is derived.
    :return list[Path]: Lock paths in deterministic string order: one output
        lock for in-memory storage, otherwise output and storage locks.
    """
    output_identity = _artifact_root_identity(experiment)
    paths = [
        _lock_path_from_material(
            Path(output_identity).name,
            {"kind": "output", "experiment_dir": output_identity},
            "output",
        )
    ]
    storage_material = _storage_run_lock_material(experiment)
    if storage_material is not None:
        paths.append(_lock_path_from_material(experiment.experiment, storage_material, "storage"))
    return sorted(paths, key=str)


@contextlib.contextmanager
def _experiment_lock(experiment: Experiment) -> Iterator[None]:
    """Hold same-host advisory locks for the whole phase-chained experiment.

    Always lock the output namespace; also lock persistent storage. Either namespace
    may be shared by configs that differ on the other, so both constrain execution.

    :param Experiment experiment: Experiment whose complete lock set is acquired.
    :raises ExperimentLockBusyError: Another process holds the output or storage lock.
    :return Iterator[None]: Context manager iterator for the held lock set.
    """
    paths = _run_lock_paths(experiment)
    handles: list[Any] = []
    try:
        for path in paths:
            handle = try_lock_file(path)
            if handle is None:
                raise ExperimentLockBusyError(
                    f"Another phasesweep process appears to be using the same "
                    f"experiment backend or output namespace for "
                    f"{experiment.experiment!r} (lock file: {path}). phasesweep "
                    f"supports one active orchestrator per experiment output "
                    f"namespace and per persistent storage identity.",
                    remediation=(
                        "Wait for the active sweep using this experiment output or storage "
                        "namespace to finish, or cancel it before retrying the same command."
                    ),
                )
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            unlock_file(handle)
