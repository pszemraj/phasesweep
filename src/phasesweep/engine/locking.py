"""Cross-process locking for experiments and suites."""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from phasesweep.config import Experiment, Suite
from phasesweep.engine.errors import (
    ExperimentLockBusyError,
)
from phasesweep.engine.paths import (
    _experiment_dir,
    _suite_dir,
)
from phasesweep.runtime.files import (
    canonical_storage_identity,
    exclusive_lock,
    storage_is_in_memory,
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
    """Lock path under the configured host lock directory.

    ``name`` is part of the lock *filename*, so two configs whose material
    hashes identically still miss each other unless their name components
    match too. Callers must therefore derive it from the same identity the
    material keys on — the resolved output namespace for the output lock —
    not from an arbitrary spelling of it (review v0.5.17 gap hunt).

    :param str name: Human-readable prefix derived from the lock identity.
    :param dict[str, str] material: Identity material hashed into the filename.
    :param str label: Short lock role such as ``"output"`` or ``"storage"``.
    :return Path: Absolute lock path; the lock file is created lazily.
    """
    return _lock_dir() / f"{name}__{label}__{_lock_digest(material)}.lock"


def _output_lock_material(experiment: Experiment) -> dict[str, str]:
    """Identity for the *output namespace* lock: which directory we write to.

    Catches the case where two configs share a workdir + experiment name but
    point at different storage backends — without an output lock those would
    silently overwrite each other's ``trial_*/``, ``winner.yaml``, and
    ``summary.yaml`` (review v0.5.6 / blocker 1). Always taken regardless of
    storage backend, including in-memory storage.

    :param Experiment experiment: Experiment supplying the output namespace.
    :return dict[str, str]: Lock material keyed on the resolved experiment directory.
    """
    # Resolve the FULL directory, leaf included: _experiment_dir resolves only
    # the workdir prefix before appending the experiment name, so a symlinked
    # experiment leaf (runs/expA -> runs/expB) would otherwise mint a second
    # lock identity for one physical namespace — and the second orchestrator's
    # preflight would reap the first one's live trials (review v0.5.17 gap
    # hunt). A leaf that does not exist yet resolves to itself.
    return {"kind": "output", "experiment_dir": str(_experiment_dir(experiment).resolve())}


def _storage_run_lock_material(experiment: Experiment) -> dict[str, str] | None:
    """Identity for the *Optuna storage* lock: which study namespace we write to.

    Catches the case where two configs share storage + experiment name but
    point at different workdirs. Returns ``None`` for in-memory storage —
    there is no shared backend, so the output lock alone is sufficient.

    :param Experiment experiment: Experiment supplying storage and study namespace.
    :return dict[str, str] | None: Canonical persistent-storage lock material,
        or ``None`` for in-memory storage.
    """
    if storage_is_in_memory(experiment.resolved_storage):
        # In-memory URLs (``sqlite:///:memory:`` and spellings thereof) have
        # no shared backend to guard; locking their canonical identity would
        # make two unrelated in-memory runs that share an experiment name
        # contend on a lock naming a backend that does not exist (review
        # v0.5.17 gap hunt).
        return None
    storage_identity = canonical_storage_identity(experiment.resolved_storage)
    if storage_identity is None:
        return None
    return {
        "kind": "persistent_storage",
        "storage": storage_identity,
        "experiment": experiment.experiment,
    }


def _run_lock_paths(experiment: Experiment) -> list[Path]:
    """All same-host locks required for one full experiment run.

    For persistent storage we take *both* locks (output + storage); for
    in-memory storage we take only the output lock. The list is sorted by
    path so acquisition order is deterministic across processes — relevant
    for clear error messages, not for deadlock avoidance (we use
    ``LOCK_NB``).

    :param Experiment experiment: Experiment whose lock set is derived.
    :return list[Path]: Lock paths in deterministic string order: one output
        lock for in-memory storage, otherwise output and storage locks.
    """
    # The output lock's filename prefix comes from the RESOLVED namespace
    # leaf, not the configured experiment name: a symlinked experiment leaf
    # spells the same physical directory differently, and a prefix mismatch
    # alone would split the lock even with identical material (review
    # v0.5.17 gap hunt).
    paths = [
        _lock_path_from_material(
            _experiment_dir(experiment).resolve().name,
            _output_lock_material(experiment),
            "output",
        )
    ]
    storage_material = _storage_run_lock_material(experiment)
    if storage_material is not None:
        paths.append(_lock_path_from_material(experiment.experiment, storage_material, "storage"))
    return sorted(paths, key=str)


@contextlib.contextmanager
def _experiment_lock(experiment: Experiment) -> Iterator[None]:
    """Take all same-host locks needed for one full experiment run.

    The phase-chained pipeline has cross-phase state — parent ``winner.yaml``,
    child fingerprints, ``summary.yaml``, ``--from-phase`` semantics — so the
    consistency domain is the entire run, not a single phase study. v0.5.7
    extends this further: the consistency domain spans both the *output
    namespace* (filesystem artifacts under ``<workdir>/<experiment>/``) and
    the *Optuna storage namespace* (review v0.5.6 / blocker 1).

    Two configs can disagree on storage but share output paths, or vice
    versa; either case can corrupt skipped-phase reuse and trial logs. We
    therefore take an output lock *always*, and a storage lock additionally
    whenever storage is persistent. In-memory storage has no shared backend,
    so the output lock alone suffices.

    Both locks are *same-host advisory only*; multi-host coordination would need durable per-trial leases and heartbeats rather than just host-local flock files.

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
                    f"namespace and per persistent storage identity."
                )
            handles.append(handle)
        yield
    finally:
        # Reverse-order release isn't required by flock semantics, but it
        # keeps the "stack" mental model intact and parallels typical
        # acquire-A-then-B / release-B-then-A discipline.
        for handle in reversed(handles):
            unlock_file(handle)


@contextlib.contextmanager
def _suite_lock(suite: Suite) -> Iterator[None]:
    """Take a same-host lock for suite-level log and summary artifacts.

    :param Suite suite: Parsed suite config whose output directory names the lock.
    :return Iterator[None]: Context manager yielding ``None`` while the suite lock is held.
    """
    material = {"kind": "suite", "suite_dir": str(_suite_dir(suite))}
    path = _lock_dir() / f"{suite.suite}__suite__{_lock_digest(material)}.lock"
    with exclusive_lock(
        path,
        busy_message=(
            f"Another phasesweep suite process appears to be using {suite.suite!r} "
            f"(lock file: {path})."
        ),
    ):
        yield
