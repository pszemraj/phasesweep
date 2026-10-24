"""Generation configuration snapshots and reproducibility provenance."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import yaml

import phasesweep.engine.fingerprints as fingerprint_ops
import phasesweep.engine.paths as path_ops
from phasesweep._metadata import __version__
from phasesweep.config.io import load_experiment
from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.errors import WinnerIntegrityError
from phasesweep.engine.state import (
    GENERATION_CONFIG_SNAPSHOT_FILENAME,
    GENERATION_SUMMARY_SCHEMA_VERSION,
    REPRODUCIBILITY_SCHEMA_VERSION,
    STUDY_SCHEMA_VERSION,
)
from phasesweep.runtime.files import atomic_text_writer, file_sha256, private_atomic_write_text
from phasesweep.runtime.json import json_object, json_text

_GIT_COMMAND_TIMEOUT_SECONDS = 10
# SHA-1 repositories name commits with 40 hex digits and SHA-256 repositories with 64.
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
# Repository-local variables (``git rev-parse --local-env-vars``). A git hook exports
# them for its own repository, so an inherited one would make the read describe that
# repository instead of the trainer's cwd.
_GIT_REPOSITORY_ENV = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CONFIG",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
        "GIT_OBJECT_DIRECTORY",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_GRAFT_FILE",
        "GIT_INDEX_FILE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_REPLACE_REF_BASE",
        "GIT_PREFIX",
        "GIT_SHALLOW_FILE",
        "GIT_COMMON_DIR",
    }
)


class CodeRevision(TypedDict):
    """Git revision of the trainer's working directory, as recorded per generation."""

    commit: str
    dirty: bool


def _git_stdout(cwd: Path, *args: str) -> str | None:
    """Run one read-only git command and return its standard output.

    Inherited repository-local variables are dropped so ``cwd`` alone selects the
    repository. ``GIT_OPTIONAL_LOCKS=0`` keeps ``git status`` from refreshing the index,
    so a read never takes the index lock away from the operator's own git commands. Standard
    input is closed because the caller may be a stdio server whose input is a protocol
    channel.

    :param Path cwd: Directory to run git in.
    :param str args: Git subcommand and its arguments.
    :return str | None: Standard output on exit status 0; ``None`` when git is not
        installed, cannot start, times out, or exits nonzero (including ``cwd`` being
        missing or outside a work tree).
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env={
                **{k: v for k, v in os.environ.items() if k not in _GIT_REPOSITORY_ENV},
                "GIT_OPTIONAL_LOCKS": "0",
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_GIT_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _code_revision(cwd: Path) -> CodeRevision | None:
    """Read the git commit checked out in ``cwd`` and whether tracked files differ from it.

    The record is provenance only: it enters no fingerprint, environment cohort, or
    resume check.

    :param Path cwd: Effective trainer working directory.
    :return CodeRevision | None: ``commit`` is HEAD's full hex id and ``dirty`` is true
        when tracked files differ from it (untracked files are ignored); ``None`` when
        ``cwd`` is not inside a git work tree with a commit, or git failed.
    """
    head = _git_stdout(cwd, "rev-parse", "HEAD")
    if head is None:
        return None
    commit = head.strip()
    if _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        return None
    changes = _git_stdout(cwd, "status", "--porcelain", "--untracked-files=no")
    if changes is None:
        return None
    return {"commit": commit, "dirty": bool(changes)}


def _parse_code_revision(record: Mapping[str, object]) -> CodeRevision | None:
    """Validate the ``code_revision`` member of a published reproducibility record.

    The member is mandatory: ``null`` is a recorded "unreadable", never a stand-in for
    absence.

    :param Mapping[str, object] record: The decoded ``reproducibility.json``.
    :raises ValueError: The member is absent, or is neither ``null`` nor an object holding
        exactly a full lowercase hex ``commit`` and a boolean ``dirty``.
    :return CodeRevision | None: The recorded revision, or ``None`` when none was readable.
    """
    if "code_revision" not in record:
        raise ValueError("the member is missing")
    raw = record["code_revision"]
    if raw is None:
        return None
    revision = json_object(raw)
    if set(revision) != {"commit", "dirty"}:
        raise ValueError("it must be null or hold exactly commit and dirty")
    commit = json_text(revision, "commit")
    if _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("commit must be a full lowercase hex git commit id")
    dirty = revision["dirty"]
    if not isinstance(dirty, bool):
        raise ValueError("dirty must be a boolean")
    return {"commit": commit, "dirty": dirty}


def _phase_config_fingerprint(phase: Phase) -> str:
    """Hash a phase's configured semantics independently of inherited winners.

    This identifies phase-level config drift before execution. Winner fingerprints
    also bind inherited effective overrides.

    :param Phase phase: Phase whose configured semantics are hashed.
    :return str: SHA-256 hex digest (64 characters) of the canonicalised payload.
    """
    payload = {
        "fingerprint_schema_version": fingerprint_ops.EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
        "name": phase.name,
        **fingerprint_ops._semantic_phase_dump(phase),
    }
    return fingerprint_ops._semantic_payload_digest(payload)


def _load_generation_experiment(experiment: Experiment | Path, generation_id: str) -> Experiment:
    """Load the producing configuration or route an unavailable snapshot as artifact damage.

    :param Experiment | Path experiment: Configuration or an already-resolved artifact root.
    :param str generation_id: Generation whose frozen launch configuration is required.
    :raises WinnerIntegrityError: The snapshot is absent, unreadable, or invalid.
    :return Experiment: Validated producing configuration.
    """
    path = path_ops._generation_config_snapshot_path(experiment, generation_id)
    try:
        return load_experiment(path)
    except (OSError, ValueError) as exc:
        raise WinnerIntegrityError(
            f"Cannot read configuration snapshot for generation {generation_id!r} at {path}: {exc}"
        ) from exc


def _write_generation_provenance(
    experiment: Experiment,
    generation_id: str,
    *,
    caller_owned_id: bool,
) -> None:
    """Freeze configuration and provenance immediately after generation claim.

    ``config.snapshot.yaml`` materializes defaults, normalizes key order, and freezes
    an omitted execution cwd as the resolved invocation directory. Its configured
    environment may contain secrets, so it is owner-only (0600).

    ``reproducibility.json`` uses ordinary umask permissions and contains versions,
    semantic fingerprints, public configured provenance, the snapshot's SHA-256,
    ``generation_id_source`` (``caller`` or ``engine``), and ``code_revision`` from
    :func:`_code_revision`. It exposes no environment or snapshot values. The source
    field preserves external launch provenance if launcher state is lost; it does
    not grant authority to a reader.

    Both files are required and hash-covered by the generation manifest.

    :param Experiment experiment: Experiment whose configuration is frozen.
    :param str generation_id: Freshly claimed generation namespace to write into.
    :param bool caller_owned_id: Whether ``generation_id`` was supplied by the
        caller rather than minted by the engine.
    :raises OSError: Either file could not be written; the claim must fail
        rather than run a generation whose configuration is unrecorded.
    :raises phasesweep.runtime.files.UnsafePrivatePathError: Something already
        occupies the snapshot path and is not a private, unshared regular file.
    :raises yaml.YAMLError: The canonicalized config could not be serialized.
    """
    snapshot_path = path_ops._generation_config_snapshot_path(experiment, generation_id)
    snapshot = experiment.model_dump(mode="json")
    trainer_cwd = fingerprint_ops._execution_identity(experiment)["cwd"]
    snapshot["execution"]["cwd"] = trainer_cwd
    private_atomic_write_text(
        snapshot_path,
        yaml.safe_dump(snapshot, sort_keys=False),
        require_private_dir=False,
    )
    reproducibility_path = path_ops._generation_reproducibility_path(experiment, generation_id)
    payload = {
        "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
        "experiment": experiment.experiment,
        "generation_id": generation_id,
        "generation_id_source": "caller" if caller_owned_id else "engine",
        "phasesweep_version": __version__,
        "schema_versions": {
            "reproducibility": REPRODUCIBILITY_SCHEMA_VERSION,
            "generation_summary": GENERATION_SUMMARY_SCHEMA_VERSION,
            "study_storage": STUDY_SCHEMA_VERSION,
            "experiment_fingerprint": fingerprint_ops.EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
            "phase_fingerprint": fingerprint_ops.FINGERPRINT_SCHEMA_VERSION,
        },
        "config_fingerprint": fingerprint_ops._experiment_semantic_fingerprint(experiment),
        "phase_config_fingerprints": [
            {"name": phase.name, "sha256": _phase_config_fingerprint(phase)}
            for phase in experiment.phases
        ],
        "provenance": dict(sorted(experiment.provenance.items())),
        "code_revision": _code_revision(Path(trainer_cwd)),
        "config_snapshot": {
            "path": GENERATION_CONFIG_SNAPSHOT_FILENAME,
            "sha256": file_sha256(snapshot_path),
        },
    }
    with atomic_text_writer(reproducibility_path) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
