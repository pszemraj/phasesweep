"""Generation configuration snapshots and reproducibility provenance."""

from __future__ import annotations

import json

import yaml

import phasesweep.engine.fingerprints as fingerprint_ops
import phasesweep.engine.paths as path_ops
from phasesweep._metadata import __version__
from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.state import (
    GENERATION_CONFIG_SNAPSHOT_FILENAME,
    GENERATION_SUMMARY_SCHEMA_VERSION,
    REPRODUCIBILITY_SCHEMA_VERSION,
    STUDY_SCHEMA_VERSION,
)
from phasesweep.runtime.files import atomic_text_writer, file_sha256, private_atomic_write_text


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
    and ``generation_id_source`` (``caller`` or ``engine``). It exposes no environment
    or snapshot values. The source field preserves external launch provenance if
    launcher state is lost; it does not grant authority to a reader.

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
    snapshot["execution"]["cwd"] = fingerprint_ops._execution_identity(experiment)["cwd"]
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
        "config_snapshot": {
            "path": GENERATION_CONFIG_SNAPSHOT_FILENAME,
            "sha256": file_sha256(snapshot_path),
        },
    }
    with atomic_text_writer(reproducibility_path) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
