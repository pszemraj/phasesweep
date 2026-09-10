"""Generation configuration snapshots and reproducibility provenance."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

import yaml

import phasesweep.engine.artifacts as artifact_io
import phasesweep.engine.fingerprints as fingerprint_ops
import phasesweep.engine.paths as path_ops
from phasesweep._metadata import __version__
from phasesweep.config import Experiment, Phase
from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.engine.state import (
    GENERATION_CONFIG_SNAPSHOT_FILENAME,
    GENERATION_SUMMARY_SCHEMA_VERSION,
    REPRODUCIBILITY_SCHEMA_VERSION,
    STUDY_SCHEMA_VERSION,
    GenerationIdSource,
)
from phasesweep.runtime.files import file_sha256, private_atomic_write_text


def _phase_config_fingerprint(phase: Phase) -> str:
    """Hash one phase's configured semantics, independent of any winner.

    This is exactly the per-phase element
    :func:`phasesweep.engine.fingerprints._experiment_semantic_fingerprint` folds
    into its own digest, hashed on its own so a reproducibility record can
    localize *which* phase's configuration differs between two generations.
    It is deliberately not ``winner.yaml``'s ``phase_fingerprint``, which
    additionally binds each inherited winner's effective overrides and
    therefore cannot exist before any phase has run.

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
    """Freeze the configuration and provenance that produced one generation.

    Written at claim time, immediately after the namespace is created and
    before any lifecycle state exists, so a generation that later fails still
    records what configuration ran (review v0.5.18 / finding F6). Two files
    land side by side:

    * ``config.snapshot.yaml`` -- the canonicalized effective config rendered
      as YAML. It is the engine's execution view, *not* a copy of the
      operator's file: key order is normalized, defaults are materialized,
      comments are not preserved, and an omitted ``execution.cwd`` is frozen
      as the resolved invocation directory the trainer inherited. Because
      ``env:`` may hold secrets it is written owner-only (0600) through the
      private atomic writer, even though its directory is deliberately
      operator-readable (see the trust-boundary note in ``docs/runtime.md``).
    * ``reproducibility.json`` -- an ordinary umask-governed artifact that is
      safe to read and share: versions, the semantic fingerprints, the
      operator-declared ``provenance`` mapping (public by design), and the
      SHA-256 of the snapshot's bytes. Digests, never values: no env values,
      no ambient values, and nothing from the snapshot's contents beyond its
      digest.

    Both files are picked up by :func:`phasesweep.engine.publication_validation._generation_artifact_manifest` and are
    therefore hash-covered by the publication manifest.

    ``reproducibility.json`` also records ``generation_id_source``: whether the
    generation's identity was supplied by an external launcher (``"caller"``)
    or minted by the engine (``"engine"``). A launcher that grants an identity
    also freezes that run's authority (e.g. the MCP server's winner-visibility
    grant) in its own state; recording the grant's *existence* here, in the
    artifact tree the results live in, lets readers detect that frozen
    authority is unaccounted for even after the launcher's state directory is
    deleted or replaced (PR #5 review / P2 missing-handle authority).

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
    artifact_io._write_json_atomic(
        path_ops._generation_reproducibility_path(experiment, generation_id),
        {
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
        },
    )


def generation_id_source(experiment: Experiment, generation_id: str) -> GenerationIdSource | None:
    """Return who supplied one generation's identity, or ``None`` when unrecorded.

    Reads the ``generation_id_source`` field frozen into the generation's
    ``reproducibility.json`` at claim time. ``"caller"`` means an external
    launcher granted the identity and holds that run's frozen authority record;
    a reader that cannot load that record must not substitute mutable current
    policy for it (PR #5 review / P2 missing-handle authority). ``None`` covers
    every record that does not positively answer the question: no
    reproducibility file (generations claimed before the file existed), a
    pre-version-2 record without the field, or an unreadable/malformed file.
    ``None`` deliberately does not fail closed -- absence is the normal state
    of every legacy tree, and tampering with the file inside the artifact tree
    is already surfaced as a failed publication by the manifest check (and a
    writer there could read the winner files directly anyway).

    :param Experiment experiment: Experiment whose artifact tree holds the generation.
    :param str generation_id: Generation namespace identifier to look up.
    :return GenerationIdSource | None: ``"caller"``, ``"engine"``, or ``None``
        when no valid record answers.
    """
    if not SAFE_NAME_PATTERN.fullmatch(generation_id):
        return None
    try:
        payload = json.loads(
            path_ops._generation_reproducibility_path(experiment, generation_id).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    source = payload.get("generation_id_source")
    return cast("GenerationIdSource", source) if source in ("caller", "engine") else None
