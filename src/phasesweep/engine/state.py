"""Engine state types, paths, logs, and persisted artifacts."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import optuna
import yaml

from phasesweep._metadata import __version__
from phasesweep.config import Experiment, Phase, Suite
from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.engine.errors import StudyFingerprintMismatchError
from phasesweep.runtime.files import (
    atomic_text_writer,
    file_sha256,
    fsync_directory,
    private_atomic_write_text,
)

if TYPE_CHECKING:
    from phasesweep.engine.read import PhaseWinnerView

log = logging.getLogger("phasesweep.engine.state")

WinnerSourceKind = Literal["phase_trial", "promotion_baseline", "suite_baseline"]

PublicationState = Literal["ok", "absent", "failed"]
"""Verdict on a last-success pointer: valid, never written, or no longer valid."""


@dataclass(frozen=True)
class WinnerSource:
    """Concrete trial that supplies an exposed winner."""

    kind: WinnerSourceKind
    phase: str
    trial_number: int
    generation_id: str | None
    attempt_id: str | None
    study: str | None = None


def _parse_winner_source(
    source_data: Mapping[str, Any], source_kind: WinnerSourceKind
) -> WinnerSource:
    """Reconstruct a persisted ``winner_source`` mapping into a :class:`WinnerSource`.

    Shared by :func:`_load_winner` (state.py) and
    :func:`phasesweep.engine.read.read_winner`, which differ only in what they
    do when this raises: the former re-raises as a strict ``RuntimeError``, the
    latter treats the winner as absent. Callers must validate ``source_kind``
    against :data:`WinnerSourceKind` themselves before calling this, since each
    site fails differently on an invalid kind.

    :param Mapping[str, Any] source_data: Parsed ``winner_source`` block from a
        persisted ``winner.yaml``.
    :param WinnerSourceKind source_kind: The already-validated source kind.
    :raises KeyError: A required field (``phase``, ``trial_number``) is missing.
    :raises TypeError | ValueError: A field cannot be coerced to its expected type.
    :return WinnerSource: The reconstructed source.
    """
    return WinnerSource(
        kind=source_kind,
        phase=str(source_data["phase"]),
        trial_number=int(source_data["trial_number"]),
        generation_id=(
            str(source_data["generation_id"])
            if isinstance(source_data.get("generation_id"), str) and source_data["generation_id"]
            else None
        ),
        attempt_id=(
            str(source_data["attempt_id"])
            if isinstance(source_data.get("attempt_id"), str) and source_data["attempt_id"]
            else None
        ),
        study=(
            str(source_data["study"])
            if isinstance(source_data.get("study"), str) and source_data["study"]
            else None
        ),
    )


@dataclass
class Winner:
    """Phase winner: sampled params, full effective overrides, and metric value.

    ``phase_fingerprint`` is the SHA-256 of the phase's semantic execution
    context at the time the winner was selected (review v0.5.6 / blocker 3).
    Persisted into ``winner.yaml`` and re-checked when ``--from-phase`` skips
    earlier phases — without that check, editing a parent phase's search
    space, fixed overrides, env, metric, or trial command and then resuming
    would silently inherit the *old* winner against the *new* parent config.

    ``None`` only on placeholder winners produced for the dry-run skip path,
    which never get persisted.
    """

    trial_number: int
    params: dict[str, Any]  # sampled params only
    effective_overrides: dict[str, Any]  # full composed overrides (fixed + inherited + sampled)
    metric: float
    constraints: dict[str, float] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    completion: dict[str, Any] = field(default_factory=dict)
    promotion: dict[str, Any] | None = None
    phase_fingerprint: str | None = None
    generation_id: str | None = None
    attempt_id: str | None = None
    source: WinnerSource | None = None
    # Frozen evidence provenance captured when the winning objective was
    # extracted: extractor config fingerprint plus source digest / frozen
    # remote summary subset (review v0.5.17 / finding F). None for dry-run
    # placeholders and winners persisted before the record existed.
    objective_provenance: dict[str, Any] | None = None
    # Identity of the environment the winning trial actually ran under
    # (review v0.5.18 / finding F3): the SHA-256 of that trial's composed
    # trainer environment, plus the ``inherit_env`` contract that produced it.
    # Variable NAMES stay on the trial attrs — the winner file keeps the
    # compact identity. None for dry-run placeholders, for winners persisted
    # before the record existed, and for trials that predate it.
    trainer_env_digest: str | None = None
    trainer_inherit_env: str | list[str] | None = None


def _winner_source_or_default(
    winner: Winner | PhaseWinnerView,
    phase: str,
    *,
    study: str | None = None,
) -> WinnerSource:
    """Return a recorded winner source or synthesize its phase-trial identity.

    :param Winner | PhaseWinnerView winner: Winner carrying optional source provenance.
    :param str phase: Exposed phase used by the fallback source.
    :param str | None study: Optional suite study used by the fallback source.
    :return WinnerSource: Explicit provenance or a complete ``phase_trial`` fallback.
    """
    return winner.source or WinnerSource(
        kind="phase_trial",
        phase=phase,
        trial_number=winner.trial_number,
        generation_id=winner.generation_id,
        attempt_id=winner.attempt_id,
        study=study,
    )


TRIAL_DIR_ATTR = "phasesweep_trial_dir"
GENERATION_ID_ATTR = "phasesweep_generation_id"
ATTEMPT_ID_ATTR = "phasesweep_attempt_id"
PHASE_FINGERPRINT_ATTR = "phasesweep_fingerprint"
STUDY_SCHEMA_ATTR = "phasesweep_study_schema_version"
STUDY_SCHEMA_VERSION = 2
TRIAL_TARGET_ATTR = "phasesweep_trial_target"
# Ordered terminal outcome used to reconstruct the failure circuit breaker
# after a restart. Every terminal trial in a current-schema study has one.
TRIAL_OUTCOME_ATTR = "phasesweep_trial_outcome"
TRIAL_OUTCOME_SCHEMA_VERSION = 1
# Durable phase-abort record. A restarted orchestrator cannot reinterpret the
# same terminal trials as a completed phase.
PHASE_ABORT_ATTR = "phasesweep_phase_abort"
# Durable boundary established when the operator explicitly raises n_trials
# after an abort. Outcomes through this sequence belong to the aborted attempt;
# later failures form the new recovery streak.
PHASE_RECOVERY_ATTR = "phasesweep_phase_recovery"
PHASE_RECOVERY_SCHEMA_VERSION = 1
# Terminal decision recorded before selecting/publishing a winner from an
# intentionally incomplete timeout. It makes selection crash-replayable
# without silently scheduling the trial slots the timeout deliberately left.
PHASE_DECISION_ATTR = "phasesweep_phase_decision"
PHASE_DECISION_SCHEMA_VERSION = 1
FEASIBLE_ATTR = "phasesweep_feasible"
GATES_ATTR = "phasesweep_gates"
# JSON-encoded frozen objective evidence provenance (review v0.5.17 /
# finding F); written when metric extraction succeeds.
OBJECTIVE_PROVENANCE_ATTR = "phasesweep_objective_provenance"
# SHA-256 of the exact trainer environment this trial's subprocess received
# (review v0.5.18 / finding F3). Written at allocation, so failed trials carry
# it too. Ambient VALUES are never stored here — the digest identifies the
# environment, the names attr says which variables it contained, and raw values
# land only in the opt-in owner-only per-trial ``environment.json``.
TRAINER_ENV_DIGEST_ATTR = "phasesweep_trainer_env_digest"
# Sorted list of the variable NAMES in that environment: diagnostic, and
# non-sensitive by construction.
TRAINER_ENV_NAMES_ATTR = "phasesweep_trainer_env_names"
RETURN_CODE_ATTR = "phasesweep_return_code"
DURATION_ATTR = "phasesweep_duration_s"
OVERRIDES_ATTR = "phasesweep_overrides"
CLEANUP_CONFIRMED_ATTR = "phasesweep_cleanup_confirmed"
CLEANUP_RECOVERED_TRIALS_ATTR = "phasesweep_cleanup_recovered_trials"
FAILURE_REASON_ATTR = "phasesweep_failure_reason"
# Study-level binding from a persistent study to the one artifact root it
# publishes into: the resolved ``<workdir>/<experiment>`` namespace as a string
# (review v0.5.19 / finding F5). ``workdir`` is deliberately outside every
# semantic fingerprint so a tree stays movable, which without this binding let
# one study back two divergent publication roots. Claimed on first contact and
# moved only by ``phasesweep rebind-workdir``; the ``_v1`` suffix leaves room
# for a future binding payload that is not a bare path string.
ARTIFACT_ROOT_ATTR = "phasesweep_artifact_root_v1"
CONSTRAINT_PREFIX = "constraint:"


def constraint_attr(name: str) -> str:
    """Return the persisted user-attr key for a constraint value.

    :param str name: Constraint name from the experiment config.
    :return str: Optuna user-attr key used to store the constraint value.
    """
    return f"{CONSTRAINT_PREFIX}{name}"


def _experiment_dir(experiment: Experiment) -> Path:
    """Return the artifact namespace for one experiment.

    :param Experiment experiment: Experiment config with workdir and name.
    :return Path: Absolute directory for experiment artifacts.
    """
    return Path(experiment.workdir).expanduser().resolve() / experiment.experiment


def _phase_dir(experiment: Experiment, phase_name: str) -> Path:
    """Return the artifact namespace for one phase.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name to append under the experiment directory.
    :return Path: Directory for phase artifacts.
    """
    return _experiment_dir(experiment) / phase_name


def _summary_path(experiment: Experiment) -> Path:
    """Return the experiment summary path.

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Path to the experiment summary YAML file.
    """
    return _experiment_dir(experiment) / "summary.yaml"


def _run_log_path(experiment: Experiment) -> Path:
    """Path to the durable run log for one experiment.

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Path to the experiment run log.
    """
    return _experiment_dir(experiment) / "run.log"


def _trial_dir_for(
    experiment: Experiment,
    phase_name: str,
    trial_number: int,
    *,
    generation_id: str | None = None,
    attempt_id: str | None = None,
) -> Path:
    """Return a trial directory, uniquely scoped when execution ids are supplied.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name containing the trial.
    :param int trial_number: Optuna trial number.
    :param str | None generation_id: Current engine invocation id.
    :param str | None attempt_id: Current subprocess attempt id.
    :return Path: Directory for the trial artifacts.
    :raises ValueError: Exactly one of ``generation_id`` and ``attempt_id`` was
        supplied; a uniquely scoped directory needs both.
    """
    if generation_id is None and attempt_id is None:
        return _phase_dir(experiment, phase_name) / f"trial_{trial_number:05d}"
    if generation_id is None or attempt_id is None:
        raise ValueError("generation_id and attempt_id must be supplied together")
    return _phase_dir(experiment, phase_name) / (
        f"trial_{trial_number:05d}__generation_{generation_id}__attempt_{attempt_id}"
    )


def _attempts_dir(experiment: Experiment) -> Path:
    """Return the experiment-level active-attempt registry directory.

    One JSON entry per nonterminal attempt, written at allocation and
    retired once the attempt's Optuna trial is durably terminal. Recovery
    scans this registry *independently of the current phase graph*, so a
    renamed or removed phase cannot hide a stale attempt from preflight
    (review v0.5.17 / blocker 3).

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Directory containing active-attempt registry entries.
    """
    return _experiment_dir(experiment) / "attempts"


def _artifact_root_binding_path(experiment: Experiment) -> Path:
    """Return the reverse artifact-root-to-storage ownership record.

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Root binding JSON path inside the experiment namespace.
    """
    return _experiment_dir(experiment) / "artifact_root_binding.json"


def _generation_path(experiment: Experiment) -> Path:
    """Return the current engine generation metadata path.

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Path to the current generation YAML file.
    """
    return _experiment_dir(experiment) / "generation.yaml"


def _generations_dir(experiment: Experiment) -> Path:
    """Return the immutable generation-record root for an experiment.

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Directory containing all immutable per-generation namespaces.
    """
    return _experiment_dir(experiment) / "generations"


def _generation_dir(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's immutable artifact namespace.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :return Path: Directory scoped to the given generation.
    """
    return _generations_dir(experiment) / generation_id


def _generation_record_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's lifecycle record path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :return Path: Path to the generation's lifecycle record YAML file.
    """
    return _generation_dir(experiment, generation_id) / "generation.yaml"


# A generation's summary is written only as the last step of a successful run,
# so its presence is what separates a published generation from a claim-time
# namespace whose run never got that far. Named once because manifest
# validation reasons about that distinction for *another* generation's
# namespace, where the helper below cannot be used (PR #5 review / reviewer 2,
# blocker 7).
GENERATION_SUMMARY_FILENAME = "summary.yaml"


def _generation_summary_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's summary path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :return Path: Path to the generation's summary YAML file.
    """
    return _generation_dir(experiment, generation_id) / GENERATION_SUMMARY_FILENAME


def _generation_winner_path(experiment: Experiment, generation_id: str, phase_name: str) -> Path:
    """Return one generation's phase-winner path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :param str phase_name: Phase name whose generation-scoped winner path is requested.
    :return Path: Path to the phase's winner YAML file within the generation namespace.
    """
    return _generation_dir(experiment, generation_id) / "phases" / phase_name / "winner.yaml"


def _generation_promotion_decision_path(
    experiment: Experiment, generation_id: str, phase_name: str
) -> Path:
    """Return one generation's phase-promotion path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :param str phase_name: Phase name whose generation-scoped promotion path is requested.
    :return Path: Path to the phase's promotion-decision YAML file within the
        generation namespace.
    """
    return _generation_dir(experiment, generation_id) / "phases" / phase_name / "promotion.yaml"


def _last_successful_generation_path(experiment: Experiment) -> Path:
    """Return the pointer to the last fully published generation.

    :param Experiment experiment: Experiment config with artifact root details.
    :return Path: Path to the YAML file recording the last-successful generation id.
    """
    return _experiment_dir(experiment) / "last_successful_generation.yaml"


GENERATION_SUMMARY_SCHEMA_VERSION = 2
SUITE_SUMMARY_SCHEMA_VERSION = 3
# Provenance files frozen into every generation namespace at claim time
# (review v0.5.18 / finding F6). The summary used to keep only the config
# *fingerprint*, so once the operator edited or lost the YAML the digest could
# prove a mismatch but could not reconstruct the search spaces, fixed
# overrides, contracts, env, or trial command behind a published winner.
GENERATION_CONFIG_SNAPSHOT_FILENAME = "config.snapshot.yaml"
GENERATION_REPRODUCIBILITY_FILENAME = "reproducibility.json"
# Version 2 added ``generation_id_source`` (PR #5 review / P2 missing-handle
# authority): "caller" marks a generation whose identity -- and therefore
# launch authority -- was granted by an external launcher, durably enough to
# survive the loss of that launcher's own state directory.
REPRODUCIBILITY_SCHEMA_VERSION = 2

GenerationIdSource = Literal["caller", "engine"]
"""Who supplied a generation's identity: an external launcher, or the engine."""
_MANIFEST_ARTIFACT_KINDS = frozenset({"winner", "promotion"})
_ARTIFACT_FILENAMES = {"winner": "winner.yaml", "promotion": "promotion.yaml"}
# Manifest kinds that name a file in the generation namespace root rather than
# a phase. Their entries carry ``path`` instead of ``phase``; a generation
# published before finding F6 lists neither kind and holds neither file, which
# is exactly what keeps it valid under the same "listed if and only if
# present" invariant.
_GENERATION_FILE_FILENAMES = {
    "config_snapshot": GENERATION_CONFIG_SNAPSHOT_FILENAME,
    "reproducibility": GENERATION_REPRODUCIBILITY_FILENAME,
}
_MANIFEST_GENERATION_FILE_KINDS = frozenset(_GENERATION_FILE_FILENAMES)


def _generation_config_snapshot_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's canonical config snapshot path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :return Path: Path to the generation's owner-only ``config.snapshot.yaml``.
    """
    return _generation_dir(experiment, generation_id) / GENERATION_CONFIG_SNAPSHOT_FILENAME


def _generation_reproducibility_path(experiment: Experiment, generation_id: str) -> Path:
    """Return one generation's shareable reproducibility-record path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str generation_id: Immutable generation namespace identifier.
    :return Path: Path to the generation's ``reproducibility.json``.
    """
    return _generation_dir(experiment, generation_id) / GENERATION_REPRODUCIBILITY_FILENAME


def _phase_config_fingerprint(phase: Phase) -> str:
    """Hash one phase's configured semantics, independent of any winner.

    This is exactly the per-phase element
    :func:`phasesweep.engine.guards._experiment_semantic_fingerprint` folds
    into its own digest, hashed on its own so a reproducibility record can
    localize *which* phase's configuration differs between two generations.
    It is deliberately not ``winner.yaml``'s ``phase_fingerprint``, which
    additionally binds each inherited winner's effective overrides and
    therefore cannot exist before any phase has run.

    :param Phase phase: Phase whose configured semantics are hashed.
    :return str: SHA-256 hex digest (64 characters) of the canonicalised payload.
    """
    # Deferred: ``engine.guards`` imports this module, so a module-level import
    # here would be circular (same pattern as :func:`_load_winner`).
    from phasesweep.engine.guards import (
        EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
        _semantic_payload_digest,
        _semantic_phase_dump,
    )

    payload = {
        "fingerprint_schema_version": EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
        "name": phase.name,
        **_semantic_phase_dump(phase),
    }
    return _semantic_payload_digest(payload)


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

    Both files are picked up by :func:`_generation_artifact_manifest` and are
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
    # Deferred for the same reason as in :func:`_phase_config_fingerprint`.
    from phasesweep.engine.guards import (
        EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
        FINGERPRINT_SCHEMA_VERSION,
        _execution_identity,
        _experiment_semantic_fingerprint,
    )

    snapshot_path = _generation_config_snapshot_path(experiment, generation_id)
    snapshot = experiment.model_dump(mode="json")
    snapshot["execution"]["cwd"] = _execution_identity(experiment)["cwd"]
    private_atomic_write_text(
        snapshot_path,
        yaml.safe_dump(snapshot, sort_keys=False),
        require_private_dir=False,
    )
    _write_json_atomic(
        _generation_reproducibility_path(experiment, generation_id),
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
                "experiment_fingerprint": EXPERIMENT_FINGERPRINT_SCHEMA_VERSION,
                "phase_fingerprint": FINGERPRINT_SCHEMA_VERSION,
            },
            "config_fingerprint": _experiment_semantic_fingerprint(experiment),
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
            _generation_reproducibility_path(experiment, generation_id).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    source = payload.get("generation_id_source")
    return cast("GenerationIdSource", source) if source in ("caller", "engine") else None


def _generation_artifact_manifest(
    experiment: Experiment, generation_id: str
) -> list[dict[str, str]]:
    """List and hash every result artifact in one generation's namespace.

    Scans the namespace itself rather than reconstructing the set from
    in-memory bookkeeping so the manifest cannot omit an artifact the run
    actually wrote (e.g. a prior promotion decision projected for a skipped
    phase). The namespace is exclusively claimed by this invocation, so
    everything present is this run's own output.

    The namespace-root provenance files written at claim time
    (:func:`_write_generation_provenance`) are listed first, under entries
    that carry ``path`` instead of ``phase``. They are absent -- and so
    unlisted -- in generations published before finding F6.

    :param Experiment experiment: Experiment whose generation is summarized.
    :param str generation_id: Immutable generation namespace to scan.
    :return list[dict[str, str]]: One ``{"kind", "path", "sha256"}`` entry per
        namespace-root provenance file, then one ``{"kind", "phase",
        "sha256"}`` entry per winner/promotion artifact ordered by phase then
        kind.
    """
    generation_dir = _generation_dir(experiment, generation_id)
    items: list[dict[str, str]] = []
    for kind, filename in _GENERATION_FILE_FILENAMES.items():
        artifact = generation_dir / filename
        if artifact.is_file():
            items.append({"kind": kind, "path": filename, "sha256": file_sha256(artifact)})
    phases_dir = generation_dir / "phases"
    if not phases_dir.is_dir():
        return items
    for phase_dir in sorted(phases_dir.iterdir()):
        if not phase_dir.is_dir():
            continue
        for kind in ("winner", "promotion"):
            artifact = phase_dir / _ARTIFACT_FILENAMES[kind]
            if artifact.is_file():
                items.append(
                    {"kind": kind, "phase": phase_dir.name, "sha256": file_sha256(artifact)}
                )
    return items


def _unreadable_artifact_permission_detail(subject: str) -> str:
    """Build the manifest-failure detail for an artifact this user may not read.

    A permission denial is not corruption, and reporting it as corruption sends
    the operator to inspect or restore a namespace that is perfectly healthy
    (re-review v0.5.19 / observation N1). Every generation's
    ``config.snapshot.yaml`` is owner-only, so a second operator reading a
    sound tree hits exactly this case; the verdict still fails closed -- an
    unvalidatable publication may not be reported as published -- but the
    remedy named is the publishing user, not the restore procedure.

    :param str subject: Artifact that could not be read, named by role
        (e.g. ``"config_snapshot artifact"``), never by path.
    :return str: Bare reason clause for the caller's manifest-validation error.
    """
    return (
        f"{subject} is not readable as this user (permission denied): validation cannot run "
        "without it, and a generation's config.snapshot.yaml is deliberately owner-only, so "
        "only the publishing user can fully validate this tree -- re-read it as that user "
        "before treating this publication as corrupt"
    )


def _validate_generation_manifest(
    generation_dir: Path,
    generation_id: str,
    summary: Mapping[str, Any],
) -> None:
    """Validate one generation's complete result graph against its summary manifest.

    "Published" must mean the immutable result is internally complete
    (review v0.5.16 / blocker 3): every artifact the summary claims exists,
    hashes to the recorded content, parses, and cross-checks against the
    summary's own winner facts — and the namespace holds nothing the
    manifest does not list. Runs both pre-commit (before the last-success
    pointer may advance) and on every read before a pointer target is trusted.

    The manifest covers two kinds of artifact: phase-scoped winners and
    promotion decisions (``kind`` + ``phase``), and the namespace-root
    provenance files frozen at claim time (``kind`` + ``path``; see
    :func:`_validate_generation_provenance_files`, review v0.5.18 / finding
    F6). Both obey the same listed-if-and-only-if-present rule, which is what
    keeps a generation published before either existed valid without a schema
    bump.

    A winner carried forward from an earlier generation additionally has that
    generation resolved in this same tree
    (:func:`_validate_winner_source_generation`, PR #5 review / reviewer 2,
    blocker 7): the cited namespace holds the evidence behind the number, so a
    publication whose source generation is absent - or which disagrees with the
    winner that source published - is not a result this tree can stand behind.

    :param Path generation_dir: The generation's immutable namespace directory.
    :param str generation_id: Generation id the summary must belong to (used
        for error text and to tell a carried-forward winner from a local one;
        ownership is checked by the caller).
    :param Mapping[str, Any] summary: Parsed generation summary payload.
    :raises RuntimeError: The manifest is missing, malformed, any artifact is
        absent, altered, unparsable, or inconsistent with the summary, or a
        winner cites a source generation this tree does not hold.
    """

    def _fail(reason: str) -> RuntimeError:
        """Build one uniformly labeled manifest-validation error.

        :param str reason: Specific validation failure being reported.
        :return RuntimeError: Error naming the generation and the reason.
        """
        return RuntimeError(f"Generation {generation_id!r} manifest validation failed: {reason}")

    if summary.get("schema_version") != GENERATION_SUMMARY_SCHEMA_VERSION:
        raise _fail(f"unsupported summary schema_version {summary.get('schema_version')!r}")
    metric = summary.get("metric")
    if (
        not isinstance(metric, Mapping)
        or not isinstance(metric.get("name"), str)
        or metric.get("goal") not in ("minimize", "maximize")
    ):
        raise _fail("summary metric block is malformed")

    raw_artifacts = summary.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise _fail("summary has no artifact manifest")
    listed: dict[tuple[str, str], Mapping[str, Any]] = {}
    listed_files: dict[str, Mapping[str, Any]] = {}
    for entry in raw_artifacts:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("sha256"), str):
            raise _fail("summary artifact entry is malformed")
        kind = entry.get("kind")
        if kind in _MANIFEST_GENERATION_FILE_KINDS:
            if entry.get("path") != _GENERATION_FILE_FILENAMES[str(kind)]:
                raise _fail(f"{kind} manifest entry names an unexpected path")
            if str(kind) in listed_files:
                raise _fail(f"duplicate artifact entry for {kind}")
            listed_files[str(kind)] = entry
            continue
        if kind not in _MANIFEST_ARTIFACT_KINDS or not isinstance(entry.get("phase"), str):
            raise _fail("summary artifact entry is malformed")
        key = (str(kind), str(entry["phase"]))
        if key in listed:
            raise _fail(f"duplicate artifact entry for {key}")
        listed[key] = entry
    _validate_generation_provenance_files(
        generation_dir,
        summary,
        listed_files,
        _fail,
    )

    raw_phases = summary.get("phases")
    if not isinstance(raw_phases, list):
        raise _fail("summary has no phase winner list")
    phase_items: dict[str, Mapping[str, Any]] = {}
    for item in raw_phases:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise _fail("summary phase winner entry is malformed")
        name = str(item["name"])
        if name in phase_items:
            raise _fail(f"duplicate phase winner entry {name!r}")
        phase_items[name] = item
        if ("winner", name) not in listed:
            raise _fail(f"phase {name!r} winner is not listed in the artifact manifest")

    raw_decisions = summary.get("promotion_decisions")
    if not isinstance(raw_decisions, list):
        raise _fail("summary has no promotion decision list")
    decision_items: dict[str, Mapping[str, Any]] = {}
    for decision in raw_decisions:
        if not isinstance(decision, Mapping) or not isinstance(decision.get("phase"), str):
            raise _fail("summary promotion decision entry is malformed")
        name = str(decision["phase"])
        decision_items[name] = decision
        if ("promotion", name) not in listed:
            raise _fail(f"phase {name!r} promotion decision is not listed in the artifact manifest")

    for kind, name in listed:
        if kind == "winner" and name not in phase_items:
            raise _fail(f"artifact manifest lists a winner for unknown phase {name!r}")

    for (kind, name), entry in listed.items():
        artifact_path = generation_dir / "phases" / name / _ARTIFACT_FILENAMES[kind]
        try:
            content = artifact_path.read_bytes()
        except PermissionError as exc:
            raise _fail(
                _unreadable_artifact_permission_detail(f"{kind} artifact for phase {name!r}")
            ) from exc
        except OSError as exc:
            raise _fail(f"{kind} artifact for phase {name!r} is missing or unreadable") from exc
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise _fail(f"{kind} artifact for phase {name!r} does not match its recorded hash")
        try:
            payload = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise _fail(f"{kind} artifact for phase {name!r} is not parseable") from exc
        if not isinstance(payload, Mapping):
            raise _fail(f"{kind} artifact for phase {name!r} is not a mapping")
        if payload.get("phase") != name:
            raise _fail(f"{kind} artifact for phase {name!r} names a different phase")
        if kind == "winner":
            item = phase_items[name]
            if payload.get("trial_number") != item.get("trial_number"):
                raise _fail(f"winner for phase {name!r} disagrees with the summary trial number")
            metric_block = payload.get("metric")
            if not isinstance(metric_block, Mapping) or metric_block.get("goal") != metric["goal"]:
                raise _fail(f"winner for phase {name!r} has a mismatched metric goal")
            value = metric_block.get(metric["name"])
            if not isinstance(value, (int, float)) or float(value) != item.get("metric"):
                raise _fail(f"winner for phase {name!r} disagrees with the summary metric value")
            # A winner is legitimately carried forward from an earlier
            # generation, so its own generation_id may name that earlier
            # generation — the check is well-formedness, not equality.
            for id_field in ("generation_id", "attempt_id"):
                recorded = payload.get(id_field)
                if not isinstance(recorded, str) or not recorded:
                    raise _fail(f"winner for phase {name!r} has no valid {id_field}")
            source = payload.get("winner_source")
            if not isinstance(source, Mapping):
                raise _fail(f"winner for phase {name!r} has no valid winner_source")
            for id_field in ("generation_id", "attempt_id"):
                if source.get(id_field) != payload.get(id_field):
                    raise _fail(
                        f"winner for phase {name!r} has a winner_source "
                        f"that disagrees with its {id_field}"
                    )
            _validate_winner_source_generation(
                generation_dir,
                generation_id,
                name,
                source.get("phase"),
                payload,
                _fail,
            )
            if not isinstance(payload.get("completion"), Mapping):
                raise _fail(f"winner for phase {name!r} has no completion metadata")
        elif name in decision_items:
            if payload.get("action") != decision_items[name].get("action"):
                raise _fail(
                    f"promotion decision for phase {name!r} disagrees with the summary action"
                )

    phases_dir = generation_dir / "phases"
    if phases_dir.is_dir():
        for phase_dir in phases_dir.iterdir():
            if not phase_dir.is_dir():
                continue
            for kind, filename in _ARTIFACT_FILENAMES.items():
                if (phase_dir / filename).is_file() and (kind, phase_dir.name) not in listed:
                    raise _fail(
                        f"namespace contains an unlisted {kind} artifact "
                        f"for phase {phase_dir.name!r}"
                    )


def _validate_winner_source_generation(
    generation_dir: Path,
    generation_id: str,
    phase_name: str,
    source_phase: object,
    payload: Mapping[str, Any],
    fail: Callable[[str], RuntimeError],
) -> None:
    """Require a carried-forward winner's source generation to exist in this tree.

    A winner's own ``generation_id`` may legitimately name an earlier
    generation - a top-up that reselects an existing trial, or a ``--from-phase``
    resume that reuses a validated parent winner - and the manifest deliberately
    checks that field for well-formedness only. That left the whole
    cross-generation claim unverified: a published winner could cite a
    generation that exists only in some *other* artifact tree, or no tree at
    all, and the publication still validated as ``ok`` (PR #5 review /
    reviewer 2, blocker 7). The cited namespace is where the evidence behind
    that number lives, so it has to resolve here.

    Deviating deliberately from a blanket "the source must also hold a winner
    record for this phase": a crash between trial completion and publication
    leaves a claim-time namespace - provenance files, no summary - whose trials
    a later top-up legitimately publishes first, citing that crashed
    generation. Requiring a winner record there would make ordinary crash
    recovery permanently unpublishable. So the record is required exactly when
    the source generation itself published (it has a summary); an unpublished
    source needs only to exist. Directory existence alone already enforces the
    same-tree invariant, and the summary carve-out still catches a published
    source that lost its winner record.

    :param Path generation_dir: The publishing generation's namespace directory.
    :param str generation_id: The publishing generation's own id.
    :param str phase_name: Phase exposing the winner being validated.
    :param object source_phase: Recorded phase that owns the source winner artifact.
    :param Mapping[str, Any] payload: Parsed winner artifact, whose
        ``generation_id``/``attempt_id``/``trial_number`` the caller has
        already checked for well-formedness and internal agreement.
    :param Callable[[str], RuntimeError] fail: Builder for the caller's
        uniformly labeled manifest-validation error.
    :raises RuntimeError: Whatever ``fail`` builds, when the cited source
        generation is unsafely named, absent from this tree, or published a
        winner record for this phase that disagrees with the carried winner.
    """
    source_generation = payload["generation_id"]
    if source_generation == generation_id:
        return
    if not isinstance(source_phase, str) or not SAFE_NAME_PATTERN.fullmatch(source_phase):
        raise fail(f"winner for phase {phase_name!r} has no valid winner_source phase")
    if not SAFE_NAME_PATTERN.fullmatch(source_generation):
        # A corrupt or hostile winner could otherwise steer the lookups below
        # out of the generations root with a traversal component.
        raise fail(
            f"winner for phase {phase_name!r} cites source generation "
            f"{source_generation!r}, which is not a valid generation name"
        )
    source_dir = generation_dir.parent / source_generation
    if not source_dir.is_dir():
        raise fail(
            f"winner for phase {phase_name!r} cites source generation "
            f"{source_generation!r} which does not exist in this tree"
        )
    source_winner_path = source_dir / "phases" / source_phase / _ARTIFACT_FILENAMES["winner"]
    if not source_winner_path.is_file():
        if (source_dir / GENERATION_SUMMARY_FILENAME).is_file():
            raise fail(
                f"winner for phase {phase_name!r} cites source generation "
                f"{source_generation!r}, which published but holds no winner record "
                f"for source phase {source_phase!r}"
            )
        # Unpublished source namespace: the crash-recovery case above.
        return
    try:
        content = source_winner_path.read_bytes()
    except PermissionError as exc:
        raise fail(
            _unreadable_artifact_permission_detail(
                f"source generation {source_generation!r} winner artifact for phase "
                f"{source_phase!r}"
            )
        ) from exc
    except OSError as exc:
        raise fail(
            f"source generation {source_generation!r} winner artifact for phase "
            f"{source_phase!r} is missing or unreadable"
        ) from exc
    try:
        source_payload = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise fail(
            f"source generation {source_generation!r} winner artifact for phase "
            f"{source_phase!r} is not parseable"
        ) from exc
    if not isinstance(source_payload, Mapping):
        raise fail(
            f"source generation {source_generation!r} winner artifact for phase "
            f"{source_phase!r} is not a mapping"
        )
    expected = {
        "phase": source_phase,
        "trial_number": payload.get("trial_number"),
        "generation_id": source_generation,
        "attempt_id": payload.get("attempt_id"),
    }
    for field_name, expected_value in expected.items():
        if source_payload.get(field_name) != expected_value:
            raise fail(
                f"winner for phase {phase_name!r} disagrees with the winner recorded by its "
                f"source generation {source_generation!r} on {field_name}"
            )


def _validate_generation_provenance_files(
    generation_dir: Path,
    summary: Mapping[str, Any],
    listed_files: Mapping[str, Mapping[str, Any]],
    fail: Callable[[str], RuntimeError],
) -> None:
    """Validate a generation's claim-time provenance files against its manifest.

    Extends the manifest invariant -- every listed artifact exists and hashes
    to its recorded content, and the namespace holds nothing the manifest does
    not list -- onto ``config.snapshot.yaml`` and ``reproducibility.json``
    (review v0.5.18 / finding F6). The two are all-or-nothing: they are
    written together at claim time, so a manifest that lists one without the
    other has been edited. A generation published before those files existed
    lists neither and holds neither, which passes every check here unchanged.

    Note that the snapshot is owner-only, so a reader who cannot read it
    cannot validate the publication at all -- the same fail-closed outcome as
    any other unreadable manifest-listed artifact. A permission denial is
    reported as its own reason rather than as "missing or unreadable"
    (:func:`_unreadable_artifact_permission_detail`, re-review v0.5.19 /
    observation N1): a second operator on a healthy tree hits it routinely,
    and the remedy is the publishing user, not a namespace restore.

    :param Path generation_dir: The generation's immutable namespace directory.
    :param Mapping[str, Any] summary: Parsed generation summary payload, whose
        identity the reproducibility record must agree with.
    :param Mapping[str, Mapping[str, Any]] listed_files: Manifest entries for
        the namespace-root provenance files, keyed by kind.
    :param Callable[[str], RuntimeError] fail: Builder for the caller's
        uniformly labeled manifest-validation error.
    :raises RuntimeError: Whatever ``fail`` builds, when a provenance file is
        listed without its partner, is absent, unreadable, altered,
        unparsable, or disagrees with the summary's own identity.
    """
    for kind, filename in _GENERATION_FILE_FILENAMES.items():
        if (generation_dir / filename).is_file() and kind not in listed_files:
            raise fail(f"namespace contains an unlisted {kind} artifact")
    if not listed_files:
        return
    if set(listed_files) != _MANIFEST_GENERATION_FILE_KINDS:
        raise fail("summary lists only part of the generation provenance record")

    digests: dict[str, str] = {}
    for kind, filename in _GENERATION_FILE_FILENAMES.items():
        try:
            content = (generation_dir / filename).read_bytes()
        except PermissionError as exc:
            raise fail(_unreadable_artifact_permission_detail(f"{kind} artifact")) from exc
        except OSError as exc:
            raise fail(f"{kind} artifact is missing or unreadable") from exc
        digest = hashlib.sha256(content).hexdigest()
        if digest != listed_files[kind]["sha256"]:
            raise fail(f"{kind} artifact does not match its recorded hash")
        digests[kind] = digest

    snapshot_path = generation_dir / GENERATION_CONFIG_SNAPSHOT_FILENAME
    try:
        snapshot = yaml.safe_load(snapshot_path.read_text())
    except PermissionError as exc:
        raise fail(_unreadable_artifact_permission_detail("config_snapshot artifact")) from exc
    except (OSError, yaml.YAMLError) as exc:
        raise fail("config_snapshot artifact is not parseable") from exc
    if not isinstance(snapshot, Mapping):
        raise fail("config_snapshot artifact is not a mapping")
    if snapshot.get("experiment") != summary.get("experiment"):
        raise fail("config_snapshot artifact names a different experiment")

    record_path = generation_dir / GENERATION_REPRODUCIBILITY_FILENAME
    try:
        record = json.loads(record_path.read_text())
    except PermissionError as exc:
        raise fail(_unreadable_artifact_permission_detail("reproducibility artifact")) from exc
    except (OSError, ValueError) as exc:
        raise fail("reproducibility artifact is not parseable") from exc
    if not isinstance(record, Mapping):
        raise fail("reproducibility artifact is not a mapping")
    recorded_identity = (record.get("experiment"), record.get("generation_id"))
    if recorded_identity != (summary.get("experiment"), summary.get("generation_id")):
        raise fail("reproducibility artifact names a different generation")
    recorded_snapshot = record.get("config_snapshot")
    if (
        not isinstance(recorded_snapshot, Mapping)
        or recorded_snapshot.get("path") != GENERATION_CONFIG_SNAPSHOT_FILENAME
        or recorded_snapshot.get("sha256") != digests["config_snapshot"]
    ):
        raise fail("reproducibility artifact does not anchor the config snapshot it published")


def _read_pointer_target(
    pointer_path: Path,
    *,
    id_key: str,
    owner_key: str,
    owner_name: str,
) -> str | None:
    """Read one last-success pointer's target id, validating the pointer itself.

    :param Path pointer_path: Pointer YAML file to read.
    :param str id_key: Payload key holding the target generation id.
    :param str owner_key: Payload key naming the owning experiment or suite.
    :param str owner_name: Expected owner name the pointer must record.
    :return str | None: A safe-name target id owned by ``owner_name``, or
        ``None`` when the pointer is missing, unreadable, malformed, names
        another owner, or carries an unsafe id.
    """
    try:
        payload = yaml.safe_load(pointer_path.read_text())
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(payload, dict) or payload.get(owner_key) != owner_name:
        return None
    target_id = payload.get(id_key)
    if not isinstance(target_id, str) or not SAFE_NAME_PATTERN.fullmatch(target_id):
        return None
    return target_id


def _read_pointer_target_summary(
    summary_path: Path,
    *,
    id_key: str,
    target_id: str,
    owner_key: str,
    owner_name: str,
) -> dict[str, Any] | None:
    """Read a pointer target's own immutable summary and confirm its identity.

    Replaces the old record-state check (``_record_is_complete``): the
    per-generation lifecycle record is now informational and written *after*
    the last-success pointer commit (review v0.5.15 / blocker 3), so requiring
    ``state == "complete"``/``"published"`` on it would create a crash window
    where a just-committed publication reads back as nothing-published. This
    fails closed on the pointer target's own immutable *summary*: it must
    parse as a mapping naming this exact owner and id. Callers holding a
    schema-versioned summary additionally validate the complete artifact
    manifest (:func:`_validate_generation_manifest`; review v0.5.16 /
    blocker 3) — this helper only performs the identity gate common to
    experiment and suite pointers.

    :param Path summary_path: Immutable generation summary YAML file to read.
    :param str id_key: Summary key holding the generation id.
    :param str target_id: Generation id the summary must name.
    :param str owner_key: Summary key naming the owning experiment or suite.
    :param str owner_name: Expected owner name the summary must carry.
    :return dict[str, Any] | None: The parsed summary when readable and
        correctly named; ``None`` otherwise.
    """
    try:
        summary = yaml.safe_load(summary_path.read_text())
    except (OSError, yaml.YAMLError):
        return None
    if (
        isinstance(summary, dict)
        and summary.get(owner_key) == owner_name
        and summary.get(id_key) == target_id
    ):
        return summary
    return None


@dataclass(frozen=True)
class PublicationPointer:
    """Tri-state resolution of one last-success publication pointer.

    "Nothing was ever published" and "the recorded publication no longer
    validates" are different operational facts with opposite remedies, and
    collapsing both into ``None`` made a corrupt result tree indistinguishable
    from a fresh one (review v0.5.18 / finding F4). The reporting surfaces
    resolve through this type; path-construction helpers keep the boolean
    :func:`_last_successful_generation_id` view, since both non-``ok`` states
    mean the same thing to them: nothing may be read as published.

    ``error`` is deliberately path-free: it names artifacts by role rather than
    location, so any reporting surface could quote it safely -- though the MCP
    layer forwards only the ``publication_integrity`` enum and leaves the
    free-text detail to the operator-facing CLI.
    """

    state: PublicationState
    generation_id: str | None
    """Pointer target id: the published generation when ``ok``, the generation
    whose validation failed when ``failed``, ``None`` when ``absent`` or when
    the pointer itself is the unreadable part."""
    error: str | None
    """Validation diagnostic; set if and only if ``state`` is ``failed``."""


def _unresolvable_pointer(pointer_path: Path, owner_label: str) -> PublicationPointer:
    """Classify a pointer whose own payload could not be resolved to a target.

    A pointer file is written once, atomically, after its target validates, so
    a file that exists but does not name a target this owner published is
    tampering or corruption -- not an owner that has published nothing.

    :param Path pointer_path: Last-success pointer whose read just failed.
    :param str owner_label: Human-readable owner for the diagnostic text.
    :return PublicationPointer: ``absent`` when no pointer file exists,
        ``failed`` when one exists but cannot be resolved.
    """
    try:
        pointer_path.stat()
    except FileNotFoundError:
        return PublicationPointer(state="absent", generation_id=None, error=None)
    except OSError:
        pass
    return PublicationPointer(
        state="failed",
        generation_id=None,
        error=(
            f"The last-success pointer for {owner_label} is unreadable, malformed, "
            "or does not name a generation this owner published."
        ),
    )


def _resolve_publication_pointer(
    experiment: Experiment,
    *,
    raise_on_manifest_error: bool = False,
) -> PublicationPointer:
    """Resolve the last-success pointer to ``ok`` / ``absent`` / ``failed``.

    The pointer is authoritative only when its target's own immutable summary
    parses and names this exact experiment and generation (review v0.5.15 /
    blocker 3) -- not when the per-generation lifecycle record says so, since
    that record is now written *after* this pointer commits and would
    otherwise create a crash window where a committed publication reads as
    nothing-published.

    For schema-versioned summaries the target's complete artifact manifest is
    additionally validated -- every listed winner/promotion artifact plus the
    claim-time provenance files must exist, hash to their recorded content,
    and cross-check against the summary (review v0.5.16 / blocker 3, review
    v0.5.18 / finding F6). Validation runs on every authoritative read:
    generation directories are write-once by PhaseSweep convention, but the
    filesystem does not enforce immutability and a long-lived reader must
    notice later corruption or operator edits. Pre-manifest legacy summaries
    keep the identity-only gate.

    Every way of failing that validation -- an unresolvable pointer, a
    missing/tampered target summary, a broken manifest -- resolves to
    ``failed``, never ``absent``: the pointer's existence is durable evidence
    that this tree once held a result, and a caller told "nothing published"
    would re-run over it and advance the pointer past the corruption.

    :param Experiment experiment: Experiment config with artifact root details.
    :param bool raise_on_manifest_error: Re-raise a versioned publication's
        manifest error for an actionable resume failure instead of reporting
        it as ``failed``, as read-only status APIs require. Only the manifest
        branch raises; every other failure still resolves to ``failed``.
    :raises RuntimeError: ``raise_on_manifest_error`` is set and the target's
        artifact manifest does not validate.
    :return PublicationPointer: The tri-state verdict for this experiment.
    """
    pointer_path = _last_successful_generation_path(experiment)
    generation_id = _read_pointer_target(
        pointer_path,
        id_key="generation_id",
        owner_key="experiment",
        owner_name=experiment.experiment,
    )
    if generation_id is None:
        return _unresolvable_pointer(pointer_path, f"experiment {experiment.experiment!r}")
    summary = _read_pointer_target_summary(
        _generation_summary_path(experiment, generation_id),
        id_key="generation_id",
        target_id=generation_id,
        owner_key="experiment",
        owner_name=experiment.experiment,
    )
    if summary is None:
        return PublicationPointer(
            state="failed",
            generation_id=generation_id,
            error=(
                f"Generation {generation_id!r} summary is missing, unreadable, or does not "
                f"name experiment {experiment.experiment!r} and this generation."
            ),
        )
    if "schema_version" in summary:
        generation_dir = _generation_dir(experiment, generation_id)
        if raise_on_manifest_error:
            _validate_generation_manifest(generation_dir, generation_id, summary)
        else:
            # A versioned summary must validate its complete artifact manifest
            # (review v0.5.16 / blocker 3). Pre-manifest legacy summaries keep
            # the identity-only gate above; see docs/config.md's upgrade notes.
            try:
                _validate_generation_manifest(generation_dir, generation_id, summary)
            except RuntimeError as exc:
                log.warning("%s", exc)
                return PublicationPointer(
                    state="failed", generation_id=generation_id, error=str(exc)
                )
    return PublicationPointer(state="ok", generation_id=generation_id, error=None)


def _last_successful_generation_id(
    experiment: Experiment,
    *,
    raise_on_manifest_error: bool = False,
) -> str | None:
    """Return the last-success generation id, failing closed on any invalidity.

    The boolean-blind view of :func:`_resolve_publication_pointer`, kept for
    the path-construction and resume callers to which "never published" and
    "published but corrupt" mean the same thing: nothing here may be read as
    published. Every caller that *reports* publication state to an operator or
    agent must use the tri-state resolver instead (review v0.5.18 / finding
    F4).

    :param Experiment experiment: Experiment config with artifact root details.
    :param bool raise_on_manifest_error: Re-raise a versioned publication's
        manifest error for an actionable resume failure instead of returning
        ``None`` as read-only status APIs require.
    :raises RuntimeError: ``raise_on_manifest_error`` is set and the target's
        artifact manifest does not validate.
    :return str | None: The last-successful generation id, or ``None`` if the
        pointer or its target summary is missing, unreadable, malformed,
        unsafely named, owned by another experiment, or fails manifest
        validation.
    """
    pointer = _resolve_publication_pointer(
        experiment,
        raise_on_manifest_error=raise_on_manifest_error,
    )
    return pointer.generation_id if pointer.state == "ok" else None


def _published_winner_path_for(
    experiment: Experiment,
    published_generation_id: str | None,
    phase_name: str,
) -> Path | None:
    """Resolve the authoritative winner path from an already-captured published id.

    Same legacy-fallback semantics as :func:`_published_winner_path`, but
    takes the caller's already-resolved last-success id instead of re-reading
    the pointer, so one status read that scopes several phases (or several
    fields) from a single captured id can never mix identities from two
    different pointer resolutions (review v0.5.15 / blocker 3).

    :param Experiment experiment: Experiment config with artifact root details.
    :param str | None published_generation_id: Already-resolved
        :func:`_last_successful_generation_id` result (or ``None``).
    :param str phase_name: Phase name whose published winner path is requested.
    :return Path | None: The generation-scoped winner path when
        ``published_generation_id`` is given; the legacy compatibility winner
        path when no generation has ever been published; ``None`` when a
        generation exists but none has completed successfully yet.
    """
    if published_generation_id is not None:
        return _generation_winner_path(experiment, published_generation_id, phase_name)
    if _generation_path(experiment).is_file():
        return None
    return _winner_path(experiment, phase_name)


def _published_winner_path(experiment: Experiment, phase_name: str) -> Path | None:
    """Return the authoritative last-success winner, with legacy fallback.

    Compatibility projections are used only for layouts that predate generation
    metadata. Once a generation has been published as current, the absence of a
    last-success pointer means no result has been published yet; a partially
    copied compatibility file must not become authoritative.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name whose published winner path is requested.
    :return Path | None: The generation-scoped winner path when a last-success
        pointer exists; the legacy compatibility winner path when no generation
        has ever been published; ``None`` when a generation exists but none has
        completed successfully yet.
    """
    return _published_winner_path_for(
        experiment, _last_successful_generation_id(experiment), phase_name
    )


def _published_summary_path_for(
    experiment: Experiment,
    published_generation_id: str | None,
) -> Path | None:
    """Resolve the authoritative summary path from an already-captured published id.

    Resolves to the generation-scoped summary path once a generation has
    published, falls back to the legacy compatibility summary path when none
    ever has, and returns ``None`` when a generation exists but none has
    completed successfully yet -- takes the caller's already-resolved
    last-success id instead of re-reading the pointer (review v0.5.15 /
    blocker 3).

    :param Experiment experiment: Experiment config with artifact root details.
    :param str | None published_generation_id: Already-resolved
        :func:`_last_successful_generation_id` result (or ``None``).
    :return Path | None: The generation-scoped summary path when
        ``published_generation_id`` is given; the legacy compatibility summary
        path when no generation has ever been published; ``None`` when a
        generation exists but none has completed successfully yet.
    """
    if published_generation_id is not None:
        return _generation_summary_path(experiment, published_generation_id)
    if _generation_path(experiment).is_file():
        return None
    return _summary_path(experiment)


def _published_promotion_decision_path(
    experiment: Experiment,
    phase_name: str,
) -> Path | None:
    """Return the authoritative last-success promotion decision, with legacy fallback.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name whose published promotion-decision path is requested.
    :return Path | None: The generation-scoped promotion-decision path when a
        last-success pointer exists; the legacy compatibility path when no
        generation has ever been published; ``None`` when a generation exists
        but none has completed successfully yet.
    """
    generation_id = _last_successful_generation_id(experiment)
    if generation_id is not None:
        return _generation_promotion_decision_path(experiment, generation_id, phase_name)
    if _generation_path(experiment).is_file():
        return None
    return _promotion_decision_path(experiment, phase_name)


def _winner_path(experiment: Experiment, phase_name: str) -> Path:
    """Return the path to a phase's persisted winner.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name whose winner path is requested.
    :return Path: Path to the persisted winner YAML file.
    """
    return _phase_dir(experiment, phase_name) / "winner.yaml"


def _promotion_decision_path(experiment: Experiment, phase_name: str) -> Path:
    """Path to the persisted phase promotion decision.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name whose promotion decision path is
        requested.
    :return Path: Path to the persisted promotion decision YAML file.
    """
    return _phase_dir(experiment, phase_name) / "promotion.yaml"


def _suite_dir(suite: Suite) -> Path:
    """Filesystem namespace for suite-level summary/log artifacts.

    :param Suite suite: Suite config with default artifact settings.
    :return Path: Absolute directory for suite artifacts.
    """
    return Path(suite.defaults.workdir).expanduser().resolve() / suite.suite


def _suite_summary_path(suite: Suite) -> Path:
    """Path to a suite-level summary.

    :param Suite suite: Suite config with artifact root details.
    :return Path: Path to the suite summary YAML file.
    """
    return _suite_dir(suite) / "suite_summary.yaml"


def _suite_generation_path(suite: Suite) -> Path:
    """Return the current suite-generation lifecycle path.

    :param Suite suite: Suite config with artifact root details.
    :return Path: Path to the current suite-generation lifecycle YAML file.
    """
    return _suite_dir(suite) / "suite_generation.yaml"


def _suite_generations_dir(suite: Suite) -> Path:
    """Return the immutable suite-generation root.

    :param Suite suite: Suite config with artifact root details.
    :return Path: Directory containing all immutable per-suite-generation namespaces.
    """
    return _suite_dir(suite) / "suite_generations"


def _suite_generation_dir(suite: Suite, generation_id: str) -> Path:
    """Return one immutable suite-generation directory.

    :param Suite suite: Suite config with artifact root details.
    :param str generation_id: Immutable suite-generation namespace identifier.
    :return Path: Directory scoped to the given suite generation.
    """
    return _suite_generations_dir(suite) / generation_id


def _suite_generation_record_path(suite: Suite, generation_id: str) -> Path:
    """Return one suite generation's lifecycle record path.

    :param Suite suite: Suite config with artifact root details.
    :param str generation_id: Immutable suite-generation namespace identifier.
    :return Path: Path to the suite generation's lifecycle record YAML file.
    """
    return _suite_generation_dir(suite, generation_id) / "generation.yaml"


def _suite_generation_summary_path(suite: Suite, generation_id: str) -> Path:
    """Return one suite generation's immutable summary path.

    :param Suite suite: Suite config with artifact root details.
    :param str generation_id: Immutable suite-generation namespace identifier.
    :return Path: Path to the suite generation's summary YAML file.
    """
    return _suite_generation_dir(suite, generation_id) / "summary.yaml"


def _last_successful_suite_generation_path(suite: Suite) -> Path:
    """Return the pointer to the last fully published suite generation.

    :param Suite suite: Suite config with artifact root details.
    :return Path: Path to the YAML file recording the last-successful suite generation id.
    """
    return _suite_dir(suite) / "last_successful_suite_generation.yaml"


def _resolve_suite_publication_pointer(suite: Suite) -> PublicationPointer:
    """Resolve the suite last-success pointer to ``ok`` / ``absent`` / ``failed``.

    Suite mirror of :func:`_resolve_publication_pointer`, including its F4
    rule that every way of failing validation resolves to ``failed`` rather
    than ``absent``: the pointer is authoritative only when its target's own
    immutable summary parses and names this exact suite and suite generation
    (review v0.5.15 / blocker 3), not when the per-suite-generation lifecycle
    record says so, and a schema-current summary must additionally anchor its
    winner facts to the hash-covered component summaries it recorded.

    :param Suite suite: Suite config with artifact root details.
    :return PublicationPointer: The tri-state verdict for this suite.
    """
    pointer_path = _last_successful_suite_generation_path(suite)
    generation_id = _read_pointer_target(
        pointer_path,
        id_key="suite_generation_id",
        owner_key="suite",
        owner_name=suite.suite,
    )
    if generation_id is None:
        return _unresolvable_pointer(pointer_path, f"suite {suite.suite!r}")
    summary = _read_pointer_target_summary(
        _suite_generation_summary_path(suite, generation_id),
        id_key="suite_generation_id",
        target_id=generation_id,
        owner_key="suite",
        owner_name=suite.suite,
    )
    if summary is None:
        return PublicationPointer(
            state="failed",
            generation_id=generation_id,
            error=(
                f"Suite generation {generation_id!r} summary is missing, unreadable, or does "
                f"not name suite {suite.suite!r} and this suite generation."
            ),
        )
    if summary.get("schema_version") not in (None, 1, 2, SUITE_SUMMARY_SCHEMA_VERSION):
        # A summary from a newer schema must not be silently misread.
        return PublicationPointer(
            state="failed",
            generation_id=generation_id,
            error=(
                f"Suite generation {generation_id!r} summary declares unsupported "
                f"schema_version {summary.get('schema_version')!r}."
            ),
        )
    if summary.get("schema_version") == SUITE_SUMMARY_SCHEMA_VERSION:
        # The summary's own winner facts must be anchored to the hash-covered
        # component artifacts it recorded at publication, so an edited or
        # partially written suite summary cannot present altered results as
        # published (review v0.5.17 gap hunt). Historical reads survive suite
        # config edits because the recorded paths and hashes are resolved
        # directly, never through the CURRENT compiled plan; pre-v3 legacy
        # summaries keep the identity-only gate above.
        try:
            _validate_suite_summary_integrity(generation_id, summary)
        except RuntimeError as exc:
            log.warning(
                "Suite last-success pointer target failed integrity validation",
                exc_info=True,
            )
            return PublicationPointer(state="failed", generation_id=generation_id, error=str(exc))
    return PublicationPointer(state="ok", generation_id=generation_id, error=None)


def _validate_suite_summary_integrity(
    generation_id: str,
    summary: Mapping[str, Any],
) -> None:
    """Validate a suite summary's winner facts against its recorded components.

    Suite mirror of :func:`_validate_generation_manifest` (review v0.5.17 gap
    hunt): the published suite summary is the authoritative read surface for
    suite winners, so its per-study results must be anchored to hash-covered
    component artifacts rather than trusted as bare text. Each study record
    names its component generation summary by absolute path plus content
    hash; the file must sit at ``generations/<component id>/summary.yaml``,
    parse, and name the recorded experiment and generation — and every
    exposed phase winner in the suite summary must match one of those verified
    component summaries. A study's own winner matches its complete compact
    payload. A promotion-adopted suite baseline matches the recorded source
    phase and trial-derived payload (trial, metric, parameters, effective
    overrides, constraints, gates, generation, and attempt), while its
    completion/source/promotion metadata may legitimately describe the
    candidate slot that exposes the clone. Runs pre-commit (before the suite
    last-success pointer advances) and read-side (before a pointer target is
    trusted).

    :param str generation_id: Suite generation id, used for error text.
    :param Mapping[str, Any] summary: Parsed suite summary payload.
    :raises RuntimeError: A study record is malformed, a component summary is
        missing, altered, or misidentified, or an exposed winner is not
        anchored to any verified component summary.
    """

    def _fail(reason: str) -> RuntimeError:
        """Build one uniformly labeled suite-integrity error.

        :param str reason: Specific validation failure being reported.
        :return RuntimeError: Error naming the suite generation and the reason.
        """
        return RuntimeError(
            f"Suite generation {generation_id!r} summary integrity validation failed: {reason}"
        )

    records = summary.get("studies")
    if not isinstance(records, list):
        raise _fail("summary has no study records")

    evidence_fields = (
        "trial_number",
        "metric",
        "params",
        "effective_overrides",
        "constraints",
        "gates",
        "generation_id",
        "attempt_id",
    )
    full_fields = (*evidence_fields, "completion", "winner_source", "promotion")

    def _winner_key(
        item: Mapping[str, Any],
        *,
        phase_name: str,
        include_exposure_metadata: bool,
    ) -> tuple[str, str]:
        """Return a type-aware canonical key for one component winner.

        :param Mapping[str, Any] item: One summary phase-winner entry.
        :param str phase_name: Source component phase that produced the winner.
        :param bool include_exposure_metadata: Include completion, source, and
            promotion fields when the suite exposes its own component winner.
        :raises RuntimeError: If the winner payload cannot be represented as safe YAML.
        :return tuple[str, str]: Source phase plus canonical winner payload.
        """
        fields = full_fields if include_exposure_metadata else evidence_fields
        payload = {field: item[field] for field in fields if field in item}
        try:
            encoded = yaml.safe_dump(payload, sort_keys=True)
        except yaml.YAMLError as exc:
            raise _fail("summary contains an invalid winner payload") from exc
        return phase_name, encoded

    component_full_winners: set[tuple[str, str]] = set()
    component_evidence_winners: set[tuple[str, str]] = set()
    legacy_component_studies: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("name"), str):
            raise _fail("summary study record is malformed")
        name = str(record["name"])
        component_generation = record.get("experiment_generation_id")
        recorded_path = record.get("component_summary_path")
        recorded_sha = record.get("component_summary_sha256")
        if (
            not isinstance(component_generation, str)
            or not component_generation
            or not isinstance(recorded_path, str)
            or not recorded_path
            or not isinstance(recorded_sha, str)
        ):
            raise _fail(f"study {name!r} has no component summary reference")
        target = Path(recorded_path)
        expected_suffix = Path("generations") / component_generation / "summary.yaml"
        if not target.is_absolute() or target.parts[-3:] != expected_suffix.parts:
            raise _fail(
                f"study {name!r} component summary path does not name its "
                "recorded generation namespace"
            )
        try:
            content = target.read_bytes()
        except OSError as exc:
            raise _fail(f"study {name!r} component summary is missing or unreadable") from exc
        if hashlib.sha256(content).hexdigest() != recorded_sha:
            raise _fail(f"study {name!r} component summary does not match its recorded hash")
        try:
            payload = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise _fail(f"study {name!r} component summary is not parseable") from exc
        if not isinstance(payload, Mapping):
            raise _fail(f"study {name!r} component summary is not a mapping")
        if (
            payload.get("experiment") != record.get("experiment")
            or payload.get("generation_id") != component_generation
        ):
            raise _fail(f"study {name!r} component summary names a different identity")
        if "schema_version" not in payload:
            # Pre-manifest legacy component summary: identity + hash anchor
            # only, mirroring the legacy rule in the publication-time
            # component-manifest chase. Its winners cannot be indexed for the
            # membership check below.
            legacy_component_studies.add(name)
            continue
        phases = payload.get("phases")
        if not isinstance(phases, list):
            raise _fail(f"study {name!r} component summary has no phase list")
        for item in phases:
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                raise _fail(f"study {name!r} component summary has a malformed phase entry")
            phase_name = str(item["name"])
            component_full_winners.add(
                _winner_key(item, phase_name=phase_name, include_exposure_metadata=True)
            )
            component_evidence_winners.add(
                _winner_key(item, phase_name=phase_name, include_exposure_metadata=False)
            )

    for record in records:
        name = str(record["name"])
        if name in legacy_component_studies:
            continue
        phases = record.get("phases")
        if not isinstance(phases, list):
            raise _fail(f"study {name!r} has no phase list")
        for item in phases:
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                raise _fail(f"study {name!r} has a malformed phase entry")
            if item.get("exposed") is not True:
                continue
            source = item.get("winner_source")
            if isinstance(source, Mapping) and source.get("kind") == "suite_baseline":
                source_phase = source.get("phase")
                if not isinstance(source_phase, str) or not source_phase:
                    raise _fail(
                        f"study {name!r} exposed winner for phase {item['name']!r} "
                        "has no baseline source phase"
                    )
                for identity_field in (
                    "trial_number",
                    "generation_id",
                    "attempt_id",
                ):
                    if type(source.get(identity_field)) is not type(
                        item.get(identity_field)
                    ) or source.get(identity_field) != item.get(identity_field):
                        raise _fail(
                            f"study {name!r} exposed winner for phase {item['name']!r} "
                            "does not match its baseline source identity"
                        )
                key = _winner_key(
                    item,
                    phase_name=source_phase,
                    include_exposure_metadata=False,
                )
                anchored = key in component_evidence_winners
            else:
                key = _winner_key(
                    item,
                    phase_name=str(item["name"]),
                    include_exposure_metadata=True,
                )
                anchored = key in component_full_winners
            if not anchored:
                raise _fail(
                    f"study {name!r} exposed winner for phase {item['name']!r} does "
                    "not match any verified component summary"
                )


def _published_suite_summary_path_for(
    suite: Suite,
    published_generation_id: str | None,
) -> Path | None:
    """Resolve the authoritative suite summary from an already-captured published id.

    Takes the caller's already-resolved suite last-success id instead of
    re-reading the pointer, so a caller that has already made a decision from
    one resolution (e.g. the CLI's publication-integrity check) cannot then
    render a summary belonging to a different one.

    :param Suite suite: Suite config with artifact root details.
    :param str | None published_generation_id: Already-resolved suite
        last-success id, or ``None`` when none validated.
    :return Path | None: The generation-scoped suite summary path when
        ``published_generation_id`` is given; the legacy compatibility summary
        path when no suite generation has ever been published; ``None`` when a
        suite generation exists but none has completed successfully yet.
    """
    if published_generation_id is not None:
        return _suite_generation_summary_path(suite, published_generation_id)
    if _suite_generation_path(suite).is_file():
        return None
    return _suite_summary_path(suite)


def _suite_log_path(suite: Suite) -> Path:
    """Path to a suite-level run log.

    :param Suite suite: Suite config with artifact root details.
    :return Path: Path to the suite run log.
    """
    return _suite_dir(suite) / "run.log"


def _write_yaml_atomic(path: Path, payload: Any) -> None:
    """Atomically write a YAML document to ``path``.

    :param Path path: Destination YAML path to replace.
    :param Any payload: YAML-serializable value to write.
    """
    with atomic_text_writer(path) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically write a JSON document to ``path`` with the artifact-tree mode.

    :param Path path: Destination JSON path to replace.
    :param Any payload: JSON-serializable value to write.
    :raises TypeError: ``payload`` is not JSON-serializable.
    :raises OSError: The document could not be staged, written, or renamed.
    """
    with atomic_text_writer(path) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_yaml_exclusive(path: Path, payload: Any) -> bool:
    """Create a YAML document at ``path`` exactly once, never overwriting it.

    Unlike :func:`_write_yaml_atomic` (always-overwrite, used for mutable
    pointers), this is the create-exclusive primitive backing truly immutable
    per-generation lifecycle records (review v0.5.15 / blocker 3): the file is
    opened with ``O_CREAT | O_EXCL`` so a second call for an already-written
    path can never clobber the first write, even a same-content rewrite. This
    is a plain create-once-and-fsync, not a full atomic-rename dance like
    :func:`atomic_text_writer` -- there is nothing to make atomic against a
    concurrent *reader* here, only against a second *writer*, and ``O_EXCL``
    already rules that out.

    :param Path path: Destination path to create; the parent directory is
        created if missing.
    :param Any payload: YAML-serializable value to write.
    :return bool: ``True`` when this call created and wrote the file;
        ``False`` when the destination already existed and nothing was
        written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        raise
    fsync_directory(path.parent)
    return True


@contextlib.contextmanager
def _file_log_handler(path: Path) -> Iterator[None]:
    """Attach a durable file handler for phasesweep logs.

    :param Path path: Log file path to append to.
    :return Iterator[None]: Context manager that removes the handler on exit.
    """
    logger = logging.getLogger("phasesweep")
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname).1s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.setLevel(logging.DEBUG)
    old_level = logger.level
    if old_level in (logging.NOTSET, 0) or old_level > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(old_level)


def _write_trials_csv(study: optuna.Study, path: Path) -> None:
    """Snapshot every trial in ``study`` to ``path`` as stdlib CSV.

    :param optuna.Study study: Study whose trials are serialized.
    :param Path path: Destination CSV path.
    """
    trials = study.get_trials(deepcopy=False)
    if not trials:
        return
    param_names = sorted({n for t in trials for n in t.params})
    attr_names = sorted({n for t in trials for n in t.user_attrs})
    fieldnames = [
        "number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration",
        *[f"param:{n}" for n in param_names],
        *[f"user_attr:{n}" for n in attr_names],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_text_writer(path, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            row: dict[str, Any] = {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "datetime_start": t.datetime_start,
                "datetime_complete": t.datetime_complete,
                "duration": t.duration,
            }
            for n in param_names:
                row[f"param:{n}"] = t.params.get(n)
            for n in attr_names:
                row[f"user_attr:{n}"] = t.user_attrs.get(n)
            writer.writerow(row)


def _save_winner(
    experiment: Experiment,
    phase_name: str,
    winner: Winner,
    *,
    generation_id: str,
) -> None:
    """Persist a phase winner into its immutable generation namespace.

    The phase fingerprint is included so ``_load_winner`` can refuse stale
    winners on ``--from-phase`` resume (review v0.5.6 / blocker 3). Real
    winners always carry a fingerprint by construction in ``_run_phase``;
    placeholder winners (dry-run skip) are never saved.

    ``trainer_env_digest`` / ``trainer_inherit_env`` record which environment
    produced the winning trial (review v0.5.18 / finding F3). Both are
    ``None`` on winners selected from trials that predate the record; neither
    ever carries ambient variable values.

    Args:
        experiment: Parsed experiment config; supplies the metric name used
            in the persisted payload.
        phase_name: Name of the phase whose winner is being saved.
        winner: The winning trial.
        generation_id: Immutable generation namespace to write into. The
            legacy compatibility projection is produced separately by
            :func:`_copy_yaml_projection` once a generation is published.

    """
    path = _generation_winner_path(experiment, generation_id, phase_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase_name,
        "metric": {experiment.metric.name: winner.metric, "goal": experiment.metric.goal},
        **_winner_common_payload(winner, phase_name),
        "phase_fingerprint": winner.phase_fingerprint,
        "objective_provenance": winner.objective_provenance,
        "trainer_env_digest": winner.trainer_env_digest,
        "trainer_inherit_env": winner.trainer_inherit_env,
    }
    _write_yaml_atomic(path, payload)


def _winner_common_payload(winner: Winner, phase_name: str) -> dict[str, Any]:
    """Serialize winner fields shared by persisted and summary representations.

    :param Winner winner: Winner whose common fields should be serialized.
    :param str phase_name: Phase exposed by this winner.
    :return dict[str, Any]: Shared trial, parameter, evidence, identity, and source fields.
    """
    payload = {
        "trial_number": winner.trial_number,
        "params": winner.params,
        "effective_overrides": winner.effective_overrides,
        "constraints": winner.constraints,
        "gates": winner.gates,
        "completion": winner.completion,
        "generation_id": winner.generation_id,
        "attempt_id": winner.attempt_id,
        "winner_source": _winner_source_payload(winner, phase_name),
    }
    if winner.promotion is not None:
        payload["promotion"] = winner.promotion
    return payload


def _winner_source_payload(winner: Winner, phase_name: str) -> dict[str, Any]:
    """Serialize the concrete source trial for an exposed winner.

    :param Winner winner: Winner whose recorded ``source`` is serialized; when
        unset, a ``phase_trial`` source is synthesized from the winner's own fields.
    :param str phase_name: Phase name used to synthesize a fallback source when
        ``winner.source`` is unset.
    :return dict[str, Any]: JSON-serializable winner-source payload with
        ``kind``, ``phase``, ``trial_number``, ``generation_id``, ``attempt_id``,
        and ``study`` keys.
    """
    source = _winner_source_or_default(winner, phase_name)
    return {
        "kind": source.kind,
        "phase": source.phase,
        "trial_number": source.trial_number,
        "generation_id": source.generation_id,
        "attempt_id": source.attempt_id,
        "study": source.study,
    }


def _save_promotion_decision(
    experiment: Experiment,
    phase_name: str,
    decision: dict[str, Any],
    *,
    generation_id: str,
) -> None:
    """Persist a phase promotion decision into its immutable generation namespace.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name whose promotion decision is being saved.
    :param dict[str, Any] decision: Promotion decision payload to persist.
    :param str generation_id: Immutable generation namespace to write into. The
        legacy compatibility projection is produced separately by
        :func:`_copy_yaml_projection` once a generation is published.
    """
    path = _generation_promotion_decision_path(experiment, generation_id, phase_name)
    _write_yaml_atomic(path, decision)


# Warn-once keys for :func:`_warn_environment_drift`. A resume can load the
# same winner twice (preflight, then the run itself) and a suite can inherit it
# across studies; the operator needs the divergence once, not once per read.
_ENVIRONMENT_DRIFT_WARNED: set[tuple[str, str, str]] = set()


def _warn_environment_drift(
    experiment: Experiment,
    phase_name: str,
    stored_digest: str | None,
) -> None:
    """Warn once when an inherited winner was produced under another environment.

    The environment digest is deliberately outside the semantic fingerprint —
    an ambient change must not invalidate a study or block a top-up — so this
    warning is the only signal that a ``--from-phase`` resume is building on a
    result produced under a different trainer environment (review v0.5.18 /
    finding F3). Winners without a recorded digest predate the record and are
    left alone.

    :param Experiment experiment: Parsed experiment supplying the current contract.
    :param str phase_name: Phase whose winner was loaded, used in the warn-once key.
    :param str | None stored_digest: Digest recorded on the loaded winner.
    """
    if stored_digest is None:
        return
    # Deferred: ``engine.trial`` pulls in the evidence/W&B stack, which the
    # read-only paths that import this module never need.
    from phasesweep.engine.trial import _environment_identity

    current_digest = _environment_identity(experiment).digest
    if stored_digest == current_digest:
        return
    key = (experiment.experiment, phase_name, stored_digest)
    if key in _ENVIRONMENT_DRIFT_WARNED:
        return
    _ENVIRONMENT_DRIFT_WARNED.add(key)
    log.warning(
        "[%s] inherited winner ran under trainer environment %s..., but this process "
        "composes %s... under execution.inherit_env=%r. The inherited result is being "
        "reused across an environment change; confirm the difference is irrelevant to "
        "the metric, or re-run the phase.",
        phase_name,
        stored_digest[:12],
        current_digest[:12],
        experiment.execution.inherit_env,
    )


def _load_winner(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> Winner:
    """Load a phase winner from disk and verify it matches the *current* config.

    ``--from-phase`` skips earlier phases by reading their persisted winners.
    Without verification, editing a parent phase's YAML between runs leaves
    the child phase silently inheriting the *old* winner against the *new*
    parent config — a correctness bug, not just a performance one.

    We re-compute the fingerprint of the current parent ``phase`` against the
    currently-resolved ``inherited_winners`` and refuse the load if either
    (a) the stored winner has no fingerprint at all (legacy or hand-edited),
    or (b) the fingerprints disagree (review v0.5.6 / blocker 3).

    A recorded trainer-environment digest that disagrees with this process's
    environment is a warning, not a refusal: the environment is outside the
    semantic fingerprint by design (see :func:`_warn_environment_drift`).
    Winners written before those fields existed load with them set to
    ``None``.

    Args:
        experiment: Parsed experiment config.
        phase: The phase whose winner is being loaded.
        inherited_winners: Winners loaded for phases earlier in the chain;
            contribute to the recomputed fingerprint.

    Returns:
        The reconstructed :class:`Winner` for ``phase``.

    Raises:
        FileNotFoundError: ``winner.yaml`` does not exist for the phase.
        RuntimeError: The file is unfingerprinted (legacy/hand-edited) or its
            fingerprint disagrees with the freshly computed one.

    """
    published_generation_id = _last_successful_generation_id(
        experiment,
        raise_on_manifest_error=True,
    )
    path = _published_winner_path_for(experiment, published_generation_id, phase.name)
    if path is None:
        raise FileNotFoundError(
            f"Winner file missing for phase {phase.name!r}: no generation has completed."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Winner file missing for phase {phase.name!r}: {path}")

    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(
            f"Winner file {path} is invalid or incomplete for skipped phase {phase.name!r}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Winner file {path} is invalid or incomplete for skipped phase "
            f"{phase.name!r}: top level must be a mapping."
        )

    from phasesweep.engine.guards import _phase_fingerprint

    current_fp = _phase_fingerprint(experiment, phase, inherited_winners)
    stored_fp = data.get("phase_fingerprint")

    if stored_fp is None:
        raise RuntimeError(
            f"Winner file {path} has no phase_fingerprint. Refusing to use it "
            f"for --from-phase because phasesweep cannot prove it matches the "
            f"current config for skipped phase {phase.name!r}. Re-run the "
            f"phase, or — if you know the config is unchanged — delete the "
            f"file and re-run to regenerate it with a fingerprint."
        )

    if stored_fp != current_fp:
        raise StudyFingerprintMismatchError(
            f"Winner file {path} was produced by a different phase config "
            f"(stored fingerprint {stored_fp[:16]}... != current "
            f"{current_fp[:16]}...). Re-run phase {phase.name!r}, change the "
            f"experiment name, or restore the matching config before resuming."
        )

    completion = data.get("completion")
    if not isinstance(completion, dict):
        raise RuntimeError(
            f"Winner file {path} is invalid or incomplete for skipped phase "
            f"{phase.name!r}: missing mapping field 'completion'."
        )
    if completion.get("incomplete") is True and not phase.allow_incomplete_on_timeout:
        raise RuntimeError(
            f"Winner file {path} records an incomplete phase result. Refusing to "
            f"use it for skipped phase {phase.name!r} unless the current config "
            "sets allow_incomplete_on_timeout: true."
        )
    generation_id = data.get("generation_id")
    attempt_id = data.get("attempt_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise RuntimeError(
            f"Winner file {path} has no valid generation_id; refusing unscoped evidence."
        )
    if not isinstance(attempt_id, str) or not attempt_id:
        raise RuntimeError(
            f"Winner file {path} has no valid attempt_id; refusing unscoped evidence."
        )
    source_data = data.get("winner_source")
    if not isinstance(source_data, dict):
        raise RuntimeError(
            f"Winner file {path} has no valid winner_source; refusing ambiguous provenance."
        )
    source_kind = source_data.get("kind")
    if source_kind not in ("phase_trial", "promotion_baseline", "suite_baseline"):
        raise RuntimeError(f"Winner file {path} has an invalid winner_source kind.")

    stored_env_digest = data.get("trainer_env_digest")
    if not isinstance(stored_env_digest, str) or not stored_env_digest:
        stored_env_digest = None
    stored_inherit_env = data.get("trainer_inherit_env")
    if not isinstance(stored_inherit_env, str | list):
        stored_inherit_env = None
    _warn_environment_drift(experiment, phase.name, stored_env_digest)

    try:
        source = _parse_winner_source(source_data, cast(WinnerSourceKind, source_kind))
        return Winner(
            trial_number=int(data["trial_number"]),
            params=dict(data["params"]),
            effective_overrides=dict(data["effective_overrides"]),
            metric=float(data["metric"][experiment.metric.name]),
            constraints={k: float(v) for k, v in (data.get("constraints") or {}).items()},
            gates=[item for item in (data.get("gates") or []) if isinstance(item, dict)],
            completion=dict(completion),
            promotion=data.get("promotion") if isinstance(data.get("promotion"), dict) else None,
            phase_fingerprint=str(stored_fp),
            generation_id=generation_id,
            attempt_id=attempt_id,
            source=source,
            objective_provenance=(
                dict(data["objective_provenance"])
                if isinstance(data.get("objective_provenance"), dict)
                else None
            ),
            trainer_env_digest=stored_env_digest,
            trainer_inherit_env=(
                [str(name) for name in stored_inherit_env]
                if isinstance(stored_inherit_env, list)
                else stored_inherit_env
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Winner file {path} is invalid or incomplete for skipped phase {phase.name!r}: {exc}"
        ) from exc
