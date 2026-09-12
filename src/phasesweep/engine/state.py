"""Shared engine winner models and persistence constants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from phasesweep.engine.read import PhaseWinnerView

WinnerSourceKind = Literal["phase_trial", "promotion_baseline", "suite_baseline"]

PublicationState = Literal["ok", "absent", "failed", "permission_denied"]
"""Verdict on a last-success pointer, including inaccessible validation evidence."""


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

    Shared by :func:`phasesweep.engine.artifacts._load_winner` and
    :func:`phasesweep.engine.read.read_winner`, which differ only in what they
    do when this raises: the former wraps it as
    :class:`phasesweep.engine.errors.WinnerIntegrityError`,
    while the latter treats the winner as absent. Callers must validate ``source_kind``
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
    # Identity of the semantic environment the winning trial ran under: the
    # SHA-256 of its composed trainer environment after explicitly classified
    # ``passthrough_env`` values are removed, plus the ``inherit_env`` contract.
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
# Versioned identity of the exact generated input consumed by the trainer.
# The record stores only format, trial-relative filename, byte length, and
# SHA-256; the generated file itself remains the evidence.
TRAINER_INPUT_ATTR = "phasesweep_trainer_input"
TRAINER_INPUT_SCHEMA_VERSION = 1
# SHA-256 of the semantic trainer environment, excluding values explicitly
# classified under ``execution.passthrough_env``. Written at allocation, so
# failed trials carry it too. Ambient VALUES are never stored here; the names
# attr lists the complete base environment (including pass-through names), and
# raw values land only in the opt-in owner-only ``environment.json``.
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


GENERATION_SUMMARY_SCHEMA_VERSION = 2
SUITE_SUMMARY_SCHEMA_VERSION = 3
PUBLICATION_POINTER_SCHEMA_VERSION = 2
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
