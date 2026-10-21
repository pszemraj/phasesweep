"""Shared engine winner models and persistence constants."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from phasesweep.config.common import _validate_safe_name
from phasesweep.runtime.json import JsonObject, json_object

if TYPE_CHECKING:
    from optuna.trial import FrozenTrial

    from phasesweep.engine.repeats import RepeatsRecord

WinnerSourceKind = Literal["phase_trial"]

PublicationState = Literal["ok", "absent", "failed", "permission_denied"]
"""Verdict on a last-success pointer, including inaccessible validation evidence."""

TrialStateName = Literal["WAITING", "RUNNING", "COMPLETE", "FAIL", "INTERRUPTED"]
"""Name of a trial state as status payloads report it, including ``INTERRUPTED``."""

TRIAL_STATE_NAMES: tuple[TrialStateName, ...] = (
    "WAITING",
    "RUNNING",
    "COMPLETE",
    "FAIL",
    "INTERRUPTED",
)
"""Every counted trial state, in the key order status payloads use.

``INTERRUPTED`` is an Optuna ``FAIL`` trial whose recorded outcome is
``interrupted``; it is counted apart from ``FAIL`` and reruns. Optuna's
``PRUNED`` is absent because the study's ``NopPruner`` never prunes and no
trainer raises ``TrialPruned``.
"""

TERMINAL_TRIAL_STATES: tuple[TrialStateName, ...] = ("COMPLETE", "FAIL")
"""The states that count toward ``n_trials``; an interrupted trial is in none of them."""

RecoveredTrialState = Literal["FAIL", "INTERRUPTED"]
"""The counted state stale recovery leaves a RUNNING trial in."""


@dataclass(frozen=True)
class WinnerSource:
    """Concrete trial that supplies an exposed winner."""

    kind: WinnerSourceKind
    phase: str
    trial_number: int
    generation_id: str
    attempt_id: str


def _parse_winner_source(data: object, *, expected_phase: str) -> WinnerSource:
    """Parse the exact winner-source key set, recognized kind, and execution identity.

    Require its phase to match ``expected_phase``. Callers apply their own error policy.

    :param object data: Parsed ``winner_source`` block from a persisted
        ``winner.yaml``.
    :param str expected_phase: Phase name the caller expects this source to
        name.
    :raises ValueError: ``data`` is not a mapping, its ``kind`` is not a
        recognized :data:`WinnerSourceKind`, it is missing a required key or
        carries an extra one, a field does not have its expected
        type, or its ``phase`` disagrees with ``expected_phase``.
    :return WinnerSource: The reconstructed source.
    """
    if not isinstance(data, Mapping):
        raise ValueError("no valid winner_source; refusing ambiguous provenance")
    kind = data.get("kind")
    if kind not in get_args(WinnerSourceKind):
        raise ValueError("invalid winner_source kind")
    fields = {"kind", "phase", "trial_number", "generation_id", "attempt_id"}
    if set(data) != fields:
        missing = sorted(fields - set(data))
        unexpected = sorted(str(key) for key in set(data) - fields)
        raise ValueError(
            f"winner_source fields are not the current set (missing {missing}, "
            f"unexpected {unexpected})"
        )
    phase = data.get("phase")
    if not isinstance(phase, str) or not phase:
        raise ValueError("invalid winner_source phase")
    _validate_safe_name("winner_source phase", phase)
    if phase != expected_phase:
        raise ValueError(f"winner_source names phase {phase!r}, expected {expected_phase!r}")
    trial_number = data["trial_number"]
    if not isinstance(trial_number, int) or isinstance(trial_number, bool):
        raise ValueError("invalid winner_source trial_number")
    generation_id = data["generation_id"]
    attempt_id = data["attempt_id"]
    if not isinstance(generation_id, str) or not generation_id:
        raise ValueError("invalid winner_source generation_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("invalid winner_source attempt_id")
    return WinnerSource(
        kind=cast(WinnerSourceKind, kind),
        phase=phase,
        trial_number=trial_number,
        generation_id=generation_id,
        attempt_id=attempt_id,
    )


@dataclass
class Winner:
    """Phase winner with sampled params, composed overrides, metric, and retained evidence.

    The semantic phase fingerprint is verified when a resume loads skipped winners.
    Dry-run placeholders lack execution evidence and are never persisted.
    """

    trial_number: int
    params: dict[str, Any]  # sampled params only
    effective_overrides: dict[str, Any]  # full composed overrides (fixed + inherited + sampled)
    metric: float
    constraints: dict[str, float] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    completion: dict[str, Any] = field(default_factory=dict)
    phase_fingerprint: str | None = None
    generation_id: str | None = None
    attempt_id: str | None = None
    source: WinnerSource | None = None
    # Frozen evidence provenance captured when the winning objective was
    # extracted: extractor config fingerprint plus source digest / frozen
    # remote summary subset.
    objective_provenance: dict[str, Any] | None = None
    # Versioned identity of the generated input consumed by the source trial.
    # Persisted with the winner so --from-phase can verify carried evidence
    # even if the source study is no longer available.
    trainer_input: dict[str, Any] | None = None
    # SHA-256 of the tracked name/value mapping stored on the source trial.
    trainer_env_digest: str | None = None
    # Advisory verdicts on the population the source selection ranked, frozen
    # at that selection (:mod:`phasesweep.engine.confound`). A carried winner
    # keeps them verbatim; no fingerprint reads them.
    confound: dict[str, Any] | None = None
    # Report-only values the selected trial recorded, by observation name; for
    # a repeats phase, the mean over the members that recorded each name.
    # Winner-only: they never enter summary.yaml, a fingerprint, or selection.
    observed: dict[str, float] = field(default_factory=dict)
    # The winning repeat group (:mod:`phasesweep.engine.repeats`): reducer,
    # spread and every member. The scalar fields above then describe the
    # group's highest-numbered member and ``metric`` is the reduced value.
    # ``None`` for a phase without repeats. Winner-only, like ``observed``.
    repeats: RepeatsRecord | None = None
    # The keys of ``effective_overrides`` whose values are dry-run stand-ins:
    # a placeholder's own searched keys plus those it carries from placeholder
    # parents. Empty on every winner a run selects or loads, and never persisted.
    placeholder_keys: frozenset[str] = frozenset()


TRIAL_DIR_ATTR = "phasesweep_trial_dir"
GENERATION_ID_ATTR = "phasesweep_generation_id"
ATTEMPT_ID_ATTR = "phasesweep_attempt_id"
PHASE_FINGERPRINT_ATTR = "phasesweep_fingerprint"
# The canonical payload PHASE_FINGERPRINT_ATTR is the digest of, recorded beside
# it so a refusal can name which keys changed instead of two opaque hashes.
PHASE_FINGERPRINT_PAYLOAD_ATTR = "phasesweep_fingerprint_payload"
STUDY_SCHEMA_ATTR = "phasesweep_study_schema_version"
STUDY_SCHEMA_VERSION = 7
TRIAL_TARGET_ATTR = "phasesweep_trial_target"
# Ordered terminal outcome used to reconstruct the failure circuit breaker
# after a restart. Every terminal trial in a current-schema study has one. The
# outcome that trips a phase abort carries that abort record under
# ``TRIAL_OUTCOME_ABORT_KEY``, so the decision is as durable as the outcome and
# a restarted orchestrator cannot reinterpret the same terminal trials as a
# completed phase.
TRIAL_OUTCOME_ATTR = "phasesweep_trial_outcome"
TRIAL_OUTCOME_SCHEMA_VERSION = 2
TRIAL_OUTCOME_ABORT_KEY = "abort"
# ``interrupted`` is the one label for a trial whose orchestrator stopped it before it
# could finish: a signal, a phase or run deadline, or recovery after the orchestrator died.
# The outcome's ``cause`` says which.
_TRIAL_OUTCOMES = frozenset({"success", "failure", "interrupted", "fatal"})
# On an interrupted rerun or failed grid-cell retry: the source's trial number,
# written in the new attempt's creation record. It takes its source's draw position.
RERUN_OF_ATTR = "phasesweep_rerun_of"
# Durable boundary established when the operator explicitly raises n_trials
# after an abort. Outcomes through this sequence belong to the aborted attempt;
# later failures form the new recovery streak.
PHASE_RECOVERY_ATTR = "phasesweep_phase_recovery"
PHASE_RECOVERY_SCHEMA_VERSION = 1
# Accepted partial-timeout decision; replay selects without allocating the remaining trials.
PHASE_DECISION_ATTR = "phasesweep_phase_decision"
PHASE_DECISION_SCHEMA_VERSION = 1
FEASIBLE_ATTR = "phasesweep_feasible"
GATES_ATTR = "phasesweep_gates"
# JSON-encoded frozen objective evidence provenance; written when metric
# extraction succeeds.
OBJECTIVE_PROVENANCE_ATTR = "phasesweep_objective_provenance"
# Versioned identity of the exact generated input consumed by the trainer.
# The record stores only format, trial-relative filename, byte length, and
# SHA-256; the generated file itself remains the evidence.
TRAINER_INPUT_ATTR = "phasesweep_trainer_input"
TRAINER_INPUT_SCHEMA_VERSION = 1
# The environment cohort key: the SHA-256 of the tracked mapping recorded under
# ``TRAINER_ENV_ATTR``, so the cohort check, winner.yaml, publication validation, the
# allocation context and the drift warning compare one string. It is a checksum of that
# mapping, not a secrecy measure. Written at allocation, so failed trials carry it too.
TRAINER_ENV_DIGEST_ATTR = "phasesweep_trainer_env_digest"
# The tracked trainer environment in plain text, as a name -> value mapping: the base
# names, the ``inherit_env`` list names, the ``env`` entries and the keys PhaseSweep composes
# from config, without the pass-through names (built-in ``TMPDIR`` and
# ``execution.passthrough_env``). It lets a cohort refusal show both values of a differing
# variable. Ledger provenance only: trials.csv and every other export leave it out.
TRAINER_ENV_ATTR = "phasesweep_trainer_env"
RETURN_CODE_ATTR = "phasesweep_return_code"
DURATION_ATTR = "phasesweep_duration_s"
OVERRIDES_ATTR = "phasesweep_overrides"
CLEANUP_CONFIRMED_ATTR = "phasesweep_cleanup_confirmed"
CLEANUP_RECOVERED_TRIALS_ATTR = "phasesweep_cleanup_recovered_trials"
FAILURE_REASON_ATTR = "phasesweep_failure_reason"
# Study-level binding from a persistent study to the one artifact root it
# publishes into: the resolved ``<workdir>/<experiment>`` namespace as a string.
# ``workdir`` stays outside semantic fingerprints, so this binding refuses a
# second divergent publication root. It is claimed only on first contact; an
# existing binding remains authoritative.
ARTIFACT_ROOT_ATTR = "phasesweep_artifact_root_v1"
CONSTRAINT_PREFIX = "constraint:"


def _study_schema_supported(version: object, has_trials: bool) -> bool:
    """Accept the current integer stamp or an empty, unmarked allocation.

    :param object version: Decoded study schema attr, or None when absent.
    :param bool has_trials: Whether the study contains any trials.
    :return bool: Whether the study's schema is supported.
    """
    return (type(version) is int and version == STUDY_SCHEMA_VERSION) or (
        version is None and not has_trials
    )


def constraint_attr(name: str) -> str:
    """Return the persisted user-attr key for a constraint value.

    :return str: ``constraint:`` key for ``name``.
    """
    return f"{CONSTRAINT_PREFIX}{name}"


OBSERVED_PREFIX = "observed:"


def observed_attr(name: str) -> str:
    """Return the persisted user-attr key for a report-only observed value.

    :return str: ``observed:`` key for ``name``.
    """
    return f"{OBSERVED_PREFIX}{name}"


def _validate_observed_block(value: object) -> dict[str, float]:
    """Check a winner's ``observed`` mapping: nonempty string names to finite numbers.

    :param object value: The decoded ``observed`` field of a winner payload.
    :raises ValueError: The value is not a mapping, a name is not a nonempty string,
        or a value is not a finite number.
    :return dict[str, float]: The mapping with every value as a float.
    """
    if not isinstance(value, Mapping):
        raise ValueError("observed must be a mapping of observation name to value")
    observed: dict[str, float] = {}
    for name, number in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"observed name {name!r} must be a nonempty string")
        if isinstance(number, bool) or not isinstance(number, int | float):
            raise ValueError(f"observed value for {name!r} must be a number")
        if not math.isfinite(number):
            raise ValueError(f"observed value for {name!r} must be finite")
        observed[name] = float(number)
    return observed


@dataclass(frozen=True)
class _ParsedTrialOutcome:
    """Validated durable terminal-outcome fields for one trial."""

    sequence: int
    outcome: str
    cause: str | None
    policy: str | None
    abort: JsonObject | None
    """The phase abort this outcome tripped, recorded with the outcome itself."""


def _valid_abort_record(value: object, *, sequence: int) -> bool:
    """Return whether the abort record stored with an outcome is well formed.

    :param object value: Raw abort record carried by a terminal outcome.
    :param int sequence: Completion sequence of the outcome that carries it.
    :return bool: Whether the record names that sequence, a positive trial
        target, and a non-empty policy and cause.
    """
    if not isinstance(value, dict):
        return False
    completion_sequence = value.get("completion_sequence")
    trial_target = value.get("trial_target")
    policy = value.get("policy")
    cause = value.get("cause")
    return (
        value.get("schema_version") == 1
        and type(completion_sequence) is int
        and completion_sequence == sequence
        and type(trial_target) is int
        and trial_target >= 1
        and isinstance(policy, str)
        and bool(policy)
        and isinstance(cause, str)
        and bool(cause)
    )


def _trial_outcome_payload(
    sequence: int,
    outcome: str,
    *,
    cause: str | None = None,
    policy: str | None = None,
    abort: JsonObject | None = None,
) -> JsonObject:
    """Serialize one ordered outcome and its optional diagnostics and abort.

    :param int sequence: Completion sequence assigned under the outcome lock.
    :param str outcome: Terminal outcome category.
    :param str | None cause: Failure explanation when present.
    :param str | None policy: Fatal policy when present.
    :param JsonObject | None abort: Phase abort tripped by this outcome.
    :return JsonObject: Current-format outcome ready for the trial user attribute.
    """
    payload: JsonObject = {
        "schema_version": TRIAL_OUTCOME_SCHEMA_VERSION,
        "sequence": sequence,
        "outcome": outcome,
    }
    if cause is not None:
        payload["cause"] = cause
    if policy is not None:
        payload["policy"] = policy
    if abort is not None:
        payload[TRIAL_OUTCOME_ABORT_KEY] = abort
    return payload


def _parsed_trial_outcome(value: object) -> _ParsedTrialOutcome | None:
    """Return the ordered outcome fields when a trial attr is well formed.

    :param object value: Raw ``TRIAL_OUTCOME_ATTR`` user-attr value to validate.
    :return _ParsedTrialOutcome | None: Validated outcome, including the fatal
        policy needed to distinguish cleanup uncertainty from ordinary errors
        and the abort record a failure or fatal outcome tripped; ``None`` when
        the payload is malformed.
    """
    if not isinstance(value, dict):
        return None
    schema_version = value.get("schema_version")
    sequence = value.get("sequence")
    outcome = value.get("outcome")
    cause = value.get("cause")
    policy = value.get("policy")
    abort = value.get(TRIAL_OUTCOME_ABORT_KEY)
    if (
        schema_version != TRIAL_OUTCOME_SCHEMA_VERSION
        or type(sequence) is not int
        or sequence < 1
        or not isinstance(outcome, str)
        or outcome not in _TRIAL_OUTCOMES
        or (cause is not None and not isinstance(cause, str))
        or (
            policy is not None and (outcome != "fatal" or not isinstance(policy, str) or not policy)
        )
        or (
            abort is not None
            and (
                outcome not in {"failure", "fatal"}
                or not _valid_abort_record(abort, sequence=sequence)
            )
        )
    ):
        return None
    try:
        abort_record = None if abort is None else json_object(abort)
    except ValueError:
        return None
    return _ParsedTrialOutcome(
        sequence=sequence,
        outcome=outcome,
        cause=cause,
        policy=policy,
        abort=abort_record,
    )


def _stale_recovery_outcome(parsed: _ParsedTrialOutcome | None) -> str:
    """Return the durable outcome to record when recovering a RUNNING trial.

    Preserve recorded failure, fatal, or interrupted outcomes. Missing or uncommitted
    success outcomes recover as interrupted. Preview uses the same rule.

    :param _ParsedTrialOutcome | None parsed: The trial's valid, unduplicated recorded
        outcome, or ``None`` when it has none.
    :return str: ``failure``, ``fatal`` or ``interrupted``.
    """
    if parsed is not None and parsed.outcome in {"failure", "fatal", "interrupted"}:
        return parsed.outcome
    return "interrupted"


def _outcome_sequence_counts(trials: Iterable[FrozenTrial]) -> Counter[int]:
    """Count how many trials' valid outcome records carry each sequence.

    :param Iterable[FrozenTrial] trials: Every trial of one study.
    :return Counter[int]: Records per sequence number.
    """
    return Counter(
        parsed.sequence
        for trial in trials
        if (parsed := _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))) is not None
    )


def _unduplicated_outcome(
    trial: FrozenTrial, sequence_counts: Counter[int]
) -> _ParsedTrialOutcome | None:
    """Return a trial's recorded outcome unless another record shares its sequence.

    Stale recovery discards a duplicated record and records ``interrupted`` with a
    fresh sequence, so its prediction must discard the same records.

    :param FrozenTrial trial: Trial whose record is read.
    :param Counter[int] sequence_counts: The study's
        :func:`_outcome_sequence_counts`.
    :return _ParsedTrialOutcome | None: The valid record, or ``None`` when it is
        missing, malformed or duplicated.
    """
    parsed = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    if parsed is None or sequence_counts[parsed.sequence] != 1:
        return None
    return parsed


def _recovered_trial_state(
    trial: FrozenTrial, sequence_counts: Counter[int]
) -> RecoveredTrialState:
    """Predict the counted state stale recovery derives from a RUNNING trial's outcome.

    :param FrozenTrial trial: RUNNING trial to predict for.
    :param Counter[int] sequence_counts: The study's
        :func:`_outcome_sequence_counts`, so a duplicated record is discarded
        as recovery discards it.
    :return RecoveredTrialState: ``FAIL`` when recovery keeps a recorded failure
        or fatal error, otherwise ``INTERRUPTED``.
    """
    outcome = _stale_recovery_outcome(_unduplicated_outcome(trial, sequence_counts))
    return "INTERRUPTED" if outcome == "interrupted" else "FAIL"


def _trial_interrupted(trial: FrozenTrial) -> bool:
    """Return whether a terminal trial was interrupted rather than measured.

    A RUNNING trial whose objective recorded ``interrupted`` is not counted yet;
    stale recovery makes it terminal before any count is taken.

    :param FrozenTrial trial: Trial to classify.
    :return bool: Whether the trial is finished and its recorded outcome is ``interrupted``.
    """
    if not trial.state.is_finished():
        return False
    parsed = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    return parsed is not None and parsed.outcome == "interrupted"


GENERATION_SUMMARY_SCHEMA_VERSION = 4
PUBLICATION_POINTER_SCHEMA_VERSION = 2
# Provenance files frozen into every generation namespace at claim time. The
# summary keeps only the config *fingerprint*: once the operator edits or loses
# the YAML, the digest can prove a mismatch but cannot reconstruct the search
# spaces, fixed overrides, execution context, env, or trial command behind a
# published winner. These files keep that.
GENERATION_CONFIG_SNAPSHOT_FILENAME = "config.snapshot.yaml"
GENERATION_REPRODUCIBILITY_FILENAME = "reproducibility.json"
# The reproducibility record carries ``generation_id_source``: "caller" marks a
# generation whose identity -- and therefore launch authority -- was granted by
# an external launcher, durably enough to survive the loss of that launcher's
# own state directory.
REPRODUCIBILITY_SCHEMA_VERSION = 2

WINNER_FILENAME = "winner.yaml"
# Manifest kinds that name a file in the generation namespace root rather than
# a phase. Their entries carry ``path`` instead of ``phase``; both are required
# in every current-format generation.
_GENERATION_FILE_FILENAMES = {
    "config_snapshot": GENERATION_CONFIG_SNAPSHOT_FILENAME,
    "reproducibility": GENERATION_REPRODUCIBILITY_FILENAME,
}
_MANIFEST_GENERATION_FILE_KINDS = frozenset(_GENERATION_FILE_FILENAMES)
