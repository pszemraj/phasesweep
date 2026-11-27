"""Shared engine winner models and persistence constants."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast, get_args

from phasesweep.config.common import _validate_safe_name
from phasesweep.engine.errors import StudySchemaMismatchError
from phasesweep.runtime.json import JsonObject, finite_nonnegative, json_object

if TYPE_CHECKING:
    from optuna import Study
    from optuna.trial import FrozenTrial, Trial

    from phasesweep.engine.confound import ConfoundBlock
    from phasesweep.engine.repeats import RepeatsRecord
    from phasesweep.runtime.process import AttemptLifecycle

WinnerSourceKind = Literal["phase_trial"]

PublicationState = Literal["ok", "absent", "failed", "permission_denied"]
"""Verdict on a last-success pointer, including inaccessible validation evidence."""

TrialStateName = Literal["WAITING", "RUNNING", "COMPLETE", "INFEASIBLE", "FAIL", "INTERRUPTED"]
"""Name of a trial state as status, trial tables and logs report it."""

TRIAL_STATE_NAMES: tuple[TrialStateName, ...] = (
    "WAITING",
    "RUNNING",
    "COMPLETE",
    "INFEASIBLE",
    "FAIL",
    "INTERRUPTED",
)
"""Every counted trial state, in the key order status payloads use.

``INFEASIBLE`` is an Optuna ``COMPLETE`` trial that violated a constraint or the
metric's bounds, so it can never win. ``INTERRUPTED`` is an Optuna ``FAIL`` trial whose
recorded outcome is ``interrupted``; it is counted apart from ``FAIL`` and reruns.
Optuna's ``PRUNED`` is absent because the study's ``NopPruner`` never prunes and no
trainer raises ``TrialPruned``.
"""

FINISHED_TRIAL_STATES: tuple[TrialStateName, ...] = ("COMPLETE", "INFEASIBLE", "FAIL")
"""The states that count toward ``n_trials``; an interrupted trial is in none of them."""

RecoveredTrialState = Literal["FAIL", "INTERRUPTED", "EVIDENCE_PENDING"]
"""Predicted recovery state; pending successful evidence requires extraction before counting."""


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
    # Address of the study that selected this winner, including its reading settings.
    study_name: str | None = None
    # Highest terminal outcome sequence when this study selected the winner.
    study_outcome_sequence: int | None = None
    generation_id: str | None = None
    attempt_id: str | None = None
    # Name of the trial directory inside its phase that the source attempt ran in.
    trial_dir: str | None = None
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
    confound: ConfoundBlock | None = None
    # Report-only values the selected trial recorded, by observation name; for
    # a repeats phase, the mean over the members that recorded each name.
    # They never enter a fingerprint or selection.
    observed: dict[str, float] = field(default_factory=dict)
    # The winning repeat group (:mod:`phasesweep.engine.repeats`): reducer,
    # spread and every member. The scalar fields above then describe the
    # group's highest-numbered member and ``metric`` is the reduced value.
    # ``None`` for a phase without repeats.
    repeats: RepeatsRecord | None = None
    # The keys of ``effective_overrides`` whose values are dry-run stand-ins:
    # a placeholder's own searched keys plus those it carries from placeholder
    # parents. Empty on every winner a run selects or loads, and never persisted.
    placeholder_keys: frozenset[str] = frozenset()


@dataclass(frozen=True)
class InterruptedReruns:
    """How far a phase's results lean on reruns of interrupted trials (:func:`_interrupted_reruns`).

    ``results`` counts the trials in a state of :data:`FINISHED_TRIAL_STATES`, ``reruns`` those
    of them whose :data:`RERUN_OF_ATTR` chain reaches an interrupted trial, ``earlier_attempt``
    those reruns scored from a value an earlier attempt wrote
    (:data:`EARLIER_ATTEMPT_RESULT_ATTR`), and ``winner`` whether a trial the winner stands on
    is among the reruns. Computed from the ledger at read time and never saved.
    """

    results: int = 0
    reruns: int = 0
    earlier_attempt: int = 0
    winner: bool = False


@dataclass(frozen=True)
class PhaseCounts:
    """A phase's trials by recorded state when the phase ended, as the run's summary reports them.

    ``earlier`` counts the trials of a state in :data:`FINISHED_TRIAL_STATES` that a previous run
    recorded, so it is a part of :attr:`counted`. ``winner_dirs`` holds the trial directory of
    each trial the phase's winner stands on, in trial order, leaving out one that recorded none;
    it is read for the summary and never saved. ``interrupted`` counts the interrupted trials
    still waiting for a rerun (:func:`_unreplaced_attempts`), as the status heading does, and
    ``reruns`` the results that reran one.
    """

    complete: int = 0
    infeasible: int = 0
    failed: int = 0
    interrupted: int = 0
    earlier: int = 0
    winner_dirs: tuple[Path, ...] = ()
    reruns: InterruptedReruns = InterruptedReruns()

    @property
    def counted(self) -> int:
        """Count the trials that count toward ``n_trials``, as the status heading does.

        An interrupted trial is in none of the :data:`FINISHED_TRIAL_STATES`: its rerun counts
        in its place.

        :return int: Trials in a state of :data:`FINISHED_TRIAL_STATES`.
        """
        by_state: dict[TrialStateName, int] = {
            "COMPLETE": self.complete,
            "INFEASIBLE": self.infeasible,
            "FAIL": self.failed,
            "INTERRUPTED": self.interrupted,
        }
        return sum(by_state[state] for state in FINISHED_TRIAL_STATES)


TRIAL_DIR_ATTR = "phasesweep_trial_dir"
GENERATION_ID_ATTR = "phasesweep_generation_id"
ATTEMPT_ID_ATTR = "phasesweep_attempt_id"
# A phase study's identity: its fingerprint, the canonical payload that fingerprint digests
# (so a refusal can name which keys changed instead of two opaque hashes), and every search
# space the study has run under with the draw position each took effect at. One attr, so a
# crash cannot leave a fingerprint beside a payload or history it does not describe.
PHASE_IDENTITY_ATTR = "phasesweep_phase_identity"
PHASE_IDENTITY_SCHEMA_VERSION = 4
STUDY_SCHEMA_ATTR = "phasesweep_study_schema_version"
STUDY_SCHEMA_VERSION = 27
TRIAL_TARGET_ATTR = "phasesweep_trial_target"
# Ordered terminal outcome used to reconstruct the failure circuit breaker
# after a restart. Every terminal trial in a current-schema study has one. The
# outcome that trips a phase abort carries that abort record under
# ``TRIAL_OUTCOME_ABORT_KEY``, so the decision is as durable as the outcome and
# a restarted orchestrator cannot reinterpret the same terminal trials as a
# completed phase. A trial whose values were measured records ``success`` with its metric,
# whatever the bounds say: the failure policy judges the values when it replays the outcomes.
TRIAL_OUTCOME_ATTR = "phasesweep_trial_outcome"
TRIAL_OUTCOME_SCHEMA_VERSION = 4
TRIAL_OUTCOME_ABORT_KEY = "abort"
# ``interrupted`` is the one label for a trial whose orchestrator stopped it before it
# could finish: a signal, a phase or run deadline, or recovery after the orchestrator died.
# The outcome's ``cause`` says which.
_TRIAL_OUTCOMES = frozenset({"success", "failure", "interrupted", "fatal"})
# On an interrupted rerun or failed grid-cell retry: the source's trial number,
# written in the new attempt's creation record. It takes its source's draw position. On a row
# adopted from another study of the phase it names the row of this study the adopted attempt
# reruns, never a row of the study that launched it.
RERUN_OF_ATTR = "phasesweep_rerun_of"
# Durable boundary established when the operator explicitly raises n_trials
# after an abort. Outcomes through this sequence belong to the aborted attempt;
# later failures form the new recovery streak.
PHASE_RECOVERY_ATTR = "phasesweep_phase_recovery"
PHASE_RECOVERY_SCHEMA_VERSION = 1
# Accepted partial-timeout decision; replay selects without allocating the remaining trials.
PHASE_DECISION_ATTR = "phasesweep_phase_decision"
PHASE_DECISION_SCHEMA_VERSION = 3
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
# ``True`` on an interrupted trial whose trainer exited 0 in a directory an earlier attempt
# used and left nothing to score there. Its rerun opens a fresh directory
# (:func:`phasesweep.engine.trial_dirs._rerun_directory_sources`), where nothing predates the
# attempt, so the same verdict cannot recur.
LEFT_NOTHING_TO_SCORE_ATTR = "phasesweep_left_nothing_to_score"
# ``True`` on a measured trial whose value evidence includes a file an earlier attempt of the
# trial wrote and the measured attempt did not rewrite; its reason line says so.
EARLIER_ATTEMPT_RESULT_ATTR = "phasesweep_earlier_attempt_result"
# How the metric extractor names itself in a failed trial's reason: ``<label>: <cause>`` when
# the objective could not be read. The producer and the remedy that recognizes the reason
# share this one string.
METRIC_EXTRACTOR_LABEL = "metric extractor"
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


class SearchSpaceEntry(TypedDict):
    """One search space a phase's study ran under, from the draw position it took effect at."""

    #: The phase's ``search_space`` as its fingerprint payload holds it, key by key.
    search_space: JsonObject
    #: The first draw position drawn under this space; zero for the study's first entry.
    first_position: int


class PhaseIdentityRecord(TypedDict):
    """The identity a phase's study was bound to, under :data:`PHASE_IDENTITY_ATTR`."""

    schema_version: int
    #: SHA-256 of ``payload``.
    fingerprint: str
    #: The canonical fingerprint payload of the config the study is bound to.
    payload: JsonObject
    #: Every search space the study ran under, oldest first; the last is ``payload``'s.
    search_spaces: list[SearchSpaceEntry]


def _parsed_search_space_entry(value: object, *, previous: int | None) -> SearchSpaceEntry:
    """Parse one recorded search-space entry.

    :param object value: Decoded entry.
    :param int | None previous: The previous entry's ``first_position``, or None for the first.
    :raises ValueError: The entry is malformed, or its position does not follow the previous one.
    :return SearchSpaceEntry: The entry.
    """
    if not isinstance(value, dict) or set(value) != {"search_space", "first_position"}:
        raise ValueError("a search-space entry needs exactly search_space and first_position")
    position = value["first_position"]
    if type(position) is not int or (position != 0 if previous is None else position < previous):
        raise ValueError(f"first_position {position!r} does not follow {previous!r}")
    return {"search_space": json_object(value["search_space"]), "first_position": position}


def _recorded_identity(study: Study) -> PhaseIdentityRecord | None:
    """Read the identity a phase's study was bound to.

    :param Study study: A phase's study, live or a read-only snapshot.
    :raises StudySchemaMismatchError: The recorded identity is malformed.
    :return PhaseIdentityRecord | None: The record, or None when the study has none yet.
    """
    return _parsed_phase_identity(study.user_attrs.get(PHASE_IDENTITY_ATTR), study.study_name)


def _parsed_phase_identity(value: object, study_name: str) -> PhaseIdentityRecord | None:
    """Parse a study's recorded identity.

    :param object value: Stored identity record, or None.
    :param str study_name: Study named in a schema refusal.
    :raises StudySchemaMismatchError: The identity is malformed.
    :return PhaseIdentityRecord | None: The identity, or None when absent.
    """
    if value is None:
        return None
    try:
        if not isinstance(value, dict) or set(value) != PhaseIdentityRecord.__required_keys__:
            raise ValueError("the record's fields differ from the current format")
        if value["schema_version"] != PHASE_IDENTITY_SCHEMA_VERSION:
            raise ValueError(f"schema_version {value['schema_version']!r}")
        fingerprint, payload, entries = (
            value["fingerprint"],
            value["payload"],
            value["search_spaces"],
        )
        if not isinstance(fingerprint, str) or not isinstance(entries, list) or not entries:
            raise ValueError("fingerprint or search_spaces is malformed")
        parsed: list[SearchSpaceEntry] = []
        for entry in entries:
            previous = parsed[-1]["first_position"] if parsed else None
            parsed.append(_parsed_search_space_entry(entry, previous=previous))
        return {
            "schema_version": PHASE_IDENTITY_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "payload": json_object(payload),
            "search_spaces": parsed,
        }
    except (KeyError, ValueError) as exc:
        raise StudySchemaMismatchError(
            f"Study {study_name!r} has a malformed {PHASE_IDENTITY_ATTR!r} record: {exc}."
        ) from exc


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
        try:
            finite = math.isfinite(number)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"observed value for {name!r} must be finite")
        observed[name] = float(number)
    return observed


@dataclass(frozen=True)
class _ParsedTrialOutcome:
    """Validated durable terminal-outcome fields for one trial."""

    sequence: int
    outcome: str
    metric: float | None
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
    metric: float | None = None,
    cause: str | None = None,
    policy: str | None = None,
    abort: JsonObject | None = None,
) -> JsonObject:
    """Serialize one ordered outcome and its optional diagnostics and abort.

    :param int sequence: Completion sequence assigned under the outcome lock.
    :param str outcome: Terminal outcome category.
    :param float | None metric: Measured scalar to tell Optuna, set exactly on a ``success``.
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
    if metric is not None:
        payload["metric"] = metric
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
        and the abort record a failure, fatal or out-of-bounds success outcome tripped;
        ``None`` when the payload is malformed.
    """
    if not isinstance(value, dict):
        return None
    schema_version = value.get("schema_version")
    sequence = value.get("sequence")
    outcome = value.get("outcome")
    metric = value.get("metric")
    cause = value.get("cause")
    policy = value.get("policy")
    abort = value.get(TRIAL_OUTCOME_ABORT_KEY)
    if (
        schema_version != TRIAL_OUTCOME_SCHEMA_VERSION
        or type(sequence) is not int
        or sequence < 1
        or not isinstance(outcome, str)
        or outcome not in _TRIAL_OUTCOMES
        or (outcome == "success") != (metric is not None)
        or (cause is not None and not isinstance(cause, str))
        or (
            policy is not None and (outcome != "fatal" or not isinstance(policy, str) or not policy)
        )
        or (
            abort is not None
            and (outcome == "interrupted" or not _valid_abort_record(abort, sequence=sequence))
        )
    ):
        return None
    try:
        if metric is not None and (
            isinstance(metric, bool)
            or not isinstance(metric, int | float)
            or not math.isfinite(metric)
        ):
            return None
        abort_record = None if abort is None else json_object(abort)
    except (ValueError, OverflowError):
        return None
    return _ParsedTrialOutcome(
        sequence=sequence,
        outcome=outcome,
        metric=None if metric is None else float(metric),
        cause=cause,
        policy=policy,
        abort=abort_record,
    )


def _unique_trial_outcomes(
    study_name: str, trials: Iterable[FrozenTrial]
) -> dict[int, _ParsedTrialOutcome]:
    """Read valid outcomes and reject duplicate completion sequences.

    :param str study_name: Study to name if its completion order is ambiguous.
    :param Iterable[FrozenTrial] trials: Trials from that study.
    :raises StudySchemaMismatchError: Two outcomes share a completion sequence.
    :return dict[int, _ParsedTrialOutcome]: Parsed outcomes keyed by trial number.
    """
    outcomes: dict[int, _ParsedTrialOutcome] = {}
    sequences: dict[int, int] = {}
    for trial in trials:
        parsed = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
        if parsed is None:
            continue
        other = sequences.get(parsed.sequence)
        if other is not None:
            raise StudySchemaMismatchError(
                f"In study {study_name!r}, trials {other} and {trial.number} "
                f"both use completion sequence {parsed.sequence}."
            )
        sequences[parsed.sequence] = trial.number
        outcomes[trial.number] = parsed
    return outcomes


def _completed_trainer_duration(
    trial: FrozenTrial | Trial,
    lifecycle: AttemptLifecycle | None,
    *,
    rechecking_evidence: bool = False,
) -> float | None:
    """Return completed trainer time without confusing SDK worker exits with trainer exits.

    Evidence workers reuse the process slot after trainer exit. The ledger records
    trainer metadata before extraction; the guardian record covers an earlier runner death,
    and only a record of the trainer's own ending carries its exit.

    :param FrozenTrial | Trial trial: Trial carrying trainer exit attributes, when committed.
    :param AttemptLifecycle | None lifecycle: Current process-slot evidence, or None to require trainer ledger attributes.
    :param bool rechecking_evidence: A committed outcome can record an extraction failure
        after a successful trainer exit. Count that trainer for a re-read, but skip a trainer
        interrupted by its per-trial timeout, even if it exited cleanly on termination.
    :return float | None: Successful trainer duration, or None when exit evidence is incomplete or invalid.
    """
    if FAILURE_REASON_ATTR in trial.user_attrs:
        from phasesweep.runtime.process import TIMEOUT_REASON_PREFIX

        reason = trial.user_attrs[FAILURE_REASON_ATTR]
        if not rechecking_evidence or (
            isinstance(reason, str) and reason.startswith(TIMEOUT_REASON_PREFIX)
        ):
            return None
    record = lifecycle if lifecycle is not None and lifecycle.ended_by == "trainer" else None
    return_code = trial.user_attrs.get(
        RETURN_CODE_ATTR, None if record is None else record.return_code
    )
    duration = trial.user_attrs.get(
        DURATION_ATTR, None if record is None else record.duration_seconds
    )
    if type(return_code) is not int or return_code != 0:
        return None
    seconds = finite_nonnegative(duration)
    return None if seconds is None else float(seconds)


def _trial_lifecycle(trial: FrozenTrial | Trial) -> AttemptLifecycle | None:
    """Read the attempt lifecycle record in the directory a trial recorded for its attempt.

    :param FrozenTrial | Trial trial: Trial whose directory and attempt id attributes name the record.
    :raises ValueError: The recorded trial directory is malformed, or the record is
        unreadable, malformed or belongs to another attempt. A caller that acts on the process
        treats that as cleanup uncertainty, not as absence.
    :return AttemptLifecycle | None: The validated record, or ``None`` when the trial recorded
        no directory or attempt id, or its directory holds no record.
    """
    from phasesweep.runtime.process import read_attempt_lifecycle

    directory = _trial_attempt_dir(trial)
    attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if directory is None or not isinstance(attempt_id, str):
        return None
    return read_attempt_lifecycle(directory, expected_attempt_id=attempt_id)


def _trial_attempt_dir(trial: FrozenTrial | Trial) -> Path | None:
    """Return the records directory of the attempt a trial recorded, beside its trial directory.

    :param FrozenTrial | Trial trial: Trial whose directory and attempt id attributes name it.
    :raises ValueError: The recorded trial directory is malformed.
    :return Path | None: The attempt directory, or ``None`` when either attribute is missing.
    """
    from phasesweep.engine.paths import _attempt_dir

    directory = _recorded_trial_dir(trial)
    attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if directory is None or not isinstance(attempt_id, str) or not attempt_id:
        return None
    return _attempt_dir(directory.parent, attempt_id)


def _recovered_trial_state(
    trial: FrozenTrial, outcome: _ParsedTrialOutcome | None
) -> RecoveredTrialState:
    """Predict the recovery disposition of a RUNNING trial.

    A lifecycle record that cannot be read predicts an interruption: this only previews what
    recovery will settle, and recovery itself refuses the unreadable record.

    :param FrozenTrial trial: Trial whose finished training may await extraction.
    :param _ParsedTrialOutcome | None outcome: Its unique committed outcome, if any.
    :return RecoveredTrialState: Pending evidence, a recorded failure or a trainer's own
        crash, or an interruption.
    """
    if outcome is not None:
        if outcome.metric is not None:
            return "EVIDENCE_PENDING"
        return "INTERRUPTED" if outcome.outcome == "interrupted" else "FAIL"
    from phasesweep.engine.trial import _trainer_failure_reason

    try:
        lifecycle = _trial_lifecycle(trial)
    except ValueError:
        return "INTERRUPTED"
    if lifecycle is not None and _completed_trainer_duration(trial, lifecycle) is not None:
        return "EVIDENCE_PENDING"
    directory = _trial_attempt_dir(trial)
    if directory is not None and _trainer_failure_reason(trial, lifecycle, directory) is not None:
        return "FAIL"
    return "INTERRUPTED"


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


def _unreplaced_attempts(trials: Iterable[FrozenTrial], *, interrupted: bool) -> list[FrozenTrial]:
    """Return the attempts no rerun names, either the interrupted ones or all the others.

    A rerun names the attempt it replaces (:data:`RERUN_OF_ATTR`): an interrupted trial, or a
    failed grid cell a raised target retries. Of the attempts no rerun names, an interrupted
    one still waits for its rerun, and every other one stands for its cell: the cell's
    measurement, its latest failure, or an attempt still queued or running.

    :param Iterable[FrozenTrial] trials: Every trial of one study.
    :param bool interrupted: ``True`` for the interrupted attempts that wait for a rerun,
        ``False`` for the attempts that stand for their cell.
    :return list[FrozenTrial]: Those attempts, in the order ``trials`` gives them.
    """
    trials = list(trials)
    named = {trial.user_attrs.get(RERUN_OF_ATTR) for trial in trials}
    return [
        trial
        for trial in trials
        if trial.number not in named and _trial_interrupted(trial) is interrupted
    ]


def _interrupted_reruns(
    trials: Iterable[FrozenTrial], standing: Mapping[int, str | None] | None = None
) -> InterruptedReruns:
    """Count a phase's results that rerun an interrupted trial, the one rule every report uses.

    A result is a trial that counts toward ``n_trials``. It reran an interrupted trial when
    any attempt its :data:`RERUN_OF_ATTR` chain names was interrupted, so a failed grid cell's
    retry of an interrupted cell counts too.

    :param Iterable[FrozenTrial] trials: Every trial of one study.
    :param Mapping[int, str | None] | None standing: The winner's trials
        (:func:`_winning_trials`), or ``None`` where the caller has no winner, defaults to None.
    :return InterruptedReruns: The counts, all zero for a phase no interruption touched.
    """
    by_number = {trial.number: trial for trial in trials}
    results = [
        trial
        for trial in by_number.values()
        if trial.state.name in FINISHED_TRIAL_STATES and not _trial_interrupted(trial)
    ]
    reruns = []
    for trial in results:
        current = trial
        # A rerun always names an earlier trial, so the walk ends.
        while (source := by_number.get(current.user_attrs.get(RERUN_OF_ATTR, -1))) is not None:
            if source.number >= current.number or _trial_interrupted(source):
                break
            current = source
        if source is not None and _trial_interrupted(source):
            reruns.append(trial)
    return InterruptedReruns(
        results=len(results),
        reruns=len(reruns),
        earlier_attempt=sum(
            1 for trial in reruns if trial.user_attrs.get(EARLIER_ATTEMPT_RESULT_ATTR) is True
        ),
        winner=standing is not None
        and any(_stands_under_winner(trial, standing) for trial in reruns),
    )


def _winning_trials(
    trial_number: int, attempt_id: str | None, repeats: RepeatsRecord | None
) -> dict[int, str | None]:
    """Map the trial numbers a phase winner stands on to the attempt each must carry.

    A repeats winner stands on every member of its group, each with the attempt it was ranked
    on; any other winner stands on its one trial.

    :param int trial_number: The winner's trial number; for a repeats winner, its
        highest-numbered member's.
    :param str | None attempt_id: That trial's attempt, or ``None`` where only the number is known.
    :param RepeatsRecord | None repeats: The winner's repeats block, if its phase has repeats.
    :return dict[int, str | None]: The attempt id to require by trial number, for
        :func:`_stands_under_winner`.
    """
    if repeats is None:
        return {trial_number: attempt_id}
    return {member["trial_number"]: member["attempt_id"] for member in repeats["members"]}


def _stands_under_winner(trial: FrozenTrial, standing: Mapping[int, str | None]) -> bool:
    """Say whether a trial is one a phase winner stands on, the attempt included.

    :param FrozenTrial trial: Trial to test.
    :param Mapping[int, str | None] standing: The winner's trials by number, each with the
        attempt id the trial must carry, ``None`` where only the number is known.
    :return bool: ``True`` when the number is listed and any attempt id recorded for it matches.
    """
    if trial.number not in standing:
        return False
    attempt = standing[trial.number]
    return attempt is None or trial.user_attrs.get(ATTEMPT_ID_ATTR) == attempt


def _recorded_trial_dir(trial: FrozenTrial | Trial) -> Path | None:
    """Return the trial directory a trial recorded when it launched, if it recorded one.

    :param FrozenTrial | Trial trial: Trial to read.
    :raises ValueError: The attribute is present but malformed (:func:`_required_trial_dir`);
        a damaged record is never read as no record.
    :return Path | None: The directory, or ``None`` when the attribute is absent.
    """
    if TRIAL_DIR_ATTR not in trial.user_attrs:
        return None
    return _required_trial_dir(trial)


def _required_trial_dir(trial: FrozenTrial | Trial) -> Path:
    """Return the trial directory a trial recorded when it launched.

    PhaseSweep records the absolute path of the directory it built, so any other value is a
    damaged ledger.

    :param FrozenTrial | Trial trial: Trial to read.
    :raises ValueError: The attribute is absent, or not a non-empty absolute path string.
    :return Path: The directory.
    """
    stored = trial.user_attrs.get(TRIAL_DIR_ATTR)
    if not isinstance(stored, str) or not stored or not Path(stored).is_absolute():
        raise ValueError(
            f"its {TRIAL_DIR_ATTR!r} user attribute is missing or invalid ({stored!r})"
        )
    return Path(stored)


GENERATION_SUMMARY_SCHEMA_VERSION = 10
PUBLICATION_POINTER_SCHEMA_VERSION = 2
# Configuration and reproducibility files frozen into every generation namespace at claim time.
GENERATION_CONFIG_SNAPSHOT_FILENAME = "config.snapshot.yaml"
GENERATION_REPRODUCIBILITY_FILENAME = "reproducibility.json"
# generation_id_source records whether the caller supplied the generation ID or the engine minted it.
REPRODUCIBILITY_SCHEMA_VERSION = 4

WINNER_FILENAME = "winner.yaml"
# Manifest kinds that name a file in the generation namespace root rather than
# a phase. Their entries carry ``path`` instead of ``phase``; both are required
# in every current-format generation.
_GENERATION_FILE_FILENAMES = {
    "config_snapshot": GENERATION_CONFIG_SNAPSHOT_FILENAME,
    "reproducibility": GENERATION_REPRODUCIBILITY_FILENAME,
}
_MANIFEST_GENERATION_FILE_KINDS = frozenset(_GENERATION_FILE_FILENAMES)
