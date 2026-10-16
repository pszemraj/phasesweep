"""Persisted study policy and continuation validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypedDict

import optuna

from phasesweep.config.models import Phase
from phasesweep.config.search import fixed_configuration_target_refusal
from phasesweep.engine.errors import (
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    TrialTargetRefusedError,
)
from phasesweep.engine.fingerprints import (
    _ABSENT,
    _MAX_REPORTED_CHANGES,
    _changed_payload_keys,
    _render_changed_values,
)
from phasesweep.engine.optuna import _completed_trial_count, _finished_trial_count
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GENERATION_ID_ATTR,
    PHASE_DECISION_ATTR,
    PHASE_DECISION_SCHEMA_VERSION,
    PHASE_RECOVERY_ATTR,
    PHASE_RECOVERY_SCHEMA_VERSION,
    RERUN_OF_ATTR,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRAINER_ENV_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_TARGET_ATTR,
    _parsed_trial_outcome,
    _trial_interrupted,
)
from phasesweep.engine.trial import EnvironmentIdentity
from phasesweep.errors import OperatorAction

_ALLOCATION_CONTEXT_ATTR = "phasesweep_allocation_context"
_ALLOCATION_CONTEXT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _PhasePolicyState:
    """Failure-policy state reconstructed from durable per-trial outcomes."""

    max_sequence: int
    consecutive_failures: int
    abort: dict[str, Any] | None
    """The first abort recorded after the recovery boundary, if any. A later
    outcome resets the streak but never this recorded decision."""


def _next_consecutive_failures(consecutive_failures: int, outcome: str) -> int:
    """Apply a terminal outcome to the consecutive-failure count.

    :param int consecutive_failures: Count observed before ``outcome``.
    :param str outcome: Terminal outcome. ``"failure"`` or ``"fatal"``
        increments the count, ``"success"`` resets it to zero, and any other
        outcome (``"interrupted"``) leaves it unchanged.
    :return int: The count after applying ``outcome``.
    """
    if outcome in {"failure", "fatal"}:
        return consecutive_failures + 1
    if outcome == "success":
        return 0
    return consecutive_failures


def _consecutive_failure_threshold_tripped(consecutive_failures: int, phase: Phase) -> bool:
    """Check the live or resumed failure count against the configured threshold.

    Replay uses the abort recorded with each outcome, not today's threshold.

    :param int consecutive_failures: Current consecutive-failure count.
    :param Phase phase: Phase config supplying ``max_consecutive_failures``.
    :return bool: True when ``consecutive_failures`` meets or exceeds the
        phase's threshold.
    """
    return consecutive_failures >= phase.max_consecutive_failures


@dataclass(frozen=True)
class _AcceptedPartialDecision:
    """Durable terminal decision to select from an incomplete timed-out phase."""

    trial_target: int
    outcome_sequence: int
    finished_trials: int
    completed_trials: int
    timeout_scope: str


class _AllocationRecord(TypedDict):
    """One validated pre-Optuna allocation boundary and its trainer environment."""

    generation_id: str
    first_trial_number: int
    trainer_environment: str


def _load_accepted_partial_decision(
    study: optuna.Study,
) -> _AcceptedPartialDecision | None:
    """Load and validate a persisted accepted-partial timeout decision.

    :param optuna.Study study: Study whose terminal phase decision is inspected.
    :raises StudySchemaMismatchError: The stored decision has an unsupported or
        internally inconsistent shape.
    :return _AcceptedPartialDecision | None: Validated decision, or ``None``
        when the study has no accepted-partial decision.
    """
    raw = study.user_attrs.get(PHASE_DECISION_ATTR)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} has malformed {PHASE_DECISION_ATTR!r}={raw!r}. "
            "Use a new experiment name, or archive/delete the inconsistent study."
        )
    trial_target = raw.get("trial_target")
    outcome_sequence = raw.get("outcome_sequence")
    finished_trials = raw.get("finished_trials")
    completed_trials = raw.get("completed_trials")
    timeout_scope = raw.get("timeout_scope")
    recovered_abort_sequence = raw.get("recovered_abort_sequence")
    if (
        raw.get("schema_version") != PHASE_DECISION_SCHEMA_VERSION
        or raw.get("decision") != "accepted_partial_timeout"
        or type(trial_target) is not int
        or trial_target < 1
        or type(outcome_sequence) is not int
        or outcome_sequence < 0
        or type(finished_trials) is not int
        or finished_trials < 0
        or finished_trials >= trial_target
        or type(completed_trials) is not int
        or completed_trials < 0
        or completed_trials > finished_trials
        or timeout_scope not in {"phase", "run"}
        or (
            recovered_abort_sequence is not None
            and (
                type(recovered_abort_sequence) is not int
                or recovered_abort_sequence < 1
                or recovered_abort_sequence > outcome_sequence
            )
        )
    ):
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} has malformed {PHASE_DECISION_ATTR!r} fields: "
            f"{raw!r}. Use a new experiment name, or archive/delete the inconsistent study."
        )
    return _AcceptedPartialDecision(
        trial_target=trial_target,
        outcome_sequence=outcome_sequence,
        finished_trials=finished_trials,
        completed_trials=completed_trials,
        timeout_scope=timeout_scope,
    )


def _phase_policy_schema_error(study: optuna.Study, detail: str) -> StudySchemaMismatchError:
    """Build the actionable error used for malformed durable policy state.

    :param optuna.Study study: Study whose durable state is inconsistent,
        used to name the study in the message.
    :param str detail: Specific description of the malformed state.
    :return StudySchemaMismatchError: Constructed error for the caller to
        raise, instructing them to use a new experiment name or
        archive/delete the inconsistent study.
    """
    return StudySchemaMismatchError(
        f"Study {study.study_name!r} has invalid durable failure-policy state: {detail}. "
        "Use a new experiment name, or archive/delete the inconsistent study before "
        "running again."
    )


class _RecoveryRecord(NamedTuple):
    """A validated ``PHASE_RECOVERY_ATTR`` record."""

    start_after_sequence: int
    trial_target: int


def _load_recovery_record(study: optuna.Study) -> _RecoveryRecord | None:
    """Validate the study's recovery record.

    :param optuna.Study study: Study whose ``PHASE_RECOVERY_ATTR`` is read.
    :raises StudySchemaMismatchError: The record is malformed.
    :return _RecoveryRecord | None: The boundary and accepted target it
        records, or ``None`` when no abort was ever recovered.
    """
    recovery = study.user_attrs.get(PHASE_RECOVERY_ATTR)
    if recovery is None:
        return None
    if not isinstance(recovery, dict):
        raise _phase_policy_schema_error(
            study, f"{PHASE_RECOVERY_ATTR!r} must be an object, got {recovery!r}"
        )
    schema_version = recovery.get("schema_version")
    start_after_sequence = recovery.get("start_after_sequence")
    recovered_abort_sequence = recovery.get("recovered_abort_sequence")
    trial_target = recovery.get("trial_target")
    if (
        schema_version != PHASE_RECOVERY_SCHEMA_VERSION
        or type(start_after_sequence) is not int
        or start_after_sequence < 1
        or type(recovered_abort_sequence) is not int
        or recovered_abort_sequence < 1
        or recovered_abort_sequence > start_after_sequence
        or type(trial_target) is not int
        or trial_target < 1
    ):
        raise _phase_policy_schema_error(
            study, f"{PHASE_RECOVERY_ATTR!r} has malformed fields: {recovery!r}"
        )
    return _RecoveryRecord(start_after_sequence, trial_target)


def _load_phase_policy_state(study: optuna.Study) -> _PhasePolicyState:
    """Validate and reconstruct the durable consecutive-failure state.

    :param optuna.Study study: Study whose ``PHASE_RECOVERY_ATTR`` and
        per-trial outcome attrs are read and validated.
    :return _PhasePolicyState: Reconstructed state: the highest recorded
        outcome sequence, the consecutive-failure count since the last
        recovery boundary, and the first abort recorded after it (if any).
    :raises StudySchemaMismatchError: ``PHASE_RECOVERY_ATTR`` or a terminal
        trial's outcome attr is malformed, two trials share a completion
        sequence, or the recovery boundary exceeds the largest recorded
        sequence.
    """
    recovery = _load_recovery_record(study)
    recovery_boundary = 0 if recovery is None else recovery.start_after_sequence

    events: list[tuple[int, str, dict[str, Any] | None]] = []
    seen_sequences: dict[int, int] = {}
    for trial in study.get_trials(deepcopy=False):
        raw = trial.user_attrs.get(TRIAL_OUTCOME_ATTR)
        parsed = _parsed_trial_outcome(raw)
        if trial.state.is_finished() and parsed is None:
            raise _phase_policy_schema_error(
                study,
                f"terminal trial {trial.number} has missing or malformed "
                f"{TRIAL_OUTCOME_ATTR!r}: {raw!r}",
            )
        if parsed is None:
            continue
        sequence = parsed.sequence
        other_trial = seen_sequences.get(sequence)
        if other_trial is not None:
            raise _phase_policy_schema_error(
                study,
                f"trials {other_trial} and {trial.number} both use completion sequence {sequence}",
            )
        seen_sequences[sequence] = trial.number
        events.append((sequence, parsed.outcome, parsed.abort))

    events.sort(key=lambda event: event[0])
    max_sequence = events[-1][0] if events else 0
    if recovery_boundary > max_sequence:
        raise _phase_policy_schema_error(
            study,
            f"{PHASE_RECOVERY_ATTR!r} starts after sequence {recovery_boundary}, "
            f"but the largest recorded sequence is {max_sequence}",
        )

    consecutive_failures = 0
    abort: dict[str, Any] | None = None
    for sequence, outcome, recorded_abort in events:
        if sequence <= recovery_boundary:
            continue
        consecutive_failures = _next_consecutive_failures(consecutive_failures, outcome)
        if abort is None:
            abort = recorded_abort

    return _PhasePolicyState(
        max_sequence=max_sequence,
        consecutive_failures=consecutive_failures,
        abort=abort,
    )


def _record_recovery_boundary(
    study: optuna.Study,
    phase: Phase,
    recovered_abort_sequence: int,
) -> None:
    """Durably acknowledge an abort and accept the raised trial target in one write.

    Outcomes through the current maximum sequence no longer count toward the failure
    streak or supply its active abort. Record the consumed abort and ``phase.n_trials``.

    :param optuna.Study study: Study whose recovery boundary is written.
    :param Phase phase: Phase whose ``n_trials`` the recovery runs toward.
    :param int recovered_abort_sequence: Completion sequence of the abort
        being acknowledged.
    :raises StudySchemaMismatchError: The study's durable failure-policy state
        is malformed.
    """
    current = _load_phase_policy_state(study)
    study.set_user_attr(
        PHASE_RECOVERY_ATTR,
        {
            "schema_version": PHASE_RECOVERY_SCHEMA_VERSION,
            "recovered_abort_sequence": recovered_abort_sequence,
            "start_after_sequence": current.max_sequence,
            "trial_target": phase.n_trials,
        },
    )


def _validate_study_schema(study: optuna.Study) -> None:
    """Initialize an empty study or reject populated incompatible storage.

    :param optuna.Study study: Study whose durable schema attr is stamped (when
        empty and unstamped) or validated against the current schema version.
    :raises StudySchemaMismatchError: The study already holds trials or a schema
        stamp from an unsupported version, or its durable failure-policy state
        is malformed.
    """
    trials = study.get_trials(deepcopy=False)
    version = study.user_attrs.get(STUDY_SCHEMA_ATTR)
    if not trials and version is None:
        study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)
        return
    if version == STUDY_SCHEMA_VERSION:
        _load_phase_policy_state(study)
        return

    trial_numbers = [trial.number for trial in trials]
    detail = "missing" if version is None else repr(version)
    raise StudySchemaMismatchError(
        f"Study {study.study_name!r} uses unsupported phasesweep storage schema {detail}; "
        f"current schema is {STUDY_SCHEMA_VERSION}. Affected trial numbers: {trial_numbers}. "
        "Use a fresh artifact root and fresh local storage with this PhaseSweep release.",
    )


def _validate_study_direction(
    study: optuna.Study,
    goal: Literal["minimize", "maximize"],
) -> None:
    """Require stored objective direction to match config before trial launch.

    Optuna's ``load_if_exists`` retains the stored direction without checking the offered one.

    :param optuna.Study study: Existing or newly created single-objective study.
    :param Literal goal: Direction required by the experiment metric.
    :raises StudySchemaMismatchError: The stored direction does not match ``goal``.
    """
    expected = (
        optuna.study.StudyDirection.MINIMIZE
        if goal == "minimize"
        else optuna.study.StudyDirection.MAXIMIZE
    )
    if study.directions == [expected]:
        return
    stored = ", ".join(direction.name.lower() for direction in study.directions)
    raise StudySchemaMismatchError(
        f"Study {study.study_name!r} optimizes {stored or 'no direction'}, but the current "
        f"config requires {goal}. Optuna does not change a persistent study's direction "
        "when load_if_exists=True. Use a new experiment/phase name or remove the "
        "incompatible study before running again."
    )


def _accepted_trial_target(study: optuna.Study) -> int:
    """Return the durable target for a current-format study.

    A recovery record accepts its target in the same write that retires an
    abort (:func:`_record_recovery_boundary`), and that write lands before
    ``TRIAL_TARGET_ATTR`` catches up. Reading both keeps a crash between the
    two from dropping the recovery's authorization or letting the old target
    run the recovered phase.

    :param optuna.Study study: Study whose accepted trial target is read.
    :return int: The larger of the stored ``phasesweep_trial_target`` user
        attr and the recovery record's target, or zero for a newly initialized
        empty study.
    :raises StudySchemaMismatchError: The stored target is not a positive int,
        is lower than the number of already-finished trials, or the recovery
        record is malformed.
    """
    finished = _finished_trial_count(study.get_trials(deepcopy=False))
    stored = study.user_attrs.get(TRIAL_TARGET_ATTR)
    if stored is None:
        if not study.get_trials(deepcopy=False):
            return 0
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} has trial state but no {TRIAL_TARGET_ATTR!r}. "
            "Use a fresh local ledger and artifact root.",
        )
    if type(stored) is not int or stored < 1 or finished > stored:
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} has invalid {TRIAL_TARGET_ATTR!r}={stored!r} "
            f"for {finished} terminal trial(s). Use a new experiment name, or archive/delete "
            "the inconsistent study before running again."
        )
    recovery = _load_recovery_record(study)
    return stored if recovery is None else max(stored, recovery.trial_target)


def _validate_trial_target(study: optuna.Study, phase: Phase) -> None:
    """Reject a target the study's durable state does not allow.

    A target lower than the accepted one is refused, and so is a fixed
    configuration's target past what it may retry (see
    :func:`_validate_fixed_configuration_target`).

    :param optuna.Study study: Existing study whose accepted target is checked.
    :param Phase phase: Phase config supplying the requested ``n_trials`` target.
    :raises TrialTargetRefusedError: ``phase.n_trials`` is lower than the
        study's durable accepted target, or a fixed configuration asks for more
        attempts than it may retry.
    """
    accepted_target = _accepted_trial_target(study)
    if phase.n_trials < accepted_target:
        raise TrialTargetRefusedError(
            f"Phase {phase.name!r} has already accepted a target of {accepted_target} terminal "
            f"trial(s), but the current config requests {phase.n_trials}. Use at least the prior "
            "target or a new experiment name."
        )
    trials = study.get_trials(deepcopy=False)
    _validate_fixed_configuration_target(
        phase,
        finished=_finished_trial_count(trials),
        completed=_completed_trial_count(trials),
    )


def _validate_fixed_configuration_target(phase: Phase, *, finished: int, completed: int) -> None:
    """Reject a fixed configuration's target that would repeat a measured configuration.

    A fixed configuration draws nothing, so every attempt runs the same trial.
    Raising ``n_trials`` retries it only while no attempt has completed: after
    ``finished`` failed attempts the target may reach ``finished + 1``, and a
    completed attempt, feasible or not, closes it. A phase that draws is not
    checked here. A study that does not exist yet counts zero finished and
    zero completed.

    :param Phase phase: Phase config supplying the requested ``n_trials`` target.
    :param int finished: Terminal attempts the phase's study holds.
    :param int completed: How many of those are COMPLETE.
    :raises TrialTargetRefusedError: The fixed configuration asks for more
        attempts than it may retry; the message names the largest allowed target.
    """
    refusal = fixed_configuration_target_refusal(phase, finished=finished, completed=completed)
    if refusal is not None:
        raise TrialTargetRefusedError(refusal)


def _record_trial_target(study: optuna.Study, phase: Phase) -> None:
    """Persist the highest accepted target before the phase launches work.

    :param optuna.Study study: Study whose accepted trial target is stored.
    :param Phase phase: Phase config supplying the new ``n_trials`` target.
    :raises TrialTargetRefusedError: ``phase.n_trials`` is lower than the
        study's already-accepted target.
    """
    _validate_trial_target(study, phase)
    if phase.n_trials != study.user_attrs.get(TRIAL_TARGET_ATTR):
        study.set_user_attr(TRIAL_TARGET_ATTR, phase.n_trials)


def _allocation_contexts(study: optuna.Study) -> list[_AllocationRecord]:
    """Return valid pre-Optuna allocation records from a study's durable context.

    :param optuna.Study study: Study holding the allocation context.
    :return list[_AllocationRecord]: Valid allocation records, ordered by creation.
    """
    raw = study.user_attrs.get(_ALLOCATION_CONTEXT_ATTR)
    if not isinstance(raw, dict) or raw.get("schema_version") != _ALLOCATION_CONTEXT_SCHEMA_VERSION:
        return []
    allocations = raw.get("allocations")
    if not isinstance(allocations, list):
        return []
    records: list[_AllocationRecord] = []
    for record in allocations:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("generation_id"), str)
            or not record["generation_id"]
            or type(record.get("first_trial_number")) is not int
            or record["first_trial_number"] < 0
            or not isinstance(record.get("trainer_environment"), str)
            or not record["trainer_environment"]
        ):
            continue
        records.append(
            {
                "generation_id": record["generation_id"],
                "first_trial_number": record["first_trial_number"],
                "trainer_environment": record["trainer_environment"],
            }
        )
    return records


def _record_allocation_context(
    study: optuna.Study,
    *,
    generation_id: str,
    trainer_environment: str,
) -> None:
    """Persist environment provenance before Optuna allocates a trial.

    A trial is allocated by ``Study.ask()`` before the objective can write any
    trial attrs. The recorded boundary identifies the environment for a trial
    that dies in precisely that gap, without assigning an environment to older
    or executed trials whose provenance is absent.

    :param optuna.Study study: Study about to allocate work.
    :param str generation_id: Generation that owns the pending allocation.
    :param str trainer_environment: Semantic trainer-environment digest.
    """
    trials = study.get_trials(deepcopy=False)
    first_trial_number = max((trial.number for trial in trials), default=-1) + 1
    records = _allocation_contexts(study)
    records.append(
        {
            "generation_id": generation_id,
            "first_trial_number": first_trial_number,
            "trainer_environment": trainer_environment,
        }
    )
    study.set_user_attr(
        _ALLOCATION_CONTEXT_ATTR,
        {
            "schema_version": _ALLOCATION_CONTEXT_SCHEMA_VERSION,
            "allocations": records,
        },
    )


def _rerun_sources(trials: Sequence[optuna.trial.FrozenTrial]) -> list[optuna.trial.FrozenTrial]:
    """Return interrupted trials not yet named by a rerun, in trial-number order.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial of one study.
    :return list[optuna.trial.FrozenTrial]: Interrupted trials without a rerun.
    """
    named = {trial.user_attrs.get(RERUN_OF_ATTR) for trial in trials}
    return sorted(
        (trial for trial in trials if _trial_interrupted(trial) and trial.number not in named),
        key=lambda trial: trial.number,
    )


def _pending_rerun_count(trials: Sequence[optuna.trial.FrozenTrial]) -> int:
    """Count the reruns a phase still owes: interrupted trials without one, and WAITING reruns.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial of one study.
    :return int: Reruns the next run that reaches the phase launches before any new draw.
    """
    waiting = sum(
        1
        for trial in trials
        if trial.state == optuna.trial.TrialState.WAITING and RERUN_OF_ATTR in trial.user_attrs
    )
    return len(_rerun_sources(trials)) + waiting


def _record_reruns(study: optuna.Study) -> list[tuple[int, int]]:
    """Schedule one WAITING rerun for every interrupted trial that has none.

    Each rerun is created in one storage record carrying its source's params,
    distributions and system attrs (a grid point, a CMA-ES generation) and
    :data:`~phasesweep.engine.state.RERUN_OF_ATTR`. Optuna's ``ask`` pops
    WAITING trials before it creates new ones and returns their stored params
    without consulting the sampler, so the reruns run first with exactly their
    sources' configurations, and each takes its source's draw position.

    :param optuna.Study study: Study about to allocate work.
    :return list[tuple[int, int]]: ``(source, rerun)`` trial numbers, one per
        rerun created, in source order.
    """
    sources = _rerun_sources(study.get_trials(deepcopy=False))
    for source in sources:
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.WAITING,
                params=source.params,
                distributions=source.distributions,
                system_attrs=dict(source.system_attrs),
                user_attrs={RERUN_OF_ATTR: source.number},
            )
        )
    if not sources:
        return []
    reruns = {
        trial.user_attrs[RERUN_OF_ATTR]: trial.number
        for trial in study.get_trials(deepcopy=False)
        if RERUN_OF_ATTR in trial.user_attrs
    }
    return [(source.number, reruns[source.number]) for source in sources]


def _prelaunch_trial_environment(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
) -> str | None:
    """Return the recorded environment for an allocation that never launched.

    A context can cover a missing trial identity only when the trial number is
    at or after its durable pre-allocation boundary and the trial never reached
    the per-attempt metadata written before any trainer can start. This keeps
    executed trials with missing environment provenance rejected.

    :param optuna.Study study: Study containing the allocation context.
    :param optuna.trial.FrozenTrial trial: Trial with a missing environment attr.
    :return str | None: Covered environment digest, or ``None`` when unknown.
    """
    attrs = trial.user_attrs
    if any(attr in attrs for attr in (TRIAL_DIR_ATTR, ATTEMPT_ID_ATTR, GENERATION_ID_ATTR)):
        return None
    for record in reversed(_allocation_contexts(study)):
        if trial.number >= record["first_trial_number"]:
            return record["trainer_environment"]
    return None


def _restore_prelaunch_environment_identity(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
) -> bool:
    """Restore environment provenance for one covered unlaunched trial.

    :param optuna.Study study: Study containing the allocation context.
    :param optuna.trial.FrozenTrial trial: Still-RUNNING prelaunch allocation.
    :return bool: Whether a missing environment identity was restored.
    """
    if (
        isinstance(trial.user_attrs.get(TRAINER_ENV_DIGEST_ATTR), str)
        and trial.user_attrs[TRAINER_ENV_DIGEST_ATTR]
    ):
        return False
    environment = _prelaunch_trial_environment(study, trial)
    if environment is None:
        return False
    optuna.Trial(study, trial._trial_id).set_user_attr(TRAINER_ENV_DIGEST_ATTR, environment)
    return True


_NO_RECORDED_ENVIRONMENT = (
    "No trial from the other cohort recorded its environment, so the difference cannot be named."
)


def _recorded_environment(trial: optuna.trial.FrozenTrial) -> dict[str, str] | None:
    """Read the tracked environment a trial recorded.

    :param optuna.trial.FrozenTrial trial: Trial from another environment cohort.
    :return dict[str, str] | None: The recorded name-to-value mapping, or ``None`` when the
        trial recorded none (an allocation that died before it landed) or holds a malformed one.
    """
    recorded = trial.user_attrs.get(TRAINER_ENV_ATTR)
    if not isinstance(recorded, dict) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in recorded.items()
    ):
        return None
    return dict(recorded)


def _describe_environment_changes(old: dict[str, str], new: dict[str, str]) -> str:
    """Render how a tracked environment differs, with the values on both sides.

    The values are cut by the window :func:`~phasesweep.engine.fingerprints._render_changed_values`
    applies to a phase fingerprint refusal, so a long ``PATH`` shows where the two differ.

    :param dict[str, str] old: Mapping a trial of the other cohort recorded.
    :param dict[str, str] new: Mapping this invocation composes.
    :return str: Each differing variable as ``NAME old -> new``, then each added name with its
        value and each removed name with its old value, up to :data:`_MAX_REPORTED_CHANGES`
        entries and a count of the rest; empty when the mappings are equal.
    """
    changed: list[str] = []
    added: list[str] = []
    removed: list[str] = []
    for name, before, after in _changed_payload_keys(old, new):
        old_text, new_text = _render_changed_values(before, after)
        if before is _ABSENT:
            added.append(f"added {name}={new_text}")
        elif after is _ABSENT:
            removed.append(f"removed {name} (was {old_text})")
        else:
            changed.append(f"{name} {old_text} -> {new_text}")
    entries = [*changed, *added, *removed]
    shown = entries[:_MAX_REPORTED_CHANGES]
    if len(entries) > _MAX_REPORTED_CHANGES:
        shown.append(f"and {len(entries) - _MAX_REPORTED_CHANGES} more")
    return "; ".join(shown)


def _environment_difference(
    current: EnvironmentIdentity, trials: Sequence[optuna.trial.FrozenTrial]
) -> str:
    """Show what differs between this invocation's environment and another cohort's trial.

    Compares the tracked mapping against the latest of ``trials`` that recorded one. Pass-through
    names are in neither mapping, since they cannot split a cohort.

    :param EnvironmentIdentity current: Environment this invocation composes.
    :param Sequence[optuna.trial.FrozenTrial] trials: Trials from other cohorts, oldest first.
    :return str: A sentence showing each differing, added and removed variable with its values,
        or one saying no trial of the other cohort recorded its environment.
    """
    for trial in reversed(trials):
        recorded = _recorded_environment(trial)
        if recorded is None:
            continue
        description = _describe_environment_changes(recorded, current.tracked)
        if description:
            return f"Compared with trial {trial.number}: {description}."
    return _NO_RECORDED_ENVIRONMENT


def _validate_environment_cohort(study: optuna.Study, current: EnvironmentIdentity) -> None:
    """Refuse to allocate into a different or unknown semantic environment cohort.

    :param optuna.Study study: Existing study whose trials define the cohort.
    :param EnvironmentIdentity current: Semantic environment this invocation composes.
    :raises StudySchemaMismatchError: A populated study has a trial with
        no usable environment identity.
    :raises StudyFingerprintMismatchError: Recorded trials belong to another
        semantic environment cohort. The message shows each variable that differs,
        was added or was removed, with its old and new value.
    """
    trials = study.get_trials(deepcopy=False)
    if not trials:
        return
    missing: list[int] = []
    recorded: set[str] = set()
    elsewhere: list[optuna.trial.FrozenTrial] = []
    for trial in trials:
        environment = trial.user_attrs.get(TRAINER_ENV_DIGEST_ATTR)
        if not (isinstance(environment, str) and environment):
            environment = _prelaunch_trial_environment(study, trial)
        if environment is None:
            missing.append(trial.number)
            continue
        recorded.add(environment)
        if environment != current.digest:
            elsewhere.append(trial)
    if missing:
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} contains populated trial(s) without a "
            f"semantic trainer-environment identity: {missing}. PhaseSweep cannot guess "
            "which environment cohort owns those results. Use a new experiment name.",
        )
    if recorded != {current.digest}:
        rendered = ", ".join(sorted(digest[:12] for digest in recorded))
        raise StudyFingerprintMismatchError(
            f"Study {study.study_name!r} contains trial(s) from semantic trainer "
            f"environment cohort(s) [{rendered}], but this invocation composes "
            f"{current.digest[:12]}. {_environment_difference(current, elsewhere)} "
            "No trial was allocated. Restore the original semantic environment, pin "
            "the value under env, classify variables that may differ, such as "
            "rotating credentials, under execution.passthrough_env, or use a new "
            "experiment name.",
            action=OperatorAction.FIX_CONFIG,
        )
