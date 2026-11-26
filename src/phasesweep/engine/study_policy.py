"""Persisted study policy and continuation validation."""

from __future__ import annotations

import os
import re
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from typing import Any, Literal, NamedTuple, TypedDict

import optuna

from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import (
    fixed_configuration_target_refusal,
    grid_cardinality,
    grid_target_refusal,
    phase_draws,
    retry_target_ceiling,
)
from phasesweep.engine.attempts import (
    _cleanup_recovered_trial_numbers,
    _trial_requires_cleanup_recovery,
)
from phasesweep.engine.bounds import bound_violations, recorded_constraints
from phasesweep.engine.errors import (
    LaunchEnvironmentError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    TrialTargetRefusedError,
)
from phasesweep.engine.evidence import _validate_selection_evidence
from phasesweep.engine.fingerprints import (
    _ABSENT,
    _MAX_REPORTED_CHANGES,
    _changed_payload_keys,
    _render_value,
    _verify_recorded_identity,
)
from phasesweep.engine.grid_order import recorded_search_space
from phasesweep.engine.optuna import (
    _any_objective_unreadable,
    _any_trial_timed_out,
    _draw_positions,
    _finished_trial_count,
    _measured_draw_count,
    _trial_count_state,
    _trial_run_state,
)
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
    _ParsedTrialOutcome,
    _recorded_identity,
    _study_schema_supported,
    _unique_trial_outcomes,
    _unreplaced_attempts,
)
from phasesweep.engine.trial import (
    EnvironmentIdentity,
    _environment_identity,
    _is_search_path,
    _missing_trainer_cwd,
)
from phasesweep.errors import (
    FRESH_STUDY,
    OTHER_FORMAT_STEP,
    REJUDGE_BOUNDS_STEP,
    RERUN_CLEANUP_RECOVERY,
    OperatorAction,
    format_mismatch,
    restore_or_new_study,
    short_digest,
    timeout_step,
    unreadable_objective_step,
)
from phasesweep.runtime.json import JsonObject, json_text_map
from phasesweep.runtime.terminal import plural

_ALLOCATION_CONTEXT_ATTR = "phasesweep_allocation_context"
_ALLOCATION_CONTEXT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _PhasePolicyState:
    """Failure-policy state reconstructed from durable per-trial outcomes."""

    max_sequence: int
    consecutive_failures: int
    abort: dict[str, Any] | None
    """The first standing abort recorded after the recovery boundary, if any. A later
    outcome resets the streak but never this recorded decision; a streak abort stands while
    its streak, judged under the current bounds, still reaches the count it recorded."""
    abort_streak: tuple[int, ...] = ()
    """The trial numbers of the failed or fatal outcomes in a row that ended at ``abort``'s
    outcome, in completion order; empty while ``abort`` is ``None``."""


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

    A resumed phase applies today's threshold only when no abort stands. A standing abort is
    judged by :func:`_config_retires_abort`, never by recounting against today's threshold.

    :param int consecutive_failures: Current consecutive-failure count.
    :param Phase phase: Phase config supplying ``max_consecutive_failures``.
    :return bool: True when ``consecutive_failures`` meets or exceeds the
        phase's threshold.
    """
    return consecutive_failures >= phase.max_consecutive_failures


def _abort_decision(
    phase: Phase,
    trial_number: int,
    outcome: str,
    *,
    completion_sequence: int,
    consecutive_failures: int,
    abort_recorded: bool,
    cause: str | None = None,
    fatal_policy: str | None = None,
) -> JsonObject | None:
    """Return the abort record that recording ``outcome`` after the given state would trip.

    Live trials, recovered outcomes and sibling re-reads apply the target study’s failure policy.

    :param Phase phase: Phase whose ``max_consecutive_failures`` and ``n_trials`` apply.
    :param int trial_number: Trial whose outcome is about to be recorded.
    :param str outcome: The terminal outcome about to be recorded.
    :param int completion_sequence: Completion sequence of the last recorded outcome.
    :param int consecutive_failures: Failure streak before ``outcome``.
    :param bool abort_recorded: Whether the phase already recorded an abort.
    :param str | None cause: The outcome's recorded cause, if any.
    :param str | None fatal_policy: The fatal policy of a ``"fatal"`` outcome.
    :return JsonObject | None: The abort record at the next completion sequence, or ``None``
        when the outcome trips nothing.
    """
    if abort_recorded:
        return None
    sequence = completion_sequence + 1
    if outcome == "fatal":
        return _abort_record(
            policy=fatal_policy or "fatal_trial_exception",
            completion_sequence=sequence,
            trial_target=phase.n_trials,
            cause=cause or f"trial {trial_number} hit a fatal objective error",
        )
    streak = _next_consecutive_failures(consecutive_failures, outcome)
    if not _consecutive_failure_threshold_tripped(streak, phase):
        return None
    return _failure_policy_abort_record(
        phase,
        consecutive_failures=streak,
        completion_sequence=sequence,
        trial_target=phase.n_trials,
    )


def _abort_record(
    *,
    policy: str,
    completion_sequence: int,
    trial_target: int,
    cause: str,
) -> JsonObject:
    """Build the common fields of a durable phase abort record.

    :param str policy: Policy that stopped the phase, such as
        ``"unsafe_process_cleanup"``.
    :param int completion_sequence: Completion sequence of the tripping outcome.
    :param int trial_target: Durable accepted ``n_trials`` target to attach
        to the record.
    :param str cause: Explanation the operator reads.
    :return JsonObject: A ``schema_version=1`` abort record ready to
        record with the tripping outcome.
    """
    return {
        "schema_version": 1,
        "policy": policy,
        "completion_sequence": completion_sequence,
        "trial_target": trial_target,
        "cause": cause,
    }


def _failure_policy_abort_record(
    phase: Phase,
    *,
    consecutive_failures: int,
    completion_sequence: int,
    trial_target: int,
) -> JsonObject:
    """Build a durable consecutive-failure abort record.

    :param Phase phase: Phase supplying ``max_consecutive_failures`` for the
        recorded threshold and cause message.
    :param int consecutive_failures: Observed consecutive failure/infeasible
        count that tripped the policy.
    :param int completion_sequence: Orchestrator completion sequence at which
        the trip was observed.
    :param int trial_target: Durable accepted ``n_trials`` target to attach
        to the record.
    :return JsonObject: A ``schema_version=1`` abort record ready to
        record with the tripping outcome.
    """
    return {
        **_abort_record(
            policy="max_consecutive_failures",
            completion_sequence=completion_sequence,
            trial_target=trial_target,
            cause=(
                f"{consecutive_failures} consecutive failed/infeasible trials reached "
                f"max_consecutive_failures={phase.max_consecutive_failures}."
            ),
        ),
        "threshold": phase.max_consecutive_failures,
        "consecutive_failures": consecutive_failures,
    }


def _replayed_streak_abort(
    phase: Phase, study: optuna.Study, policy_state: _PhasePolicyState
) -> JsonObject | None:
    """Return the abort a replayed failure streak at the current limit would record.

    :param Phase phase: The phase being judged.
    :param optuna.Study study: The phase's study.
    :param _PhasePolicyState policy_state: The study's replayed failure-policy state.
    :return JsonObject | None: The abort, or ``None`` when one is already recorded, the phase
        has no work remaining, or the streak is under the limit.
    """
    if (
        policy_state.abort is not None
        or _finished_trial_count(study.get_trials(deepcopy=False)) >= phase.n_trials
        or not _consecutive_failure_threshold_tripped(policy_state.consecutive_failures, phase)
    ):
        return None
    return _failure_policy_abort_record(
        phase,
        consecutive_failures=policy_state.consecutive_failures,
        completion_sequence=policy_state.max_sequence,
        trial_target=_accepted_trial_target(study),
    )


@dataclass(frozen=True)
class _AcceptedPartialDecision:
    """Durable terminal decision to select from an incomplete timed-out phase."""

    trial_target: int
    outcome_sequence: int
    finished_trials: int
    measured_trials: int
    """Finished trials that measured their values, a count no bound changes."""
    timeout_scope: str


def _at_accepted_partial_target(
    decision: _AcceptedPartialDecision | None, phase: Phase
) -> typing.TypeGuard[_AcceptedPartialDecision]:
    """Check whether the phase is still at its accepted partial target.

    :param _AcceptedPartialDecision | None decision: Recorded decision, if any.
    :param Phase phase: Phase whose current target is being judged.
    :return bool: Whether the decision governs this target before failure-policy checks.
    """
    return decision is not None and phase.n_trials == decision.trial_target


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
    return _parse_accepted_partial_decision(
        study.user_attrs.get(PHASE_DECISION_ATTR), study.study_name
    )


def _record_accepted_partial_decision(
    study: optuna.Study,
    trial_target: int,
    *,
    outcome_sequence: int,
    finished_trials: int,
    measured_trials: int,
    timeout_scope: str,
    recovered_abort_sequence: int | None,
) -> None:
    """Write the accepted timeout boundary a replay selects from without launching.

    :param optuna.Study study: The study accepting the boundary.
    :param int trial_target: The ``n_trials`` the boundary was accepted at.
    :param int outcome_sequence: The last completion sequence inside the boundary.
    :param int finished_trials: Finished trials inside the boundary.
    :param int measured_trials: COMPLETE trials inside the boundary.
    :param str timeout_scope: ``"phase"`` or ``"run"``, the deadline that stopped the phase.
    :param int | None recovered_abort_sequence: Sequence of the abort the run recovered from.
    """
    study.set_user_attr(
        PHASE_DECISION_ATTR,
        {
            "schema_version": PHASE_DECISION_SCHEMA_VERSION,
            "decision": "accepted_partial_timeout",
            "trial_target": trial_target,
            "outcome_sequence": outcome_sequence,
            "finished_trials": finished_trials,
            "measured_trials": measured_trials,
            "timeout_scope": timeout_scope,
            "recovered_abort_sequence": recovered_abort_sequence,
        },
    )


def _parse_accepted_partial_decision(
    raw: object, study_name: str
) -> _AcceptedPartialDecision | None:
    """Decode a study's accepted timeout boundary.

    :param object raw: Stored decision, or None.
    :param str study_name: Study named in a schema refusal.
    :raises StudySchemaMismatchError: The decision is malformed.
    :return _AcceptedPartialDecision | None: The decision, or None when absent.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise StudySchemaMismatchError(
            f"Study {study_name!r} has malformed {PHASE_DECISION_ATTR!r}={raw!r}."
        )
    trial_target = raw.get("trial_target")
    outcome_sequence = raw.get("outcome_sequence")
    finished_trials = raw.get("finished_trials")
    measured_trials = raw.get("measured_trials")
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
        or type(measured_trials) is not int
        or measured_trials < 0
        or measured_trials > finished_trials
        or timeout_scope not in ("phase", "run")
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
            f"Study {study_name!r} has malformed {PHASE_DECISION_ATTR!r} fields: {raw!r}."
        )
    return _AcceptedPartialDecision(
        trial_target=trial_target,
        outcome_sequence=outcome_sequence,
        finished_trials=finished_trials,
        measured_trials=measured_trials,
        timeout_scope=timeout_scope,
    )


def _phase_policy_schema_error(study: optuna.Study, detail: str) -> StudySchemaMismatchError:
    """Build the actionable error used for malformed durable policy state.

    :param optuna.Study study: Study whose durable state is inconsistent,
        used to name the study in the message.
    :param str detail: Specific description of the malformed state.
    :return StudySchemaMismatchError: Diagnostic carrying the study's remediation action.
    """
    return StudySchemaMismatchError(
        f"Study {study.study_name!r} has invalid durable failure-policy state: {detail}."
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


def _counted_outcome(
    outcome: str, metric: float | None, trial: optuna.trial.FrozenTrial, experiment: Experiment
) -> str:
    """Return the outcome the failure streak counts for a recorded one.

    A ``success`` records a measurement; the streak counts it as a failure while its
    measured values violate a current bound, so correcting a bound re-judges the streak.

    :param str outcome: The recorded outcome.
    :param float | None metric: The metric a ``success`` recorded.
    :param optuna.trial.FrozenTrial trial: The trial, whose constraint values are read.
    :param Experiment experiment: Config supplying the current bounds.
    :return str: ``"failure"`` for a success outside the bounds, otherwise ``outcome``.
    """
    if outcome != "success" or metric is None:
        return outcome
    violations = bound_violations(metric, recorded_constraints(trial, experiment), experiment)
    return "failure" if violations else "success"


def _recorded_outcome_events(
    study: optuna.Study,
) -> tuple[list[tuple[_ParsedTrialOutcome, optuna.trial.FrozenTrial]], int]:
    """Validate a study's durable outcomes and return them in completion order.

    :param optuna.Study study: Study whose ``PHASE_RECOVERY_ATTR`` and per-trial outcome
        attrs are read and validated.
    :return tuple[list[tuple[_ParsedTrialOutcome, optuna.trial.FrozenTrial]], int]: Each
        outcome with its trial, by sequence, and the recovery boundary's sequence (0 without
        one).
    :raises StudySchemaMismatchError: ``PHASE_RECOVERY_ATTR`` or a terminal trial's outcome
        attr is malformed, two trials share a completion sequence, or the recovery boundary
        exceeds the largest recorded sequence.
    """
    recovery = _load_recovery_record(study)
    recovery_boundary = 0 if recovery is None else recovery.start_after_sequence

    trials = study.get_trials(deepcopy=False)
    outcomes = _unique_trial_outcomes(study.study_name, trials)
    for trial in trials:
        if trial.state.is_finished() and trial.number not in outcomes:
            raw = trial.user_attrs.get(TRIAL_OUTCOME_ATTR)
            raise _phase_policy_schema_error(
                study,
                f"terminal trial {trial.number} has missing or malformed "
                f"{TRIAL_OUTCOME_ATTR!r}: {raw!r}",
            )
    by_number = {trial.number: trial for trial in trials}
    events = sorted(
        ((parsed, by_number[number]) for number, parsed in outcomes.items()),
        key=lambda event: event[0].sequence,
    )
    max_sequence = events[-1][0].sequence if events else 0
    if recovery_boundary > max_sequence:
        raise _phase_policy_schema_error(
            study,
            f"{PHASE_RECOVERY_ATTR!r} starts after sequence {recovery_boundary}, "
            f"but the largest recorded sequence is {max_sequence}",
        )
    return events, recovery_boundary


def _recorded_outcome_sequence(study: optuna.Study) -> int:
    """Return the highest completion sequence a study's validated outcomes record.

    :param optuna.Study study: Study whose outcomes are read.
    :return int: The highest sequence, 0 without outcomes.
    :raises StudySchemaMismatchError: The durable outcomes are malformed
        (:func:`_recorded_outcome_events`).
    """
    events, _ = _recorded_outcome_events(study)
    return events[-1][0].sequence if events else 0


def _load_phase_policy_state(study: optuna.Study, experiment: Experiment) -> _PhasePolicyState:
    """Validate and reconstruct the durable consecutive-failure state under the current bounds.

    Each outcome counts as :func:`_counted_outcome` says. A recorded streak abort stands
    only while the streak that ended at its outcome, so re-judged, still reaches the count it
    recorded: loosening a bound that turns the streak's trials feasible clears it, and a run
    then judges the replayed streak against the current limit instead.

    :param optuna.Study study: Study whose ``PHASE_RECOVERY_ATTR`` and
        per-trial outcome attrs are read and validated.
    :param Experiment experiment: Config supplying the current bounds.
    :return _PhasePolicyState: Reconstructed state: the highest recorded
        outcome sequence, the consecutive-failure count since the last
        recovery boundary, and the first standing abort recorded after it (if any)
        with the trials of the failure streak that ended there.
    :raises StudySchemaMismatchError: ``PHASE_RECOVERY_ATTR`` or a terminal
        trial's outcome attr is malformed, two trials share a completion
        sequence, or the recovery boundary exceeds the largest recorded
        sequence.
    """
    events, recovery_boundary = _recorded_outcome_events(study)
    max_sequence = events[-1][0].sequence if events else 0

    consecutive_failures = 0
    streak: list[int] = []
    abort: dict[str, Any] | None = None
    abort_streak: tuple[int, ...] = ()
    for parsed, trial in events:
        if parsed.sequence <= recovery_boundary:
            continue
        previous = consecutive_failures
        consecutive_failures = _next_consecutive_failures(
            previous, _counted_outcome(parsed.outcome, parsed.metric, trial, experiment)
        )
        if consecutive_failures > previous:
            streak.append(trial.number)
        elif consecutive_failures == 0:
            streak.clear()
        recorded_abort = parsed.abort
        tripped = None if recorded_abort is None else _tripped_streak(recorded_abort)
        if (
            abort is None
            and recorded_abort is not None
            and (tripped is None or consecutive_failures >= tripped)
        ):
            abort = recorded_abort
            abort_streak = tuple(streak)

    return _PhasePolicyState(
        max_sequence=max_sequence,
        consecutive_failures=consecutive_failures,
        abort=abort,
        abort_streak=abort_streak,
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
    max_sequence = _recorded_outcome_sequence(study)
    study.set_user_attr(
        PHASE_RECOVERY_ATTR,
        {
            "schema_version": PHASE_RECOVERY_SCHEMA_VERSION,
            "recovered_abort_sequence": recovered_abort_sequence,
            "start_after_sequence": max_sequence,
            "trial_target": phase.n_trials,
        },
    )


def _validate_study_schema(study: optuna.Study) -> None:
    """Validate the recorded study schema and current failure-policy state without writing.

    :param optuna.Study study: Study whose durable schema attr and policy state are checked.
    :raises StudySchemaMismatchError: The study already holds trials or a schema
        stamp from an unsupported version, or its durable failure-policy state
        is malformed.
    """
    trials = study.get_trials(deepcopy=False)
    version = study.user_attrs.get(STUDY_SCHEMA_ATTR)
    if _study_schema_supported(version, bool(trials)):
        if version is not None:
            _recorded_outcome_events(study)
        return

    trial_numbers = [trial.number for trial in trials]
    raise StudySchemaMismatchError(
        f"Study {study.study_name!r} {format_mismatch(version, STUDY_SCHEMA_VERSION)}. "
        f"Affected trial numbers: {trial_numbers}.",
        remediation=OTHER_FORMAT_STEP,
    )


def _initialize_study_schema(study: optuna.Study) -> None:
    """Validate a live study and stamp an empty, unmarked allocation.

    :param optuna.Study study: Live study being prepared for execution.
    :raises StudySchemaMismatchError: The schema or current policy state is unsupported.
    """
    _validate_study_schema(study)
    if study.user_attrs.get(STUDY_SCHEMA_ATTR) is None:
        study.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION)


def _validate_study_direction(
    study: optuna.Study,
    goal: Literal["minimize", "maximize"],
) -> None:
    """Require stored objective direction to match config before trial launch.

    Optuna's ``load_if_exists`` retains the stored direction without checking the offered one.

    :param optuna.Study study: Existing or newly created single-objective study.
    :param Literal goal: Direction required by the experiment metric.
    :raises StudySchemaMismatchError: The stored direction does not match ``goal``; the step
        restores ``metric.goal`` to the stored direction when the study has one.
    """
    expected = (
        optuna.study.StudyDirection.MINIMIZE
        if goal == "minimize"
        else optuna.study.StudyDirection.MAXIMIZE
    )
    if study.directions == [expected]:
        return
    stored = ", ".join(direction.name.lower() for direction in study.directions)
    # Only a single-objective study's direction is one PhaseSweep can write back as metric.goal.
    restorable = len(study.directions) == 1 and stored in ("minimize", "maximize")
    raise StudySchemaMismatchError(
        f"Study {study.study_name!r} optimizes {stored or 'no direction'}, but the config "
        f"requires {goal}.",
        remediation=restore_or_new_study(f"`metric.goal: {stored}`") if restorable else None,
    )


def _accepted_trial_target(study: optuna.Study) -> int:
    """Return the target a study's run accepted or its finished trials already hold.

    A run stores ``TRIAL_TARGET_ATTR`` only once its launch checks pass, and recovery records
    advance before it, so the larger target preserves the accepted budget across an
    interrupted recovery. Kept attempts a study adopts before any launch add finished trials
    without accepting a target, so those trials are a floor of their own.

    :param optuna.Study study: Study whose accepted trial target is read.
    :return int: The largest of the stored ``phasesweep_trial_target`` user attr, the recovery
        record's target and the finished trial count; zero for an empty study.
    :raises StudySchemaMismatchError: The stored target is not a positive int, or the recovery
        record is malformed.
    """
    finished = _finished_trial_count(study.get_trials(deepcopy=False))
    stored = study.user_attrs.get(TRIAL_TARGET_ATTR)
    if stored is not None and (type(stored) is not int or stored < 1):
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} has invalid {TRIAL_TARGET_ATTR!r}={stored!r}."
        )
    recovery = _load_recovery_record(study)
    return max(stored or 0, finished, 0 if recovery is None else recovery.trial_target)


def _validate_trial_target(study: optuna.Study, phase: Phase) -> None:
    """Reject a target the study's durable state does not allow.

    A target must keep the accepted budget, fit the available attempts, and reserve
    enough draws to cover a full grown grid.

    :param optuna.Study study: Existing study whose accepted target is checked.
    :param Phase phase: Phase config supplying the requested ``n_trials`` target.
    :raises TrialTargetRefusedError: The target is too low for the accepted budget or
        full grown grid, or too high for a fixed configuration or grid.
    """
    accepted_target = _accepted_trial_target(study)
    _validate_accepted_trial_target(phase, accepted_target)
    trials = study.get_trials(deepcopy=False)
    _validate_fixed_configuration_target(
        phase,
        finished=_finished_trial_count(trials),
        measured=_measured_draw_count(trials),
        accepted=accepted_target,
    )
    _validate_grown_grid_target(study, phase, trials)


def _validate_accepted_trial_target(phase: Phase, accepted_target: int) -> None:
    """Refuse a requested budget below the phase's accepted trial target.

    :param Phase phase: Phase with the configured target.
    :param int accepted_target: Trial budget already accepted by its study.
    :raises TrialTargetRefusedError: The requested budget is lower.
    """
    if phase.n_trials < accepted_target:
        raise TrialTargetRefusedError(
            f"Phase {phase.name!r} has already accepted a target of "
            f"{plural(accepted_target, 'finished trial')}; the config requests {phase.n_trials}.",
            remediation=f"Set phases[{phase.name}].n_trials to at least {accepted_target}.",
        )


def _validate_fixed_configuration_target(
    phase: Phase, *, finished: int, measured: int, accepted: int
) -> None:
    """Reject a fixed configuration or grid target that would repeat measured cells.

    A fixed configuration draws nothing, so every attempt runs the same trial.
    Raising ``n_trials`` retries it only while no attempt is COMPLETE: after
    ``finished`` failed attempts the target may reach ``finished + 1``, and a
    COMPLETE attempt, feasible or not, closes it. A grid applies the same rule
    per cell: its target may add one attempt per cell without a COMPLETE result.
    Other sampled phases are unrestricted. A missing study counts zero finished
    and zero measured. When the accepted target equals the largest allowed one, the remedy
    names that one target, since any lower value falls below the accepted target.

    :param Phase phase: Phase config supplying the requested ``n_trials`` target.
    :param int finished: Finished attempts the phase's study holds.
    :param int measured: How many draws have a COMPLETE attempt, feasible or not.
    :param int accepted: The study's accepted target (:func:`_accepted_trial_target`), 0 for
        no study; every caller passes it, so run, validate and status give one remedy.
    :raises TrialTargetRefusedError: The fixed configuration asks for more
        attempts than it may retry; the message names the largest allowed target.
    """
    refusal = fixed_configuration_target_refusal(phase, finished=finished, measured=measured)
    # A completed fixed configuration's remedy already names its one target.
    bounded = refusal is None or measured == 0
    if refusal is None:
        refusal = grid_target_refusal(phase, finished=finished, measured=measured)
    if refusal is not None:
        message, remediation = refusal
        if (
            bounded
            and retry_target_ceiling(phase, finished=finished, measured=measured) == accepted
        ):
            remediation = f"Set phases[{phase.name}].n_trials to {accepted}."
        raise TrialTargetRefusedError(message, remediation=remediation)


def _record_trial_target(study: optuna.Study, phase: Phase) -> None:
    """Persist the target the phase was validated against before it launches work.

    The caller has already run :func:`_validate_trial_target` on this study under the
    experiment lock, before claiming the generation.

    :param optuna.Study study: Study whose accepted trial target is stored.
    :param Phase phase: Phase config supplying the new ``n_trials`` target.
    """
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


def _rerun_sources(
    trials: Sequence[optuna.trial.FrozenTrial], *, assumed_complete: frozenset[int] = frozenset()
) -> list[optuna.trial.FrozenTrial]:
    """Return interrupted trials that still need a result, in trial-number order.

    A reread may find a COMPLETE result in an earlier attempt of an interrupted retry.
    That draw is already measured, so its interrupted attempt needs no rerun.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial of one study.
    :param frozenset[int] assumed_complete: Unread trials that might already provide a
        result for their draw. Treat them as measured to count only certain reruns.
    :return list[optuna.trial.FrozenTrial]: Interrupted trials without a rerun or a measurement.
    """
    sources = _unreplaced_attempts(trials, interrupted=True)
    if not sources:
        return []
    positions = _draw_positions(trials)
    measured = {
        positions[trial.number]
        for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE or trial.number in assumed_complete
    }
    return sorted(
        (trial for trial in sources if positions[trial.number] not in measured),
        key=lambda trial: trial.number,
    )


def _undrawn_grown_grid_cells(
    study: optuna.Study, phase: Phase, trials: Sequence[optuna.trial.FrozenTrial]
) -> int:
    """Count positions an enlarged full grid still has to allocate.

    :param optuna.Study study: Study whose first search space defines its original grid.
    :param Phase phase: Current phase whose grid may have grown.
    :param Sequence[optuna.trial.FrozenTrial] trials: Existing attempts, including reruns.
    :return int: Unallocated cells of an enlarged full grid, or zero for any other phase.
    """
    if phase.sampler.type != "grid" or phase.allow_partial_grid or not phase_draws(phase):
        return 0
    identity = _recorded_identity(study)
    if identity is None:
        return 0
    original = phase.model_copy(
        update={"search_space": recorded_search_space(identity["search_spaces"][0]["search_space"])}
    )
    size = grid_cardinality(phase)
    if size <= grid_cardinality(original):
        return 0
    return max(0, size - len(set(_draw_positions(trials).values())))


def _validate_grown_grid_target(
    study: optuna.Study,
    phase: Phase,
    trials: Sequence[optuna.trial.FrozenTrial],
) -> None:
    """Require enough trial slots for every cell of an enlarged full grid.

    A RUNNING attempt uses one slot whether it recovers as finished or needs a rerun,
    so this check also works on the dry run's unrecovered study snapshot.

    :param optuna.Study study: Study whose first search space defines its original grid.
    :param Phase phase: Phase whose current target is checked.
    :param Sequence[optuna.trial.FrozenTrial] trials: Existing attempts.
    :raises TrialTargetRefusedError: The target leaves too few slots for the undrawn cells.
    """
    undrawn = _undrawn_grown_grid_cells(study, phase, trials)
    if not undrawn:
        return
    running = sum(trial.state == optuna.trial.TrialState.RUNNING for trial in trials)
    minimum = _finished_trial_count(trials) + running + _pending_rerun_count(study) + undrawn
    if phase.n_trials < minimum:
        raise TrialTargetRefusedError(
            f"Phase {phase.name!r}: n_trials={phase.n_trials} leaves too few trials for "
            f"the enlarged grid's {plural(undrawn, 'undrawn cell')}.",
            remediation=f"Set phases[{phase.name}].n_trials to at least {minimum}.",
        )


def _scheduled_reruns(
    study: optuna.Study,
    trials: Sequence[optuna.trial.FrozenTrial],
    phase: Phase | None = None,
    *,
    possible: bool = False,
) -> list[optuna.trial.FrozenTrial]:
    """Return the trials the next run reruns before any new draw and has not yet queued.

    Interrupted trials without a rerun come first, then failed trials a raised target retries,
    each in trial-number order. A retry takes one of the slots the target leaves after the
    finished trials and reruns already counted. For a full grown grid, reserve one slot per
    undrawn cell first.

    :param optuna.Study study: Study holding the trials and grid history.
    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial of one study.
    :param Phase | None phase: Phase whose target may retry a failed fixed trial or grid cell.
    :param bool possible: Include failures whose retry eligibility depends on pending reads.
    :return list[optuna.trial.FrozenTrial]: The trials to rerun, in scheduling order.
    """
    sources = _rerun_sources(trials)
    if phase is None:
        return sources
    if not phase_draws(phase):
        if (
            (possible or not sources)
            and trials
            and _trial_run_state(trials[-1]) == "FAIL"
            and _finished_trial_count(trials) < phase.n_trials
        ):
            sources.append(trials[-1])
    elif phase.sampler.type == "grid" and phase.n_trials > grid_cardinality(phase):
        waiting = sum(trial.state == optuna.trial.TrialState.WAITING for trial in trials)
        undrawn = _undrawn_grown_grid_cells(study, phase, trials)
        slots = max(
            0, phase.n_trials - _finished_trial_count(trials) - len(sources) - waiting - undrawn
        )
        failed = _failed_grid_cells(trials)
        sources.extend(failed if possible else failed[:slots])
    return sources


def _pending_rerun_count(study: optuna.Study, phase: Phase | None = None) -> int:
    """Count the reruns a phase's next run launches before any new draw.

    That is every WAITING rerun already queued plus each trial :func:`_scheduled_reruns`
    names: interrupted trials without a rerun and failed trials a raised target retries.
    All of them are among the trials the target still leaves to run.

    :param optuna.Study study: Study holding the trials and grid history.
    :param Phase | None phase: Phase whose target may retry a failed fixed trial or grid cell.
    :return int: Reruns the next run that reaches the phase launches before any new draw.
    """
    trials = study.get_trials(deepcopy=False)
    waiting = sum(
        1
        for trial in trials
        if trial.state == optuna.trial.TrialState.WAITING and RERUN_OF_ATTR in trial.user_attrs
    )
    return len(_scheduled_reruns(study, trials, phase)) + waiting


def _failed_grid_cells(
    trials: Sequence[optuna.trial.FrozenTrial],
) -> list[optuna.trial.FrozenTrial]:
    """Return the latest failure of each grid cell still eligible for retry.

    A cell with a pending or running retry, a completed attempt or an interrupted
    one is not listed: the first two are already handled and the last reruns itself.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial of one study.
    :return list[optuna.trial.FrozenTrial]: The latest failed attempt of each failed cell.
    """
    latest: dict[int, optuna.trial.FrozenTrial] = {}
    for trial in trials:
        grid_id = trial.system_attrs.get("grid_id")
        if grid_id is not None and (
            grid_id not in latest or latest[grid_id].state != optuna.trial.TrialState.COMPLETE
        ):
            latest[grid_id] = trial
    return sorted(
        (trial for trial in latest.values() if _trial_run_state(trial) == "FAIL"),
        key=lambda trial: trial.number,
    )


def _tripped_streak(abort: Mapping[str, object]) -> int | None:
    """Return the consecutive-failure count a streak abort recorded.

    :param Mapping[str, object] abort: A recorded abort.
    :return int | None: The count when ``max_consecutive_failures`` tripped, else ``None``.
    """
    if abort.get("policy") != "max_consecutive_failures":
        return None
    streak = abort.get("consecutive_failures")
    return streak if type(streak) is int else None


def _streak_trials_all_infeasible(study: optuna.Study, policy_state: _PhasePolicyState) -> bool:
    """Say whether a streak abort tripped on trials that all ended outside the bounds.

    The tripping trials are the state's ``abort_streak``. A trial whose trainer failed ends
    FAIL and has a stderr log. One that is INFEASIBLE, a COMPLETE trial outside the metric
    bounds or constraints, has none.

    :param optuna.Study study: The phase's study.
    :param _PhasePolicyState policy_state: The study's replayed failure-policy state.
    :return bool: Whether its abort is a streak abort and none of the tripping trials failed.
    """
    if policy_state.abort is None or _tripped_streak(policy_state.abort) is None:
        return False
    trials = {trial.number: trial for trial in study.get_trials(deepcopy=False)}
    return all(
        trials[number].state == optuna.trial.TrialState.COMPLETE
        for number in policy_state.abort_streak
    )


def _config_retires_abort(phase: Phase, study: optuna.Study, abort: JsonObject) -> bool:
    """Say whether the current config lets a run recover from a recorded abort.

    A raised ``n_trials`` retires any abort. A ``max_consecutive_failures`` above the
    count a streak abort recorded retires it while the target still leaves trials to run.

    :param Phase phase: Phase whose current ``n_trials`` and limit are judged.
    :param optuna.Study study: The phase's study whose finished trials are counted.
    :param JsonObject abort: The abort recorded after the recovery boundary.
    :return bool: Whether the run writes the recovery boundary before any new work.
    """
    trial_target = abort["trial_target"]
    if isinstance(trial_target, int) and phase.n_trials > trial_target:
        return True
    streak = _tripped_streak(abort)
    return (
        streak is not None
        and phase.max_consecutive_failures > streak
        and _finished_trial_count(study.get_trials(deepcopy=False)) < phase.n_trials
    )


def _awaits_cleanup_recovery(study: optuna.Study, abort: Mapping[str, object]) -> bool:
    """Say whether an unsafe-cleanup abort still waits for the recovery that retires it.

    Only a run's preflight or a confirmed ``recover-run`` proving the process group gone
    does; a raised ``n_trials`` does not.

    :param optuna.Study study: Study whose cleanup recovery ledger and trials are read.
    :param Mapping[str, object] abort: The recorded abort.
    :return bool: Whether the abort's policy is ``unsafe_process_cleanup`` and a terminal
        trial still lacks confirmed cleanup.
    """
    if abort.get("policy") != "unsafe_process_cleanup":
        return False
    recovered = _cleanup_recovered_trial_numbers(study)
    return any(
        trial.state.is_finished()
        and trial.number not in recovered
        and _trial_requires_cleanup_recovery(trial)
        for trial in study.get_trials(deepcopy=False)
    )


def _recovery_action(study: optuna.Study, abort: Mapping[str, object]) -> OperatorAction:
    """Name the operator step the recovery advice for ``abort`` leads with.

    :param optuna.Study study: The phase's study.
    :param Mapping[str, object] abort: The recorded abort.
    :return OperatorAction: Cleanup recovery while an unsafe-cleanup abort awaits it, else
        a config change.
    """
    if _awaits_cleanup_recovery(study, abort):
        return OperatorAction.RUN_RECOVER_RUN
    return OperatorAction.FIX_CONFIG


def _raised_target_step(phase: Phase, trials: Sequence[optuna.trial.FrozenTrial]) -> str | None:
    """Name the raised ``n_trials`` that retries a phase's failed work, or say none does.

    A grid or fixed configuration retries each failed cell once up to the ceiling
    :func:`~phasesweep.config.search.retry_target_ceiling` allows, and a grid runs its
    unvisited cells too, so its step names the exact target. Any other sampler only adds
    draws, so any larger target will do. The no-winner remedy and the recovery remedies give
    this one step; docs/runtime.md ("Retry fixed configurations and grids") says what a retry
    repeats.

    :param Phase phase: Phase whose configured ``n_trials`` the step raises.
    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial of the phase's study.
    :return str | None: The step, in lower case so a caller can lead into it, ending with a
        period; ``None`` when a grid or fixed configuration is at its retry ceiling.
    """
    name, target = phase.name, phase.n_trials
    finished = _finished_trial_count(trials)
    ceiling = retry_target_ceiling(phase, finished=finished, measured=_measured_draw_count(trials))
    if ceiling is None:
        return f"raise phases[{name}].n_trials above {target} and rerun."
    if ceiling <= target:
        return None
    raise_to = f"raise phases[{name}].n_trials from {target} to {ceiling} and rerun to"
    if not phase_draws(phase):
        return f"{raise_to} retry its failed attempt."
    failed = len(_failed_grid_cells(trials))
    remaining = ceiling - finished - failed
    actions = []
    if failed:
        actions.append(f"retry its {plural(failed, 'failed cell')}")
    if remaining:
        actions.append(f"run its {plural(remaining, 'remaining cell')}")
    return f"{raise_to} {' and '.join(actions)}."


def _recovery_advice(
    experiment: Experiment, phase: Phase, study: optuna.Study, abort: Mapping[str, object]
) -> str | None:
    """Say how to continue a phase whose study recorded ``abort``, or that nothing can.

    Recovery messages take their advice from here, and :func:`_recovery_action` routes it.
    Recovery retains recorded results. Bounds are not part of the study's identity: every read
    judges the recorded values under the current ones, so correcting a bound the tripping
    trials violated re-judges them. A fix that changes the study's identity meets its own
    refusal, which gives its own step.

    :param Experiment experiment: Config whose bounds judge the recorded trials.
    :param Phase phase: Phase that aborted.
    :param optuna.Study study: The phase's study.
    :param Mapping[str, object] abort: The recorded abort.
    :return str | None: The advice, or ``None`` when no remedy or explanation applies.
    """
    if _streak_trials_all_infeasible(study, _load_phase_policy_state(study, experiment)):
        remedies = _recovery_remedies(experiment, phase, study, abort, lead_in="Otherwise, ")
        if remedies is None:
            return None
        return (
            "The tripping trials were outside the metric bounds or constraints. "
            f"{REJUDGE_BOUNDS_STEP} {remedies}"
        )
    trials = study.get_trials(deepcopy=False)
    unreadable = _any_objective_unreadable(trials)
    # Corrected metric settings re-read every finished trial, so the retry is the
    # alternative for a trainer-side failure, not a step after the correction.
    objective_lead = f"{unreadable_objective_step(persistent=True)} " if unreadable else ""
    fix = (
        f"{timeout_step(phase.name)} and fix any other cause"
        if _any_trial_timed_out(trials)
        else "Fix the cause"
    )
    lead_in = f"Otherwise, {fix[0].lower()}{fix[1:]}, then " if unreadable else f"{fix}, then "
    remedies = _recovery_remedies(experiment, phase, study, abort, lead_in=lead_in)
    if remedies is None:
        return None
    return f"{objective_lead}{remedies}"


def _recovery_remedies(
    experiment: Experiment,
    phase: Phase,
    study: optuna.Study,
    abort: Mapping[str, object],
    *,
    lead_in: str,
) -> str | None:
    """Name the step that continues a phase whose study recorded ``abort``, or say none does.

    An unconfirmed unsafe-cleanup abort needs cleanup recovery before any target retires
    it. Otherwise the step is a raised ``n_trials``, which retires any abort, a grid or fixed
    configuration raising it only as far as its retry ceiling allows. At that ceiling a
    streak abort with trials left is retired by a ``max_consecutive_failures`` above the
    recorded count instead.

    :param Experiment experiment: Config whose bounds judge the recorded trials.
    :param Phase phase: Phase that aborted.
    :param optuna.Study study: The phase's study.
    :param Mapping[str, object] abort: The recorded abort.
    :param str lead_in: Words the caller puts before the raised ``n_trials`` or limit step.
    :return str | None: The step, or ``None`` when no remedy or explanation applies.
    """
    name, target = phase.name, phase.n_trials
    if _awaits_cleanup_recovery(study, abort):
        aborted_at = abort["trial_target"]
        then = (
            f"Then n_trials={target} retires the abort."
            if isinstance(aborted_at, int) and target > aborted_at
            else f"Then raise phases[{name}].n_trials above {target}."
        )
        return f"First confirm the cleanup: {RERUN_CLEANUP_RECOVERY}. {then}"
    trials = study.get_trials(deepcopy=False)
    finished = _finished_trial_count(trials)
    ceiling = retry_target_ceiling(phase, finished=finished, measured=_measured_draw_count(trials))
    streak = _tripped_streak(abort)
    step = _raised_target_step(phase, trials)
    if step is None and streak is not None and finished < target:
        step = (
            f"raise phases[{name}].max_consecutive_failures above {streak} and rerun to run the "
            f"{plural(target - finished, 'trial')} still left."
        )
    if step is not None:
        return f"{lead_in}{step}"
    if ceiling is None or ceiling > finished:
        return None
    if any(_trial_count_state(trial, experiment) == "COMPLETE" for trial in trials):
        return "Nothing is left to run or retry."
    if phase_draws(phase):
        return "Every cell was measured and none is feasible, so change the search space or bounds."
    return (
        "Its configuration was measured and is not feasible, so change fixed_overrides or "
        "the bounds."
    )


def _record_reruns(study: optuna.Study, phase: Phase | None = None) -> list[tuple[int, int]]:
    """Schedule interrupted reruns and the grid retries a raised target permits.

    Each rerun is created in one storage record carrying its source's params,
    distributions and system attrs (a grid point, a CMA-ES generation) and
    :data:`~phasesweep.engine.state.RERUN_OF_ATTR`. Optuna's ``ask`` pops
    WAITING trials before it creates new ones and returns their stored params
    without consulting the sampler, so the reruns run first with exactly their
    sources' configurations, and each takes its source's draw position.

    :param optuna.Study study: Study about to allocate work.
    :param Phase | None phase: Phase whose target above grid cardinality may retry failed cells.
    :return list[tuple[int, int]]: ``(source, rerun)`` trial numbers, interrupted
        sources first, then failed cells, each in trial-number order.
    """
    sources = _scheduled_reruns(study, study.get_trials(deepcopy=False), phase)
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


class _EnvironmentChange(NamedTuple):
    """One variable a tracked environment differs in, as the refusal shows it."""

    name: str
    #: ``NAME old -> new``, ``NAME: only recorded: ...``, ``added NAME=value`` or
    #: ``removed NAME (was value)``.
    text: str
    #: Whether the other cohort's trial recorded the variable, so that it can be restored.
    recorded: bool


def _environment_entries(old: dict[str, str], new: dict[str, str]) -> list[_EnvironmentChange]:
    """List each variable a tracked environment changed, with how it changed.

    A variable present on both sides shows both values whole, so it can be restored, except
    a search-path list such as ``PATH``, whose two values run to hundreds of characters
    apiece and differ in an entry or two: it shows only the entries that differ
    (:func:`_path_list_difference`).

    :param dict[str, str] old: Mapping a trial of the other cohort recorded.
    :param dict[str, str] new: Mapping this invocation composes.
    :return list[_EnvironmentChange]: Each differing variable, then each added name with its
        value and each removed name with its old value.
    """
    changed: list[_EnvironmentChange] = []
    added: list[_EnvironmentChange] = []
    removed: list[_EnvironmentChange] = []
    for name, before, after in _changed_payload_keys(old, new):
        if before is _ABSENT:
            added.append(
                _EnvironmentChange(name, f"added {name}={_render_value(after)}", recorded=False)
            )
        elif after is _ABSENT:
            removed.append(
                _EnvironmentChange(
                    name, f"removed {name} (was {_render_value(before)})", recorded=True
                )
            )
        elif _is_search_path(name) and isinstance(before, str) and isinstance(after, str):
            changed.append(
                _EnvironmentChange(
                    name, f"{name}: {_path_list_difference(before, after)}", recorded=True
                )
            )
        else:
            changed.append(
                _EnvironmentChange(
                    name, f"{name} {_render_value(before)} -> {_render_value(after)}", recorded=True
                )
            )
    return [*changed, *added, *removed]


def _path_list_difference(before: str, after: str) -> str:
    """Name the entries two search-path lists do not share, or show the recorded order.

    Both values come from :func:`phasesweep.engine.trial._environment_identity`, which keeps
    each entry once, so lists with the same entries differ only in order. Order decides which
    entry a lookup finds first, and the entries alone do not say how to restore it, so that
    case shows the recorded value whole.

    :param str before: Recorded value.
    :param str after: Current value.
    :return str: ``only recorded: a, b; only current: c``, with a side left out when it has
        none, or the recorded value for the same entries in another order.
    """
    recorded, current = before.split(os.pathsep), after.split(os.pathsep)
    parts = [
        f"only {side}: {', '.join(entries)}"
        for side, entries in (
            ("recorded", [entry for entry in recorded if entry not in current]),
            ("current", [entry for entry in current if entry not in recorded]),
        )
        if entries
    ]
    if parts:
        return "; ".join(parts)
    return f"the same entries in another order (recorded {_render_value(before)})"


def _describe_environment_changes(old: dict[str, str], new: dict[str, str]) -> str:
    """Render how a tracked environment differs, for the cohort refusal and the drift warning.

    :param dict[str, str] old: Mapping a trial of the other cohort recorded.
    :param dict[str, str] new: Mapping this invocation composes.
    :return str: The entries of :func:`_environment_entries` up to :data:`_MAX_REPORTED_CHANGES`
        and a count of the rest; empty when the mappings are equal.
    """
    entries = [change.text for change in _environment_entries(old, new)]
    shown = entries[:_MAX_REPORTED_CHANGES]
    if len(entries) > _MAX_REPORTED_CHANGES:
        shown.append(f"and {len(entries) - _MAX_REPORTED_CHANGES} more")
    return "; ".join(shown)


def _environment_difference(
    current: EnvironmentIdentity, trials: Sequence[optuna.trial.FrozenTrial]
) -> tuple[str, list[str], list[str]]:
    """Show what differs between this invocation's environment and another cohort's trial.

    Compares the tracked mapping against the latest of ``trials`` that recorded one. Pass-through
    names are in neither mapping, since they cannot split a cohort.

    :param EnvironmentIdentity current: Environment this invocation composes.
    :param Sequence[optuna.trial.FrozenTrial] trials: Trials from other cohorts, oldest first.
    :return tuple[str, list[str], list[str]]: A sentence showing each differing, added and
        removed variable (:func:`_environment_entries`), or one saying no trial of the other
        cohort recorded its environment; the names whose recorded value restores them; and the
        added names, which restoring unsets. The lists are empty in the second case.
    """
    for trial in reversed(trials):
        recorded = json_text_map(trial.user_attrs.get(TRAINER_ENV_ATTR))
        if recorded is None:
            # Allocation stopped before recording the mapping, or the mapping is malformed.
            continue
        changes = _environment_entries(recorded, current.tracked)
        if changes:
            restore = [change.name for change in changes if change.recorded]
            unset = [change.name for change in changes if not change.recorded]
            description = _describe_environment_changes(recorded, current.tracked)
            return f"Compared with trial {trial.number}: {description}.", restore, unset
    return _NO_RECORDED_ENVIRONMENT, [], []


def _environment_cohort_remediation(restore: Sequence[str], unset: Sequence[str]) -> str:
    """Name the way back to a study's recorded environment, or the way out of it.

    :param Sequence[str] restore: Variables whose recorded value the run needs again.
    :param Sequence[str] unset: Variables the recorded environment lacks. Both are empty when
        no trial recorded its environment.
    :return str: The step that continues the study and the one that starts a fresh study.
    """
    # Where an MCP run's environment comes from, and that pinning a variable under `env` or
    # `execution.passthrough_env` is itself a change of identity, are in docs/mcp_setup.md and
    # docs/config.md ("Trainer contract"); the fingerprint refusal gives that edit's own step.
    if restore or unset:
        parts = []
        if restore:
            parts.append("with the recorded " + ", ".join(f"`{name}`" for name in restore))
        if unset:
            parts.append("without " + ", ".join(f"`{name}`" for name in unset))
        way_back = f"Run {' and '.join(parts)} to continue this study"
    else:
        way_back = "Run in the environment that created this study to continue it"
    return f"{way_back}, or {FRESH_STUDY}."


def _validate_environment_cohort(study: optuna.Study, current: EnvironmentIdentity) -> None:
    """Refuse to allocate into a different or unknown semantic environment cohort.

    :param optuna.Study study: Existing study whose trials define the cohort.
    :param EnvironmentIdentity current: Semantic environment this invocation composes.
    :raises StudySchemaMismatchError: A populated study has a trial with
        no usable environment identity.
    :raises StudyFingerprintMismatchError: Recorded trials belong to another
        semantic environment cohort. The message shows each variable that differs,
        was added or was removed (:func:`_environment_entries`), and the remediation names
        those to restore and those to unset.
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
            f"semantic trainer-environment identity: {missing}.",
        )
    if recorded != {current.digest}:
        difference, restore, unset = _environment_difference(current, elsewhere)
        # The digests stand in for the difference only when no trial recorded its mapping.
        cohorts = (
            ""
            if restore or unset
            else f" ({', '.join(sorted(short_digest(digest) for digest in recorded))}; this run "
            f"composes {short_digest(current.digest)})"
        )
        raise StudyFingerprintMismatchError(
            f"Study {study.study_name!r} ran its trials under a different trainer "
            f"environment{cohorts}. {difference}",
            action=OperatorAction.FIX_CONFIG,
            remediation=_environment_cohort_remediation(restore, unset),
        )


#: Reserved words and builtins of dash and bash, either of which may be ``/bin/sh``.
_SHELL_OWN_WORDS = frozenset(
    [
        "case",
        "coproc",
        "do",
        "done",
        "elif",
        "else",
        "esac",
        "fi",
        "for",
        "function",
        "if",
        "in",
        "select",
        "then",
        "time",
        "until",
        "while",
        ".",
        "alias",
        "bg",
        "bind",
        "break",
        "builtin",
        "caller",
        "cd",
        "chdir",
        "command",
        "compgen",
        "complete",
        "compopt",
        "continue",
        "declare",
        "dirs",
        "disown",
        "echo",
        "enable",
        "eval",
        "exec",
        "exit",
        "export",
        "false",
        "fc",
        "fg",
        "getopts",
        "hash",
        "help",
        "history",
        "jobs",
        "kill",
        "let",
        "local",
        "logout",
        "mapfile",
        "popd",
        "printf",
        "pushd",
        "pwd",
        "read",
        "readarray",
        "readonly",
        "return",
        "set",
        "shift",
        "shopt",
        "source",
        "suspend",
        "test",
        "times",
        "trap",
        "true",
        "type",
        "typeset",
        "ulimit",
        "umask",
        "unalias",
        "unset",
        "wait",
    ]
)
_PLAIN_PROGRAM = re.compile(r"[A-Za-z0-9_./+-]+")


def _require_launch_target(experiment: Experiment, *, path: str | None = None) -> None:
    """Refuse a trainer directory that does not exist or a trial program the shell cannot run.

    Without ``path`` no shell's ``PATH`` changes the answer, so ``validate`` and ``mcp check``
    call it that way. A launch passes the trainer's ``PATH`` in
    :func:`_launch_preflight`, because the ``PATH`` that counts is the one the run starts with.
    With ``execution.cwd`` unset the trainer starts where PhaseSweep was started, so a relative
    program is judged against that directory, as a run started there would resolve it.

    :param Experiment experiment: Experiment whose ``execution.cwd`` and trial command are checked.
    :param str | None path: ``PATH`` the trainer receives, or ``None`` to judge only a program
        written with a ``/``, which no ``PATH`` changes.
    :raises LaunchEnvironmentError: The trainer directory or plain program is known
        to be unavailable.
    """
    # MCP shows a step without the diagnosis, so each step names its directory itself.
    try:
        trainer_cwd = experiment.execution.resolved_cwd()
        missing = _missing_trainer_cwd(experiment)
        if missing is not None:
            raise LaunchEnvironmentError(
                missing,
                remediation=(
                    f"Create {trainer_cwd}, or set execution.cwd to an existing directory."
                ),
            )
        cwd = trainer_cwd or Path.cwd()
    except PermissionError as exc:
        where = exc.filename or "the trainer directory"
        raise LaunchEnvironmentError(
            f"Cannot access the trainer directory: {exc}.",
            remediation=f"Restore access to {where}.",
        ) from exc
    except OSError:
        return
    refusal = _trial_program_refusal(
        experiment.trial_command,
        path=path,
        cwd=cwd,
        path_from_config="PATH" in experiment.env,
    )
    if refusal is not None:
        message, remediation = refusal
        raise LaunchEnvironmentError(message, remediation=remediation)


def _is_executable_file(path: Path) -> bool | None:
    """Check for a regular executable when its path can be inspected.

    :param Path path: Program candidate as resolved from the trainer's directory.
    :return bool | None: Whether the candidate is executable, or ``None`` if unknown.
    """
    try:
        return S_ISREG(path.stat().st_mode) and os.access(path, os.X_OK)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return None


def _trial_program_refusal(
    command: str, *, path: str | None, cwd: Path, path_from_config: bool = False
) -> tuple[str, str] | None:
    """Judge whether the shell that runs a trial command can find its program.

    A plain first word is required even with quoted arguments, redirection or an AND-list.
    Leave command substitution, parameter assignments and other control flow to the shell.

    :param str command: The trial command template.
    :param str | None path: ``PATH`` the trainer receives, or ``None`` to judge only a program
        written with a ``/``.
    :param Path cwd: Directory the trainer starts in, which relative paths resolve against.
    :param bool path_from_config: The config's ``env`` sets ``PATH``, so neither the shell nor
        the environment the MCP server is started with changes the one searched.
    :return tuple[str, str] | None: Refusal diagnosis and remedy, or ``None`` when the program
        is found or cannot be judged.
    """
    text = command
    if "#" not in text:
        # The shell removes a continued newline without inserting a word boundary.
        text = re.sub(r"(?<!\\)\\\n", "", text)
    text = text.strip(" \t\n")
    # Only an unescaped `&&` or descriptor duplication can account for a bare ampersand.
    control_text = re.sub(r"(?<!\\)(?:&&|[<>]&(?:[0-9]+|-)(?=$|[ \t<>]))", "", text)
    if any(char in control_text for char in "\n\r();|&`") or "${" in text:
        return None
    program = re.split(r"[ \t]", text, maxsplit=1)[0]
    if program in _SHELL_OWN_WORDS or _PLAIN_PROGRAM.fullmatch(program) is None:
        return None
    if "/" in program:
        resolved = cwd / program
        if _is_executable_file(resolved) is not False:
            return None
        where = (
            f"is {program!r}" if str(resolved) == program else f"{program!r} resolves to {resolved}"
        )
        return (
            f"The trial command's program {where}, which is not an executable file.",
            f"Correct {program!r} in trial_command, or make {resolved} an executable file.",
        )
    if path is None:
        return None
    if any(
        _is_executable_file(cwd / entry / program) is not False for entry in path.split(os.pathsep)
    ):
        return None
    # The other routes, `env.PATH` and an absolute program path, change a recorded study's
    # identity; docs/config.md ("Trainer contract") names them.
    source = (
        "the PATH the config's `env` sets"
        if path_from_config
        else "the PATH PhaseSweep is launched with"
    )
    # MCP shows the step without the diagnosis, so the step names the program itself.
    return (
        f"The trial command's program {program!r} is not on the trainer's PATH.",
        f"Put {program!r} on {source}.",
    )


def _launch_preflight(
    experiment: Experiment, phase_name: str, study: optuna.Study | None
) -> EnvironmentIdentity:
    """Refuse a phase launch whose trials could not start, before any trial is allocated.

    The cohort check comes first, so a changed ``PATH`` reads as a cohort refusal that names
    the entries that differ, not as a missing program.

    :param Experiment experiment: Experiment supplying the trainer environment, working
        directory and trial command.
    :param str phase_name: Phase about to allocate trials.
    :param optuna.Study | None study: The phase's existing study, or ``None`` when it has none.
    :return EnvironmentIdentity: The trainer environment the phase launches with.
    :raises StudySchemaMismatchError: A populated study has a trial with no usable
        environment identity.
    :raises StudyFingerprintMismatchError: The study's trials ran under another environment
        cohort.
    :raises LaunchEnvironmentError: The trainer environment cannot be composed,
        ``execution.cwd`` is not a directory, or the trial command's program cannot be found.
    """
    identity = _environment_identity(experiment, phase_name)
    if study is not None:
        _validate_environment_cohort(study, identity)
    _require_launch_target(experiment, path=identity.values.get("PATH"))
    return identity


class ReachedStudy(NamedTuple):
    """A reached phase's existing study, as the continuation preflight checks it."""

    phase: Phase
    study: optuna.Study
    siblings: Sequence[optuna.Study] = ()


def _reached_study_refusals(
    experiment: Experiment, reached: Sequence[ReachedStudy]
) -> list[Exception]:
    """Check each reached study before any phase runs: identity, target, launch, then evidence.

    A run (:func:`phasesweep.engine.guards._preflight_existing_studies`) and its dry run
    (:func:`phasesweep.engine.preview._refuse_as_a_launch_would`) both call this, so they refuse
    in one order, and each step assumes the ones before it passed. A changed config comes first
    and alone, for the first study in phase order that recorded another, because restoring it or
    renaming the phase decides every later check. Target, launch and selection-evidence refusals
    (:func:`phasesweep.engine.evidence._validate_selection_evidence`, which also notes trials
    whose directory is gone) are collected per study. A phase's actual inherited values, and
    the evidence of the study they address, are checked once its parents have selected winners.

    :param Experiment experiment: Current experiment config.
    :param Sequence[ReachedStudy] reached: The reached phases' studies, in phase order, each
        already past its direction and schema checks.
    :return list[Exception]: The refusals in phase order; empty when every study passes.
    """
    from phasesweep.engine.fingerprints import _recorded_inheritance
    from phasesweep.engine.reread import plan_adoption

    for item in reached:
        try:
            _verify_recorded_identity(item.study, experiment, item.phase)
        except StudyFingerprintMismatchError as exc:
            return [exc]
    errors: list[Exception] = []
    for item in reached:
        phase, study = item.phase, item.study
        try:
            record = _recorded_identity(study)
            # The run judges the study once it adopts kept attempts: a re-read assumed to
            # fail checks the target, and one assumed to measure decides the launch.
            plan = plan_adoption(
                experiment,
                phase,
                {} if record is None else _recorded_inheritance(record),
                study,
                item.siblings,
                study_name=study.study_name,
            )
            judged = study if plan is None else plan.failed
            _validate_trial_target(judged, phase)
            partial = _load_accepted_partial_decision(judged)
            finished = _finished_trial_count(judged.get_trials(deepcopy=False))
            needs_new_trials = (
                phase.n_trials > finished and not _at_accepted_partial_target(partial, phase)
                if plan is None
                else plan.launches[0] > 0
            )
            # Preflight a definite launch. A kept result may cover a budgeted retry, but
            # if its read cannot measure the trial, the reached phase preflights before launching.
            if needs_new_trials:
                _launch_preflight(experiment, phase.name, study if plan is None else plan.complete)
            _validate_selection_evidence(experiment, {phase.name: study})
        except Exception as exc:
            errors.append(exc)
    return errors
