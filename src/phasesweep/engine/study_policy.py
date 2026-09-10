"""Persisted study policy and continuation validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import optuna

from phasesweep.config import Phase
from phasesweep.config.search import NON_RESUMABLE_SAMPLERS
from phasesweep.engine.attempts import _parsed_trial_outcome
from phasesweep.engine.errors import (
    SamplerContinuationUnsupportedError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    TrialTargetRegressionError,
)
from phasesweep.engine.state import (
    PHASE_DECISION_ATTR,
    PHASE_DECISION_SCHEMA_VERSION,
    PHASE_RECOVERY_ATTR,
    PHASE_RECOVERY_SCHEMA_VERSION,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
    TRAINER_ENV_DIGEST_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_TARGET_ATTR,
)


@dataclass(frozen=True)
class _PhasePolicyState:
    """Failure-policy state reconstructed from durable per-trial outcomes."""

    max_sequence: int
    consecutive_failures: int
    recovered_abort_sequence: int | None
    fatal_trial_number: int | None
    fatal_sequence: int | None
    fatal_cause: str | None
    fatal_policy: str | None


@dataclass(frozen=True)
class _AcceptedPartialDecision:
    """Durable terminal decision to select from an incomplete timed-out phase."""

    trial_target: int
    outcome_sequence: int
    finished_trials: int
    completed_trials: int
    timeout_scope: str
    recovered_abort_sequence: int | None


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
        recovered_abort_sequence=recovered_abort_sequence,
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


def _load_phase_policy_state(study: optuna.Study) -> _PhasePolicyState:
    """Validate and reconstruct the durable consecutive-failure state.

    :param optuna.Study study: Study whose ``PHASE_RECOVERY_ATTR`` and
        per-trial outcome attrs are read and validated.
    :return _PhasePolicyState: Reconstructed state: the highest recorded
        outcome sequence, the consecutive-failure count since the last
        recovery boundary, the recovered abort sequence (if any), and the
        first fatal trial's number/sequence/cause (if any).
    :raises StudySchemaMismatchError: ``PHASE_RECOVERY_ATTR`` or a terminal
        trial's outcome attr is malformed, two trials share a completion
        sequence, or the recovery boundary exceeds the largest recorded
        sequence.
    """
    recovery = study.user_attrs.get(PHASE_RECOVERY_ATTR)
    recovery_boundary = 0
    recovered_abort_sequence: int | None = None
    if recovery is not None:
        if not isinstance(recovery, dict):
            raise _phase_policy_schema_error(
                study, f"{PHASE_RECOVERY_ATTR!r} must be an object, got {recovery!r}"
            )
        schema_version = recovery.get("schema_version")
        raw_recovery_boundary = recovery.get("start_after_sequence")
        raw_recovered_abort_sequence = recovery.get("recovered_abort_sequence")
        recovery_target = recovery.get("trial_target")
        if (
            schema_version != PHASE_RECOVERY_SCHEMA_VERSION
            or type(raw_recovery_boundary) is not int
            or raw_recovery_boundary < 1
            or type(raw_recovered_abort_sequence) is not int
            or raw_recovered_abort_sequence < 1
            or raw_recovered_abort_sequence > raw_recovery_boundary
            or type(recovery_target) is not int
            or recovery_target < 1
        ):
            raise _phase_policy_schema_error(
                study, f"{PHASE_RECOVERY_ATTR!r} has malformed fields: {recovery!r}"
            )
        recovery_boundary = raw_recovery_boundary
        recovered_abort_sequence = raw_recovered_abort_sequence

    events: list[tuple[int, int, str, str | None, str | None]] = []
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
        outcome = parsed.outcome
        cause = parsed.cause
        policy = parsed.policy
        other_trial = seen_sequences.get(sequence)
        if other_trial is not None:
            raise _phase_policy_schema_error(
                study,
                f"trials {other_trial} and {trial.number} both use completion sequence {sequence}",
            )
        seen_sequences[sequence] = trial.number
        events.append((sequence, trial.number, outcome, cause, policy))

    events.sort()
    max_sequence = events[-1][0] if events else 0
    if recovery_boundary > max_sequence:
        raise _phase_policy_schema_error(
            study,
            f"{PHASE_RECOVERY_ATTR!r} starts after sequence {recovery_boundary}, "
            f"but the largest recorded sequence is {max_sequence}",
        )

    consecutive_failures = 0
    fatal_trial_number: int | None = None
    fatal_sequence: int | None = None
    fatal_cause: str | None = None
    fatal_policy: str | None = None
    for sequence, trial_number, outcome, cause, policy in events:
        if sequence <= recovery_boundary:
            continue
        if outcome == "success":
            consecutive_failures = 0
        elif outcome in {"failure", "fatal"}:
            consecutive_failures += 1
        if outcome == "fatal" and fatal_trial_number is None:
            fatal_trial_number = trial_number
            fatal_sequence = sequence
            fatal_cause = cause
            fatal_policy = policy

    return _PhasePolicyState(
        max_sequence=max_sequence,
        consecutive_failures=consecutive_failures,
        recovered_abort_sequence=recovered_abort_sequence,
        fatal_trial_number=fatal_trial_number,
        fatal_sequence=fatal_sequence,
        fatal_cause=fatal_cause,
        fatal_policy=fatal_policy,
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
        "Use a new experiment name, or archive/delete the old study before running again."
    )


def _validate_study_direction(
    study: optuna.Study,
    goal: Literal["minimize", "maximize"],
) -> None:
    """Reject a durable study whose objective direction differs from the config.

    Optuna's ``load_if_exists=True`` keeps the stored direction and silently
    ignores the direction supplied by a later caller. PhaseSweep therefore
    validates the durable value explicitly before any trial can be launched.

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
    """Return the durable target, inferring old current-schema studies from history.

    :param optuna.Study study: Study whose accepted trial target is read.
    :return int: The stored ``phasesweep_trial_target`` user attr, or the
        number of finished trials when no target has been recorded yet.
    :raises StudySchemaMismatchError: The stored target is not a positive int,
        or is lower than the number of already-finished trials.
    """
    finished = sum(1 for trial in study.get_trials(deepcopy=False) if trial.state.is_finished())
    stored = study.user_attrs.get(TRIAL_TARGET_ATTR)
    if stored is None:
        return finished
    if type(stored) is not int or stored < 1 or finished > stored:
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} has invalid {TRIAL_TARGET_ATTR!r}={stored!r} "
            f"for {finished} terminal trial(s). Use a new experiment name, or archive/delete "
            "the inconsistent study before running again."
        )
    return stored


def _validate_trial_target(study: optuna.Study, phase: Phase) -> None:
    """Reject a target lower than the study's durable accepted target.

    :param optuna.Study study: Existing study whose accepted target is checked.
    :param Phase phase: Phase config supplying the requested ``n_trials`` target.
    :raises TrialTargetRegressionError: ``phase.n_trials`` is lower than the
        study's durable accepted target.
    """
    accepted_target = _accepted_trial_target(study)
    if phase.n_trials < accepted_target:
        raise TrialTargetRegressionError(
            f"Phase {phase.name!r} has already accepted a target of {accepted_target} terminal "
            f"trial(s), but the current config requests {phase.n_trials}. Use at least the prior "
            "target or a new experiment name."
        )


def _validate_sampler_continuation(study: optuna.Study, phase: Phase) -> None:
    """Reject any cross-process continuation of a partially complete stateful study.

    TPE and CMA-ES suggestions depend on process-local RNG/optimizer state that
    Optuna storage does not persist. Recreating a seeded sampler in a fresh
    process restarts that stream, so a mid-target resume can exactly repeat
    already-evaluated startup suggestions and spend the remaining budget on
    duplicates. Until PhaseSweep persists real sampler continuation state, a
    stateful phase is restartable only before its first terminal trial or after
    reaching its accepted target.

    :param optuna.Study study: Existing study whose finished-trial count is checked.
    :param Phase phase: Phase config supplying the sampler type and trial target.
    :raises SamplerContinuationUnsupportedError: The phase uses a stateful sampler
        (``tpe`` or ``cmaes``) and either raises its previously accepted trial
        target or was interrupted before reaching it.
    """
    finished = sum(1 for trial in study.get_trials(deepcopy=False) if trial.state.is_finished())
    if phase.sampler.type not in NON_RESUMABLE_SAMPLERS or finished == 0:
        return

    accepted_target = _accepted_trial_target(study)
    partial_decision = _load_accepted_partial_decision(study)
    if partial_decision is not None and phase.n_trials == partial_decision.trial_target:
        # An accepted partial timeout is terminal at its frozen target.
        # Identical replay launches no suggestions, so no process-local
        # sampler state needs to be reconstructed.
        return
    if phase.n_trials > accepted_target:
        raise SamplerContinuationUnsupportedError(
            f"Phase {phase.name!r} uses {phase.sampler.type!r} and raises its accepted target "
            f"from {accepted_target} to {phase.n_trials} terminal trial(s). PhaseSweep cannot "
            "reproduce this sampler's process-local continuation state safely. Use a new "
            "experiment name, or run the full target in one invocation."
        )
    if finished < accepted_target:
        raise SamplerContinuationUnsupportedError(
            f"Phase {phase.name!r} uses {phase.sampler.type!r} and was interrupted at "
            f"{finished}/{accepted_target} terminal trial(s). PhaseSweep cannot reconstruct "
            "this sampler's exact process-local continuation state, so resuming could "
            "re-evaluate identical suggestions and waste the remaining budget. Use a new "
            "experiment name (optionally with a stateless random/grid sampler), or run the "
            "full target in one uninterrupted invocation."
        )


def _record_trial_target(study: optuna.Study, phase: Phase) -> None:
    """Persist the highest accepted target before the phase launches work.

    :param optuna.Study study: Study whose accepted trial target is stored.
    :param Phase phase: Phase config supplying the new ``n_trials`` target.
    :raises TrialTargetRegressionError: ``phase.n_trials`` is lower than the
        study's already-accepted target.
    """
    _validate_trial_target(study, phase)
    if phase.n_trials != study.user_attrs.get(TRIAL_TARGET_ATTR):
        study.set_user_attr(TRIAL_TARGET_ATTR, phase.n_trials)


def _validate_environment_cohort(study: optuna.Study, current_digest: str) -> None:
    """Refuse to allocate into a different or unknown semantic environment cohort.

    :param optuna.Study study: Existing study whose trials define the cohort.
    :param str current_digest: Semantic environment digest for this invocation.
    :raises StudySchemaMismatchError: A populated legacy study has a trial with
        no usable environment identity.
    :raises StudyFingerprintMismatchError: Recorded trials belong to another
        semantic environment cohort.
    """
    trials = study.get_trials(deepcopy=False)
    if not trials:
        return
    missing = [
        trial.number
        for trial in trials
        if not isinstance(trial.user_attrs.get(TRAINER_ENV_DIGEST_ATTR), str)
        or not trial.user_attrs.get(TRAINER_ENV_DIGEST_ATTR)
    ]
    if missing:
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} contains populated legacy trial(s) without a "
            f"semantic trainer-environment identity: {missing}. PhaseSweep cannot guess "
            "which environment cohort owns those results. Use a new experiment name, or "
            "archive/delete the legacy study before running again."
        )
    recorded = {trial.user_attrs[TRAINER_ENV_DIGEST_ATTR] for trial in trials}
    if recorded != {current_digest}:
        rendered = ", ".join(sorted(digest[:12] for digest in recorded))
        raise StudyFingerprintMismatchError(
            f"Study {study.study_name!r} contains trial(s) from semantic trainer "
            f"environment cohort(s) [{rendered}], but this invocation composes "
            f"{current_digest[:12]}. No trial was allocated. Restore the original "
            "semantic environment, classify rotating credentials under "
            "execution.passthrough_env, or use a new experiment name."
        )
