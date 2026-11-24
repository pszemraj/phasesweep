"""Phase execution through Optuna."""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple
from uuid import uuid4

import optuna

from phasesweep.config.models import Constraint, Experiment, Metric, Phase
from phasesweep.engine.artifacts import _warn_environment_drift, _write_trials_csv
from phasesweep.engine.attempts import (
    _active_attempt_persistence_error,
    _cleanup_recovered_trial_numbers,
    _register_active_attempt,
    _retire_active_attempt,
    _trial_requires_cleanup_recovery,
)
from phasesweep.engine.cleanup import _reap_stale_trials
from phasesweep.engine.errors import (
    ActiveAttemptPersistenceError,
    PhaseTimeoutError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    WinnerIntegrityError,
    phase_timeout,
    tracker_unreadable,
)
from phasesweep.engine.evidence import (
    _earlier_attempts,
    _validate_selection_evidence,
    _verify_winner_evidence,
)
from phasesweep.engine.fingerprints import (
    _inherited_override_payload,
    _inherited_override_suppliers,
    _verify_fingerprint,
    new_study_note,
    phase_study_name,
)
from phasesweep.engine.generation import InvocationGeneration
from phasesweep.engine.ledger import ClaimedLedger, open_phase_study
from phasesweep.engine.optuna import (
    _completed_trial_count,
    _finished_trial_count,
    _suggest_all,
)
from phasesweep.engine.paths import _attempt_dir, _phase_dir, _trials_csv_path
from phasesweep.engine.progress import (
    _log_finished_trial,
    _log_reruns,
    _log_trial_start,
    _warn_deferred_extraction,
)
from phasesweep.engine.repeats import repeat_axis
from phasesweep.engine.selection import (
    NoFeasibleTrialError,
    PhaseTrialsFailedError,
    PhaseTrialsInfeasibleError,
    ReplayWithoutFeasibleTrialError,
    _phase_refusal,
    select_winner,
)
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    DURATION_ATTR,
    FAILURE_REASON_ATTR,
    GENERATION_ID_ATTR,
    OVERRIDES_ATTR,
    PHASE_DECISION_ATTR,
    PHASE_DECISION_SCHEMA_VERSION,
    RERUN_OF_ATTR,
    RETURN_CODE_ATTR,
    TRAINER_ENV_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    Winner,
    WinnerSource,
    _completed_trainer_duration,
    _parsed_trial_outcome,
    _trial_attempt_dir,
    _trial_lifecycle,
    _trial_outcome_payload,
)
from phasesweep.engine.study_policy import (
    _abort_record,
    _AcceptedPartialDecision,
    _at_accepted_partial_target,
    _consecutive_failure_threshold_tripped,
    _failure_policy_abort_record,
    _initialize_study_schema,
    _launch_preflight,
    _load_accepted_partial_decision,
    _load_phase_policy_state,
    _next_consecutive_failures,
    _pending_rerun_count,
    _PhasePolicyState,
    _record_allocation_context,
    _record_recovery_boundary,
    _record_reruns,
    _record_trial_target,
    _recovery_action,
    _recovery_advice,
    _replayed_streak_abort,
    _streak_trials_all_infeasible,
    _validate_study_direction,
    _validate_trial_target,
)
from phasesweep.engine.trial import (
    _STOPPED_BY_PHASESWEEP,
    EnvironmentIdentity,
    TrialExecutionError,
    _trainer_failure_reason,
    extract_trial_result,
    launch_trial,
    prepare_trainer_input,
    record_trial_result,
    trial_run_name,
)
from phasesweep.engine.trial_dirs import _trial_dir_for_attempt
from phasesweep.engine.workers import _await_worker_trials, _WorkerTrials
from phasesweep.errors import (
    ABORT_START_AGAIN_STEP,
    ACCEPT_INCOMPLETE_STEP,
    OperatorAction,
    PhaseSweepError,
    ProcessCleanupUncertainError,
    UnsafeProcessCleanupError,
    as_sentence,
)
from phasesweep.run_report import phase_start, report
from phasesweep.runtime.gpu import GpuLeaseCancelledError, GpuLeaseTimeoutError, GpuPool
from phasesweep.runtime.json import JsonObject
from phasesweep.runtime.process import write_attempt_lifecycle
from phasesweep.runtime.shutdown import (
    PhaseSweepShutdown,
    ShutdownCleanupReport,
    ShutdownLaunchRefusedError,
    interruption_cause,
    servicing_shutdown_signal,
    shutdown_raised_over,
)
from phasesweep.runtime.terminal import plural

if TYPE_CHECKING:
    from phasesweep.tracking.session import DeliveryTally

log = logging.getLogger("phasesweep.engine.phase")


def _winner_completion(
    *,
    requested_trials: int,
    finished_trials: int,
    completed_trials: int,
    incomplete: bool,
    reason: str | None,
    timeout_scope: str | None,
) -> dict[str, Any]:
    """Build the completion metadata a phase's :class:`Winner` persists.

    :param int requested_trials: Trial target the phase was configured to reach.
    :param int finished_trials: Finished trial count backing this completion.
    :param int completed_trials: Completed trial count backing this completion, the
        finished trials inside every metric bound and constraint.
    :param bool incomplete: Whether the phase stopped short of its trial target.
    :param str | None reason: Why the phase is incomplete, or ``None`` when complete.
    :param str | None timeout_scope: Timeout guard that stopped the phase, or ``None``
        when the phase did not stop on a timeout.
    :return dict[str, Any]: Completion metadata in the shape ``Winner.completion`` expects.
    """
    return {
        "requested_trials": requested_trials,
        "finished_trials": finished_trials,
        "completed_trials": completed_trials,
        "incomplete": incomplete,
        "reason": reason,
        "timeout_scope": timeout_scope,
    }


def _partial_completion_for_replay(
    study: optuna.Study,
    decision: _AcceptedPartialDecision,
    *,
    outcome_sequence: int,
) -> dict[str, Any]:
    """Re-prove and return the frozen completion metadata for a decision replay.

    :param optuna.Study study: Study whose current finished and completed counts are verified.
    :param _AcceptedPartialDecision decision: Persisted accepted-partial decision.
    :param int outcome_sequence: Current durable outcome sequence for the study.
    :raises StudySchemaMismatchError: Trial counts or the outcome sequence changed
        after the accepted-partial decision was committed.
    :return dict[str, Any]: Frozen incomplete-completion metadata for publication.
    """
    trials = study.get_trials(deepcopy=False)
    if (
        outcome_sequence != decision.outcome_sequence
        or _finished_trial_count(trials) != decision.finished_trials
        or _completed_trial_count(trials) != decision.completed_trials
    ):
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} changed after its accepted partial-timeout "
            "decision was committed."
        )
    return _winner_completion(
        requested_trials=decision.trial_target,
        finished_trials=decision.finished_trials,
        completed_trials=decision.completed_trials,
        incomplete=True,
        reason="timeout",
        timeout_scope=decision.timeout_scope,
    )


class _DeadlineTrialExecutionError(TrialExecutionError):
    """Raised when a trial fails specifically because a run deadline expired."""


class _UnlaunchedAbortError(TrialExecutionError):
    """Raised when a phase abort stops a trial before its trainer launched.

    The trial has no trainer outcome to count, so it is recorded as
    interrupted and reruns once the abort is recovered by raising ``n_trials``.
    """


class _InterruptedTrialError(TrialExecutionError):
    """Raised when a shutdown stopped a trial's trainer, or kept it from starting.

    A trainer that ended by itself keeps its own verdict even while a shutdown is serviced.
    """


_UNLAUNCHED_ABORT_CAUSE = "phase aborted before this trial's trainer launched"


class _TrialLeftRunning(BaseException):
    """Keep a trial RUNNING for evidence or journal recovery and stop the phase with ``error``.

    Optuna does not catch this BaseException subclass, so it commits no terminal
    state. A preparation worker with unresolved cleanup leaves its unlaunched trial for recovery.
    After workers drain, ``_run_phase`` raises ``error``, unless a worker's
    unconfirmed process cleanup outranks it. Stale recovery
    then settles the trial: it records an outcome that could not be persisted
    before committing the terminal state, or extracts finished training once
    the tracker that gave no answer about it answers.
    """

    def __init__(self, error: PhaseSweepError) -> None:
        """Carry the operator error the phase stops with.

        :param PhaseSweepError error: Error ``_run_phase`` raises after workers drain.
        """
        super().__init__(str(error))
        self.error = error


_OUTCOME_WRITE_RETRY_DELAYS = (0.05, 0.25)
_OUTCOME_WRITE_ATTEMPTS = len(_OUTCOME_WRITE_RETRY_DELAYS) + 1


@dataclass
class CsvSnapshotThrottle:
    """Debounce expensive full ``trials.csv`` snapshots during a phase."""

    min_trials: int = 10
    min_seconds: float = 30.0
    last_finished: int = 0
    last_write_at: float = 0.0

    def should_write(self, finished: int, now: float) -> bool:
        """Return whether another full CSV snapshot should be written.

        :param int finished: Current number of finished trials.
        :param float now: Current timestamp in the throttle's clock domain.
        :return bool: Whether either the trial or elapsed-time threshold was reached.
        """
        return (
            finished - self.last_finished >= self.min_trials
            or now - self.last_write_at >= self.min_seconds
        )

    def mark_written(self, finished: int, now: float) -> None:
        """Record a successful snapshot write.

        :param int finished: Number of finished trials included in the snapshot.
        :param float now: Snapshot timestamp in the throttle's clock domain.
        """
        self.last_finished = finished
        self.last_write_at = now


def _composed_overrides(
    experiment: Experiment,
    phase: Phase,
    sampled: dict[str, object],
    inherited_winners: dict[str, Winner],
) -> dict[str, object]:
    """Merge inherited winners, fixed overrides, and sampled params.

    :param Experiment experiment: The experiment whose phase graph says which
        inherited keys an ancestor sampled.
    :param Phase phase: The phase whose ``fixed_overrides`` and inheritance list apply.
    :param dict[str, object] sampled: Values Optuna suggested for this trial.
    :param dict[str, Winner] inherited_winners: Parent-phase winners supplying
        the lowest-priority override layer.
    :raises ValueError: If the sampled parameters try to replace a value that
        an ancestor phase sampled. A sampled value replaces an inherited fixed
        one, and a repeat value replaces any inherited one.
    :return dict[str, object]: Fully composed overrides for the trainer command.
    """
    repeat_keys = set() if phase.repeats is None else {phase.repeats.key}
    resampled = experiment.sampled_locked_keys(phase.name) & (set(sampled) - repeat_keys)
    if resampled:
        raise ValueError(
            f"Phase {phase.name!r} re-samples inherited winner key(s) {sorted(resampled)} "
            "that an ancestor sampled."
        )
    out = {
        key: inherited_winners[parent].effective_overrides[key]
        for key, parent in _inherited_override_suppliers(phase, inherited_winners).items()
    }
    out.update(phase.fixed_overrides)
    out.update(sampled)
    return out


def _winner_overrides(
    experiment: Experiment,
    phase: Phase,
    params: dict[str, object],
    inherited_winners: dict[str, Winner],
) -> dict[str, object]:
    """Compose a winner's effective overrides, which never carry the repeat key.

    A repeats winner stands on members that each ran a different repeat value,
    so no single value of that key, and no inherited value it replaced, describes
    the winner; children never inherit it.

    :param Experiment experiment: The experiment owning the phase graph.
    :param Phase phase: The phase whose winner is composed.
    :param dict[str, object] params: The winner's params, without the repeat key.
    :param dict[str, Winner] inherited_winners: Parent-phase winners.
    :return dict[str, object]: Composed overrides without the repeat key.
    """
    effective: dict[str, object] = _composed_overrides(
        experiment, phase, dict(params), inherited_winners
    )
    if phase.repeats is not None:
        effective.pop(phase.repeats.key, None)
    return effective


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

    A live phase execution and stale-trial recovery both decide here, so an outcome trips the
    same abort whichever of them records it.

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


@dataclass(frozen=True)
class _PhaseResume:
    """A live phase's durable state, validated before it may launch new work."""

    policy_state: _PhasePolicyState
    phase_fingerprint: str
    partial_decision: _AcceptedPartialDecision | None
    # A durable abort that the raised n_trials or max_consecutive_failures authorizes this run
    # to recover from.
    recovery_abort: dict[str, Any] | None


def _resume_phase(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    study: optuna.Study,
) -> _PhaseResume | Winner:
    """Validate a live phase's durable state and replay what it already decided.

    Reaps stale trials, replays the failure policy from the durable outcomes,
    and verifies the phase fingerprint and trial target. An accepted
    partial-timeout decision at the current target replays as the phase's
    winner while the current config still accepts incomplete results. An abort
    recorded in the outcome ledger, or a replayed streak that meets the current
    limit while work remains, is refused unless the raised ``n_trials`` or, for
    a streak abort with trials left to run, a ``max_consecutive_failures``
    above the recorded count authorizes recovering from it.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: The phase being resumed.
    :param dict[str, Winner] inherited_winners: Winners loaded for phases
        earlier in the chain.
    :param optuna.Study study: The phase's live study.
    :raises StudySchemaMismatchError: The study's schema or stored direction is
        unsupported; what the recorded state contradicts is listed with
        :func:`_judge_recorded_phase`, whose other refusals also propagate.
    :raises StudyFingerprintMismatchError: The study was produced by a
        different config.
    :raises ProcessCleanupUncertainError: A stale trial's process could not be
        proven gone.
    :return _PhaseResume | Winner: The validated state, or the winner an
        accepted partial-timeout decision replays.
    """
    _validate_study_direction(study, experiment.metric.goal)
    _initialize_study_schema(study)
    _reap_stale_trials(study, experiment, phase.name, confirm=True)
    policy_state = _load_phase_policy_state(study)
    phase_fingerprint = _verify_fingerprint(study, experiment, phase, inherited_winners)
    return _judge_recorded_phase(
        experiment,
        phase,
        inherited_winners,
        study,
        policy_state=policy_state,
        phase_fingerprint=phase_fingerprint,
        verify_evidence=True,
    )


def _no_feasible_before_deadline(
    experiment: Experiment, phase_name: str, timeout_scope: str
) -> PhaseTimeoutError:
    """Describe a deadline that prevented the phase from producing a feasible winner.

    :param Experiment experiment: Experiment whose storage decides what a rerun keeps.
    :param str phase_name: Phase whose winner cannot be selected.
    :param str timeout_scope: Deadline responsible for exhausting feasible work.
    :return PhaseTimeoutError: Refusal shared by live completion and recorded-result replay.
    """
    return phase_timeout(
        f"Phase {phase_name!r} hit its {timeout_scope} deadline "
        "before any trial completed inside the metric bounds and constraints; no winner can "
        "be selected.",
        scope=timeout_scope,
        experiment=experiment,
    )


def _judge_recorded_phase(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    study: optuna.Study,
    *,
    policy_state: _PhasePolicyState,
    phase_fingerprint: str,
    verify_evidence: bool,
) -> _PhaseResume | Winner:
    """Apply the trial-target, partial-decision and abort rules to a phase's recorded study.

    Replays the study without writing it. When ``verify_evidence`` is true,
    the selected winner is checked against its frozen evidence and its
    environment is compared with the current run.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: The phase being judged.
    :param dict[str, Winner] inherited_winners: Winners for phases earlier in the chain.
    :param optuna.Study study: The phase's study, live or a read-only snapshot.
    :param _PhasePolicyState policy_state: The study's replayed failure-policy state.
    :param str phase_fingerprint: Fingerprint carried on a replayed winner.
    :param bool verify_evidence: Whether a replayed winner's evidence is re-proved;
        see :func:`_select_phase_winner` for everything it covers.
    :raises TrialTargetRefusedError: ``phase.n_trials`` is below the accepted
        target, or a fixed configuration asks for more attempts than it may retry.
    :raises PhaseAbortRefusedError: The phase aborted and the current config does not
        retire the abort (see :func:`~phasesweep.engine.study_policy._config_retires_abort`),
        a trial of its unsafe-cleanup abort still awaits cleanup recovery, or its streak
        meets the current limit with work remaining, at an accepted target its current
        ``n_trials`` does not exceed.
    :raises WinnerIntegrityError: An accepted partial-timeout decision would
        replay while the current config disables ``allow_incomplete_on_timeout``.
    :raises TimeoutError: An accepted partial-timeout decision replays with no
        feasible trial.
    :raises StudySchemaMismatchError: Durable phase state is malformed or
        contradicts the study's trials.
    :raises TrialEvidenceMissingError: A replayed winner's evidence is missing
        or altered; only when ``verify_evidence`` is set.
    :return _PhaseResume | Winner: The validated state, or the winner an
        accepted partial-timeout decision replays.
    """
    _validate_trial_target(study, phase)
    partial_decision = _load_accepted_partial_decision(study)
    if _at_accepted_partial_target(partial_decision, phase):
        # Replay the accepted timeout boundary only while incomplete results remain enabled.
        if not phase.allow_incomplete_on_timeout:
            raise WinnerIntegrityError(
                f"Phase {phase.name!r} accepted an incomplete result at "
                f"{partial_decision.finished_trials}/{partial_decision.trial_target} finished "
                f"trials when its {partial_decision.timeout_scope} deadline expired.",
                action=OperatorAction.FIX_CONFIG,
                remediation=ACCEPT_INCOMPLETE_STEP,
            )
        completion = _partial_completion_for_replay(
            study,
            partial_decision,
            outcome_sequence=policy_state.max_sequence,
        )
        try:
            return _select_phase_winner(
                experiment,
                phase,
                inherited_winners,
                study,
                phase_fingerprint=phase_fingerprint,
                completion=completion,
                verify_evidence=verify_evidence,
            )
        except NoFeasibleTrialError as exc:
            raise _no_feasible_before_deadline(
                experiment, phase.name, partial_decision.timeout_scope
            ) from exc
    # A later success or a lowered failure limit does not clear a recorded abort. A live run's
    # preflight confirms cleanup before this point; a dry run cannot.
    refusal = _phase_refusal(phase, study, policy_state)
    if refusal is not None:
        raise refusal
    active_abort = policy_state.abort
    if active_abort is None:
        active_abort = _replayed_streak_abort(phase, study, policy_state)
    return _PhaseResume(
        policy_state=policy_state,
        phase_fingerprint=phase_fingerprint,
        partial_decision=partial_decision,
        recovery_abort=active_abort,
    )


class _PhaseBudget(NamedTuple):
    """The wallclock budget a phase's optimize loop runs under; ``None`` means unbounded."""

    timeout: float | None
    deadline: float | None
    source: str | None


def _phase_budget(experiment: Experiment, phase: Phase, run_deadline: float | None) -> _PhaseBudget:
    """Combine the phase timeout and the remaining run wallclock into one budget.

    :param Experiment experiment: Experiment whose storage decides what a rerun keeps.
    :param Phase phase: Phase supplying ``timeout_seconds_per_phase``.
    :param float | None run_deadline: Whole-run ``time.monotonic()`` deadline,
        when configured.
    :raises TimeoutError: The run deadline has already passed.
    :return _PhaseBudget: The optimize timeout in seconds, its monotonic
        deadline, and which guard (``"phase"`` or ``"run"``) sets it.
    """
    timeout_source: str | None = None
    optimize_timeout = phase.timeout_seconds_per_phase
    if optimize_timeout is not None:
        timeout_source = "phase"
    if run_deadline is not None:
        remaining_run_seconds = max(0.0, run_deadline - time.monotonic())
        if optimize_timeout is None or remaining_run_seconds <= optimize_timeout:
            timeout_source = "run"
        optimize_timeout = (
            remaining_run_seconds
            if optimize_timeout is None
            else min(optimize_timeout, remaining_run_seconds)
        )
    if optimize_timeout is not None and optimize_timeout <= 0.0:
        raise phase_timeout(
            f"Run wallclock deadline reached before phase {phase.name!r} could launch.",
            scope="run",
            experiment=experiment,
        )
    optimize_deadline = None if optimize_timeout is None else time.monotonic() + optimize_timeout
    return _PhaseBudget(optimize_timeout, optimize_deadline, timeout_source)


@dataclass
class _PhaseExecution:
    """One live phase execution, shared by its objective workers and callbacks.

    Built once the phase has remaining work, its GPU pool, and its wallclock
    budget, and before the durable trial target is recorded. The context
    fields never change afterwards. The state fields are what the workers
    share: ``outcome_lock`` orders outcome recording, ``fatal_lock`` guards the
    first fatal error, ``csv_lock`` orders CSV snapshots, and every flag only ever goes from ``False`` to
    ``True``. The methods change memory and Optuna's stop flag only; every
    durable write stays with its caller.
    """

    experiment: Experiment
    phase: Phase
    study: optuna.Study
    inherited_winners: dict[str, Winner]
    generation_id: str
    environment_identity: EnvironmentIdentity
    gpu_pool: GpuPool
    optimize_deadline: float | None
    timeout_source: str | None
    completion_sequence: int = 0
    consecutive_failures: int = 0
    recorded_outcomes: dict[int, str] = field(default_factory=dict)
    # Trials that count toward n_trials: the study's at phase start, then one per recorded
    # outcome other than ``interrupted``. ``finished_ordinals`` keeps the count each trial's
    # outcome took, so a trial's progress line states its own position however callbacks interleave.
    finished_trials: int = 0
    finished_ordinals: dict[int, int] = field(default_factory=dict)
    # Shared by the consecutive-failure and fatal-error paths. Queued
    # objectives check it inside the GPU lease and stop, interrupted, before launching.
    aborted: bool = False
    # Set once an outcome carrying this execution's abort record is durable.
    abort_recorded: bool = False
    # An outcome write could not be verified, so the next completion sequence is unknown.
    unconfirmed_outcome: StudyStorageUnavailableError | None = None
    deadline_exhausted: bool = False
    # Retain the first fatal worker exception for propagation after all workers drain.
    fatal_error: BaseException | None = None
    csv_throttle: CsvSnapshotThrottle = field(default_factory=CsvSnapshotThrottle)
    csv_lock: threading.Lock = field(default_factory=threading.Lock)
    outcome_lock: threading.Lock = field(default_factory=threading.Lock)
    fatal_lock: threading.Lock = field(default_factory=threading.Lock)
    workers: _WorkerTrials = field(default_factory=_WorkerTrials)

    def resume_from(self, policy_state: _PhasePolicyState) -> None:
        """Continue the completion order the durable outcome ledger replays.

        :param _PhasePolicyState policy_state: State replayed by
            :func:`phasesweep.engine.study_policy._load_phase_policy_state`.
        """
        self.completion_sequence = policy_state.max_sequence
        self.consecutive_failures = policy_state.consecutive_failures

    def abort_decision(
        self,
        trial_number: int,
        outcome: str,
        *,
        cause: str | None,
        fatal_policy: str | None,
    ) -> JsonObject | None:
        """Return the abort record that recording ``outcome`` next would trip.

        Changes nothing: the caller holds ``outcome_lock`` and writes the
        record with the outcome, before :meth:`advance` applies it, so the
        decision is exactly as durable as the outcome. Only the phase
        execution's first abort is recorded.

        :param int trial_number: Trial whose outcome is about to be recorded.
        :param str outcome: The terminal outcome about to be recorded.
        :param str | None cause: The outcome's recorded cause, if any.
        :param str | None fatal_policy: The fatal policy of a ``"fatal"`` outcome.
        :return JsonObject | None: The abort record for the outcome's
            completion sequence, or ``None`` when it trips nothing new.
        """
        return _abort_decision(
            self.phase,
            trial_number,
            outcome,
            completion_sequence=self.completion_sequence,
            consecutive_failures=self.consecutive_failures,
            abort_recorded=self.abort_recorded,
            cause=cause,
            fatal_policy=fatal_policy,
        )

    def advance(self, trial_number: int, outcome: str) -> None:
        """Apply one durably recorded outcome in completion order.

        The caller holds ``outcome_lock`` and has already written the outcome
        at sequence ``completion_sequence + 1``. The transition is the one the
        durable replay applies, so a restarted process reconstructs the same
        streak.

        :param int trial_number: Trial whose outcome was recorded.
        :param str outcome: The recorded terminal outcome.
        """
        self.completion_sequence += 1
        self.recorded_outcomes[trial_number] = outcome
        if outcome != "interrupted":
            self.finished_trials += 1
        self.finished_ordinals[trial_number] = self.finished_trials
        self.consecutive_failures = _next_consecutive_failures(self.consecutive_failures, outcome)

    def stop_launches(self) -> None:
        """Stop every queued and future trial from launching."""
        self.aborted = True
        self.gpu_pool.cancel_waiters()

    def stop_for_deadline(self) -> None:
        """Record causal deadline exhaustion and stop scheduling peer trials."""
        self.deadline_exhausted = True
        with contextlib.suppress(Exception):
            self.study.stop()

    def record_fatal(self, error: BaseException) -> None:
        """Record the first fatal objective or post-trial exception and stop peer launches.

        First-writer wins so the root failure is not replaced by secondary
        peer-abort or persistence fallout, except that unconfirmed process cleanup
        replaces a trial left RUNNING: a leaked group may hold the GPU, so it must
        stop all work until recovery whichever worker recorded first. The original
        exception object keeps its worker traceback for the post-optimize re-raise.

        :param BaseException error: The fatal exception a worker observed.
        """
        with self.fatal_lock:
            recorded = self.fatal_error is None or (
                isinstance(self.fatal_error, _TrialLeftRunning)
                and isinstance(error, ProcessCleanupUncertainError)
            )
            if recorded:
                self.fatal_error = error
        if recorded:
            # The error re-raised once the workers drain reaches the console in operator words;
            # run.log keeps this line with the exception type.
            log.info(
                "phase=%s FATAL OBJECTIVE ERROR (%s): %s",
                self.phase.name,
                type(error).__name__,
                error,
            )
        self.stop_launches()
        with contextlib.suppress(Exception):
            self.study.stop()

    def raise_if_fatal(self) -> None:
        """Re-raise the first fatal worker exception with its original traceback.

        A no-op when no worker recorded a fatal abort.

        :raises BaseException: The exception :meth:`record_fatal` captured
            first, re-raised with its original worker traceback.
        """
        with self.fatal_lock:
            error = self.fatal_error
        if error is not None:
            raise error.with_traceback(error.__traceback__)


def _refuse_launch_after_deadline(run: _PhaseExecution) -> None:
    """Refuse to launch a trial once the optimize deadline has passed.

    :param _PhaseExecution run: The phase execution whose deadline is checked.
    :raises _DeadlineTrialExecutionError: The deadline has passed; peer
        launches are stopped first.
    """
    if run.optimize_deadline is not None and run.optimize_deadline - time.monotonic() <= 0.0:
        run.stop_for_deadline()
        raise _DeadlineTrialExecutionError(
            f"{run.timeout_source or 'wallclock'} deadline reached before trial launch."
        )


def _refuse_launch_during_shutdown() -> None:
    """Refuse to start a trial while a shutdown signal is being serviced.

    :raises _InterruptedTrialError: A shutdown is being serviced, so
        :func:`_objective` records the trial as interrupted by that signal.
    """
    signum = servicing_shutdown_signal()
    if signum is not None:
        raise _InterruptedTrialError(interruption_cause(signum))


def _trial_recording_error(
    run: _PhaseExecution, trial: optuna.Trial, record: str, detail: str
) -> StudyStorageUnavailableError:
    """Report an unconfirmed trial write and what the operator can recover.

    :param _PhaseExecution run: Phase execution whose study owns the trial.
    :param optuna.Trial trial: Trial whose evidence or outcome could not be confirmed.
    :param str record: Name of the trial record being written.
    :param str detail: Cause of the write or readback failure.
    :return StudyStorageUnavailableError: The operator refusal carried out of optimization.
    """
    message = (
        f"Could not confirm {record} for trial {trial.number} "
        f"in study {run.study.study_name!r} {detail}."
    )
    if not run.experiment.persistent:
        return StudyStorageUnavailableError(
            f"{message} The trial result was lost.",
            action=OperatorAction.INSPECT_LOGS,
            remediation="Resolve the error and rerun.",
        )
    return StudyStorageUnavailableError(
        f"{message} The trial stays RUNNING; the next run recovers it once the ledger is writable."
    )


def _record_outcome(
    run: _PhaseExecution,
    trial: optuna.Trial,
    outcome: str,
    *,
    metric: float | None = None,
    cause: str | None = None,
    fatal_policy: str | None = None,
) -> None:
    """Persist an outcome and its abort decision before the trial becomes terminal.

    Sequence numbers follow lock-acquisition order, independent of trial number.
    Read back a failed write under the lock before another writer can assign its sequence.
    Only a confirmed absent outcome retries with a new decision after backoff.
    Already-recorded trials are ignored. A later success cannot clear an abort.

    :param _PhaseExecution run: Phase execution whose counters advance after the durable write.
    :param optuna.Trial trial: Trial receiving the outcome record.
    :param str outcome: ``success``, ``failure``, ``interrupted``, or ``fatal``.
    :param float | None metric: Accepted objective, also recorded when it violates bounds.
    :param str | None cause: Optional explanation stored in the outcome.
    :param str | None fatal_policy: Optional policy stored in the outcome and fatal abort.
    :raises _TrialLeftRunning: Retries exhausted or an outcome remains unconfirmed; leave
        the trial RUNNING and stop launches for stale-attempt recovery.
    """
    write_error: Exception | None = None
    abort_record: JsonObject | None = None
    for attempt_index in range(_OUTCOME_WRITE_ATTEMPTS):
        if attempt_index:
            time.sleep(_OUTCOME_WRITE_RETRY_DELAYS[attempt_index - 1])
        with run.outcome_lock:
            if trial.number in run.recorded_outcomes:
                return
            if run.unconfirmed_outcome is not None:
                raise _TrialLeftRunning(run.unconfirmed_outcome)
            abort_record = run.abort_decision(
                trial.number, outcome, cause=cause, fatal_policy=fatal_policy
            )
            payload = _trial_outcome_payload(
                run.completion_sequence + 1,
                outcome,
                metric=metric,
                cause=cause,
                policy=fatal_policy,
                abort=abort_record,
            )
            try:
                trial.set_user_attr(TRIAL_OUTCOME_ATTR, payload)
            except Exception as exc:
                write_error = exc
                # Trial.user_attrs comes from Optuna's local cache, which a failed
                # acknowledgment may leave behind the journal.
                stored: object
                try:
                    stored = run.study._storage.get_trial(trial._trial_id).user_attrs.get(
                        TRIAL_OUTCOME_ATTR
                    )
                except Exception as read_error:
                    stored = None
                    detail = f"readback failed ({type(read_error).__name__}: {read_error})"
                else:
                    if stored is None:
                        continue
                    detail = "readback found a different outcome"
                if stored != payload:
                    run.unconfirmed_outcome = _trial_recording_error(
                        run,
                        trial,
                        "the terminal outcome",
                        f"after its write failed ({type(exc).__name__}: {exc}) and {detail}",
                    )
                    unconfirmed = _TrialLeftRunning(run.unconfirmed_outcome)
                    run.record_fatal(unconfirmed)
                    raise unconfirmed from exc

            write_error = None
            run.advance(trial.number, outcome)
            if abort_record is None:
                return
            run.abort_recorded = True
            run.stop_launches()
            break
    if write_error is not None:
        unrecorded = _TrialLeftRunning(
            _trial_recording_error(
                run,
                trial,
                "the terminal outcome",
                f"after {_OUTCOME_WRITE_ATTEMPTS} write attempts "
                f"({type(write_error).__name__}: {write_error})",
            )
        )
        run.record_fatal(unrecorded)
        raise unrecorded from write_error

    with contextlib.suppress(Exception):
        run.study.stop()


@contextlib.contextmanager
def _persist_trial_evidence(run: _PhaseExecution, trial: optuna.Trial) -> Iterator[None]:
    """Leave the trial RUNNING when the ledger cannot save its evidence.

    :param _PhaseExecution run: Live phase whose workers must stop on a storage failure.
    :param optuna.Trial trial: Trial whose trainer exit or extraction evidence is being saved.
    :raises _TrialLeftRunning: The trial's evidence could not be written.
    :return Iterator[None]: Context covering only evidence reads and writes to the ledger.
    """
    try:
        yield
    except Exception as exc:
        refusal = _trial_recording_error(
            run,
            trial,
            "trial evidence",
            f"after a ledger operation failed ({type(exc).__name__}: {exc})",
        )
        error = _TrialLeftRunning(refusal)
        run.record_fatal(error)
        raise error from exc


def _execute_objective(run: _PhaseExecution, trial: optuna.Trial) -> float:
    """Optuna objective: sample, launch trial subprocess, extract, return metric.

    :param _PhaseExecution run: The phase execution the trial belongs to.
    :param optuna.Trial trial: The active Optuna trial being evaluated.
    :raises _UnlaunchedAbortError: A peer trial tripped a soft abort (max_consecutive_failures)
        or a fatal error before this trial's trainer launched, so it must not start.
    :raises UnsafeProcessCleanupError: The subprocess's cleanup could not be confirmed; the
        phase hard-aborts before another trial can acquire the just-released GPU lease.
    :raises TrialExecutionError: The subprocess returned non-zero or produced no metric.
        Caught by ``study.optimize(catch=...)``.
    :raises PhaseSweepShutdown: Shutdown interrupted the trial or left completed training pending extraction.
    :raises ActiveAttemptPersistenceError: Required attempt recovery metadata could not be
        persisted; nothing was launched and no GPU lease was consumed.
    :raises _TrialLeftRunning: Evidence or the outcome could not be persisted, tracking preparation could not confirm cleanup, or the tracker gave no answer about finished training in a persistent study.
    :return float: The extracted metric value (Optuna minimizes/maximizes per ``direction``).
    """
    experiment = run.experiment
    phase = run.phase

    # Record the allocation cohort even if preparation or a peer abort prevents launch.
    trial.set_user_attr(TRAINER_ENV_DIGEST_ATTR, run.environment_identity.digest)
    trial.set_user_attr(TRAINER_ENV_ATTR, dict(run.environment_identity.tracked))

    if run.aborted:
        raise _UnlaunchedAbortError(_UNLAUNCHED_ABORT_CAUSE)
    _refuse_launch_during_shutdown()

    sampled = _suggest_all(trial, phase)
    overrides = _composed_overrides(experiment, phase, sampled, run.inherited_winners)

    attempt_id = uuid4().hex
    # A rerun of an interrupted or failed trial continues in its directory, under the opener's names.
    try:
        trials = {frozen.number: frozen for frozen in run.study.get_trials(deepcopy=False)}
        directory = _trial_dir_for_attempt(
            experiment,
            phase.name,
            trials,
            trials[trial.number],
            attempt_id=attempt_id,
            recovered=frozenset(_cleanup_recovered_trial_numbers(run.study)),
        )
        earlier_attempts = _earlier_attempts(trials, trials[trial.number])
    except PhaseSweepError as exc:
        # Unconfirmed cleanup of the source, or a trial path the rerun cannot follow: the
        # operator acts before anything launches.
        run.record_fatal(exc)
        raise _TrialLeftRunning(exc) from exc
    trial_dir = directory.path
    opened_by = (directory.opener_number, directory.opener_attempt_id)
    attempt_dir = _attempt_dir(_phase_dir(experiment, phase.name), attempt_id)
    # A broken link was refused above, so a rerun's path that does not exist is missing.
    reusing_directory = directory.opener_number != trial.number
    directory_missing = reusing_directory and not trial_dir.exists()
    # Persist allocated before discovery and GPU waiting; no identity then means no launch.
    try:
        if directory_missing:
            log.warning(
                "The original trial directory %s is missing. Trial %d reruns trial %s with the "
                "same parameters in a new directory at that path, without its previous local "
                "checkpoints.",
                trial_dir,
                trial.number,
                trial.user_attrs.get(RERUN_OF_ATTR),
            )
        trial_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        write_attempt_lifecycle(attempt_dir, attempt_id=attempt_id, state="allocated")
    except OSError as exc:
        raise _active_attempt_persistence_error(
            "persist",
            f"the allocated lifecycle marker for trial {trial.number} (attempt {attempt_id})",
            attempt_dir,
            exc,
        ) from exc
    # Register before leasing or launch; retain unsafe-cleanup entries until recovery.
    _register_active_attempt(
        experiment,
        attempt_id=attempt_id,
        phase_name=phase.name,
        study_name=run.study.study_name,
        trial_number=trial.number,
        trial_dir=trial_dir,
        attempt_dir=attempt_dir,
        generation_id=run.generation_id,
    )
    trial.set_user_attr(GENERATION_ID_ATTR, run.generation_id)
    trial.set_user_attr(ATTEMPT_ID_ATTR, attempt_id)
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    trial.set_user_attr(OVERRIDES_ATTR, json.dumps(overrides, default=str, sort_keys=True))
    # GPU lease covers only subprocess lifetime, not extraction.
    try:
        _refuse_launch_after_deadline(run)
        prepared_input = prepare_trainer_input(
            experiment=experiment,
            phase_name=phase.name,
            trial_id=opened_by[0],
            attempt_id=opened_by[1],
            trial_dir=trial_dir,
            overrides=overrides,
        )
        trial.set_user_attr(TRAINER_INPUT_ATTR, prepared_input.record())
        if experiment.mlflow is not None:
            from phasesweep.tracking.session import prepare_attempt

            try:
                prepare_attempt(
                    experiment,
                    phase=phase.name,
                    generation_id=run.generation_id,
                    attempt_id=attempt_id,
                    run_name=trial_run_name(experiment, phase.name, *opened_by),
                    overrides_sha256=prepared_input.sha256,
                    inherited=run.inherited_winners,
                    environment=run.environment_identity.values,
                    deadline=run.optimize_deadline,
                )
            except ProcessCleanupUncertainError as exc:
                # The registered tracking worker owns its cleanup. This trainer never launched.
                run.record_fatal(exc)
                raise _TrialLeftRunning(exc) from exc
            except (OSError, ValueError, RuntimeError) as exc:
                # A shutdown stops the preparation worker too; that is no failure of the trial.
                _refuse_launch_during_shutdown()
                reason = f"MLflow run preparation: {exc}"
                log.warning("[%s/trial_%d] %s", phase.name, trial.number, reason)
                trial.set_user_attr(FAILURE_REASON_ATTR, reason)
                if run.optimize_deadline is not None and time.monotonic() >= run.optimize_deadline:
                    raise _DeadlineTrialExecutionError(reason) from exc
                raise TrialExecutionError(reason) from exc
        with run.gpu_pool.acquire(deadline=run.optimize_deadline) as gpu_assignment:
            # Queued workers must observe aborts that occurred while waiting for the lease.
            if run.aborted:
                raise _UnlaunchedAbortError(_UNLAUNCHED_ABORT_CAUSE)
            # A shutdown's kill sweep frees leases; a queued worker that takes
            # one afterwards must not start a trainer no handler will kill.
            _refuse_launch_during_shutdown()

            _refuse_launch_after_deadline(run)
            _log_trial_start(
                experiment,
                phase,
                trial,
                gpu=gpu_assignment.visible_devices,
                in_place=reusing_directory and not directory_missing,
            )
            try:
                executed = launch_trial(
                    experiment=experiment,
                    phase_name=phase.name,
                    trial_id=trial.number,
                    generation_id=run.generation_id,
                    attempt_id=attempt_id,
                    trial_dir=trial_dir,
                    attempt_dir=attempt_dir,
                    opened_by=opened_by,
                    earlier_attempts=earlier_attempts,
                    overrides=overrides,
                    timeout_seconds=phase.timeout_seconds_per_trial,
                    wallclock_deadline=run.optimize_deadline,
                    gpu_id=gpu_assignment.visible_devices,
                    gpu_lease_fds=gpu_assignment.lease_fds,
                    prepared_input=prepared_input,
                    environment_identity=run.environment_identity,
                )
            except ShutdownLaunchRefusedError as exc:
                # The shutdown began between the check above and the launch lock.
                raise _InterruptedTrialError(interruption_cause(exc.signum)) from exc

            # Block queued launches before releasing a GPU whose process cleanup is uncertain.
            if not executed.process.cleanup_confirmed:
                cleanup_error = UnsafeProcessCleanupError(
                    f"Trial {trial.number} cleanup could not be confirmed: "
                    f"{executed.process.failure_reason or 'no reason was recorded'} "
                    f"(attempt_dir={attempt_dir}, pid={executed.process.pid})."
                )
                run.record_fatal(cleanup_error)

                # Diagnostic writes must not mask the fatal cleanup error already recorded.
                with contextlib.suppress(Exception):
                    trial.set_user_attr(
                        FAILURE_REASON_ATTR,
                        executed.process.failure_reason or "process cleanup could not be confirmed",
                    )

                raise cleanup_error
    except GpuLeaseTimeoutError as exc:
        run.stop_for_deadline()
        raise _DeadlineTrialExecutionError(str(exc)) from exc
    except GpuLeaseCancelledError as exc:
        _refuse_launch_during_shutdown()
        raise _UnlaunchedAbortError(_UNLAUNCHED_ABORT_CAUSE) from exc

    # SDK readers reuse the process slot; preserve the trainer's result before extraction.
    with _persist_trial_evidence(run, trial):
        if executed.process.failure_reason is not None:
            trial.set_user_attr(FAILURE_REASON_ATTR, executed.process.failure_reason)
        trial.set_user_attr(DURATION_ATTR, executed.process.duration_seconds)
        trial.set_user_attr(RETURN_CODE_ATTR, executed.process.return_code)

    def check_shutdown_after_trainer_exit() -> None:
        """Preserve finished training, keep a crash's own failure, and interrupt a stopped trainer.

        :raises PhaseSweepShutdown: Shutdown prevents extraction of a successful trainer's result.
        :raises _InterruptedTrialError: Shutdown stopped the trainer before it finished.
        :raises TrialExecutionError: The trainer failed by itself before the shutdown.
        """
        signum = servicing_shutdown_signal()
        if signum is None:
            return
        if _completed_trainer_duration(trial, None) is not None:
            raise PhaseSweepShutdown(signum, ShutdownCleanupReport(signum, True, ()))
        try:
            lifecycle = _trial_lifecycle(trial)
        except ValueError:
            lifecycle = None
        reason = _trainer_failure_reason(trial, lifecycle, attempt_dir)
        if reason is None:
            raise _InterruptedTrialError(interruption_cause(signum))
        raise TrialExecutionError(reason)

    # Extraction releases the GPU but still consumes the phase/run budget.
    check_shutdown_after_trainer_exit()
    if not executed.process.timed_out:
        # A trainer PhaseSweep stopped outside a shutdown, as after its guardian died, is
        # interrupted by the rule recovery applies (_trainer_failure_reason), and reruns in
        # its directory. A deadline cap is a timeout, which the extraction below reports.
        try:
            lifecycle = _trial_lifecycle(trial)
        except ValueError:
            lifecycle = None
        if lifecycle is not None and lifecycle.ended_by in _STOPPED_BY_PHASESWEEP:
            raise _InterruptedTrialError(
                executed.process.failure_reason or "PhaseSweep stopped the trainer"
            )
    try:
        result = extract_trial_result(
            experiment=experiment,
            executed=executed,
            gates=phase.gates,
            deadline=run.optimize_deadline,
        )
    except ShutdownLaunchRefusedError as exc:
        raise PhaseSweepShutdown(exc.signum, ShutdownCleanupReport(exc.signum, True, ())) from exc
    check_shutdown_after_trainer_exit()
    if result.evidence_unreadable and not result.deadline_exhausted and experiment.persistent:
        # No answer from the tracker says nothing about the training, so no outcome
        # is recorded: recovery extracts it once the tracker answers. An in-memory
        # study has no later run to do that, so there the trial fails with the reason.
        _warn_deferred_extraction(attempt_id, phase.name, trial.number, persistent=True)
        left_running = _TrialLeftRunning(
            tracker_unreadable(
                f"Trial {trial.number} in phase {phase.name!r} finished training, but the "
                f"tracker could not be read: {(result.failure_reason or '').rstrip('.')}."
            )
        )
        run.record_fatal(left_running)
        raise left_running
    if result.deadline_exhausted:
        # Only a deadline-caused failure counts as a timeout.
        run.stop_for_deadline()
        if experiment.persistent and _completed_trainer_duration(trial, None) is not None:
            # Finished training that the deadline kept from being read stays pending, as after
            # a tracker outage: no outcome is recorded, so the run stops before any partial
            # decision freezes the phase's counts, and recovery extracts it on the next run.
            _warn_deferred_extraction(attempt_id, phase.name, trial.number, persistent=True)
            left_running = _TrialLeftRunning(
                phase_timeout(
                    f"Phase {phase.name!r} timed out via {run.timeout_source or 'wallclock'} "
                    f"guard while reading the evidence of trial {trial.number}, which finished "
                    "training.",
                    scope=run.timeout_source,
                    experiment=experiment,
                )
            )
            run.record_fatal(left_running)
            raise left_running

    with _persist_trial_evidence(run, trial):
        record_trial_result(
            trial, result, overrides_json=json.dumps(overrides, default=str, sort_keys=True)
        )

    # Process/extractor failures -> Optuna FAIL state, not COMPLETE with inf.
    if result.failure_reason:
        if result.deadline_exhausted:
            with _persist_trial_evidence(run, trial):
                trial.set_user_attr(FAILURE_REASON_ATTR, result.failure_reason)
            raise _DeadlineTrialExecutionError(result.failure_reason)
        raise TrialExecutionError(result.failure_reason)

    assert result.metric is not None  # guaranteed when failure_reason is None
    violations: list[str] = []
    if not result.feasible:
        bound_checks: list[tuple[str, float, Metric | Constraint]] = [
            (f"metric {experiment.metric.name!r}", result.metric, experiment.metric),
            *(
                (f"constraint {constraint.name!r}", result.constraints[constraint.name], constraint)
                for constraint in experiment.constraints
            ),
        ]
        for label, value, bounds in bound_checks:
            if bounds.min is not None and value < bounds.min:
                violations.append(f"{label}={value} < min={bounds.min}")
            if bounds.max is not None and value > bounds.max:
                violations.append(f"{label}={value} > max={bounds.max}")
    _record_outcome(
        run,
        trial,
        "success" if result.feasible else "failure",
        metric=result.metric,
        cause=(
            None
            if result.feasible
            else f"metric or evidence constraints were infeasible: {'; '.join(violations)}"
        ),
    )

    return result.metric


def _objective(run: _PhaseExecution, trial: optuna.Trial) -> float:
    """Record objective outcomes, preserving completed training when shutdown interrupts extraction.

    :param _PhaseExecution run: Phase execution receiving the outcome and any fatal error.
    :param optuna.Trial trial: Active Optuna trial.
    :return float: Extracted metric.
    """
    with run.workers.track(trial.number):
        try:
            return _execute_objective(run, trial)
        except _TrialLeftRunning:
            # Recovery owns the unresolved trial and any outcome already in the journal.
            raise
        except (_DeadlineTrialExecutionError, _UnlaunchedAbortError) as exc:
            _record_outcome(run, trial, "interrupted", cause=str(exc))
            raise
        except TrialExecutionError as exc:
            # Under n_jobs > 1 the shutdown handler runs in the main thread and kills this
            # worker's trainer; the worker's lifecycle check names that an interruption.
            interrupted = isinstance(exc, _InterruptedTrialError)
            cause = str(exc)
            _record_outcome(run, trial, "interrupted" if interrupted else "failure", cause=cause)
            # An earlier diagnostic would prevent recovery from reading a finished trainer's
            # evidence if its outcome could not be committed.
            if interrupted or FAILURE_REASON_ATTR not in trial.user_attrs:
                with _persist_trial_evidence(run, trial):
                    trial.set_user_attr(FAILURE_REASON_ATTR, cause)
            raise
        except UnsafeProcessCleanupError as exc:
            run.record_fatal(exc)
            # Both trainer and evidence-worker cleanup failures retain this
            # diagnostic; the fatal policy below remains the recovery authority.
            with contextlib.suppress(Exception):
                trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
            _record_outcome(
                run,
                trial,
                "fatal",
                cause=str(exc),
                fatal_policy="unsafe_process_cleanup",
            )
            raise
        except ActiveAttemptPersistenceError as exc:
            run.record_fatal(exc)
            _record_outcome(
                run,
                trial,
                "fatal",
                cause=str(exc),
                fatal_policy="active_attempt_persistence",
            )
            raise
        except PhaseSweepShutdown as exc:
            attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
            try:
                lifecycle = _trial_lifecycle(trial)
            except ValueError as lifecycle_error:
                raise exc from lifecycle_error
            if _completed_trainer_duration(trial, lifecycle) is None:
                directory = _trial_attempt_dir(trial)
                reason = (
                    None
                    if directory is None
                    else _trainer_failure_reason(trial, lifecycle, directory)
                )
                if reason is None:
                    _record_outcome(run, trial, "interrupted", cause=interruption_cause(exc.signum))
                else:
                    # A trainer that crashed before the shutdown keeps its own failure.
                    _record_outcome(run, trial, "failure", cause=reason)
                    with contextlib.suppress(Exception):
                        trial.set_user_attr(FAILURE_REASON_ATTR, reason)
            elif isinstance(attempt_id, str) and not (
                run.experiment.persistent and trial.number in run.recorded_outcomes
            ):
                # A persistent trial whose outcome is already recorded needs no extraction:
                # the exit settlement commits it and logs it as kept. Only training that
                # nothing recorded is left to the next run.
                _warn_deferred_extraction(
                    attempt_id,
                    run.phase.name,
                    trial.number,
                    persistent=run.experiment.persistent,
                )
            raise
        except BaseException as exc:
            run.record_fatal(exc)
            _record_outcome(
                run,
                trial,
                "fatal",
                cause=f"{type(exc).__name__}: {exc}",
                fatal_policy="unexpected_objective_exception",
            )
            raise


def _after_trial(
    run: _PhaseExecution,
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
) -> None:
    """Settle one finished trial, routing an escaping failure through the fatal channel.

    Optuna never retrieves the futures of its final ``n_jobs > 1`` batch, so
    an exception escaping this callback there would vanish and the phase
    could publish with cleanup reported confirmed. Recording it first makes
    every ``n_jobs`` setting raise it after the workers drain, as for
    objective errors, and stops queued launches. The trial's accepted Optuna
    state is untouched. A shutdown is a cancellation, never a fatal error.

    :param _PhaseExecution run: The phase execution the trial belongs to.
    :param optuna.Study study: The running Optuna study.
    :param optuna.trial.FrozenTrial trial: The just-finished trial.
    """
    try:
        _settle_finished_trial(run, study, trial)
    except PhaseSweepShutdown:
        raise
    except BaseException as exc:
        run.record_fatal(exc)
        raise


def _settle_finished_trial(
    run: _PhaseExecution,
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
) -> None:
    """Log a finished trial, retire cleaned attempts, snapshot CSV, and mirror MLflow.

    Repeat ``study.stop()`` after an abort; the objective owns the abort decision.
    Write local results before attempting MLflow publication.

    :param _PhaseExecution run: Phase execution.
    :param optuna.Study study: Running study to stop after an abort.
    :param optuna.trial.FrozenTrial trial: Finished trial whose attempt may be retired.
    """
    if run.aborted:
        with contextlib.suppress(Exception):
            study.stop()
    finished = _finished_trial_count(study.get_trials(deepcopy=False))
    # Peers finish while this callback waits, so the study's count here can repeat or skip a
    # value; the ordinal this trial's outcome took is its own.
    _log_finished_trial(
        run.experiment, run.phase, trial, run.finished_ordinals.get(trial.number, finished)
    )
    outcome = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    if outcome is not None and outcome.abort is not None:
        # The raised abort error reaches the console with its fix; run.log keeps this line.
        log.info("phase=%s ABORTED: %s", run.phase.name, outcome.abort["cause"])
    # Keep unsafe-cleanup entries until recovery. Run reconciliation handles skipped callbacks.
    finished_attempt = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if (
        isinstance(finished_attempt, str)
        and finished_attempt
        and not _trial_requires_cleanup_recovery(trial)
    ):
        _retire_active_attempt(run.experiment, finished_attempt)
    with run.csv_lock:
        now = time.monotonic()
        if not run.experiment.persistent or run.csv_throttle.should_write(finished, now):
            with contextlib.suppress(Exception):
                _write_trials_csv(
                    study,
                    _trials_csv_path(run.experiment, run.phase.name),
                    repeats=repeat_axis(run.phase),
                )
                run.csv_throttle.mark_written(finished, now)
    if run.experiment.mlflow is not None and (
        trial.state == optuna.trial.TrialState.COMPLETE
        or not _trial_requires_cleanup_recovery(trial)
    ):
        from phasesweep.tracking.publication import publish_trial

        publish_trial(run.experiment, run.generation_id, trial, run.environment_identity.values)


def _finish_phase(run: _PhaseExecution, *, phase_fingerprint: str) -> Winner:
    """Turn how the optimize loop ended into the phase's winner or refusal.

    :param _PhaseExecution run: The drained phase execution.
    :param str phase_fingerprint: Verified semantic fingerprint for the phase.
    :raises NoFeasibleTrialError: An abort tripped and no timeout left work
        undone, or no finished trial is feasible; the subclass says whether a trial failed.
    :raises TimeoutError: The deadline left the phase incomplete and partial
        publication is disabled, or no trial completed before it.
    :raises RuntimeError: Optuna returned short of the trial target without a
        timeout or abort.
    :return Winner: The selected phase winner.
    """
    experiment = run.experiment
    phase = run.phase
    study = run.study
    trials_after = study.get_trials(deepcopy=False)
    finished_after = _finished_trial_count(trials_after)
    completed_after = _completed_trial_count(trials_after)
    # An expired budget with unfinished slots takes precedence over a failure-streak abort.
    scheduler_deadline_exhausted = (
        run.optimize_deadline is not None
        and finished_after < phase.n_trials
        and time.monotonic() >= run.optimize_deadline
    )
    if scheduler_deadline_exhausted:
        run.deadline_exhausted = True
    timeout_observed = run.deadline_exhausted
    timed_out_incomplete = timeout_observed and finished_after < phase.n_trials
    accepted_partial_timeout = (
        phase.allow_incomplete_on_timeout and timeout_observed and finished_after < phase.n_trials
    )
    if run.aborted and not timed_out_incomplete:
        aborted_state = _load_phase_policy_state(study)
        abort = aborted_state.abort
        assert abort is not None  # an aborted run recorded its abort with the tripping outcome
        # Without storage there is no study to continue, so the whole run starts again.
        advice = (
            _recovery_advice(phase, study, abort)
            if experiment.persistent
            else ABORT_START_AGAIN_STEP
        )
        all_infeasible = _streak_trials_all_infeasible(study, aborted_state)
        evidence = (
            "The tripping trials were INFEASIBLE: outside the metric bounds or constraints."
            if all_infeasible
            else f"Inspect {_phase_dir(experiment, phase.name) / 'attempts'} for stderr logs."
        )
        abort_error = PhaseTrialsInfeasibleError if all_infeasible else PhaseTrialsFailedError
        raise abort_error(
            f"Phase {phase.name!r} aborted: {as_sentence(abort['cause'])} {evidence}",
            action=(
                _recovery_action(study, abort)
                if experiment.persistent
                else OperatorAction.INSPECT_LOGS
            ),
            remediation=advice,
        )
    if (
        finished_after < phase.n_trials
        and not accepted_partial_timeout
        and not timed_out_incomplete
    ):
        interrupted = _pending_rerun_count(study)
        if interrupted:
            # A pass whose every trial PhaseSweep stopped finished none, so the run stops
            # rather than rerun them without end; the next run reruns them in place.
            raise PhaseSweepError(
                f"Phase {phase.name!r} stopped after {finished_after}/{phase.n_trials} "
                f"finished trials with {plural(interrupted, 'interrupted trial')} left to rerun.",
                action=OperatorAction.RETRY,
                remediation=(
                    f"Inspect {_phase_dir(experiment, phase.name) / 'attempts'} and run.log, "
                    "then run the same command again; the interrupted trials rerun in place."
                ),
            )
        raise RuntimeError(
            f"Phase {phase.name!r} stopped after {finished_after}/{phase.n_trials} "
            "finished trials without an accepted timeout; refusing to publish "
            "an incomplete winner."
        )
    current_policy_state = _load_phase_policy_state(study)
    recorded_abort = current_policy_state.abort
    if accepted_partial_timeout:
        recovered_abort_sequence = (
            recorded_abort["completion_sequence"] if recorded_abort is not None else None
        )
        # Persist the frozen timeout boundary before selection; retries replay it without launch.
        study.set_user_attr(
            PHASE_DECISION_ATTR,
            {
                "schema_version": PHASE_DECISION_SCHEMA_VERSION,
                "decision": "accepted_partial_timeout",
                "trial_target": phase.n_trials,
                "outcome_sequence": current_policy_state.max_sequence,
                "finished_trials": finished_after,
                "completed_trials": completed_after,
                "timeout_scope": run.timeout_source,
                "recovered_abort_sequence": recovered_abort_sequence,
            },
        )
    if recorded_abort is not None and timed_out_incomplete:
        # Consume the abort before timeout handling so a larger timeout can resume refused work.
        _record_recovery_boundary(study, phase, recorded_abort["completion_sequence"])
    if timed_out_incomplete and not phase.allow_incomplete_on_timeout:
        raise phase_timeout(
            f"Phase {phase.name!r} timed out via {run.timeout_source or 'wallclock'} guard "
            f"after {finished_after}/{phase.n_trials} finished trials "
            f"({completed_after} completed).",
            scope=run.timeout_source,
            experiment=experiment,
        )
    try:
        return _select_phase_winner(
            experiment,
            phase,
            run.inherited_winners,
            study,
            phase_fingerprint=phase_fingerprint,
            completion=_winner_completion(
                requested_trials=phase.n_trials,
                finished_trials=finished_after,
                completed_trials=completed_after,
                incomplete=accepted_partial_timeout,
                reason="timeout" if accepted_partial_timeout else None,
                timeout_scope=run.timeout_source if accepted_partial_timeout else None,
            ),
        )
    except NoFeasibleTrialError as exc:
        if run.deadline_exhausted:
            raise _no_feasible_before_deadline(
                experiment, phase.name, run.timeout_source or "wallclock"
            ) from exc
        raise


def _run_phase(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    *,
    generation: InvocationGeneration,
    ledger: ClaimedLedger,
    run_deadline: float | None = None,
    tracking: DeliveryTally | None = None,
) -> Winner:
    """Execute one phase end-to-end (sampler, study.optimize, winner selection).

    Opens the study the parents' winners address (:func:`phase_study_name`), saying so when
    it starts a new one beside the phase's other studies, which stay as they are.
    Validates the phase's durable state (:func:`_resume_phase`), runs Optuna's
    optimize loop with every worker sharing one :class:`_PhaseExecution`, and
    turns how that loop ended into the phase's winner or refusal
    (:func:`_finish_phase`). A phase with trials to launch claims ``generation``
    once every launch check has passed, before its first durable trial-target write.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: The phase to execute.
    :param dict[str, Winner] inherited_winners: Winners loaded for phases earlier in the chain.
    :param InvocationGeneration generation: The current engine invocation's generation.
    :param ClaimedLedger ledger: The claimed ledger, for ``experiment``, that this phase opens
        its live study through.
    :param float | None run_deadline: Optional ``time.monotonic()`` deadline inherited from the
        experiment-level wallclock guard.
    :param DeliveryTally | None tracking: The run's MLflow delivery, which the pass after the
        run's first target check counts into, or ``None`` for a run without MLflow.
    :raises NoFeasibleTrialError: ``max_consecutive_failures`` tripped, no finished trial is
        feasible, or the recorded trials a met target replays hold no feasible one.
    :raises PhaseAbortRefusedError: A recorded abort or a replayed failure streak stops the
        phase before it launches a trial.
    :raises UnsafeProcessCleanupError: A trial's process group could not be confirmed dead;
        phase hard-aborted.
    :raises ProcessCleanupUncertainError: An MLflow worker of the target check, or of the
        delivery after it, may still be running; no trial was started.
    :raises ArtifactRootConflictError: This phase's persistent study is bound to a different
        artifact root than the config's workdir offers.
    :raises TrialEvidenceMissingError: The selected winner's evidence directory, audit
        artifacts, or objective source are missing or no longer match the provenance frozen at
        extraction.
    :raises ActiveAttemptPersistenceError: A trial's required recovery metadata could not be
        persisted, so it was never launched.
    :raises StudyStorageUnavailableError: A trial's evidence or outcome could not be persisted.
        A persistent trial is left ``RUNNING`` with its attempt record intact and recovers on
        the next run once storage is writable.
    :raises StudySchemaMismatchError: Persisted phase recovery state contradicts the study's
        finished trial count.
    :raises TimeoutError: The phase or run deadline expires before the requested trial budget
        completes and partial publication is disabled, or before any trial completes.
    :raises PhaseSweepShutdown: The orchestrator receives a handled shutdown signal;
        propagated after recording a non-failure trial outcome, also when Optuna's own
        cleanup raised another error over it.
    :raises RuntimeError: Optuna returned before the requested trial budget without a timeout,
        abort, or exception, which violates the runner invariant.
    :return Winner: The selected phase winner.
    """
    inherited = _inherited_override_payload(phase, inherited_winners)
    study_name = phase_study_name(experiment, phase, inherited)
    others = [
        other for other in ledger.studies.get(phase.name, ()) if other.study_name != study_name
    ]
    # The live opener also claims a newly created study's publication root,
    # before any inspection, reaping, or trial work below can run against it.
    study = open_phase_study(ledger, phase, study_name)
    if not study.get_trials(deepcopy=False):
        note = new_study_note(phase, inherited, others)
        if note is not None:
            report.info("%s  %s", phase.name, note)
    resumed = _resume_phase(experiment, phase, inherited_winners, study)
    if phase.inherits:
        # Its study was unknown before the run selected its parents' winners.
        _validate_selection_evidence(experiment, {phase.name: study})
    if others:
        # Recovery refreshes trials.csv from each study it settles, which can leave it showing
        # a study this run does not address. Like every trials.csv snapshot it is best-effort.
        with contextlib.suppress(Exception):
            _write_trials_csv(
                study, _trials_csv_path(experiment, phase.name), repeats=repeat_axis(phase)
            )
    if isinstance(resumed, Winner):
        return resumed
    resume = resumed

    trials = study.get_trials(deepcopy=False)
    finished = _finished_trial_count(trials)
    remaining = max(0, phase.n_trials - finished)
    reruns = _pending_rerun_count(study, phase)
    report.info(
        phase_start(experiment, phase, finished=finished, remaining=remaining, reruns=reruns)
    )

    if remaining == 0:
        return _replay_met_target(
            experiment, phase, inherited_winners, study, resume, verify_evidence=True
        )

    # No-op replay needs no launch environment; new allocations must be able to start.
    environment_identity = _launch_preflight(experiment, phase.name, study)

    from phasesweep.config.models import _wandb_query
    from phasesweep.evidence.wandb import require_wandb_sdk
    from phasesweep.tracking.mlflow import require_mlflow_sdk

    if _wandb_query(experiment, phase.gates) is not None:
        require_wandb_sdk()
    if experiment.mlflow is not None:
        require_mlflow_sdk()

    gpu_pool = GpuPool.for_phase(
        phase,
        cuda_visible_devices=experiment.env.get("CUDA_VISIBLE_DEVICES"),
    )
    budget = _phase_budget(experiment, phase, run_deadline)
    if experiment.mlflow is not None:
        from phasesweep.tracking.session import require_tracking_target

        log.info("phase=%s checking MLflow target %s", phase.name, experiment.mlflow.tracking_uri)
        bound = require_tracking_target(
            experiment,
            generation_id=generation.generation_id,
            environment=environment_identity.values,
            deadline=budget.deadline,
        )
        if bound and tracking is not None:
            from phasesweep.tracking.publication import deliver_ended_generations

            # Once per run, before its first trial: what an interrupted earlier run left reaches
            # MLflow now rather than after this run's training. Its requests are cut at the run
            # deadline, the phase's budget starts after it, and the closeout sends what is left.
            deliver_ended_generations(
                experiment, ledger, generation.generation_id, tracking, deadline=run_deadline
            )
            budget = _phase_budget(experiment, phase, run_deadline)
    generation_id = generation.claim()
    run = _PhaseExecution(
        experiment=experiment,
        phase=phase,
        study=study,
        inherited_winners=inherited_winners,
        generation_id=generation_id,
        environment_identity=environment_identity,
        gpu_pool=gpu_pool,
        optimize_deadline=budget.deadline,
        timeout_source=budget.source,
        finished_trials=finished,
    )
    run.resume_from(resume.policy_state)
    # Accept the target only after environment, tracking, GPU, and budget checks pass.
    if resume.recovery_abort is not None:
        # Atomically accept the recovery target and retire the abort before updating the target attr.
        _record_recovery_boundary(study, phase, resume.recovery_abort["completion_sequence"])
        run.resume_from(_load_phase_policy_state(study))
    _record_trial_target(study, phase)
    if (
        resume.partial_decision is not None
        and phase.n_trials > resume.partial_decision.trial_target
    ):
        study.set_user_attr(PHASE_DECISION_ATTR, None)
    try:
        try:
            _record_allocation_context(
                study,
                generation_id=generation_id,
                trainer_environment=environment_identity.digest,
            )
            # WAITING reruns precede new draws and share the allocation context.
            recorded_reruns = _record_reruns(study, phase)
            timeout = budget.timeout
            try:
                while True:
                    _log_reruns(experiment, phase, study, recorded_reruns)
                    finished_before = _finished_trial_count(study.get_trials(deepcopy=False))
                    study.optimize(
                        functools.partial(_objective, run),
                        n_trials=remaining,
                        n_jobs=phase.n_jobs,
                        timeout=timeout,
                        gc_after_trial=True,
                        callbacks=[functools.partial(_after_trial, run)],
                        catch=(TrialExecutionError,),
                    )
                    # A trainer PhaseSweep stopped while the run went on, as after its guardian
                    # died, is interrupted; it reruns in its directory in this invocation while
                    # each pass still finishes trials, and otherwise in the next one.
                    finished_after = _finished_trial_count(study.get_trials(deepcopy=False))
                    remaining = phase.n_trials - finished_after
                    if (
                        remaining <= 0
                        or finished_after == finished_before
                        or run.aborted
                        or run.deadline_exhausted
                        or run.fatal_error is not None
                    ):
                        break
                    if budget.deadline is not None:
                        timeout = budget.deadline - time.monotonic()
                        if timeout <= 0:
                            break
                    recorded_reruns = _record_reruns(study)
                    if not recorded_reruns:
                        break
            except BaseException as exc:
                # Drain workers before run reconciliation can reap their trials.
                if phase.n_jobs > 1:
                    _await_worker_trials(run.workers, study)
                if not isinstance(exc, _TrialLeftRunning):
                    # The handler raises in the main thread at whatever bytecode is running,
                    # and a library's cleanup code can raise over it: Optuna's tell raises
                    # UnboundLocalError from its finally block. The run is still stopping.
                    shutdown = shutdown_raised_over(exc)
                    if shutdown is None or shutdown is exc:
                        raise
                    raise shutdown from None
                # A trial left RUNNING stops the phase with whichever recorded fatal
                # error outranks it once every worker has recorded its own.
                run.record_fatal(exc)
        finally:
            # Snapshot CSV on every exit without masking the optimize error.
            with contextlib.suppress(Exception):
                _write_trials_csv(
                    study,
                    _trials_csv_path(experiment, phase.name),
                    repeats=repeat_axis(phase),
                )

        # Propagate swallowed worker/callback errors before selection or timeout classification.
        run.raise_if_fatal()
    except _TrialLeftRunning as exc:
        raise exc.error from exc

    return _finish_phase(run, phase_fingerprint=resume.phase_fingerprint)


def _replay_met_target(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    study: optuna.Study,
    resume: _PhaseResume,
    *,
    verify_evidence: bool,
) -> Winner:
    """Select the winner of a phase whose trial target is already met, launching nothing.

    The winner is selected from the recorded trials again. Replay must neither
    require launch resources (GPU discovery on a CPU-only host) nor mutate the
    durable accepted target.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: The phase whose recorded trials meet its target.
    :param dict[str, Winner] inherited_winners: Winners from earlier phases in the chain.
    :param optuna.Study study: The phase's study, live or a read-only snapshot.
    :param _PhaseResume resume: The phase's validated durable state.
    :param bool verify_evidence: Whether the winner's evidence is re-proved; see
        :func:`_select_phase_winner` for everything it covers.
    :raises StudySchemaMismatchError: The study accepted an abort recovery target
        but has no remaining trial slots.
    :raises ReplayWithoutFeasibleTrialError: No recorded trial is complete, feasible and
        inside the metric's bounds, or with ``repeats`` no group is complete.
    :raises TrialEvidenceMissingError: The selected winner's evidence is missing or
        altered; only when ``verify_evidence`` is set.
    :return Winner: The winner the recorded trials select.
    """
    if resume.recovery_abort is not None:
        raise StudySchemaMismatchError(
            f"Phase {phase.name!r} accepted an abort recovery target but has no "
            "remaining trial slots."
        )
    trials = study.get_trials(deepcopy=False)
    try:
        return _select_phase_winner(
            experiment,
            phase,
            inherited_winners,
            study,
            phase_fingerprint=resume.phase_fingerprint,
            completion=_winner_completion(
                requested_trials=phase.n_trials,
                finished_trials=_finished_trial_count(trials),
                completed_trials=_completed_trial_count(trials),
                incomplete=False,
                reason=None,
                timeout_scope=None,
            ),
            verify_evidence=verify_evidence,
            emit_advisories=False,
        )
    except NoFeasibleTrialError as exc:
        raise ReplayWithoutFeasibleTrialError.rewrap(
            exc,
            f"Phase {phase.name!r} launched no trial: its recorded trials already meet "
            f"n_trials={phase.n_trials}. {exc}",
        ) from exc


def _select_phase_winner(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    study: optuna.Study,
    *,
    phase_fingerprint: str,
    completion: dict[str, Any],
    verify_evidence: bool = True,
    emit_advisories: bool = True,
) -> Winner:
    """Select the phase winner and preserve its execution identity.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: Phase whose winner is selected.
    :param dict[str, Winner] inherited_winners: Winners from earlier phases in the chain.
    :param optuna.Study study: Study whose completed trials supply the winner.
    :param str phase_fingerprint: Verified semantic fingerprint for the phase.
    :param dict[str, Any] completion: Completion metadata persisted with the winner.
    :param bool verify_evidence: Whether to verify the selected trial against its
        frozen evidence and warn if its environment differs from the current run.
    :param bool emit_advisories: Whether to log fresh selection diagnostics instead of silently replaying them.
    :raises NoFeasibleTrialError: No trial is complete, feasible and inside the metric's
        bounds, or with ``repeats`` no group is complete.
    :raises TrialEvidenceMissingError: The evidence directory, audit artifacts,
        generated trainer input, or objective source of the selected trial, or of
        any member of its group, are missing or no longer match the provenance
        frozen when its metric was extracted.
    :return Winner: The selected winner with composed overrides and source identity.
    """
    selected = select_winner(
        study,
        experiment,
        phase_name=phase.name,
        repeats=repeat_axis(phase),
        emit_advisories=emit_advisories,
    )
    if verify_evidence:
        _verify_winner_evidence(experiment, phase, selected)
        # Replayed winners may predate the current environment cohort.
        _warn_environment_drift(
            experiment, phase.name, selected.trainer_env_digest, selected.trainer_env
        )
    effective = _winner_overrides(experiment, phase, selected.params, inherited_winners)
    return Winner(
        trial_number=selected.trial_number,
        params=selected.params,
        effective_overrides=effective,
        metric=selected.metric,
        constraints=selected.constraints,
        gates=selected.gates,
        completion=completion,
        phase_fingerprint=phase_fingerprint,
        generation_id=selected.generation_id,
        attempt_id=selected.attempt_id,
        trial_dir=selected.trial_dir,
        source=WinnerSource(
            kind="phase_trial",
            phase=phase.name,
            trial_number=selected.trial_number,
            generation_id=selected.generation_id,
            attempt_id=selected.attempt_id,
        ),
        objective_provenance=selected.objective_provenance,
        trainer_input=dict(selected.trainer_input),
        # Preserve the winning trial's environment, including on no-op replay.
        trainer_env_digest=selected.trainer_env_digest,
        confound=selected.confound,
        observed=selected.observed,
        repeats=selected.repeats,
    )
