"""Phase execution through Optuna."""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, NamedTuple
from uuid import uuid4

import optuna

from phasesweep.config import Experiment, Phase
from phasesweep.config.search import _placeholder_values_for
from phasesweep.engine.artifacts import _write_trials_csv
from phasesweep.engine.attempts import (
    _register_active_attempt,
    _retire_active_attempt,
    _trial_requires_cleanup_recovery,
)
from phasesweep.engine.cleanup import _reap_stale_trials
from phasesweep.engine.errors import (
    ActiveAttemptPersistenceError,
    OperatorAction,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
)
from phasesweep.engine.evidence import _verify_winner_objective_evidence
from phasesweep.engine.fingerprints import _verify_fingerprint
from phasesweep.engine.ledger import ClaimedLedger, open_phase_study, open_preview_study
from phasesweep.engine.optuna import (
    _completed_trial_count,
    _finished_trial_count,
    _phase_study_name,
    _suggest,
)
from phasesweep.engine.paths import _phase_dir, _trial_dir_for
from phasesweep.engine.selection import NoFeasibleTrialError, select_winner
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    DURATION_ATTR,
    FAILURE_REASON_ATTR,
    FEASIBLE_ATTR,
    GATES_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    OVERRIDES_ATTR,
    PHASE_DECISION_ATTR,
    PHASE_DECISION_SCHEMA_VERSION,
    RETURN_CODE_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_ENV_NAMES_ATTR,
    TRAINER_INPUT_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ABORT_KEY,
    TRIAL_OUTCOME_ATTR,
    TRIAL_OUTCOME_SCHEMA_VERSION,
    Winner,
    WinnerSource,
    constraint_attr,
)
from phasesweep.engine.study_policy import (
    _accepted_trial_target,
    _AcceptedPartialDecision,
    _consecutive_failure_threshold_tripped,
    _load_accepted_partial_decision,
    _load_phase_policy_state,
    _next_consecutive_failures,
    _PhasePolicyState,
    _record_allocation_context,
    _record_recovery_boundary,
    _record_trial_target,
    _validate_environment_cohort,
    _validate_study_direction,
    _validate_study_schema,
    _validate_trial_target,
)
from phasesweep.engine.trial import (
    EnvironmentIdentity,
    TrialExecutionError,
    UnsafeProcessCleanupError,
    _environment_identity,
    _inherit_env_contract,
    extract_trial_result,
    launch_trial,
    prepare_trainer_input,
)
from phasesweep.runtime.commands import render_command
from phasesweep.runtime.gpu import GpuLeaseCancelledError, GpuLeaseTimeoutError, GpuPool
from phasesweep.runtime.process import write_attempt_lifecycle
from phasesweep.runtime.shutdown import PhaseSweepShutdown

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
    :param int finished_trials: Terminal trial count backing this completion.
    :param int completed_trials: COMPLETE trial count backing this completion.
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

    :param optuna.Study study: Study whose current terminal counts are verified.
    :param _AcceptedPartialDecision decision: Persisted accepted-partial decision.
    :param int outcome_sequence: Current durable outcome sequence for the study.
    :raises StudySchemaMismatchError: Trial counts or the outcome sequence changed
        after the accepted-partial decision was committed.
    :return dict[str, Any]: Frozen incomplete-completion metadata for publication.
    """
    trials = study.get_trials(deepcopy=False)
    finished_trials = _finished_trial_count(trials)
    completed_trials = _completed_trial_count(trials)
    if (
        outcome_sequence != decision.outcome_sequence
        or finished_trials != decision.finished_trials
        or completed_trials != decision.completed_trials
    ):
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} changed after its accepted partial-timeout "
            "decision was committed. Use a new experiment name, or archive/delete "
            "the inconsistent study."
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


class _TrialOutcomeUnrecordedAbort(BaseException):
    """Control-flow abort for a trial whose outcome record could not be written.

    Derives directly from :class:`BaseException` on purpose, and must keep
    doing so. Optuna's ``_run_trial`` catches ``optuna.TrialPruned`` and
    ``(Exception, KeyboardInterrupt)`` around the objective and then calls
    ``_tell_with_warning`` unconditionally, so *any* ``Exception`` leaving the
    objective commits that trial's terminal state. A terminal row whose
    ``TRIAL_OUTCOME_ATTR`` is missing cannot be repaired afterwards - Optuna
    raises ``UpdateFinishedTrialError`` for user-attr writes on a finished
    trial - and permanently wedges the study, because every later invocation
    rejects it in ``_load_phase_policy_state``. Raising from outside Optuna's
    catch set is what keeps the trial ``RUNNING`` instead, where the standard
    stale-attempt recovery writes the outcome *before* the terminal
    transition (PR #5 review / reviewer 2 pass 2, blocker 4).

    Deliberately not a :class:`PhaseSweepError`: it never reaches an operator.
    ``_run_phase`` converts it into :class:`StudyStorageUnavailableError` once
    the optimize loop has drained.
    """


# Delays between the bounded retries of the per-trial outcome write. The write
# is a single small user-attr row, so a transient backend fault (a locked
# journal ledger, a momentary connection drop) usually clears within a few
# hundred milliseconds; anything longer is an outage the phase must not
# outrun. len() + 1 total attempts.
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
    phase: Phase,
    sampled: dict[str, Any],
    inherited_winners: dict[str, Winner],
) -> dict[str, Any]:
    """Merge inherited winners, fixed overrides, and sampled params.

    :param Phase phase: The phase whose ``fixed_overrides`` and inheritance list apply.
    :param dict[str, Any] sampled: Values Optuna suggested for this trial.
    :param dict[str, Winner] inherited_winners: Parent-phase winners supplying
        the lowest-priority override layer.
    :raises ValueError: If the sampled parameters try to replace an inherited
        winner value.
    :return dict[str, Any]: Fully composed overrides for the trainer command.
    """
    inherited_overrides: dict[str, Any] = {}
    for parent in phase.inherits:
        inherited_overrides.update(inherited_winners[parent].effective_overrides)
    resampled = set(inherited_overrides) & set(sampled)
    if resampled:
        raise ValueError(
            f"Phase {phase.name!r} re-samples inherited winner key(s) {sorted(resampled)}."
        )
    out = dict(inherited_overrides)
    out.update(phase.fixed_overrides)
    out.update(sampled)
    return out


def _failure_policy_abort_record(
    phase: Phase,
    *,
    consecutive_failures: int,
    completion_sequence: int,
    trial_target: int,
) -> dict[str, Any]:
    """Build a durable consecutive-failure abort record.

    :param Phase phase: Phase supplying ``max_consecutive_failures`` for the
        recorded threshold and cause message.
    :param int consecutive_failures: Observed consecutive failure/infeasible
        count that tripped the policy.
    :param int completion_sequence: Orchestrator completion sequence at which
        the trip was observed.
    :param int trial_target: Durable accepted ``n_trials`` target to attach
        to the record.
    :return dict[str, Any]: A ``schema_version=1`` abort record ready to
        record with the tripping outcome.
    """
    return {
        "schema_version": 1,
        "policy": "max_consecutive_failures",
        "threshold": phase.max_consecutive_failures,
        "consecutive_failures": consecutive_failures,
        "completion_sequence": completion_sequence,
        "trial_target": trial_target,
        "cause": (
            f"{consecutive_failures} consecutive failed/infeasible trials reached "
            f"max_consecutive_failures={phase.max_consecutive_failures}."
        ),
    }


def _fatal_abort_record(
    *,
    policy: str,
    completion_sequence: int,
    trial_target: int,
    cause: str,
) -> dict[str, Any]:
    """Build a durable fatal-outcome abort record.

    :param str policy: Fatal policy that stopped the phase, such as
        ``"unsafe_process_cleanup"``.
    :param int completion_sequence: Completion sequence of the fatal outcome.
    :param int trial_target: Durable accepted ``n_trials`` target to attach
        to the record.
    :param str cause: Explanation the operator reads.
    :return dict[str, Any]: A ``schema_version=1`` abort record ready to
        record with the fatal outcome.
    """
    return {
        "schema_version": 1,
        "policy": policy,
        "completion_sequence": completion_sequence,
        "trial_target": trial_target,
        "cause": cause,
    }


def _raise_prior_phase_abort(phase: Phase, abort_record: dict[str, Any]) -> None:
    """Explain how to make an explicit recovery attempt after a durable abort.

    :param Phase phase: Phase whose accepted trial target is quoted in the
        message.
    :param dict[str, Any] abort_record: Recorded abort; supplies
        ``trial_target`` and ``cause``.
    :raises NoFeasibleTrialError: Always; this function never returns
        normally.
    """
    trial_target = abort_record["trial_target"]
    raise NoFeasibleTrialError(
        f"Phase {phase.name!r} previously aborted at its accepted n_trials={trial_target}: "
        f"{abort_record['cause']} Refusing to reinterpret those terminal attempts as a "
        f"successful phase. Increase n_trials above {trial_target} to explicitly schedule "
        "new recovery attempts, or use a new experiment name.",
        action=OperatorAction.FIX_CONFIG,
    )


def _raise_streak_at_limit(phase: Phase, consecutive_failures: int, trial_target: int) -> None:
    """Refuse new work while the resumed failure streak meets the current limit.

    :param Phase phase: Phase whose current ``max_consecutive_failures`` the
        streak meets.
    :param int consecutive_failures: Replayed consecutive-failure count.
    :param int trial_target: Durable accepted ``n_trials`` target.
    :raises NoFeasibleTrialError: Always; this function never returns
        normally.
    """
    raise NoFeasibleTrialError(
        f"Phase {phase.name!r} has {consecutive_failures} consecutive failed/infeasible "
        f"trials, which meets max_consecutive_failures={phase.max_consecutive_failures}, so "
        f"it launches no more trials at its accepted n_trials={trial_target}. Increase "
        f"n_trials above {trial_target} to explicitly schedule new recovery attempts, raise "
        "max_consecutive_failures, or use a new experiment name.",
        action=OperatorAction.FIX_CONFIG,
    )


@dataclass(frozen=True)
class _PhaseResume:
    """A live phase's durable state, validated before it may launch new work."""

    policy_state: _PhasePolicyState
    phase_fingerprint: str
    partial_decision: _AcceptedPartialDecision | None
    # A durable abort that the raised n_trials authorizes this run to recover from.
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
    winner. An abort recorded in the outcome ledger, or a replayed streak that
    meets the current limit while work remains, is refused unless the raised
    ``n_trials`` authorizes recovering from it.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: The phase being resumed.
    :param dict[str, Winner] inherited_winners: Winners loaded for phases
        earlier in the chain.
    :param optuna.Study study: The phase's live study.
    :raises NoFeasibleTrialError: The phase aborted, or its streak meets the
        current limit with work remaining, at an accepted target its current
        ``n_trials`` does not exceed.
    :raises TimeoutError: An accepted partial-timeout decision replays with
        no feasible trial.
    :raises StudySchemaMismatchError: Durable phase state is malformed or
        contradicts the study's trials.
    :raises StudyFingerprintMismatchError: The study was produced by a
        different phase config.
    :raises ProcessCleanupUncertainError: A stale trial's process could not be
        proven gone.
    :return _PhaseResume | Winner: The validated state, or the winner an
        accepted partial-timeout decision replays.
    """
    _validate_study_direction(study, experiment.metric.goal)
    _validate_study_schema(study)
    _reap_stale_trials(study, experiment, phase.name, confirm=True)
    policy_state = _load_phase_policy_state(study)
    phase_fingerprint = _verify_fingerprint(study, experiment, phase, inherited_winners)
    _validate_trial_target(study, phase)
    partial_decision = _load_accepted_partial_decision(study)
    if partial_decision is not None and phase.n_trials == partial_decision.trial_target:
        # This terminal decision is the authority even if a crash landed
        # before the timeout's recovery boundary consumed a recorded abort.
        # Replay deterministic selection; never reinterpret the timeout's
        # unused slots as permission to launch more trainers.
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
            )
        except NoFeasibleTrialError as exc:
            raise TimeoutError(
                f"Phase {phase.name!r} hit its {partial_decision.timeout_scope} deadline "
                "before any feasible trial could complete; no winner can be selected."
            ) from exc
    # An abort is the decision recorded with the outcome that tripped it, so a
    # later success resetting the streak, or a changed max_consecutive_failures,
    # never undoes it.
    active_abort = policy_state.abort
    if active_abort is not None and phase.n_trials <= active_abort["trial_target"]:
        _raise_prior_phase_abort(phase, active_abort)
    if (
        active_abort is None
        and _finished_trial_count(study.get_trials(deepcopy=False)) < phase.n_trials
        and _consecutive_failure_threshold_tripped(policy_state.consecutive_failures, phase)
    ):
        # The current limit governs new work only. A replayed streak that
        # already meets it (failures recovered from a stopped orchestrator, or
        # a lowered limit) launches nothing until n_trials is raised. Nothing is
        # recorded here, so a phase with no work left publishes, and restoring
        # the limit restores the phase.
        accepted_target = _accepted_trial_target(study)
        if phase.n_trials <= accepted_target:
            _raise_streak_at_limit(phase, policy_state.consecutive_failures, accepted_target)
        active_abort = _failure_policy_abort_record(
            phase,
            consecutive_failures=policy_state.consecutive_failures,
            completion_sequence=policy_state.max_sequence,
            trial_target=accepted_target,
        )
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


def _phase_budget(phase: Phase, run_deadline: float | None) -> _PhaseBudget:
    """Combine the phase timeout and the remaining run wallclock into one budget.

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
        raise TimeoutError(
            f"Run wallclock deadline reached before phase {phase.name!r} could launch."
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
    first fatal error, and every flag only ever goes from ``False`` to
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
    # Shared by the consecutive-failure and fatal-error paths. Queued
    # objectives check it inside the GPU lease and prune before launching.
    aborted: bool = False
    # Set once an outcome carrying this execution's abort record is durable.
    abort_recorded: bool = False
    deadline_exhausted: bool = False
    # Optuna's threaded optimize path can log an uncaught objective exception,
    # mark that one trial FAIL, and still return normally. The first fatal
    # exception is kept here so every n_jobs setting presents the same failure
    # to the caller after workers drain.
    fatal_error: BaseException | None = None
    csv_throttle: CsvSnapshotThrottle = field(default_factory=CsvSnapshotThrottle)
    outcome_lock: threading.Lock = field(default_factory=threading.Lock)
    fatal_lock: threading.Lock = field(default_factory=threading.Lock)

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
    ) -> dict[str, Any] | None:
        """Return the abort record that recording ``outcome`` next would trip.

        Changes nothing: the caller holds ``outcome_lock`` and writes the
        record with the outcome, before :meth:`advance` applies it, so the
        decision is exactly as durable as the outcome. Only the phase
        execution's first abort is recorded.

        :param int trial_number: Trial whose outcome is about to be recorded.
        :param str outcome: The terminal outcome about to be recorded.
        :param str | None cause: The outcome's recorded cause, if any.
        :param str | None fatal_policy: The fatal policy of a ``"fatal"`` outcome.
        :return dict[str, Any] | None: The abort record for the outcome's
            completion sequence, or ``None`` when it trips nothing new.
        """
        if self.abort_recorded:
            return None
        sequence = self.completion_sequence + 1
        if outcome == "fatal":
            return _fatal_abort_record(
                policy=fatal_policy or "fatal_trial_exception",
                completion_sequence=sequence,
                trial_target=self.phase.n_trials,
                cause=cause or f"trial {trial_number} hit a fatal objective error",
            )
        streak = _next_consecutive_failures(self.consecutive_failures, outcome)
        if self.aborted or not _consecutive_failure_threshold_tripped(streak, self.phase):
            return None
        return _failure_policy_abort_record(
            self.phase,
            consecutive_failures=streak,
            completion_sequence=sequence,
            trial_target=self.phase.n_trials,
        )

    def advance(self, trial_number: int, outcome: str) -> bool:
        """Apply one durably recorded outcome in completion order.

        The caller holds ``outcome_lock`` and has already written the outcome
        at sequence ``completion_sequence + 1``. The transition is the one the
        durable replay applies, so a restarted process reconstructs the same
        streak.

        :param int trial_number: Trial whose outcome was recorded.
        :param str outcome: The recorded terminal outcome.
        :return bool: Whether the outcome trips an abort: any fatal outcome,
            or the consecutive-failure threshold while no abort has tripped.
        """
        self.completion_sequence += 1
        self.recorded_outcomes[trial_number] = outcome
        self.consecutive_failures = _next_consecutive_failures(self.consecutive_failures, outcome)
        threshold_tripped = not self.aborted and _consecutive_failure_threshold_tripped(
            self.consecutive_failures, self.phase
        )
        return threshold_tripped or outcome == "fatal"

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
        """Record the first fatal objective exception and stop peer launches.

        First-writer wins so the root failure is not replaced by secondary
        peer-prune or persistence fallout. The original exception object keeps
        its worker traceback for the post-optimize re-raise.

        :param BaseException error: The fatal exception a worker observed.
        """
        with self.fatal_lock:
            first = self.fatal_error is None
            if first:
                self.fatal_error = error
        if first:
            log.error(
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


def _record_outcome(
    run: _PhaseExecution,
    trial: optuna.Trial,
    outcome: str,
    *,
    cause: str | None = None,
    fatal_policy: str | None = None,
) -> None:
    """Persist one terminal trial outcome in orchestrator completion order.

    "Consecutive" means consecutive in the order outcomes are recorded
    here — the order objective threads pass through ``run.outcome_lock`` —
    not trial-number order. The per-trial outcome is written before the
    counter changes or the objective returns, so a restarted process can
    reconstruct the same streak. An outcome that trips the phase's first
    abort carries that abort record in the same write, so the decision is
    exactly as durable as the outcome: a later peer success resets the
    streak but never the recorded abort. Any persistence failure aborts this
    invocation; it is never downgraded to a warning.

    A no-op if ``trial.number`` is already in ``run.recorded_outcomes``.
    Advances ``run`` through :meth:`_PhaseExecution.advance` and, once a
    threshold or fatal outcome trips, stops launches.

    Args:
        run: The phase execution the outcome belongs to.
        trial: The Optuna trial whose terminal outcome is being recorded.
        outcome: One of ``"success"``, ``"failure"``, ``"pruned"``,
            ``"cancelled"``, or ``"fatal"``.
        cause: Optional human-readable explanation stored on the outcome
            payload.
        fatal_policy: Optional policy name stored on the outcome payload
            and, when ``outcome`` is ``"fatal"``, on the resulting abort
            record.

    Raises:
        _TrialOutcomeUnrecordedAbort: The per-trial outcome row could not
            be persisted, even after ``_OUTCOME_WRITE_ATTEMPTS`` tries.
            The fatal-abort slot is set first, and the trial is left
            ``RUNNING`` for stale-attempt recovery.

    """
    # INVARIANT: no terminal Optuna row may exist without its outcome
    # record. Optuna refuses user-attr writes on a finished trial
    # (UpdateFinishedTrialError), so a FAIL committed after this write
    # failed could never be repaired. Retry any Exception first — storage
    # backends signal transient faults with assorted types — but never a
    # BaseException, which is a shutdown/control-flow abort of its own.
    #
    # The sequencing lock covers each write attempt and the state update it
    # authorizes, but not retry backoff. A worker waiting for storage must
    # not prevent a peer from durably recording an independent outcome.
    # Because a peer may advance the completion sequence during that wait,
    # every retry rebuilds its payload from the then-current sequence.
    write_error: Exception | None = None
    abort_record: dict[str, Any] | None = None
    for attempt_index in range(_OUTCOME_WRITE_ATTEMPTS):
        if attempt_index:
            time.sleep(_OUTCOME_WRITE_RETRY_DELAYS[attempt_index - 1])
        with run.outcome_lock:
            if trial.number in run.recorded_outcomes:
                return
            abort_record = run.abort_decision(
                trial.number, outcome, cause=cause, fatal_policy=fatal_policy
            )
            payload: dict[str, Any] = {
                "schema_version": TRIAL_OUTCOME_SCHEMA_VERSION,
                "sequence": run.completion_sequence + 1,
                "outcome": outcome,
            }
            if cause is not None:
                payload["cause"] = cause
            if fatal_policy is not None:
                payload["policy"] = fatal_policy
            if abort_record is not None:
                payload[TRIAL_OUTCOME_ABORT_KEY] = abort_record
            try:
                trial.set_user_attr(TRIAL_OUTCOME_ATTR, payload)
            except Exception as exc:
                write_error = exc
                continue

            write_error = None
            if not run.advance(trial.number, outcome):
                return
            run.stop_launches()
            if abort_record is None:
                return
            run.abort_recorded = True
            break
    if write_error is not None:
        unrecorded = _TrialOutcomeUnrecordedAbort(
            f"Could not persist the terminal outcome for trial {trial.number} "
            f"in study {run.study.study_name!r} after {_OUTCOME_WRITE_ATTEMPTS} "
            f"attempts ({type(write_error).__name__}: {write_error}). Trial "
            f"{trial.number} was deliberately left RUNNING, with its durable "
            "attempt record intact, so no terminal trial exists without its "
            "outcome record. Once storage accepts writes again, the next run "
            "recovers it through the standard stale-attempt protocol, which "
            "records the failure before marking the trial FAIL."
        )
        run.record_fatal(unrecorded)
        raise unrecorded from write_error

    assert abort_record is not None
    log.error("phase=%s ABORTED: %s", run.phase.name, abort_record["cause"])
    with contextlib.suppress(Exception):
        run.study.stop()


def _execute_objective(run: _PhaseExecution, trial: optuna.Trial) -> float:
    """Optuna objective: sample, launch trial subprocess, extract, return metric.

    Args:
        run: The phase execution the trial belongs to.
        trial: The active Optuna trial being evaluated.

    Returns:
        The extracted metric value (Optuna minimizes/maximizes per ``direction``).

    Raises:
        optuna.TrialPruned: A peer trial tripped a soft abort
            (max_consecutive_failures) and we should not start a new trial.
        UnsafeProcessCleanupError: The subprocess's cleanup could not be
            confirmed; we hard-abort the phase before another trial can
            acquire the just-released GPU lease.
        TrialExecutionError: The subprocess returned non-zero / produced
            no metric. Caught by ``study.optimize(catch=...)``.
        ActiveAttemptPersistenceError: Required attempt recovery metadata
            could not be persisted; nothing was launched and no GPU lease
            was consumed.
        _TrialOutcomeUnrecordedAbort: A successful trial's terminal
            outcome could not be persisted, so the trial is deliberately
            left ``RUNNING`` for stale-attempt recovery.

    """
    experiment = run.experiment
    phase = run.phase

    # Stamp the environment as the first durable fact of allocation. Even
    # a peer abort or failure creating lifecycle/registry metadata must remain in
    # the study's known semantic cohort on the next invocation.
    trial.set_user_attr(TRAINER_ENV_DIGEST_ATTR, run.environment_identity.digest)
    trial.set_user_attr(TRAINER_ENV_NAMES_ATTR, list(run.environment_identity.names))

    if run.aborted:
        raise optuna.TrialPruned("phase aborted")

    sampled = {name: _suggest(trial, name, p) for name, p in phase.search_space.items()}
    overrides = _composed_overrides(phase, sampled, run.inherited_winners)

    # Persist the resolved trial directory BEFORE launching the subprocess
    # so a later reaper can locate identity files even if the user moved
    # workdir or invoked phasesweep from a different cwd (review v0.5.3 /
    # blocker 4). Setting this attribute is what creates the trial in
    # Optuna storage with a known directory binding.
    attempt_id = uuid4().hex
    trial_dir = _trial_dir_for(
        experiment,
        phase.name,
        trial.number,
        generation_id=run.generation_id,
        attempt_id=attempt_id,
    )
    # Durable 'allocated' marker BEFORE the trial's attrs make it
    # discoverable and BEFORE the (arbitrarily long) GPU wait. A worker
    # killed while queued leaves a RUNNING trial with this marker and no
    # process identity; recovery can then prove no process was ever
    # created instead of failing closed forever (review v0.5.17 /
    # blocker 2 gap A). This is a pre-launch durability requirement: a
    # write failure would otherwise create an attempt that recovery can
    # never distinguish from a process whose identity write was torn.
    try:
        trial_dir.mkdir(parents=True, exist_ok=True)
        write_attempt_lifecycle(trial_dir, attempt_id=attempt_id, state="allocated")
    except OSError as exc:
        raise ActiveAttemptPersistenceError(
            f"Could not persist the allocated lifecycle marker for trial "
            f"{trial.number} (attempt {attempt_id}) at {trial_dir}: {exc}. "
            "No trainer was started and no GPU lease was consumed. Restore "
            "write access to the experiment workdir, then run again. With persistent "
            "storage this refusal durably aborts the phase, so that run also needs "
            "n_trials raised above its accepted target, or a new experiment name."
        ) from exc
    # Experiment-level registration is what keeps this attempt visible to
    # recovery even if the phase is later renamed/removed or the storage
    # URL changes (review v0.5.17 / blocker 3). Retired by the post-trial
    # callback once Optuna's terminal state is durable; preflight GCs
    # entries the callback never reached. It raises rather than warns, and
    # deliberately runs here — before the GPU lease and the launch — so a
    # trainer PhaseSweep could not record is never started (PR #5 review /
    # reviewer 2 pass 2, blocker 5). Unsafe-cleanup entries deliberately
    # survive the callback until preflight positively resolves the process
    # and commits the cleanup-recovery ledger.
    _register_active_attempt(
        experiment,
        attempt_id=attempt_id,
        phase_name=phase.name,
        study_name=run.study.study_name,
        trial_number=trial.number,
        trial_dir=trial_dir,
        generation_id=run.generation_id,
    )
    trial.set_user_attr(GENERATION_ID_ATTR, run.generation_id)
    trial.set_user_attr(ATTEMPT_ID_ATTR, attempt_id)
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    # GPU lease covers only subprocess lifetime, not extraction (#2).
    try:
        with run.gpu_pool.acquire(deadline=run.optimize_deadline) as gpu_assignment:
            # Re-check abort flags inside the lease (review v0.5.2 / blocker 8,
            # extended to fatal objective errors). Without this, queued
            # objective threads that passed the outer check before a peer
            # flipped the flag would still launch trials after the abort
            # fires — defeating max_consecutive_failures whenever n_jobs
            # exceeds the GPU-pool size, and defeating unsafe-cleanup abort
            # whenever any sibling thread is between launch_trial() return
            # and the cleanup_confirmed check.
            if run.aborted:
                raise optuna.TrialPruned("phase aborted")

            _refuse_launch_after_deadline(run)
            prepared_input = prepare_trainer_input(
                experiment=experiment,
                phase_name=phase.name,
                trial_id=trial.number,
                attempt_id=attempt_id,
                trial_dir=trial_dir,
                overrides=overrides,
            )
            # Persist the historical input identity before the trainer is
            # started. If this ledger write fails, no subprocess consumes
            # evidence that the study cannot later verify.
            trial.set_user_attr(TRAINER_INPUT_ATTR, prepared_input.record())
            _refuse_launch_after_deadline(run)
            executed = launch_trial(
                experiment=experiment,
                phase_name=phase.name,
                trial_id=trial.number,
                generation_id=run.generation_id,
                attempt_id=attempt_id,
                trial_dir=trial_dir,
                overrides=overrides,
                timeout_seconds=phase.timeout_seconds_per_trial,
                wallclock_deadline=run.optimize_deadline,
                gpu_id=gpu_assignment.visible_devices,
                gpu_lease_fds=gpu_assignment.lease_fds,
                prepared_input=prepared_input,
            )

            # CRITICAL: this check must happen INSIDE the GPU lease (review
            # v0.5.11 / blocker 3). Releasing the lease before observing
            # ``cleanup_confirmed=False`` lets a queued worker acquire the
            # GPU and launch a new trial onto the still-leaked process
            # group. ``run.record_fatal`` sets ``run.aborted`` while we
            # still hold the lease, so the next thread to enter sees the
            # flag and prunes before launch.
            if not executed.process.cleanup_confirmed:
                message = (
                    f"Trial {trial.number} cleanup could not be confirmed. "
                    f"trial_dir={trial_dir} pid={executed.process.pid}. "
                    f"reason={executed.process.failure_reason or 'process cleanup could not be confirmed'}. "
                    "Refusing to launch additional trials because a leaked "
                    "process group may still hold GPU/CPU resources."
                )
                cleanup_error = UnsafeProcessCleanupError(message)
                run.record_fatal(cleanup_error)

                # Best-effort forensic attrs. A storage write failure here
                # must not mask the safety-critical state: the fatal error
                # is already recorded and ``run.raise_if_fatal`` will fire
                # after ``study.optimize`` returns regardless.
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
        raise optuna.TrialPruned("phase aborted") from exc

    # Extraction happens outside GPU lease but INSIDE the phase/run
    # wallclock budget: the configured timeouts bound the whole trial,
    # not just the trainer (review v0.5.17 / blocker 8).
    result = extract_trial_result(
        experiment=experiment,
        executed=executed,
        gates=phase.gates,
        deadline=run.optimize_deadline,
    )
    if result.deadline_exhausted:
        # Preserve causal attribution carried by the result: a trainer
        # killed by the wallclock-capped budget, extraction, or gate
        # enforcement. Merely observing another failure after the clock
        # elapsed must not relabel it as a timeout.
        run.stop_for_deadline()

    trial.set_user_attr(FEASIBLE_ATTR, result.feasible)
    trial.set_user_attr(RETURN_CODE_ATTR, result.return_code)
    trial.set_user_attr(DURATION_ATTR, result.duration_seconds)
    trial.set_user_attr(OVERRIDES_ATTR, json.dumps(overrides, default=str, sort_keys=True))
    if result.objective_provenance is not None:
        trial.set_user_attr(
            OBJECTIVE_PROVENANCE_ATTR,
            json.dumps(result.objective_provenance, sort_keys=True),
        )
    if result.gate_results is not None:
        trial.set_user_attr(
            GATES_ATTR,
            json.dumps(
                [
                    {
                        "type": gate.gate_type,
                        "passed": gate.passed,
                        "detail": gate.detail,
                    }
                    for gate in result.gate_results
                ],
                sort_keys=True,
            ),
        )

    # Process/extractor failures -> Optuna FAIL state, not COMPLETE with inf (#4).
    if result.failure_reason:
        trial.set_user_attr(FAILURE_REASON_ATTR, result.failure_reason)
        error_type = (
            _DeadlineTrialExecutionError if result.deadline_exhausted else TrialExecutionError
        )
        raise error_type(result.failure_reason)

    for cname, cval in result.constraints.items():
        trial.set_user_attr(constraint_attr(cname), cval)

    _record_outcome(
        run,
        trial,
        "success" if result.feasible else "failure",
        cause=None if result.feasible else "metric or evidence constraints were infeasible",
    )

    assert result.metric is not None  # guaranteed when failure_reason is None
    return result.metric


def _objective(run: _PhaseExecution, trial: optuna.Trial) -> float:
    """Classify and durably order every terminal objective outcome.

    Delegates to ``_execute_objective``, then routes any exception it
    raises through ``_record_outcome`` (and, for fatal cases,
    ``run.record_fatal``) before re-raising the original exception
    unchanged.

    Args:
        run: The phase execution the trial belongs to.
        trial: The active Optuna trial being evaluated.

    Returns:
        The metric from ``_execute_objective`` when the trial completes
        without raising.

    """
    try:
        return _execute_objective(run, trial)
    except _TrialOutcomeUnrecordedAbort:
        # Must stay first. Every handler below answers by calling
        # ``_record_outcome``, which is exactly what just failed, and the
        # catch-all would additionally relabel this as an unexpected
        # objective exception. Re-raising it untouched is what keeps the
        # trial RUNNING for stale-attempt recovery (PR #5 review /
        # reviewer 2 pass 2, blocker 4). Only the direct call path from
        # ``_execute_objective`` needs the guard: the same abort raised
        # from inside a sibling handler already bypasses the rest.
        raise
    except optuna.TrialPruned as exc:
        _record_outcome(run, trial, "pruned", cause=str(exc))
        raise
    except _DeadlineTrialExecutionError as exc:
        _record_outcome(run, trial, "cancelled", cause=str(exc))
        raise
    except TrialExecutionError as exc:
        _record_outcome(run, trial, "failure", cause=str(exc))
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
        # Named rather than left to the catch-all below: the durable abort
        # record is the operator's only account of why the phase stopped,
        # and "unexpected_objective_exception" would point them at a
        # PhaseSweep bug instead of at their unwritable workdir (PR #5
        # review / reviewer 2 pass 2, blocker 5).
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
        # A host shutdown cancels the orchestration attempt; it is not a
        # trainer failure or an internal objective bug. Optuna propagates
        # SystemExit without promising a terminal trial transition, so the
        # durable outcome is the recovery authority: a later stale-attempt
        # pass may close the row, but must not turn the interruption into a
        # failure streak or durable phase abort.
        _record_outcome(
            run,
            trial,
            "cancelled",
            cause=f"orchestrator interrupted by signal {exc.signum}",
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
    """Post-trial callback: re-assert a tripped abort and snapshot ``trials.csv``.

    The threshold decision itself lives in ``_record_outcome``, inside the
    objective's completion-order critical section — a callback observing a
    mutable aggregate counter is race-prone under ``n_jobs > 1`` (review
    v0.5.17 / blocker 7). This callback only repeats ``study.stop()`` in
    case the tripping thread's stop call failed transiently.

    Args:
        run: The phase execution the trial belongs to.
        study: The running Optuna study (used to call ``study.stop``).
        trial: The just-finished trial; supplies the attempt id whose
            registry entry is retired.

    """
    if run.aborted:
        with contextlib.suppress(Exception):
            study.stop()
    # Terminal state alone is insufficient for an unsafe-cleanup trial: its
    # registry entry is the cross-phase/storage recovery locator and must
    # survive until preflight durably consumes cleanup evidence.
    # Optuna skips callbacks for uncaught objective errors; the enclosing
    # run's reconciliation retires those attempts after checking cleanup.
    finished_attempt = trial.user_attrs.get(ATTEMPT_ID_ATTR)
    if (
        isinstance(finished_attempt, str)
        and finished_attempt
        and not _trial_requires_cleanup_recovery(trial)
    ):
        _retire_active_attempt(run.experiment, finished_attempt)
    finished = _finished_trial_count(study.get_trials(deepcopy=False))
    now = time.monotonic()
    if run.csv_throttle.should_write(finished, now):
        with contextlib.suppress(Exception):
            _write_trials_csv(study, _phase_dir(run.experiment, run.phase.name) / "trials.csv")
            run.csv_throttle.mark_written(finished, now)


def _finish_phase(run: _PhaseExecution, *, phase_fingerprint: str) -> Winner:
    """Turn how the optimize loop ended into the phase's winner or refusal.

    :param _PhaseExecution run: The drained phase execution.
    :param str phase_fingerprint: Verified semantic fingerprint for the phase.
    :raises NoFeasibleTrialError: An abort tripped and no timeout left work
        undone, or every trial was infeasible.
    :raises TimeoutError: The deadline left the phase incomplete and partial
        publication is disabled, or no feasible trial completed before it.
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
    # Scheduler-level deadline causality: the phase is short of its trial
    # target and the clock is spent, so the budget — not the observation order
    # — is what left work undone. Being short of the target is the whole test:
    # a phase whose every requested trial is terminal is not relabelled just
    # because the clock happened to elapse before the last one was observed.
    # A tripped ``run.aborted`` deliberately does NOT veto this. When a
    # wallclock timeout and ``max_consecutive_failures`` become true in the
    # same phase, timeout handling takes precedence (docs/runtime.md), which is
    # what lets ``allow_incomplete_on_timeout`` publish the partial winner the
    # earlier successful trials already earned instead of discarding it.
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
        raise NoFeasibleTrialError(
            f"Phase {phase.name!r} aborted after "
            f"{phase.max_consecutive_failures} consecutive failures. "
            f"Inspect {_phase_dir(experiment, phase.name)} for stderr logs."
        )
    if (
        finished_after < phase.n_trials
        and not accepted_partial_timeout
        and not timed_out_incomplete
    ):
        raise RuntimeError(
            f"Phase {phase.name!r} stopped after {finished_after}/{phase.n_trials} "
            "terminal trials without an accepted timeout; refusing to publish "
            "an incomplete winner."
        )
    current_policy_state = _load_phase_policy_state(study)
    recorded_abort = current_policy_state.abort
    if accepted_partial_timeout:
        recovered_abort_sequence = (
            recorded_abort["completion_sequence"] if recorded_abort is not None else None
        )
        # This is the terminal-decision transaction boundary. If the write
        # fails, selection/publication never starts and a retry may spend a
        # fresh timeout budget. Once it succeeds, identical retries replay
        # selection from this frozen trial boundary without launching work.
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
        # A deadline takes precedence over a failure streak whether or not the
        # operator accepts a partial winner. Record that the timeout consumed
        # this exact abort before either selection or TimeoutError: otherwise
        # the documented remedy for a refused partial result (retry with a
        # larger timeout) would replay the recorded failure abort.
        _record_recovery_boundary(study, phase, recorded_abort["completion_sequence"])
    if timed_out_incomplete and not phase.allow_incomplete_on_timeout:
        raise TimeoutError(
            f"Phase {phase.name!r} timed out via {run.timeout_source or 'wallclock'} guard "
            f"after {completed_after}/{phase.n_trials} completed evaluations "
            f"({finished_after} terminal trials). Refusing to select a winner "
            "from an incomplete phase; set allow_incomplete_on_timeout: true "
            "only when a partial decision is intentional."
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
            # The deadline (not the trainer) is what prevented feasible work
            # — e.g. every launch was refused or cut short after the budget
            # expired (review v0.5.16 / blocker 6). Reporting this as
            # "no feasible trial" would misdirect operators at trainer logs.
            raise TimeoutError(
                f"Phase {phase.name!r} hit its {run.timeout_source or 'wallclock'} deadline "
                "before any feasible trial could complete; no winner can be selected."
            ) from exc
        raise


def _run_phase(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    *,
    generation_id: str | None,
    ledger: ClaimedLedger | None,
    run_deadline: float | None = None,
) -> Winner:
    """Execute one phase end-to-end (sampler, study.optimize, winner selection).

    Validates the phase's durable state (:func:`_resume_phase`), runs Optuna's
    optimize loop with every worker sharing one :class:`_PhaseExecution`, and
    turns how that loop ended into the phase's winner or refusal
    (:func:`_finish_phase`).

    Args:
        experiment: Parsed experiment config.
        phase: The phase to execute.
        inherited_winners: Winners loaded for phases earlier in the chain.
        generation_id: Identity of the current engine invocation, or ``None`` for dry-run.
        ledger: The claimed ledger, for ``experiment``, that this phase opens
            its live study through. ``None`` is a dry run: render an example
            trial command against an in-memory preview study and return a
            placeholder winner instead of launching any subprocesses.
        run_deadline: Optional ``time.monotonic()`` deadline inherited from
            the experiment-level wallclock guard.

    Returns:
        The selected phase :class:`Winner`.

    Raises:
        NoFeasibleTrialError: ``max_consecutive_failures`` tripped or every
            trial was infeasible.
        UnsafeProcessCleanupError: A trial's process group could not be
            confirmed dead; phase hard-aborted (review v0.5.11).
        ArtifactRootConflictError: This phase's persistent study is bound to a
            different artifact root than the config's workdir offers.
        TrialEvidenceMissingError: The selected winner's evidence directory,
            audit artifacts, or objective source are missing or no longer match
            the provenance frozen at extraction.
        ActiveAttemptPersistenceError: A trial's required recovery metadata
            could not be persisted, so it was never launched.
        StudyStorageUnavailableError: A trial's terminal outcome could not be
            persisted. That trial is left ``RUNNING`` with its attempt record
            intact and recovers on the next run once storage is writable
            (PR #5 review / reviewer 2 pass 2, blocker 4).
        StudySchemaMismatchError: Persisted phase recovery state contradicts
            the study's terminal trial count.
        TimeoutError: The phase or run deadline expires before the requested
            trial budget completes and partial publication is disabled, or
            before any feasible trial completes.
        PhaseSweepShutdown: The orchestrator receives a handled shutdown
            signal; propagated after recording a non-failure trial outcome.
        RuntimeError: Optuna returned before the requested trial budget without
            a timeout, abort, or exception, which violates the runner invariant.

    """
    dry_run = ledger is None
    study_name = _phase_study_name(experiment, phase)
    # The live opener also claims a newly created study's publication root,
    # before any inspection, reaping, or trial work below can run against it.
    study = (
        open_preview_study(experiment, phase) if ledger is None else open_phase_study(ledger, phase)
    )
    resume: _PhaseResume | None = None
    if not dry_run:
        resumed = _resume_phase(experiment, phase, inherited_winners, study)
        if isinstance(resumed, Winner):
            return resumed
        resume = resumed

    completed = _finished_trial_count(study.get_trials(deepcopy=False))
    remaining = max(0, phase.n_trials - completed)
    log.info(
        "phase=%s study=%s completed=%d remaining=%d n_jobs=%d",
        phase.name,
        study_name,
        completed,
        remaining,
        phase.n_jobs,
    )

    if dry_run:
        return _dry_run_phase(experiment, phase, inherited_winners, study, remaining)
    assert resume is not None

    if remaining == 0:
        # A no-op invocation republishes the existing result. It must neither
        # require launch resources (GPU discovery on a CPU-only host) nor
        # mutate the durable accepted target (review v0.5.14 / blocker 4).
        if resume.recovery_abort is not None:
            raise StudySchemaMismatchError(
                f"Phase {phase.name!r} accepted an abort recovery target but has no "
                "remaining trial slots. Use a new experiment name rather than reusing "
                "this inconsistent study."
            )
        trials_after = study.get_trials(deepcopy=False)
        return _select_phase_winner(
            experiment,
            phase,
            inherited_winners,
            study,
            phase_fingerprint=resume.phase_fingerprint,
            completion=_winner_completion(
                requested_trials=phase.n_trials,
                finished_trials=_finished_trial_count(trials_after),
                completed_trials=_completed_trial_count(trials_after),
                incomplete=False,
                reason=None,
                timeout_scope=None,
            ),
        )

    # Compose and validate the launch environment only after recovery proves
    # this phase has work remaining. Published/no-op replay must not require an
    # online W&B environment, while every new allocation still joins the
    # study's established semantic environment cohort.
    environment_identity = _environment_identity(experiment, phase.name)
    _validate_environment_cohort(study, environment_identity.digest)

    from phasesweep.config.models import _wandb_query
    from phasesweep.evidence.wandb import require_wandb_sdk

    if _wandb_query(experiment, phase.gates) is not None:
        require_wandb_sdk()

    gpu_pool = GpuPool.create(
        n_jobs=phase.n_jobs,
        explicit_ids=phase.gpu_ids,
        explicit_devices=phase.gpu_devices,
        allow_no_gpu=phase.allow_no_gpu_isolation,
        policy=phase.gpu_policy,
        cuda_visible_devices=experiment.env.get("CUDA_VISIBLE_DEVICES"),
    )
    budget = _phase_budget(phase, run_deadline)
    assert generation_id is not None
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
    )
    run.resume_from(resume.policy_state)
    # Accept the (possibly larger) durable target only after every launch
    # prerequisite — GPU discovery and the wallclock budget above — has passed.
    # Recording it earlier would strand the study at a target no invocation
    # ever launched work toward, making the previously working config a
    # rejected regression (review v0.5.14 / blocker 4).
    _record_trial_target(study, phase)
    if (
        resume.partial_decision is not None
        and phase.n_trials > resume.partial_decision.trial_target
    ):
        # Raising n_trials is the explicit authorization to resume work after
        # a terminal partial decision. The old record is no longer actionable
        # once the larger target is durable.
        study.set_user_attr(PHASE_DECISION_ATTR, None)
    if resume.recovery_abort is not None:
        # The boundary retires the recorded abort and starts a new streak, so
        # this execution's own first abort is recorded afresh.
        _record_recovery_boundary(study, phase, resume.recovery_abort["completion_sequence"])
        run.resume_from(_load_phase_policy_state(study))
    try:
        try:
            _record_allocation_context(
                study,
                generation_id=generation_id,
                trainer_environment=environment_identity.digest,
            )
            study.optimize(
                functools.partial(_objective, run),
                n_trials=remaining,
                n_jobs=phase.n_jobs,
                timeout=budget.timeout,
                gc_after_trial=True,
                callbacks=[functools.partial(_after_trial, run)],
                catch=(TrialExecutionError,),
            )
        finally:
            # Always snapshot trials.csv, even if ``study.optimize`` raises
            # (n_jobs=1 fatal-objective path) or some other transient backend
            # error escapes. Forensic data must survive every exit path.
            # Best-effort: a write failure here must not mask the actual
            # exception from ``study.optimize``.
            with contextlib.suppress(Exception):
                _write_trials_csv(study, _phase_dir(experiment, phase.name) / "trials.csv")

        # Optuna's threaded path can absorb any uncaught objective exception
        # after marking that trial FAIL. Re-raise the first one before timeout,
        # soft abort, completeness, or winner-selection logic can relabel it.
        run.raise_if_fatal()
    except _TrialOutcomeUnrecordedAbort as exc:
        # The single conversion site for the outcome-write abort, covering
        # both routes out of the optimize loop: n_jobs=1 propagates it
        # straight through ``study.optimize``, while n_jobs>1 may swallow the
        # worker's exception and leave ``run.raise_if_fatal`` to re-raise it
        # from the fatal-abort slot. Operators and the MCP runner see a
        # retryable storage outage, which is what it is; the affected trial is
        # still RUNNING and recovers on the next run (PR #5 review /
        # reviewer 2 pass 2, blocker 4).
        raise StudyStorageUnavailableError(str(exc)) from exc

    return _finish_phase(run, phase_fingerprint=resume.phase_fingerprint)


def _select_phase_winner(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    study: optuna.Study,
    *,
    phase_fingerprint: str,
    completion: dict[str, Any],
) -> Winner:
    """Select the phase winner and bind it to its verified execution identity.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: Phase whose winner is selected.
    :param dict[str, Winner] inherited_winners: Winners from earlier phases in the chain.
    :param optuna.Study study: Study whose terminal trials supply the winner.
    :param str phase_fingerprint: Verified semantic fingerprint for the phase.
    :param dict[str, Any] completion: Completion metadata persisted with the winner.
    :raises NoFeasibleTrialError: Every terminal trial was infeasible.
    :raises TrialEvidenceMissingError: The selected trial's evidence directory,
        audit artifacts, or objective source are missing or no longer match the
        provenance frozen when its metric was extracted.
    :return Winner: The selected winner with composed overrides and source identity.
    """
    selected = select_winner(study, experiment, phase_name=phase.name)
    # The evidence behind the number about to be published is re-proved here,
    # digest and all (PR #5 review / reviewer 2, blocker 7). Selection itself
    # reads only Optuna, so nothing before this point has looked at whether the
    # winning trial's directory still holds the bytes its metric came from.
    _verify_winner_objective_evidence(experiment, phase.name, selected)
    effective = _composed_overrides(phase, selected.params, inherited_winners)
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
        source=WinnerSource(
            kind="phase_trial",
            phase=phase.name,
            trial_number=selected.trial_number,
            generation_id=selected.generation_id,
            attempt_id=selected.attempt_id,
        ),
        objective_provenance=selected.objective_provenance,
        trainer_input=dict(selected.trainer_input),
        # The digest comes from the winning TRIAL — a top-up can select a
        # trial an earlier invocation ran under another environment — while
        # the environment contract is config, identical for every trial in the study
        # because it is fingerprinted.
        trainer_env_digest=selected.trainer_env_digest,
        trainer_inherit_env=_inherit_env_contract(experiment),
    )


def _dry_run_phase(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    study: optuna.Study,
    remaining: int,
) -> Winner:
    """Render and log one example trial command for the phase without launching anything.

    Args:
        experiment: Parsed experiment config.
        phase: The phase being previewed.
        inherited_winners: Winners loaded for phases earlier in the chain.
        study: An in-memory Optuna study used to ``ask`` for one sample.
        remaining: Number of trials that *would* run; logged for the user.

    Returns:
        A :class:`Winner` placeholder built from the previewed sample, or from
        each parameter's low bound or first choice, so downstream dry-run
        previews see consistent inherited context.

    """
    log.info("DRY RUN phase=%s would launch %d trials", phase.name, remaining)
    sampled: dict[str, Any] | None = None
    if remaining > 0:
        sample_trial = study.ask()
        sampled = {name: _suggest(sample_trial, name, p) for name, p in phase.search_space.items()}
        study.tell(sample_trial, state=optuna.trial.TrialState.FAIL)
        overrides = _composed_overrides(phase, sampled, inherited_winners)
        preview_dir = _phase_dir(experiment, phase.name) / "trial_dryrun"
        cmd = render_command(
            experiment.trial_command,
            overrides,
            experiment.override_format,
            trial_dir=preview_dir,
            trial_id=-1,
            phase=phase.name,
            run_name=f"{experiment.experiment}-{phase.name}-DRYRUN",
            trainer_config=experiment.trainer_config,
        )
        log.info("DRY RUN example command:\n  %s", cmd)

    return _placeholder_winner(
        phase,
        inherited_winners,
        sampled_params=sampled,
    )


def _placeholder_winner(
    phase: Phase,
    inherited_winners: dict[str, Winner],
    *,
    sampled_params: dict[str, Any] | None = None,
) -> Winner:
    """Synthesize a placeholder winner for dry-run mode.

    A phase whose command was previewed reuses that command's sampled values so
    downstream previews inherit one coherent hypothetical chain. A skipped
    phase without a preview uses each parameter's low bound or first choice.
    Both paths include inherited effective overrides.

    Args:
        phase: The phase whose placeholder winner is needed.
        inherited_winners: Winners from earlier phases in the chain.
        sampled_params: Values used in the displayed preview command, or
            ``None`` to synthesize deterministic placeholder values.

    Returns:
        A :class:`Winner` with ``trial_number=-1`` and ``metric=NaN`` so any
        accidental use in non-dry contexts surfaces obviously.

    """
    placeholder_params = (
        _placeholder_values_for(phase.search_space)
        if sampled_params is None
        else dict(sampled_params)
    )
    effective = _composed_overrides(phase, placeholder_params, inherited_winners)
    return Winner(
        trial_number=-1,
        params=placeholder_params,
        effective_overrides=effective,
        metric=float("nan"),
        constraints={},
        gates=[],
        completion=_winner_completion(
            requested_trials=phase.n_trials,
            finished_trials=0,
            completed_trials=0,
            incomplete=True,
            reason="dry_run",
            timeout_scope=None,
        ),
    )
