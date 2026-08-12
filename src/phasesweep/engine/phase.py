"""Phase execution through Optuna."""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import optuna

from phasesweep.config import Experiment, Gate, Phase
from phasesweep.config.models import _iter_fixed_override_layers
from phasesweep.config.search import _placeholder_values_for
from phasesweep.engine.errors import (
    ActiveAttemptPersistenceError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
)
from phasesweep.engine.guards import (
    _accepted_trial_target,
    _AcceptedPartialDecision,
    _bind_study_artifact_root,
    _load_accepted_partial_decision,
    _load_phase_policy_state,
    _reap_stale_trials,
    _record_trial_target,
    _register_active_attempt,
    _retire_active_attempt,
    _trial_requires_cleanup_recovery,
    _validate_study_direction,
    _validate_study_schema,
    _validate_trial_target,
    _verify_fingerprint,
    _verify_winner_objective_evidence,
)
from phasesweep.engine.optuna import _create_phase_study, _phase_study_name, _suggest
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
    PHASE_ABORT_ATTR,
    PHASE_DECISION_ATTR,
    PHASE_DECISION_SCHEMA_VERSION,
    PHASE_RECOVERY_ATTR,
    PHASE_RECOVERY_SCHEMA_VERSION,
    RETURN_CODE_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_ENV_NAMES_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_OUTCOME_SCHEMA_VERSION,
    Winner,
    WinnerSource,
    _phase_dir,
    _trial_dir_for,
    _write_trials_csv,
    constraint_attr,
)
from phasesweep.engine.trial import (
    TrialExecutionError,
    UnsafeProcessCleanupError,
    _environment_identity,
    _inherit_env_contract,
    extract_trial_result,
    launch_trial,
)
from phasesweep.runtime.commands import render_command
from phasesweep.runtime.gpu import GpuLeaseTimeoutError, GpuPool
from phasesweep.runtime.process import PhaseSweepShutdown, write_attempt_lifecycle

log = logging.getLogger("phasesweep.engine.phase")


def _partial_completion_for_replay(
    study: optuna.Study,
    decision: _AcceptedPartialDecision,
    *,
    outcome_sequence: int,
) -> dict[str, Any]:
    """Re-prove and return the frozen completion metadata for a decision replay."""
    trials = study.get_trials(deepcopy=False)
    finished_trials = _finished_trial_count(trials)
    completed_trials = sum(1 for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE)
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
    return {
        "requested_trials": decision.trial_target,
        "finished_trials": decision.finished_trials,
        "completed_trials": decision.completed_trials,
        "incomplete": True,
        "reason": "timeout",
        "timeout_scope": decision.timeout_scope,
    }


class _PolicyStateWriteError(RuntimeError):
    """Raised when durable failure-policy state cannot be persisted."""


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
# SQLite file, a momentary connection drop) usually clears within a few
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


def _finished_trial_count(trials: Iterable[optuna.trial.FrozenTrial]) -> int:
    """Return the number of terminal trials in ``trials``.

    :param Iterable[optuna.trial.FrozenTrial] trials: Trials whose states should be counted.
    :return int: Number of trials with a finished state.
    """
    return sum(1 for trial in trials if trial.state.is_finished())


def _composed_overrides(
    experiment: Experiment,
    phase: Phase,
    sampled: dict[str, Any],
    inherited_winners: dict[str, Winner],
) -> dict[str, Any]:
    """Merge inherited winners, contracts, fixed overrides, and sampled params.

    Args:
        experiment: Parsed experiment; provides named contracts.
        phase: The phase whose ``fixed_overrides`` and inheritance list apply.
        sampled: The values Optuna just suggested for this trial.
        inherited_winners: Parent-phase winners; their ``effective_overrides``
            are the base layer (lowest priority).

    Returns:
        The fully-composed override dict that gets handed to the trial command.
        Later layers (later keys in the merge order) overwrite earlier ones.

    """
    out: dict[str, Any] = {}
    for parent in phase.inherits:
        out.update(inherited_winners[parent].effective_overrides)
    for _origin, fixed_overrides in _iter_fixed_override_layers(experiment, phase):
        out.update(fixed_overrides)
    out.update(sampled)
    return out


def _phase_gates(experiment: Experiment, phase: Phase) -> list[Gate]:
    """Return contract gates followed by phase-local gates.

    :param Experiment experiment: Parsed experiment config containing named contracts.
    :param Phase phase: Phase whose contract list and local gates are resolved.
    :return list[Gate]: Gates in evaluation order.
    """
    gates: list[Gate] = []
    for contract_name in phase.contracts:
        gates.extend(experiment.contracts[contract_name].gates)
    gates.extend(phase.gates)
    return gates


def _active_phase_abort(
    study: optuna.Study,
    *,
    recovered_abort_sequence: int | None,
) -> dict[str, Any] | None:
    """Return the validated active abort record, clearing a superseded marker.

    :param optuna.Study study: Study whose ``PHASE_ABORT_ATTR`` user attr is
        read and, when superseded, cleared.
    :param int | None recovered_abort_sequence: Completion sequence of an
        already-acknowledged recovery, or ``None``. When it equals the
        persisted record's ``completion_sequence``, that record is cleared
        and treated as superseded.
    :raises StudySchemaMismatchError: The persisted attr is not a dict, or its
        fields fail schema validation.
    :return dict[str, Any] | None: The validated abort record, or ``None`` if
        no abort is persisted or the persisted one was just superseded.
    """
    raw = study.user_attrs.get(PHASE_ABORT_ATTR)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} has malformed {PHASE_ABORT_ATTR!r}={raw!r}. "
            "Use a new experiment name, or archive/delete the inconsistent study."
        )
    sequence = raw.get("completion_sequence")
    trial_target = raw.get("trial_target")
    policy = raw.get("policy")
    cause = raw.get("cause")
    if (
        raw.get("schema_version") != 1
        or type(sequence) is not int
        or sequence < 1
        or type(trial_target) is not int
        or trial_target < 1
        or not isinstance(policy, str)
        or not policy
        or not isinstance(cause, str)
        or not cause
    ):
        raise StudySchemaMismatchError(
            f"Study {study.study_name!r} has malformed {PHASE_ABORT_ATTR!r} fields: "
            f"{raw!r}. Use a new experiment name, or archive/delete the inconsistent study."
        )
    if recovered_abort_sequence == sequence:
        study.set_user_attr(PHASE_ABORT_ATTR, None)
        return None
    return raw


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
        persist as ``PHASE_ABORT_ATTR``.
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


def _raise_prior_phase_abort(phase: Phase, abort_record: dict[str, Any]) -> None:
    """Explain how to make an explicit recovery attempt after a durable abort.

    :param Phase phase: Phase whose accepted trial target is quoted in the
        message.
    :param dict[str, Any] abort_record: Persisted abort record; supplies
        ``trial_target`` and ``cause``.
    :raises NoFeasibleTrialError: Always; this function never returns
        normally.
    """
    trial_target = abort_record["trial_target"]
    raise NoFeasibleTrialError(
        f"Phase {phase.name!r} previously aborted at its accepted n_trials={trial_target}: "
        f"{abort_record['cause']} Refusing to reinterpret those terminal attempts as a "
        f"successful phase. Increase n_trials above {trial_target} to explicitly schedule "
        "new recovery attempts, or use a new experiment name."
    )


def _run_phase(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    *,
    generation_id: str | None,
    dry_run: bool = False,
    run_deadline: float | None = None,
) -> Winner:
    """Execute one phase end-to-end (sampler, study.optimize, winner selection).

    Defines nested objective, callback, and fatal-abort closures to encapsulate
    per-phase mutable state.

    Args:
        experiment: Parsed experiment config.
        phase: The phase to execute.
        inherited_winners: Winners loaded for phases earlier in the chain.
        generation_id: Identity of the current engine invocation, or ``None`` for dry-run.
        dry_run: When ``True``, render an example trial command and return a
            placeholder midpoint winner instead of launching any subprocesses.
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
        LegacyArtifactRootMigrationRequiredError: This phase's persistent study
            holds trials but predates artifact-root binding, so the workdir
            that owns its evidence cannot be inferred.
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
        RuntimeError: Optuna returned before the requested trial budget without
            a timeout, abort, or exception, which violates the runner invariant.

    """
    study_name = _phase_study_name(experiment, phase)
    study = _create_phase_study(experiment, phase, dry_run=dry_run)
    phase_fingerprint: str
    policy_state = None
    recovery_abort: dict[str, Any] | None = None
    partial_decision: _AcceptedPartialDecision | None = None
    # The trainer environment is a property of this process and its config, not
    # of any one trial, so it is composed once per phase execution and stamped
    # onto every trial (review v0.5.18 / finding F3).
    environment_identity = _environment_identity(experiment)

    if not dry_run:
        # A study this invocation just created was invisible to preflight, so
        # it claims its publication root here — before any inspection, reaping,
        # or trial work (review v0.5.19 / finding F5).
        _bind_study_artifact_root(study, experiment)
        _validate_study_direction(study, experiment.metric.goal)
        _validate_study_schema(study)
        _reap_stale_trials(study, experiment, phase.name)
        policy_state = _load_phase_policy_state(study)
        phase_fingerprint = _verify_fingerprint(study, experiment, phase, inherited_winners)
        _validate_trial_target(study, phase)
        partial_decision = _load_accepted_partial_decision(study)
        if partial_decision is not None and phase.n_trials == partial_decision.trial_target:
            # This terminal decision is the authority even if a crash landed
            # before the separate historical abort marker was cleared. Replay
            # deterministic selection; never reinterpret the timeout's unused
            # slots as permission to launch more trainers.
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
        accepted_target = _accepted_trial_target(study)
        active_abort = _active_phase_abort(
            study,
            recovered_abort_sequence=policy_state.recovered_abort_sequence,
        )
        if active_abort is None and policy_state.fatal_trial_number is not None:
            active_abort = {
                "schema_version": 1,
                "policy": "fatal_trial_exception",
                "completion_sequence": policy_state.fatal_sequence,
                "trial_target": accepted_target,
                "cause": (
                    f"trial {policy_state.fatal_trial_number} hit an unexpected fatal "
                    f"objective error: {policy_state.fatal_cause or 'no cause recorded'}"
                ),
            }
            active_abort["policy"] = policy_state.fatal_policy or "fatal_trial_exception"
            study.set_user_attr(PHASE_ABORT_ATTR, active_abort)
        if (
            active_abort is None
            and policy_state.consecutive_failures >= phase.max_consecutive_failures
        ):
            active_abort = _failure_policy_abort_record(
                phase,
                consecutive_failures=policy_state.consecutive_failures,
                completion_sequence=policy_state.max_sequence,
                trial_target=accepted_target,
            )
            study.set_user_attr(PHASE_ABORT_ATTR, active_abort)
        if active_abort is not None:
            if phase.n_trials <= active_abort["trial_target"]:
                _raise_prior_phase_abort(phase, active_abort)
            recovery_abort = active_abort

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

    if remaining == 0:
        # A no-op invocation republishes the existing result. It must neither
        # require launch resources (GPU discovery on a CPU-only host) nor
        # mutate the durable accepted target (review v0.5.14 / blocker 4).
        if recovery_abort is not None:
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
            phase_fingerprint=phase_fingerprint,
            completion={
                "requested_trials": phase.n_trials,
                "finished_trials": _finished_trial_count(trials_after),
                "completed_trials": sum(
                    1 for t in trials_after if t.state == optuna.trial.TrialState.COMPLETE
                ),
                "incomplete": False,
                "reason": None,
                "timeout_scope": None,
            },
        )

    gpu_pool = GpuPool.create(
        n_jobs=phase.n_jobs,
        explicit_ids=phase.gpu_ids,
        explicit_devices=phase.gpu_devices,
        allow_no_gpu=phase.allow_no_gpu_isolation,
        policy=phase.gpu_policy,
        cuda_visible_devices=experiment.env.get("CUDA_VISIBLE_DEVICES"),
    )

    assert policy_state is not None
    _failure_lock = threading.Lock()
    _consecutive_failures = policy_state.consecutive_failures
    _completion_sequence = policy_state.max_sequence
    recorded_outcomes: dict[int, str] = {}
    # ``abort["flag"]`` is shared by the consecutive-failure and fatal-error
    # paths. Queued objectives check it inside the GPU lease and prune before
    # launching.
    abort = {"flag": False}
    abort_recorded = {"flag": False}

    # Optuna's threaded optimize path can log an uncaught objective exception,
    # mark that one trial FAIL, and still return normally. Keep the first fatal
    # exception in orchestrator-owned state so every n_jobs setting presents
    # the same failure to the caller after workers drain.
    _fatal_abort_lock = threading.Lock()
    fatal_abort: dict[str, BaseException | None] = {"exception": None}
    deadline_exhausted = {"flag": False}
    csv_throttle = CsvSnapshotThrottle()

    def _record_fatal_abort(error: BaseException) -> None:
        """Record the first fatal objective exception and stop peer launches.

        First-writer wins so the root failure is not replaced by secondary
        peer-prune or persistence fallout. The original exception object keeps
        its worker traceback for the post-optimize re-raise.
        """
        with _fatal_abort_lock:
            first = fatal_abort["exception"] is None
            if first:
                fatal_abort["exception"] = error
        if first:
            log.error(
                "phase=%s FATAL OBJECTIVE ERROR (%s): %s",
                phase.name,
                type(error).__name__,
                error,
            )
        abort["flag"] = True
        with contextlib.suppress(Exception):
            study.stop()

    def _raise_if_fatal_aborted() -> None:
        """Re-raise the first fatal worker exception with its original traceback.

        A no-op when no worker recorded a fatal abort.

        Raises:
            BaseException: The exception ``_record_fatal_abort`` captured
                first, re-raised with its original worker traceback.

        """
        with _fatal_abort_lock:
            error = fatal_abort["exception"]
        if error is not None:
            raise error.with_traceback(error.__traceback__)

    def _record_outcome(
        trial: optuna.Trial,
        outcome: str,
        *,
        cause: str | None = None,
        fatal_policy: str | None = None,
    ) -> None:
        """Persist one terminal trial outcome in orchestrator completion order.

        "Consecutive" means consecutive in the order outcomes are recorded
        here — the order objective threads pass through ``_failure_lock`` —
        not trial-number order. The per-trial outcome is written before the
        counter changes or the objective returns, so a restarted process can
        reconstruct the same streak. Any persistence failure aborts this
        invocation; it is never downgraded to a warning. The two writes fail
        differently on purpose: the trial ledger row must exist before Optuna
        can be allowed to commit any terminal state for the trial, while the
        phase abort marker is written after that row is already durable and
        can therefore be reconstructed from it on the next run.

        A no-op if ``trial.number`` is already in ``recorded_outcomes``.
        Mutates ``_run_phase``'s enclosing ``_consecutive_failures``,
        ``_completion_sequence``, and ``recorded_outcomes``, and — once a
        threshold or fatal outcome trips — sets ``abort["flag"]`` and, the
        first time only, persists an abort record to ``study``'s
        ``PHASE_ABORT_ATTR``.

        Args:
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
            _PolicyStateWriteError: The phase abort marker could not be
                persisted. This trial's outcome row is already durable, so the
                next run reconstructs the abort from the ledger.

        """
        nonlocal _consecutive_failures, _completion_sequence
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
            with _failure_lock:
                if trial.number in recorded_outcomes:
                    return
                next_sequence = _completion_sequence + 1
                payload: dict[str, Any] = {
                    "schema_version": TRIAL_OUTCOME_SCHEMA_VERSION,
                    "sequence": next_sequence,
                    "outcome": outcome,
                }
                if cause is not None:
                    payload["cause"] = cause
                if fatal_policy is not None:
                    payload["policy"] = fatal_policy
                try:
                    trial.set_user_attr(TRIAL_OUTCOME_ATTR, payload)
                except Exception as exc:
                    write_error = exc
                    continue

                write_error = None
                _completion_sequence = next_sequence
                recorded_outcomes[trial.number] = outcome
                if outcome in {"failure", "fatal"}:
                    _consecutive_failures += 1
                elif outcome == "success":
                    _consecutive_failures = 0
                threshold_tripped = (not abort["flag"]) and (
                    _consecutive_failures >= phase.max_consecutive_failures
                )
                fatal_tripped = outcome == "fatal"
                if not threshold_tripped and not fatal_tripped:
                    return
                abort["flag"] = True
                if abort_recorded["flag"]:
                    return
                if fatal_tripped:
                    abort_record = {
                        "schema_version": 1,
                        "policy": fatal_policy or "fatal_trial_exception",
                        "completion_sequence": _completion_sequence,
                        "trial_target": phase.n_trials,
                        "cause": cause or f"trial {trial.number} hit a fatal objective error",
                    }
                else:
                    abort_record = _failure_policy_abort_record(
                        phase,
                        consecutive_failures=_consecutive_failures,
                        completion_sequence=_completion_sequence,
                        trial_target=phase.n_trials,
                    )
                try:
                    study.set_user_attr(PHASE_ABORT_ATTR, abort_record)
                except Exception as exc:
                    error = _PolicyStateWriteError(
                        f"Trial {trial.number} durably recorded completion sequence "
                        f"{_completion_sequence}, but the phase abort marker could not be "
                        "persisted. Refusing to continue; the durable outcome ledger will "
                        "reconstruct the abort on the next run."
                    )
                    _record_fatal_abort(error)
                    raise error from exc
                abort_recorded["flag"] = True
                break
        if write_error is not None:
            unrecorded = _TrialOutcomeUnrecordedAbort(
                f"Could not persist the terminal outcome for trial {trial.number} "
                f"in study {study.study_name!r} after {_OUTCOME_WRITE_ATTEMPTS} "
                f"attempts ({type(write_error).__name__}: {write_error}). Trial "
                f"{trial.number} was deliberately left RUNNING, with its durable "
                "attempt record intact, so no terminal trial exists without its "
                "outcome record. Once storage accepts writes again, the next run "
                "recovers it through the standard stale-attempt protocol, which "
                "records the failure before marking the trial FAIL."
            )
            _record_fatal_abort(unrecorded)
            raise unrecorded from write_error

        assert abort_record is not None
        log.error("phase=%s ABORTED: %s", phase.name, abort_record["cause"])
        with contextlib.suppress(Exception):
            study.stop()

    def _execute_objective(trial: optuna.Trial) -> float:
        """Optuna objective: sample, launch trial subprocess, extract, return metric.

        Args:
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
        assert generation_id is not None

        if abort["flag"]:
            raise optuna.TrialPruned("phase aborted")

        sampled = {name: _suggest(trial, name, p) for name, p in phase.search_space.items()}
        overrides = _composed_overrides(experiment, phase, sampled, inherited_winners)

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
            generation_id=generation_id,
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
                "write access to the experiment workdir, then run again."
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
            study_name=study.study_name,
            trial_number=trial.number,
            trial_dir=trial_dir,
            generation_id=generation_id,
        )
        trial.set_user_attr(GENERATION_ID_ATTR, generation_id)
        trial.set_user_attr(ATTEMPT_ID_ATTR, attempt_id)
        trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
        # Recorded at allocation so failed trials carry their environment too:
        # a phase whose every trial died under a broken CUDA stack is only
        # diagnosable if the failures name the environment they ran under.
        trial.set_user_attr(TRAINER_ENV_DIGEST_ATTR, environment_identity.digest)
        trial.set_user_attr(TRAINER_ENV_NAMES_ATTR, list(environment_identity.names))

        # GPU lease covers only subprocess lifetime, not extraction (#2).
        try:
            with gpu_pool.acquire(deadline=optimize_deadline) as gpu_assignment:
                # Re-check abort flags inside the lease (review v0.5.2 / blocker 8,
                # extended to fatal objective errors). Without this, queued
                # objective threads that passed the outer check before a peer
                # flipped the flag would still launch trials after the abort
                # fires — defeating max_consecutive_failures whenever n_jobs
                # exceeds the GPU-pool size, and defeating unsafe-cleanup abort
                # whenever any sibling thread is between launch_trial() return
                # and the cleanup_confirmed check.
                if abort["flag"]:
                    raise optuna.TrialPruned("phase aborted")

                timeout_seconds = phase.timeout_seconds_per_trial
                timeout_capped_by_wallclock = False
                if optimize_deadline is not None:
                    remaining_wallclock = optimize_deadline - time.monotonic()
                    if remaining_wallclock <= 0.0:
                        deadline_exhausted["flag"] = True
                        raise TrialExecutionError(
                            f"{timeout_source or 'wallclock'} deadline reached before trial launch."
                        )
                    if timeout_seconds is None or remaining_wallclock < timeout_seconds:
                        timeout_seconds = remaining_wallclock
                        timeout_capped_by_wallclock = True

                executed = launch_trial(
                    experiment=experiment,
                    phase_name=phase.name,
                    trial_id=trial.number,
                    generation_id=generation_id,
                    attempt_id=attempt_id,
                    trial_dir=trial_dir,
                    overrides=overrides,
                    timeout_seconds=timeout_seconds,
                    gpu_id=gpu_assignment.visible_devices,
                    gpu_lease_fds=gpu_assignment.lease_fds,
                )

                # CRITICAL: this check must happen INSIDE the GPU lease (review
                # v0.5.11 / blocker 3). Releasing the lease before observing
                # ``cleanup_confirmed=False`` lets a queued worker acquire the
                # GPU and launch a new trial onto the still-leaked process
                # group. ``_record_fatal_abort`` flips the shared abort flag while
                # we still hold the lease, so the next thread to enter sees the
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
                    _record_fatal_abort(cleanup_error)

                    # Best-effort forensic attrs. A storage write failure here
                    # must not mask the safety-critical state: the fatal error
                    # is already recorded and ``_raise_if_fatal_aborted`` will
                    # fire after ``study.optimize`` returns regardless.
                    with contextlib.suppress(Exception):
                        trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
                        trial.set_user_attr(
                            FAILURE_REASON_ATTR,
                            executed.process.failure_reason
                            or "process cleanup could not be confirmed",
                        )

                    raise cleanup_error
        except GpuLeaseTimeoutError as exc:
            deadline_exhausted["flag"] = True
            raise TrialExecutionError(str(exc)) from exc

        # Extraction happens outside GPU lease but INSIDE the phase/run
        # wallclock budget: the configured timeouts bound the whole trial,
        # not just the trainer (review v0.5.17 / blocker 8).
        result = extract_trial_result(
            experiment=experiment,
            executed=executed,
            gates=_phase_gates(experiment, phase),
            enforce_gates=phase.promotion is None or phase.promotion.requires_gates,
            deadline=optimize_deadline,
            trainer_timeout_is_deadline=timeout_capped_by_wallclock,
        )
        if result.deadline_exhausted:
            # Preserve causal attribution carried by the result: a trainer
            # killed by the wallclock-capped budget, extraction, or gate
            # enforcement. Merely observing another failure after the clock
            # elapsed must not relabel it as a timeout.
            deadline_exhausted["flag"] = True

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
            raise TrialExecutionError(result.failure_reason)

        for cname, cval in result.constraints.items():
            trial.set_user_attr(constraint_attr(cname), cval)

        _record_outcome(
            trial,
            "success" if result.feasible else "failure",
            cause=None if result.feasible else "metric or evidence constraints were infeasible",
        )

        assert result.metric is not None  # guaranteed when failure_reason is None
        return result.metric

    def objective(trial: optuna.Trial) -> float:
        """Classify and durably order every terminal objective outcome.

        Delegates to ``_execute_objective``, then routes any exception it
        raises through ``_record_outcome`` (and, for fatal cases,
        ``_record_fatal_abort``) before re-raising the original exception
        unchanged.

        Args:
            trial: The active Optuna trial being evaluated.

        Returns:
            The metric from ``_execute_objective`` when the trial completes
            without raising.

        """
        try:
            return _execute_objective(trial)
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
        except _PolicyStateWriteError as exc:
            _record_fatal_abort(exc)
            raise
        except optuna.TrialPruned as exc:
            _record_outcome(trial, "pruned", cause=str(exc))
            raise
        except TrialExecutionError as exc:
            _record_outcome(trial, "failure", cause=str(exc))
            raise
        except UnsafeProcessCleanupError as exc:
            _record_fatal_abort(exc)
            _record_outcome(
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
            _record_fatal_abort(exc)
            _record_outcome(
                trial,
                "fatal",
                cause=str(exc),
                fatal_policy="active_attempt_persistence",
            )
            raise
        except PhaseSweepShutdown as exc:
            # A host shutdown cancels the orchestration attempt; it is not a
            # trainer failure or an internal objective bug. Optuna will still
            # close the active trial while propagating SystemExit, so give the
            # terminal row an ordered, non-failure outcome without installing
            # a durable phase-abort marker that would wedge the next run.
            _record_outcome(
                trial,
                "cancelled",
                cause=f"orchestrator interrupted by signal {exc.signum}",
            )
            raise
        except BaseException as exc:
            _record_fatal_abort(exc)
            _record_outcome(
                trial,
                "fatal",
                cause=f"{type(exc).__name__}: {exc}",
                fatal_policy="unexpected_objective_exception",
            )
            raise

    def abort_callback(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        """Post-trial callback: re-assert a tripped abort and snapshot ``trials.csv``.

        The threshold decision itself lives in ``_record_outcome``, inside the
        objective's completion-order critical section — a callback observing a
        mutable aggregate counter is race-prone under ``n_jobs > 1`` (review
        v0.5.17 / blocker 7). This callback only repeats ``study.stop()`` in
        case the tripping thread's stop call failed transiently.

        Args:
            study: The running Optuna study (used to call ``study.stop``).
            _trial: The just-finished trial; supplies the attempt id whose
                registry entry is retired.

        """
        if abort["flag"]:
            with contextlib.suppress(Exception):
                study.stop()
        # Terminal state alone is insufficient for an unsafe-cleanup trial: its
        # registry entry is the cross-phase/storage recovery locator and must
        # survive until preflight durably consumes cleanup evidence.
        finished_attempt = _trial.user_attrs.get(ATTEMPT_ID_ATTR)
        if (
            isinstance(finished_attempt, str)
            and finished_attempt
            and not _trial_requires_cleanup_recovery(_trial)
        ):
            _retire_active_attempt(experiment, finished_attempt)
        finished = _finished_trial_count(study.get_trials(deepcopy=False))
        now = time.monotonic()
        if csv_throttle.should_write(finished, now):
            with contextlib.suppress(Exception):
                _write_trials_csv(study, _phase_dir(experiment, phase.name) / "trials.csv")
                csv_throttle.mark_written(finished, now)

    timeout_source: str | None = None
    optimize_deadline: float | None = None
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
    if optimize_timeout is not None:
        optimize_deadline = time.monotonic() + optimize_timeout
    # Accept the (possibly larger) durable target only after every launch
    # prerequisite — GPU discovery above, wallclock budget here — has passed.
    # Recording it earlier would strand the study at a target no invocation
    # ever launched work toward, making the previously working config a
    # rejected regression (review v0.5.14 / blocker 4).
    _record_trial_target(study, phase)
    if partial_decision is not None and phase.n_trials > partial_decision.trial_target:
        # Raising n_trials is the explicit authorization to resume work after
        # a terminal partial decision. The old record is no longer actionable
        # once the larger target is durable.
        study.set_user_attr(PHASE_DECISION_ATTR, None)
    if recovery_abort is not None:
        current_policy_state = _load_phase_policy_state(study)
        study.set_user_attr(
            PHASE_RECOVERY_ATTR,
            {
                "schema_version": PHASE_RECOVERY_SCHEMA_VERSION,
                "recovered_abort_sequence": recovery_abort["completion_sequence"],
                "start_after_sequence": current_policy_state.max_sequence,
                "trial_target": phase.n_trials,
            },
        )
        # The recovery boundary is durable before the marker is cleared. If
        # clearing fails or this process dies here, the next invocation sees
        # that this exact abort was already acknowledged and safely retries
        # the clear instead of resetting the streak again.
        study.set_user_attr(PHASE_ABORT_ATTR, None)
        current_policy_state = _load_phase_policy_state(study)
        _completion_sequence = current_policy_state.max_sequence
        _consecutive_failures = current_policy_state.consecutive_failures
    try:
        try:
            study.optimize(
                objective,
                n_trials=remaining,
                n_jobs=phase.n_jobs,
                timeout=optimize_timeout,
                gc_after_trial=True,
                callbacks=[abort_callback],
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
        _raise_if_fatal_aborted()
    except _TrialOutcomeUnrecordedAbort as exc:
        # The single conversion site for the outcome-write abort, covering
        # both routes out of the optimize loop: n_jobs=1 propagates it
        # straight through ``study.optimize``, while n_jobs>1 may swallow the
        # worker's exception and leave ``_raise_if_fatal_aborted`` to re-raise
        # it from the fatal-abort slot. Operators and the MCP runner see a
        # retryable storage outage, which is what it is; the affected trial is
        # still RUNNING and recovers on the next run (PR #5 review /
        # reviewer 2 pass 2, blocker 4).
        raise StudyStorageUnavailableError(str(exc)) from exc

    trials_after = study.get_trials(deepcopy=False)
    finished_after = _finished_trial_count(trials_after)
    completed_after = sum(1 for t in trials_after if t.state == optuna.trial.TrialState.COMPLETE)
    # Scheduler-level deadline causality: the phase is short of its trial
    # target and the clock is spent, so the budget — not the observation order
    # — is what left work undone. Being short of the target is the whole test:
    # a phase whose every requested trial is terminal is not relabelled just
    # because the clock happened to elapse before the last one was observed.
    # A tripped ``abort["flag"]`` deliberately does NOT veto this. When a
    # wallclock timeout and ``max_consecutive_failures`` become true in the
    # same phase, timeout handling takes precedence (docs/runtime.md), which is
    # what lets ``allow_incomplete_on_timeout`` publish the partial winner the
    # earlier successful trials already earned instead of discarding it.
    scheduler_deadline_exhausted = (
        optimize_deadline is not None
        and finished_after < phase.n_trials
        and time.monotonic() >= optimize_deadline
    )
    if scheduler_deadline_exhausted:
        deadline_exhausted["flag"] = True
    timeout_observed = deadline_exhausted["flag"]
    timed_out_incomplete = timeout_observed and finished_after < phase.n_trials
    accepted_partial_timeout = (
        phase.allow_incomplete_on_timeout and timeout_observed and finished_after < phase.n_trials
    )
    if abort["flag"] and not timed_out_incomplete:
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
    persisted_abort = study.user_attrs.get(PHASE_ABORT_ATTR)
    if accepted_partial_timeout:
        current_policy_state = _load_phase_policy_state(study)
        recovered_abort_sequence = (
            persisted_abort["completion_sequence"] if persisted_abort is not None else None
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
                "timeout_scope": timeout_source,
                "recovered_abort_sequence": recovered_abort_sequence,
            },
        )
    if persisted_abort is not None:
        # A deadline takes precedence over a failure streak whether or not the
        # operator accepts a partial winner. Record that the timeout consumed
        # this exact abort before either selection or TimeoutError: otherwise
        # the documented remedy for a refused partial result (retry with a
        # larger timeout) would reconstruct and raise the stale failure abort.
        if timed_out_incomplete:
            current_policy_state = _load_phase_policy_state(study)
            study.set_user_attr(
                PHASE_RECOVERY_ATTR,
                {
                    "schema_version": PHASE_RECOVERY_SCHEMA_VERSION,
                    "recovered_abort_sequence": persisted_abort["completion_sequence"],
                    "start_after_sequence": current_policy_state.max_sequence,
                    "trial_target": phase.n_trials,
                },
            )
        # Clear before selection. Selection is deterministic from durable
        # trial data, so a crash during it re-derives the same winner.
        study.set_user_attr(PHASE_ABORT_ATTR, None)
    if timed_out_incomplete and not phase.allow_incomplete_on_timeout:
        raise TimeoutError(
            f"Phase {phase.name!r} timed out via {timeout_source or 'wallclock'} guard "
            f"after {completed_after}/{phase.n_trials} completed evaluations "
            f"({finished_after} terminal trials). Refusing to select a winner "
            "from an incomplete phase; set allow_incomplete_on_timeout: true "
            "only when a partial decision is intentional."
        )
    try:
        return _select_phase_winner(
            experiment,
            phase,
            inherited_winners,
            study,
            phase_fingerprint=phase_fingerprint,
            completion={
                "requested_trials": phase.n_trials,
                "finished_trials": finished_after,
                "completed_trials": completed_after,
                "incomplete": accepted_partial_timeout,
                "reason": "timeout" if accepted_partial_timeout else None,
                "timeout_scope": timeout_source if accepted_partial_timeout else None,
            },
        )
    except NoFeasibleTrialError as exc:
        if deadline_exhausted["flag"]:
            # The deadline (not the trainer) is what prevented feasible work
            # — e.g. every launch was refused or cut short after the budget
            # expired (review v0.5.16 / blocker 6). Reporting this as
            # "no feasible trial" would misdirect operators at trainer logs.
            raise TimeoutError(
                f"Phase {phase.name!r} hit its {timeout_source or 'wallclock'} deadline "
                "before any feasible trial could complete; no winner can be selected."
            ) from exc
        raise


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
    effective = _composed_overrides(experiment, phase, selected.params, inherited_winners)
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
        # The digest comes from the winning TRIAL — a top-up can select a
        # trial an earlier invocation ran under another environment — while
        # the contract is config, identical for every trial in the study
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
        A :class:`Winner` placeholder built from midpoint params so downstream
        dry-run previews see consistent inherited context.

    """
    log.info("DRY RUN phase=%s would launch %d trials", phase.name, remaining)
    sampled: dict[str, Any] | None = None
    if remaining > 0:
        sample_trial = study.ask()
        sampled = {name: _suggest(sample_trial, name, p) for name, p in phase.search_space.items()}
        study.tell(sample_trial, state=optuna.trial.TrialState.FAIL)
        overrides = _composed_overrides(experiment, phase, sampled, inherited_winners)
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
            write_files=False,
        )
        log.info("DRY RUN example command:\n  %s", cmd)

    return _placeholder_winner(
        experiment,
        phase,
        inherited_winners,
        sampled_params=sampled,
    )


def _placeholder_winner(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    *,
    sampled_params: dict[str, Any] | None = None,
) -> Winner:
    """Synthesize a placeholder winner for dry-run mode.

    A phase whose command was previewed reuses that command's sampled values so
    downstream previews inherit one coherent hypothetical chain. A skipped
    phase without a preview uses deterministic midpoint/first-choice values.
    Both paths include inherited effective overrides.

    Args:
        experiment: Parsed experiment; supplies named contracts.
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
    effective = _composed_overrides(experiment, phase, placeholder_params, inherited_winners)
    return Winner(
        trial_number=-1,
        params=placeholder_params,
        effective_overrides=effective,
        metric=float("nan"),
        constraints={},
        gates=[],
        completion={
            "requested_trials": phase.n_trials,
            "finished_trials": 0,
            "completed_trials": 0,
            "incomplete": True,
            "reason": "dry_run",
            "timeout_scope": None,
        },
        source=WinnerSource(
            kind="phase_trial",
            phase=phase.name,
            trial_number=-1,
            generation_id=None,
            attempt_id=None,
        ),
    )
