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

import optuna

from phasesweep.config import Experiment, Gate, Phase
from phasesweep.config.search import _placeholder_values_for
from phasesweep.engine.guards import (
    _phase_fingerprint,
    _reap_stale_trials,
    _verify_fingerprint,
)
from phasesweep.engine.optuna import _create_phase_study, _phase_study_name, _suggest
from phasesweep.engine.selection import NoFeasibleTrialError, select_winner
from phasesweep.engine.state import (
    CLEANUP_CONFIRMED_ATTR,
    DURATION_ATTR,
    FAILURE_REASON_ATTR,
    FEASIBLE_ATTR,
    GATES_ATTR,
    OVERRIDES_ATTR,
    RETURN_CODE_ATTR,
    TRIAL_DIR_ATTR,
    Winner,
    _phase_dir,
    _trial_dir_for,
    _write_trials_csv,
    constraint_attr,
)
from phasesweep.engine.trial import (
    TrialExecutionError,
    UnsafeProcessCleanupError,
    extract_trial_result,
    launch_trial,
)
from phasesweep.runtime.commands import render_command
from phasesweep.runtime.gpu import GpuPool

log = logging.getLogger("phasesweep.engine.phase")


@dataclass
class CsvSnapshotThrottle:
    """Debounce expensive full ``trials.csv`` snapshots during a phase."""

    min_trials: int = 10
    min_seconds: float = 30.0
    last_finished: int = 0
    last_write_at: float = 0.0

    def should_write(self, finished: int, now: float) -> bool:
        """Return whether another full CSV snapshot should be written."""
        return (
            finished - self.last_finished >= self.min_trials
            or now - self.last_write_at >= self.min_seconds
        )

    def mark_written(self, finished: int, now: float) -> None:
        """Record a successful snapshot write."""
        self.last_finished = finished
        self.last_write_at = now


def _finished_trial_count(trials: Iterable[optuna.trial.FrozenTrial]) -> int:
    """Return the number of terminal trials in ``trials``."""
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
    for contract_name in phase.contracts:
        out.update(experiment.contracts[contract_name].fixed_overrides)
    out.update(phase.fixed_overrides)
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


def _run_phase(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    *,
    dry_run: bool = False,
    run_deadline: float | None = None,
) -> Winner:
    """Execute one phase end-to-end (sampler, study.optimize, winner selection).

    Defines nested
    closures ``objective``, ``abort_callback``, ``_record_hard_abort``, and
    ``_raise_if_hard_aborted`` to encapsulate per-phase mutable state.

    Args:
        experiment: Parsed experiment config.
        phase: The phase to execute.
        inherited_winners: Winners loaded for phases earlier in the chain.
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
        RuntimeError: Storage / fingerprint / stale-reaper inconsistency.

    """
    study_name = _phase_study_name(experiment, phase)
    study = _create_phase_study(experiment, phase, dry_run=dry_run)

    if not dry_run:
        # Reap first, fingerprint second (review item #7). A config-mismatch RuntimeError
        # must not leave a previous orchestrator's training process holding GPU memory.
        _reap_stale_trials(study, experiment, phase.name)
        _verify_fingerprint(study, experiment, phase, inherited_winners)

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

    gpu_pool = GpuPool.create(
        n_jobs=phase.n_jobs,
        explicit_ids=phase.gpu_ids,
        explicit_devices=phase.gpu_devices,
        allow_no_gpu=phase.allow_no_gpu_isolation,
        policy=phase.gpu_policy,
    )

    _failure_lock = threading.Lock()
    _consecutive_failures = 0
    # ``abort["flag"]`` is the soft-abort flag, set by max_consecutive_failures
    # and (for defense in depth) by ``_record_hard_abort`` below. Queued
    # objectives check it inside the GPU lease and prune before launching.
    abort = {"flag": False}

    # Hard-abort state for unsafe process cleanup. Optuna's threaded
    # ``n_jobs>1`` optimize path does NOT propagate uncaught objective
    # exceptions: it logs them and marks the trial FAIL (verified against
    # optuna._optimize._run_trial in v0.5.11 review). Propagation only works
    # for ``n_jobs=1``. We therefore record the unsafe-cleanup condition in
    # orchestrator-owned state and re-raise after ``study.optimize()``
    # returns. See review v0.5.11.
    _hard_abort_lock = threading.Lock()
    hard_abort: dict[str, str | None] = {"message": None}
    deadline_exhausted = {"flag": False}
    csv_throttle = CsvSnapshotThrottle()

    def _record_hard_abort(message: str) -> None:
        """Record a safety-critical phase abort.

        First-writer wins on ``hard_abort['message']``. Flips the soft
        ``abort['flag']`` so queued objectives prune before launch, and asks
        Optuna to stop scheduling new trials. ``study.stop`` is best-effort:
        we do not want a storage hiccup to mask the safety-critical state.
        """
        with _hard_abort_lock:
            first = hard_abort["message"] is None
            if first:
                hard_abort["message"] = message
        if first:
            log.error("phase=%s HARD ABORT: %s", phase.name, message)
        abort["flag"] = True
        with contextlib.suppress(Exception):
            study.stop()

    def _raise_if_hard_aborted() -> None:
        """Raise ``UnsafeProcessCleanupError`` if any peer recorded a hard abort.

        Raises:
            UnsafeProcessCleanupError: ``hard_abort['message']`` is set.

        """
        with _hard_abort_lock:
            message = hard_abort["message"]
        if message is not None:
            raise UnsafeProcessCleanupError(message)

    def objective(trial: optuna.Trial) -> float:
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

        """
        nonlocal _consecutive_failures

        # Hard abort takes priority. For n_jobs=1 this matches the old
        # behavior of relying on exception propagation; for n_jobs>1 this
        # is the only mechanism that surfaces unsafe cleanup, since Optuna
        # swallows non-caught objective exceptions in threaded mode.
        _raise_if_hard_aborted()
        if abort["flag"]:
            raise optuna.TrialPruned("phase aborted")

        sampled = {name: _suggest(trial, name, p) for name, p in phase.search_space.items()}
        overrides = _composed_overrides(experiment, phase, sampled, inherited_winners)

        # Persist the resolved trial directory BEFORE launching the subprocess
        # so a later reaper can locate identity files even if the user moved
        # workdir or invoked phasesweep from a different cwd (review v0.5.3 /
        # blocker 4). Setting this attribute is what creates the trial in
        # Optuna storage with a known directory binding.
        trial_dir = _trial_dir_for(experiment, phase.name, trial.number)
        trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))

        # GPU lease covers only subprocess lifetime, not extraction (#2).
        try:
            with gpu_pool.acquire(deadline=optimize_deadline) as gpu_id:
                # Re-check abort flags inside the lease (review v0.5.2 / blocker 8,
                # extended in v0.5.11 for hard_abort). Without this, queued
                # objective threads that passed the outer check before a peer
                # flipped the flag would still launch trials after the abort
                # fires — defeating max_consecutive_failures whenever n_jobs
                # exceeds the GPU-pool size, and defeating unsafe-cleanup abort
                # whenever any sibling thread is between launch_trial() return
                # and the cleanup_confirmed check.
                _raise_if_hard_aborted()
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
                    trial_dir=trial_dir,
                    overrides=overrides,
                    timeout_seconds=timeout_seconds,
                    gpu_id=gpu_id,
                )
                if timeout_capped_by_wallclock and executed.process.timed_out:
                    deadline_exhausted["flag"] = True

                # CRITICAL: this check must happen INSIDE the GPU lease (review
                # v0.5.11 / blocker 3). Releasing the lease before observing
                # ``cleanup_confirmed=False`` lets a queued worker acquire the
                # GPU and launch a new trial onto the still-leaked process
                # group. ``_record_hard_abort`` flips the soft abort flag while
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
                    _record_hard_abort(message)

                    # Best-effort forensic attrs. A storage write failure here
                    # must not mask the safety-critical state: ``hard_abort``
                    # is already recorded and ``_raise_if_hard_aborted`` will
                    # fire after ``study.optimize`` returns regardless.
                    with contextlib.suppress(Exception):
                        trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
                        trial.set_user_attr(
                            FAILURE_REASON_ATTR,
                            executed.process.failure_reason
                            or "process cleanup could not be confirmed",
                        )

                    raise UnsafeProcessCleanupError(message)
        except TimeoutError as exc:
            deadline_exhausted["flag"] = True
            raise TrialExecutionError(str(exc)) from exc

        # Extraction happens outside GPU lease.
        result = extract_trial_result(
            experiment=experiment,
            executed=executed,
            gates=_phase_gates(experiment, phase),
            enforce_gates=phase.promotion is None or phase.promotion.requires_gates,
        )

        trial.set_user_attr(FEASIBLE_ATTR, result.feasible)
        trial.set_user_attr(RETURN_CODE_ATTR, result.return_code)
        trial.set_user_attr(DURATION_ATTR, result.duration_seconds)
        trial.set_user_attr(OVERRIDES_ATTR, json.dumps(overrides, default=str, sort_keys=True))
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
            with _failure_lock:
                _consecutive_failures += 1
            raise TrialExecutionError(result.failure_reason)

        for cname, cval in result.constraints.items():
            trial.set_user_attr(constraint_attr(cname), cval)

        with _failure_lock:
            if result.feasible:
                _consecutive_failures = 0
            else:
                _consecutive_failures += 1

        assert result.metric is not None  # guaranteed when failure_reason is None
        return result.metric

    def abort_callback(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        """Post-trial callback: trip the soft abort flag if ``max_consecutive_failures`` reached.

        Args:
            study: The running Optuna study (used to call ``study.stop``).
            _trial: The just-finished trial; unused (we read the shared
                ``_consecutive_failures`` counter instead, which the objective
                maintains under ``_failure_lock``).

        """
        with _failure_lock:
            count = _consecutive_failures
        if count >= phase.max_consecutive_failures:
            if not abort["flag"]:
                log.error(
                    "phase=%s ABORTED after %d consecutive failed/infeasible trials",
                    phase.name,
                    count,
                )
            abort["flag"] = True
            study.stop()
        finished = _finished_trial_count(study.get_trials(deepcopy=False))
        now = time.monotonic()
        if csv_throttle.should_write(finished, now):
            with contextlib.suppress(Exception):
                _write_trials_csv(study, _phase_dir(experiment, phase.name) / "trials.csv")
                csv_throttle.mark_written(finished, now)

    timeout_source: str | None = None
    optimize_deadline: float | None = None
    if remaining > 0:
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
            # (n_jobs=1 hard-abort path) or some other transient backend
            # error escapes. Forensic data must survive every exit path.
            # Best-effort: a write failure here must not mask the actual
            # exception from ``study.optimize``.
            with contextlib.suppress(Exception):
                _write_trials_csv(study, _phase_dir(experiment, phase.name) / "trials.csv")

    # Re-raise unsafe cleanup BEFORE the soft abort check. Optuna's threaded
    # n_jobs>1 optimize path can swallow non-caught objective exceptions when
    # ``n_trials == n_jobs`` and every trial fails (it logs them and marks
    # the trial FAIL — verified against optuna 4.8.0 in v0.5.11 review). We
    # cannot rely on exception propagation alone to surface this safety-
    # critical condition; the orchestrator owns the abort state and re-raises
    # here. For n_jobs=1 the original UnsafeProcessCleanupError already
    # propagated out of study.optimize above; this re-raise is a no-op then.
    # Review v0.5.11 / v0.5.12.
    _raise_if_hard_aborted()

    trials_after = study.get_trials(deepcopy=False)
    finished_after = _finished_trial_count(trials_after)
    completed_after = sum(1 for t in trials_after if t.state == optuna.trial.TrialState.COMPLETE)
    timeout_observed = deadline_exhausted["flag"] or (
        optimize_deadline is not None and time.monotonic() >= optimize_deadline
    )
    timed_out_incomplete = timeout_observed and finished_after < phase.n_trials
    accepted_partial_timeout = (
        phase.allow_incomplete_on_timeout and timeout_observed and finished_after < phase.n_trials
    )
    if timed_out_incomplete and not phase.allow_incomplete_on_timeout:
        raise TimeoutError(
            f"Phase {phase.name!r} timed out via {timeout_source or 'wallclock'} guard "
            f"after {completed_after}/{phase.n_trials} completed evaluations "
            f"({finished_after} terminal trials). Refusing to select a winner "
            "from an incomplete phase; set allow_incomplete_on_timeout: true "
            "only when a partial decision is intentional."
        )
    if abort["flag"] and not timed_out_incomplete:
        raise NoFeasibleTrialError(
            f"Phase {phase.name!r} aborted after "
            f"{phase.max_consecutive_failures} consecutive failures. "
            f"Inspect {_phase_dir(experiment, phase.name)} for stderr logs."
        )
    completion = {
        "requested_trials": phase.n_trials,
        "finished_trials": finished_after,
        "completed_trials": completed_after,
        "incomplete": accepted_partial_timeout,
        "reason": "timeout" if accepted_partial_timeout else None,
        "timeout_scope": timeout_source if accepted_partial_timeout else None,
    }

    # Build winner with effective_overrides (#9). Stamp it with the phase
    # fingerprint so a later --from-phase resume can detect stale parent
    # config (review v0.5.6 / blocker 3). The fingerprint is the same one
    # _verify_fingerprint stamps on the Optuna study; recomputing here keeps
    # _save_winner independent of study state.
    selected = select_winner(study, experiment)
    effective = _composed_overrides(experiment, phase, selected.params, inherited_winners)
    winner = Winner(
        trial_number=selected.trial_number,
        params=selected.params,
        effective_overrides=effective,
        metric=selected.metric,
        constraints=selected.constraints,
        gates=selected.gates,
        completion=completion,
        phase_fingerprint=_phase_fingerprint(experiment, phase, inherited_winners),
    )
    return winner


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
    )
