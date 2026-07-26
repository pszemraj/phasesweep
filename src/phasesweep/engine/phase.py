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
from phasesweep.config.search import _placeholder_values_for
from phasesweep.engine.guards import (
    _reap_stale_trials,
    _record_trial_target,
    _validate_study_schema,
    _validate_trial_target,
    _verify_fingerprint,
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
    OVERRIDES_ATTR,
    PHASE_ABORT_ATTR,
    RETURN_CODE_ATTR,
    TRIAL_DIR_ATTR,
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
    extract_trial_result,
    launch_trial,
)
from phasesweep.runtime.commands import render_command
from phasesweep.runtime.gpu import GpuPool
from phasesweep.runtime.process import write_attempt_lifecycle

log = logging.getLogger("phasesweep.engine.phase")


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
    generation_id: str | None,
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
        RuntimeError: Storage / fingerprint / stale-reaper inconsistency.

    """
    study_name = _phase_study_name(experiment, phase)
    study = _create_phase_study(experiment, phase, dry_run=dry_run)
    phase_fingerprint: str

    if not dry_run:
        _validate_study_schema(study)
        _reap_stale_trials(study, experiment, phase.name)
        phase_fingerprint = _verify_fingerprint(study, experiment, phase, inherited_winners)
        _validate_trial_target(study, phase)

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
        # It must also consult the durable abort record: the same terminal
        # trial set that previously ended in an abort must not be
        # reinterpreted as a completed phase merely because the in-memory
        # abort flag died with the aborting process (review v0.5.17 /
        # blocker 1). Publishing again requires new work — a top-up that
        # reaches winner selection clears the record.
        abort_record = study.user_attrs.get(PHASE_ABORT_ATTR)
        if abort_record is not None:
            cause = (
                abort_record.get("cause")
                if isinstance(abort_record, dict)
                else f"malformed abort record {abort_record!r}"
            )
            raise NoFeasibleTrialError(
                f"Phase {phase.name!r} previously aborted: {cause} "
                "An identical re-run cannot convert that abort into a published "
                "result. Raise n_trials to schedule new attempts, or use a new "
                "experiment name."
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
    )

    _failure_lock = threading.Lock()
    _consecutive_failures = 0
    _completion_sequence = 0
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

    def _record_outcome(*, failed: bool) -> None:
        """Record one terminal trial outcome in orchestrator completion order.

        "Consecutive" means consecutive in the order outcomes are recorded
        here — the order objective threads pass through ``_failure_lock`` —
        not trial-number order. The threshold decision, the abort flag flip,
        and the durable abort record are all made inside the same critical
        section, so a later success can never reset the counter between two
        earlier failures and their policy evaluation, and a crash after the
        trip can never lose the abort (review v0.5.17 / blockers 1+7).

        The durable write is best-effort *within* the critical section: a
        storage failure is loudly logged but must not mask the in-memory
        abort, which still fails this invocation.
        """
        nonlocal _consecutive_failures, _completion_sequence
        with _failure_lock:
            _completion_sequence += 1
            if failed:
                _consecutive_failures += 1
            else:
                _consecutive_failures = 0
            tripped = (not abort["flag"]) and (
                _consecutive_failures >= phase.max_consecutive_failures
            )
            if not tripped:
                return
            abort["flag"] = True
            record = {
                "policy": "max_consecutive_failures",
                "threshold": phase.max_consecutive_failures,
                "consecutive_failures": _consecutive_failures,
                "completion_sequence": _completion_sequence,
                "cause": (
                    f"{_consecutive_failures} consecutive failed/infeasible trials "
                    f"reached max_consecutive_failures={phase.max_consecutive_failures}."
                ),
            }
            try:
                study.set_user_attr(PHASE_ABORT_ATTR, record)
            except Exception:  # noqa: BLE001 - durability failure must not mask the abort
                log.exception(
                    "phase=%s could not persist the durable abort record; "
                    "an identical no-op re-run of this phase may not see the abort",
                    phase.name,
                )
        log.error(
            "phase=%s ABORTED after %d consecutive failed/infeasible trials",
            phase.name,
            record["consecutive_failures"],
        )
        with contextlib.suppress(Exception):
            study.stop()

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
        assert generation_id is not None

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
        # blocker 2 gap A). Best-effort: a write failure only degrades that
        # trial back to the old fail-closed recovery semantics.
        try:
            trial_dir.mkdir(parents=True, exist_ok=True)
            write_attempt_lifecycle(trial_dir, attempt_id=attempt_id, state="allocated")
        except OSError:
            log.warning(
                "Could not persist the 'allocated' lifecycle marker for trial %d "
                "(attempt %s); recovery of a pre-launch crash will fail closed.",
                trial.number,
                attempt_id,
            )
        trial.set_user_attr(GENERATION_ID_ATTR, generation_id)
        trial.set_user_attr(ATTEMPT_ID_ATTR, attempt_id)
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
                    generation_id=generation_id,
                    attempt_id=attempt_id,
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
            _record_outcome(failed=True)
            raise TrialExecutionError(result.failure_reason)

        for cname, cval in result.constraints.items():
            trial.set_user_attr(constraint_attr(cname), cval)

        _record_outcome(failed=not result.feasible)

        assert result.metric is not None  # guaranteed when failure_reason is None
        return result.metric

    def abort_callback(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        """Post-trial callback: re-assert a tripped abort and snapshot ``trials.csv``.

        The threshold decision itself lives in ``_record_outcome``, inside the
        objective's completion-order critical section — a callback observing a
        mutable aggregate counter is race-prone under ``n_jobs > 1`` (review
        v0.5.17 / blocker 7). This callback only repeats ``study.stop()`` in
        case the tripping thread's stop call failed transiently.

        Args:
            study: The running Optuna study (used to call ``study.stop``).
            _trial: The just-finished trial; unused.

        """
        if abort["flag"]:
            with contextlib.suppress(Exception):
                study.stop()
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
    if finished_after < phase.n_trials and not accepted_partial_timeout:
        raise RuntimeError(
            f"Phase {phase.name!r} stopped after {finished_after}/{phase.n_trials} "
            "terminal trials without an accepted timeout; refusing to publish "
            "an incomplete winner."
        )
    try:
        winner = _select_phase_winner(
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
    if study.user_attrs.get(PHASE_ABORT_ATTR) is not None:
        # This invocation ran new work (the no-op path raises on an abort
        # record) and reached a successful selection, so the durable abort
        # no longer describes the study's terminal state (review v0.5.17 /
        # blocker 1). ``None`` reads back as "absent" through ``.get``.
        study.set_user_attr(PHASE_ABORT_ATTR, None)
    return winner


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
    :return Winner: The selected winner with composed overrides and source identity.
    """
    selected = select_winner(study, experiment)
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
