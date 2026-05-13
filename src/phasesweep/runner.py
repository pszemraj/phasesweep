"""Run a single trial as a supervised subprocess and extract its result.

Split into two phases:
  launch_trial  — needs GPU lease, runs subprocess
  extract_trial — no GPU needed, reads result files / polls W&B
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phasesweep.config import Experiment, Gate, check_bounds
from phasesweep.extractors import ExtractorError, TrialContext, run_extractor
from phasesweep.gates import GateResult, evaluate_gates
from phasesweep.overrides import render_command
from phasesweep.process import ProcessResult, run_supervised

log = logging.getLogger("phasesweep.runner")


@dataclass
class ExecutedTrial:
    """Result of launching a trial subprocess. Does NOT yet contain metrics."""

    ctx: TrialContext
    process: ProcessResult


@dataclass
class TrialResult:
    """Final result after extraction. metric is None when the trial failed."""

    metric: float | None
    constraints: dict[str, float]
    return_code: int
    duration_seconds: float
    feasible: bool
    failure_reason: str | None = None
    gate_results: list[GateResult] | None = None


class TrialExecutionError(RuntimeError):
    """Raised when a trial subprocess crashes or extraction fails.

    Caught by study.optimize(catch=...) so Optuna marks the trial FAIL,
    not COMPLETE with a sentinel value.
    """


class UnsafeProcessCleanupError(RuntimeError):
    """Raised when a trial process group may still be alive after cleanup.

    This must NOT be included in Optuna's ``catch`` tuple. The correct behavior
    is to abort the phase/run, not mark one trial FAIL and continue — a leaked
    process group can hold GPU memory, write conflicting outputs, or starve
    the host scheduler (review v0.5.9 / blocker 3).
    """


def _failed_trial(
    *,
    rc: int,
    duration: float,
    failure_reason: str,
    constraints: dict[str, float] | None = None,
    gate_results: list[GateResult] | None = None,
) -> TrialResult:
    """Build a ``TrialResult`` representing a failed trial.

    Centralizes the ``metric=None / feasible=False`` shape that the five
    failure exits in :func:`extract_trial_result` were repeating. ``constraints``
    is empty for failures that occur before constraint extraction starts; it
    carries the partial dict for failures that surface mid-loop.

    Args:
        rc: Subprocess return code.
        duration: Wall-clock seconds the subprocess ran for.
        failure_reason: Human-readable cause; surfaced in logs and Optuna user attrs.
        constraints: Constraint readings collected before the failure, if any.
        gate_results: Evidence gate results collected before the failure, if any.

    Returns:
        A :class:`TrialResult` with ``metric=None`` and ``feasible=False``.

    """
    return TrialResult(
        metric=None,
        constraints=constraints if constraints is not None else {},
        return_code=rc,
        duration_seconds=duration,
        feasible=False,
        failure_reason=failure_reason,
        gate_results=gate_results,
    )


def launch_trial(
    *,
    experiment: Experiment,
    phase_name: str,
    trial_id: int,
    trial_dir: Path,
    overrides: dict[str, Any],
    timeout_seconds: float | None,
    gpu_id: int | None = None,
) -> ExecutedTrial:
    """Launch the trial subprocess. Call this while holding the GPU lease.

    ``trial_dir`` is passed in (not recomputed) so the caller can persist its
    resolved absolute path as an Optuna user attribute *before* the subprocess
    starts. The stale-trial reaper then reads back that exact path on a later
    run, even if the user changed ``experiment.workdir`` or invoked phasesweep
    from a different cwd (review v0.5.3 / blocker 4).

    Args:
        experiment: Parsed experiment config; provides ``trial_command``,
            ``override_format``, and ``env``.
        phase_name: Name of the running phase (used in ``run_name`` and logs).
        trial_id: Optuna's numeric trial number.
        trial_dir: Resolved per-trial directory; created if missing.
        overrides: Composed overrides (inherited + fixed + sampled) for this trial.
        timeout_seconds: Wall-clock timeout passed to :func:`run_supervised`,
            or ``None`` for no timeout.
        gpu_id: GPU index from the pool, or ``None`` for inactive pool;
            written into ``CUDA_VISIBLE_DEVICES`` if not ``None``.

    Returns:
        :class:`ExecutedTrial` bundling the trial context, the supervised
        :class:`ProcessResult`.

    """
    workdir = trial_dir
    workdir.mkdir(parents=True, exist_ok=True)

    run_name = f"{experiment.experiment}-{phase_name}-{trial_id}"
    cmd = render_command(
        experiment.trial_command,
        overrides,
        experiment.override_format,
        trial_dir=workdir,
        trial_id=trial_id,
        phase=phase_name,
        run_name=run_name,
    )

    (workdir / "overrides_resolved.json").write_text(_json_dump_overrides(overrides))
    (workdir / "command.txt").write_text(cmd + "\n")

    env = os.environ.copy()
    env.update(experiment.env)
    env["PHASESWEEP_TRIAL_DIR"] = str(workdir)
    env["PHASESWEEP_TRIAL_ID"] = str(trial_id)
    env["PHASESWEEP_PHASE"] = phase_name
    env["PHASESWEEP_RUN_NAME"] = run_name

    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log.debug("[%s/trial_%d] GPU assigned: %d", phase_name, trial_id, gpu_id)

    log.info("[%s/trial_%d] %s", phase_name, trial_id, cmd)

    with (workdir / "stdout.log").open("w") as fout, (workdir / "stderr.log").open("w") as ferr:
        proc_result = run_supervised(
            cmd,
            env=env,
            stdout=fout,
            stderr=ferr,
            timeout=timeout_seconds,
            trial_dir=workdir,
        )

    ctx = TrialContext(
        experiment=experiment.experiment,
        phase=phase_name,
        trial_id=trial_id,
        trial_dir=workdir,
        run_name=run_name,
        return_code=proc_result.return_code,
        duration_seconds=proc_result.duration_seconds,
    )

    return ExecutedTrial(ctx=ctx, process=proc_result)


def extract_trial_result(
    *,
    experiment: Experiment,
    executed: ExecutedTrial,
    gates: list[Gate] | None = None,
    enforce_gates: bool = True,
) -> TrialResult:
    """Extract metrics from a completed trial. Call AFTER releasing the GPU lease.

    Failure modes that produce a `failure_reason` (and thus an Optuna FAIL via
    TrialExecutionError):
      * non-zero return code from the subprocess
      * metric extractor raised ExtractorError
      * metric extractor returned a non-finite value
      * any constraint extractor raised ExtractorError (review item #2)
      * any constraint extractor returned a non-finite value (review item #3)

    A trial that produced a finite metric and finite constraint values but violated
    a bound is COMPLETE+infeasible — that's a valid evaluation, not an instrumentation
    failure, and should still inform the sampler.

    Args:
        experiment: Parsed config; supplies the metric and constraint extractors.
        executed: Output of :func:`launch_trial`; provides the trial context
            and the :class:`ProcessResult`.
        gates: Evidence gates that must pass for the trial to count.
        enforce_gates: If ``True``, failed gates fail the trial. If ``False``,
            gates are advisory and are recorded without changing the metric
            result.

    Returns:
        :class:`TrialResult` with either a finite metric and feasibility flag,
        or ``metric=None`` plus a ``failure_reason``.

    """
    rc = executed.process.return_code
    duration = executed.process.duration_seconds
    failure_reason = executed.process.failure_reason

    if rc != 0 and failure_reason is None:
        failure_reason = f"non-zero exit code {rc}"

    if failure_reason is not None:
        return _failed_trial(rc=rc, duration=duration, failure_reason=failure_reason)

    try:
        metric_value = run_extractor(executed.ctx, experiment.metric.extractor)
    except ExtractorError as exc:
        log.warning(
            "[%s/trial_%d] metric extraction failed: %s",
            executed.ctx.phase,
            executed.ctx.trial_id,
            exc,
        )
        return _failed_trial(rc=rc, duration=duration, failure_reason=f"metric extractor: {exc}")

    if not math.isfinite(metric_value):
        log.warning(
            "[%s/trial_%d] metric extractor returned non-finite value: %r",
            executed.ctx.phase,
            executed.ctx.trial_id,
            metric_value,
        )
        return _failed_trial(
            rc=rc,
            duration=duration,
            failure_reason=f"metric extractor returned non-finite value: {metric_value!r}",
        )

    constraint_values: dict[str, float] = {}
    feasible = True
    for c in experiment.constraints:
        try:
            v = run_extractor(executed.ctx, c.extractor)
        except ExtractorError as exc:
            log.warning(
                "[%s/trial_%d] constraint %s extraction failed: %s",
                executed.ctx.phase,
                executed.ctx.trial_id,
                c.name,
                exc,
            )
            return _failed_trial(
                rc=rc,
                duration=duration,
                failure_reason=f"constraint extractor {c.name!r}: {exc}",
                constraints=constraint_values,
            )
        if not math.isfinite(v):
            log.warning(
                "[%s/trial_%d] constraint %s returned non-finite value: %r",
                executed.ctx.phase,
                executed.ctx.trial_id,
                c.name,
                v,
            )
            return _failed_trial(
                rc=rc,
                duration=duration,
                failure_reason=(
                    f"constraint extractor {c.name!r} returned non-finite value: {v!r}"
                ),
                constraints=constraint_values,
            )
        constraint_values[c.name] = v
        if not check_bounds(v, min_value=c.min, max_value=c.max):
            feasible = False

    gate_results = evaluate_gates(executed.ctx, gates or [])
    failed_gates = [gate for gate in gate_results if not gate.passed]
    if failed_gates and enforce_gates:
        detail = "; ".join(gate.detail for gate in failed_gates)
        log.warning(
            "[%s/trial_%d] evidence gate(s) failed: %s",
            executed.ctx.phase,
            executed.ctx.trial_id,
            detail,
        )
        return _failed_trial(
            rc=rc,
            duration=duration,
            failure_reason=f"evidence gates failed: {detail}",
            constraints=constraint_values,
            gate_results=gate_results,
        )

    return TrialResult(
        metric=metric_value,
        constraints=constraint_values,
        return_code=rc,
        duration_seconds=duration,
        feasible=feasible,
        failure_reason=None,
        gate_results=gate_results,
    )


def _json_dump_overrides(overrides: dict[str, Any]) -> str:
    """Serialize resolved overrides to indented JSON for ``overrides_resolved.json``.

    Args:
        overrides: The composed (inherited + fixed + sampled) overrides dict.

    Returns:
        Trailing-newline-terminated, sorted, two-space-indented JSON. Non-JSON
        scalars fall back through ``default=str`` (Path, etc.).

    """
    import json

    return json.dumps(overrides, indent=2, sort_keys=True, default=str) + "\n"
