"""Run a single trial as a supervised subprocess and extract its result.

Split into two phases:
  launch_trial  — needs GPU lease, runs subprocess
  extract_trial — no GPU needed, reads result files / polls W&B
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phasesweep.config import Experiment, Gate, check_bounds
from phasesweep.engine.errors import PhaseSweepError
from phasesweep.engine.state import TRAINER_INPUT_SCHEMA_VERSION
from phasesweep.evidence.evaluation import (
    DeadlineExceededError,
    ExtractorError,
    GateResult,
    TrialContext,
    evaluate_gates,
    run_extractor,
)
from phasesweep.evidence.models import JsonEnvelopeExtractor
from phasesweep.runtime.commands import (
    dump_json_file_overrides,
    dump_trial_trainer_config_yaml,
    render_command,
)
from phasesweep.runtime.files import atomic_write_text, private_atomic_write_text
from phasesweep.runtime.process import ProcessResult, run_supervised

log = logging.getLogger("phasesweep.engine.trial")


@dataclass
class ExecutedTrial:
    """Result of launching a trial subprocess. Does NOT yet contain metrics."""

    ctx: TrialContext
    process: ProcessResult
    trainer_input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedTrainerInput:
    """Exact generated trainer input, materialized before subprocess launch."""

    format: str
    filename: str
    content: bytes

    @property
    def sha256(self) -> str:
        """Return the identity injected into the trainer environment."""
        return hashlib.sha256(self.content).hexdigest()

    def record(self) -> dict[str, Any]:
        """Return the versioned Optuna user-attribute payload."""
        return {
            "schema_version": TRAINER_INPUT_SCHEMA_VERSION,
            "format": self.format,
            "filename": self.filename,
            "size_bytes": len(self.content),
            "sha256": self.sha256,
        }


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
    # Frozen evidence provenance captured when the objective was extracted:
    # extractor config fingerprint plus source digest / remote summary subset
    # (review v0.5.17 / finding F). None when metric extraction failed.
    objective_provenance: dict[str, Any] | None = None
    # True only when the phase/run deadline directly caused this result to
    # fail, rather than merely having elapsed by the time another failure was
    # observed.
    deadline_exhausted: bool = False


# Minimal environment base used when the execution contract narrows
# inheritance below "all": enough for a shell + interpreter to start and
# write temp files, nothing that can carry semantic configuration.
#
# CUDA_VISIBLE_DEVICES is deliberately NOT here. It decides whether a trial
# trains on CPU or GPU, so admitting the ambient value would let two operators
# running the identical config from differently-exported shells write
# CPU-trained and GPU-trained evaluations into one study under one semantic
# fingerprint — exactly the unfingerprinted ambient input the narrowed
# contracts exist to exclude (the fingerprint covers the contract's mode/names
# and configured ``env``, never ambient values). Bind trainer visibility with
# a top-level ``env.CUDA_VISIBLE_DEVICES`` entry, which is fingerprinted and
# also drives GPU pool discovery, or name the variable in an ``inherit_env``
# list to opt its ambient value in explicitly.
_BASE_INHERITED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "LOGNAME", "TZ")

# Warn-once keys for :func:`_warn_dropped_cuda_visibility`, so a narrowed
# contract reports the divergence once per phase instead of once per trial.
_DROPPED_CUDA_VISIBILITY_WARNED: set[tuple[str, str, str]] = set()


def _trainer_environment(experiment: Experiment) -> dict[str, str]:
    """Compose the trainer environment per the execution contract.

    ``inherit_env: all`` preserves the historical full-inheritance behavior.
    ``none`` starts from the minimal base; a list adds exactly the named
    ambient variables on top of that base. Configured ``experiment.env``
    values always apply last (review v0.5.17 / blocker 4).

    :param Experiment experiment: Parsed experiment supplying the contract.
    :return dict[str, str]: The composed trainer environment.
    """
    contract = experiment.execution.inherit_env
    if contract == "all":
        env = os.environ.copy()
    else:
        names = list(_BASE_INHERITED_ENV)
        if isinstance(contract, list):
            names.extend(contract)
        env = {name: os.environ[name] for name in names if name in os.environ}
    env.update(experiment.env)
    return env


def _inherit_env_contract(experiment: Experiment) -> str | list[str]:
    """Return the ``inherit_env`` contract in its canonical persisted form.

    A list contract is a *set* of names — order carries no meaning — so it is
    sorted, matching how :func:`phasesweep.engine.guards._execution_identity`
    canonicalises the same field for fingerprints.

    :param Experiment experiment: Parsed experiment supplying the contract.
    :return str | list[str]: ``"all"``, ``"none"``, or the sorted name list.
    """
    contract = experiment.execution.inherit_env
    return sorted(contract) if isinstance(contract, list) else contract


@dataclass(frozen=True)
class EnvironmentIdentity:
    """Identity of the exact environment a trial subprocess receives.

    ``values`` exists so one composition serves both the launch and the
    opt-in ``environment.json`` record; it holds secrets (``HF_TOKEN``,
    ``AWS_*``, ...) and must never be persisted or logged outside the
    owner-only record ``execution.record_env`` enables.
    """

    digest: str
    names: tuple[str, ...]
    values: dict[str, str]


def _environment_identity(experiment: Experiment) -> EnvironmentIdentity:
    """Fingerprint the environment :func:`_trainer_environment` composes.

    Two trials run under materially different environments (a rotated
    ``HF_TOKEN``, a different ``PYTHONHASHSEED``, another CUDA stack) are
    otherwise indistinguishable in the ledger, and a winner gives no clue
    which environment produced it (review v0.5.18 / finding F3).

    The digest is the SHA-256 of the compact JSON encoding of the
    ``[[name, value], ...]`` pairs sorted by name. JSON is used rather than a
    delimiter join because ``experiment.env`` keys are arbitrary config
    strings: encoding ``{"A": "B=1"}`` and ``{"A=B": "1"}`` as ``A=B=1``
    would collide, while JSON string quoting keeps the encoding injective.
    Sorting makes the digest independent of mapping order.

    This digest is deliberately NOT part of any semantic fingerprint: an
    environment change must not invalidate a resumable study or block a
    top-up. Divergence is reported by the selection and winner-load warnings
    instead.

    :param Experiment experiment: Parsed experiment supplying the contract.
    :return EnvironmentIdentity: Digest, sorted variable names, and the
        composed name-to-value mapping the digest covers.
    """
    env = _trainer_environment(experiment)
    items = sorted(env.items())
    encoded = json.dumps(items, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return EnvironmentIdentity(
        digest=hashlib.sha256(encoded).hexdigest(),
        names=tuple(name for name, _ in items),
        values=env,
    )


def _record_trainer_environment(
    experiment: Experiment,
    identity: EnvironmentIdentity,
    trial_dir: Path,
) -> None:
    """Write the opt-in raw environment record for one trial.

    A no-op unless ``execution.record_env`` is set. The record holds ambient
    *values*, so it is written owner-only (0600) through
    :func:`phasesweep.runtime.files.private_atomic_write_text` rather than the
    umask-governed artifact writer the rest of the trial directory uses. Only
    the file is hardened: the trial directory itself stays operator-trusted, so
    logs and evidence remain readable by ordinary tooling.

    The recorded mapping is exactly the digest's preimage — the contract-composed
    environment before the per-trial ``PHASESWEEP_*`` and GPU bindings, which
    vary by trial and are recoverable from the trial directory itself.

    :param Experiment experiment: Parsed experiment supplying the contract.
    :param EnvironmentIdentity identity: Identity of the composed environment.
    :param Path trial_dir: Existing trial directory to write into.
    :raises OSError: The record could not be written.
    :raises UnsafePrivatePathError: An existing ``environment.json`` is not a
        private, unshared regular file, so replacing it could leak values.
    """
    if not experiment.execution.record_env:
        return
    payload = {
        "digest": identity.digest,
        "inherit_env": _inherit_env_contract(experiment),
        "env": identity.values,
    }
    private_atomic_write_text(
        trial_dir / "environment.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        require_private_dir=False,
    )


def _warn_dropped_cuda_visibility(
    *,
    experiment: Experiment,
    phase_name: str,
    env: dict[str, str],
    gpu_id: int | str | None,
) -> None:
    """Warn once when the trainer's GPU visibility diverges from the parent's.

    The narrowed ``inherit_env`` contracts drop the ambient
    ``CUDA_VISIBLE_DEVICES`` on purpose (see :data:`_BASE_INHERITED_ENV`). The
    GPU pool still reads that ambient value for discovery: it leases and
    re-binds device tokens, and pins a disable sentinel (``""``/``-1``) into
    every trial. The remaining unbound case is ``gpu_policy: none``, where the
    phase deliberately assigns nothing: the child then has no visibility
    binding and can use host GPUs the parent had hidden, holding no GPU host
    lock. That is a real hazard, but it is a *configuration* hazard: the fix is
    a fingerprinted ``env`` entry, not a silent ambient read. Say so once
    rather than papering over it.

    Args:
        experiment: Parsed experiment supplying the inherit contract.
        phase_name: Phase whose trials are affected, used for the warn-once key.
        env: Composed trainer environment, after GPU assignment.
        gpu_id: Device token assigned by the pool, or ``None`` when the pool
            leased nothing for this trial.

    """
    if gpu_id is not None or "CUDA_VISIBLE_DEVICES" in env:
        return
    ambient = os.environ.get("CUDA_VISIBLE_DEVICES")
    if ambient is None:
        return
    key = (experiment.experiment, phase_name, ambient)
    if key in _DROPPED_CUDA_VISIBILITY_WARNED:
        return
    _DROPPED_CUDA_VISIBILITY_WARNED.add(key)
    log.warning(
        "[%s] ambient CUDA_VISIBLE_DEVICES=%r is outside the "
        "execution.inherit_env=%r contract and this phase leases no device, so trials "
        "will see every host GPU and hold no GPU host lock. Set a top-level "
        "env.CUDA_VISIBLE_DEVICES to bind trainer visibility — it is fingerprinted and "
        "also drives GPU pool discovery, so lock set and trainer visibility stay in step.",
        phase_name,
        ambient,
        experiment.execution.inherit_env,
    )


def _resolved_execution_cwd(experiment: Experiment) -> Path | None:
    """Resolve the configured trainer working directory, if any.

    :param Experiment experiment: Parsed experiment supplying ``execution.cwd``.
    :raises TrialExecutionError: The configured directory does not exist.
    :return Path | None: Resolved absolute directory, or ``None`` when the
        contract leaves the cwd unbound (invocation cwd, historical behavior).
    """
    configured = experiment.execution.cwd
    if configured is None:
        return None
    resolved = Path(configured).expanduser().resolve()
    if not resolved.is_dir():
        raise TrialExecutionError(
            f"execution.cwd {configured!r} resolves to {resolved}, which is not a "
            "directory on this host."
        )
    return resolved


class TrialExecutionError(RuntimeError):
    """Raised when a trial subprocess crashes or extraction fails.

    Caught by study.optimize(catch=...) so Optuna marks the trial FAIL,
    not COMPLETE with a sentinel value.
    """


class ProcessCleanupUncertainError(PhaseSweepError):
    """Base class for failures where a subprocess group may still be alive."""


class UnsafeProcessCleanupError(ProcessCleanupUncertainError):
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
    deadline_exhausted: bool = False,
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
        deadline_exhausted: Whether the phase/run deadline caused the failure.

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
        deadline_exhausted=deadline_exhausted,
    )


def prepare_trainer_input(
    *,
    experiment: Experiment,
    phase_name: str,
    trial_id: int,
    attempt_id: str,
    trial_dir: Path,
    overrides: dict[str, Any],
) -> PreparedTrainerInput:
    """Atomically write and identify the exact input one trainer will consume.

    One serialized byte string is used for the atomic write, environment
    digest, and durable trial record. CLI modes consume the resolved-overrides
    audit file; file modes consume their generated YAML or JSON file.

    :param Experiment experiment: Experiment supplying format and base config.
    :param str phase_name: Phase label used for runtime placeholders.
    :param int trial_id: Numeric trial identifier used for placeholders.
    :param str attempt_id: Attempt identity used in ``run_name`` placeholders.
    :param Path trial_dir: Per-trial artifact directory.
    :param dict[str, Any] overrides: Fully composed trial overrides.
    :return PreparedTrainerInput: Materialized input bytes and historical identity.
    """
    trial_dir.mkdir(parents=True, exist_ok=True)
    resolved_text = _json_dump_overrides(
        overrides,
        strict=experiment.override_format in {"json_file", "yaml_file"},
    )
    resolved_path = trial_dir / "overrides_resolved.json"
    atomic_write_text(resolved_path, resolved_text)

    run_name = f"{experiment.experiment}-{phase_name}-{trial_id}-{attempt_id}"
    if experiment.override_format == "yaml_file":
        filename = "trainer_config.yaml"
        text = dump_trial_trainer_config_yaml(
            experiment.trainer_config,
            overrides,
            substitutions={
                "{trial_dir}": str(trial_dir),
                "{trial_id}": str(trial_id),
                "{phase}": phase_name,
                "{run_name}": run_name,
            },
        )
    elif experiment.override_format == "json_file":
        filename = "overrides.json"
        text = dump_json_file_overrides(overrides)
    else:
        filename = "overrides_resolved.json"
        text = resolved_text

    if filename != resolved_path.name:
        atomic_write_text(trial_dir / filename, text)
    return PreparedTrainerInput(
        format=experiment.override_format,
        filename=filename,
        content=text.encode("utf-8"),
    )


def launch_trial(
    *,
    experiment: Experiment,
    phase_name: str,
    trial_id: int,
    generation_id: str,
    attempt_id: str,
    trial_dir: Path,
    overrides: dict[str, Any],
    timeout_seconds: float | None,
    gpu_id: int | str | None = None,
    gpu_lease_fds: tuple[int, ...] = (),
    prepared_input: PreparedTrainerInput | None = None,
) -> ExecutedTrial:
    """Launch the trial subprocess. Call this while holding the GPU lease.

    ``trial_dir`` is passed in (not recomputed) so the caller can persist its
    resolved absolute path as an Optuna user attribute *before* the subprocess
    starts. The stale-trial reaper then reads back that exact path on a later
    run, even if the user changed ``experiment.workdir`` or invoked phasesweep
    from a different cwd (review v0.5.3 / blocker 4).

    Args:
        experiment: Parsed experiment config; provides ``trial_command``,
            ``trainer_config``, ``override_format``, and ``env``.
        phase_name: Name of the running phase (used in ``run_name`` and logs).
        trial_id: Optuna's numeric trial number.
        generation_id: Identity of the current engine invocation.
        attempt_id: Immutable identity of this subprocess attempt.
        trial_dir: Resolved per-trial directory; created if missing.
        overrides: Composed overrides (inherited + fixed + sampled) for this trial.
        timeout_seconds: Total wall-clock budget passed to
            :func:`run_supervised`, or ``None`` for no timeout. The budget
            covers the whole supervised launch (supervisor startup, identity
            persistence, payload delivery) as well as trainer execution, so
            an already-expired phase/run deadline can never start new
            trainer work (review v0.5.16 / blocker 6).
        gpu_id: CUDA device token from the pool, or ``None`` for inactive pool;
            written into ``CUDA_VISIBLE_DEVICES`` if not ``None``.
        gpu_lease_fds: Host GPU-lock descriptors inherited by the trusted
            supervisor guardian (not trainer code) so exclusion lasts until
            the whole trainer process group exits, even if the orchestrator
            exits abruptly.
        prepared_input: Exact input already materialized and persisted by the
            Optuna objective. Direct callers may omit it; launch then prepares
            the input itself before starting the subprocess.

    Returns:
        :class:`ExecutedTrial` bundling the trial context, the supervised
        :class:`ProcessResult`.

    Raises:
        TrialExecutionError: The configured ``execution.cwd`` does not resolve
            to a directory on this host.
        OSError: A trial artifact (generated trainer config, resolved overrides,
            command record, the opt-in ``environment.json``, or the captured
            output streams) could not be written. Raised before the trainer
            starts, so no subprocess is left behind.
        UnsafePrivatePathError: ``execution.record_env`` is set and an existing
            ``environment.json`` in the trial directory is not a private,
            unshared regular file.

    """
    workdir = trial_dir
    workdir.mkdir(parents=True, exist_ok=True)
    # Every engine artifact under ``workdir``, including process-control
    # records written later, shares this operator-trusted boundary. These
    # files use ordinary paths rather than the validated O_NOFOLLOW helpers
    # reserved for the lock namespace and MCP state; forcing trial dirs
    # private would break normal operator/tool visibility into logs and
    # resolved overrides. See docs/runtime.md's trust-boundary note.
    run_name = f"{experiment.experiment}-{phase_name}-{trial_id}-{attempt_id}"
    trainer_input = prepared_input or prepare_trainer_input(
        experiment=experiment,
        phase_name=phase_name,
        trial_id=trial_id,
        attempt_id=attempt_id,
        trial_dir=workdir,
        overrides=overrides,
    )
    cmd = render_command(
        experiment.trial_command,
        overrides,
        experiment.override_format,
        trial_dir=workdir,
        trial_id=trial_id,
        phase=phase_name,
        run_name=run_name,
        trainer_config=experiment.trainer_config,
        materialized_input_path=workdir / trainer_input.filename,
    )
    overrides_sha256 = trainer_input.sha256
    # Orchestrator-created trial artifact, not symlink-hardened by design (see
    # the trust-boundary comment above).
    (workdir / "command.txt").write_text(cmd + "\n")

    # Environment and working directory follow the explicit execution
    # contract instead of unbounded ambient inheritance (review v0.5.17 /
    # blocker 4). The identity is computed from the same composition the
    # subprocess receives, and the opt-in raw record is written before any
    # per-trial binding is layered on, so the file stays the digest's exact
    # preimage (review v0.5.18 / finding F3).
    identity = _environment_identity(experiment)
    _record_trainer_environment(experiment, identity, workdir)
    env = dict(identity.values)
    trainer_cwd = _resolved_execution_cwd(experiment)
    env["PHASESWEEP_TRIAL_DIR"] = str(workdir)
    env["PHASESWEEP_TRIAL_ID"] = str(trial_id)
    env["PHASESWEEP_PHASE"] = phase_name
    env["PHASESWEEP_RUN_NAME"] = run_name
    env["PHASESWEEP_GENERATION_ID"] = generation_id
    env["PHASESWEEP_ATTEMPT_ID"] = attempt_id
    env["PHASESWEEP_OVERRIDES_SHA256"] = overrides_sha256
    env["WANDB_RUN_ID"] = attempt_id
    if isinstance(experiment.metric.extractor, JsonEnvelopeExtractor):
        env["PHASESWEEP_OBJECTIVE_PATH"] = str(workdir / experiment.metric.extractor.path)
    else:
        # This is a reserved, extractor-dependent value. Do not let an ambient
        # variable direct a log- or W&B-backed trial to an unrelated path.
        env.pop("PHASESWEEP_OBJECTIVE_PATH", None)

    if gpu_id is not None:
        visible_devices = str(gpu_id)
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
        if any(token.strip().isdigit() for token in visible_devices.split(",")):
            # Numeric tokens are locked against nvidia-smi's PCI-order index
            # map, so allowing FASTEST_FIRST here could run on a different
            # physical card than the one whose host lock we hold.
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        else:
            env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        log.debug("[%s/trial_%d] GPU assigned: %s", phase_name, trial_id, gpu_id)

    _warn_dropped_cuda_visibility(
        experiment=experiment, phase_name=phase_name, env=env, gpu_id=gpu_id
    )

    log.info("[%s/trial_%d] %s", phase_name, trial_id, cmd)

    # Orchestrator-created trial artifacts, not symlink-hardened by design
    # (see the trust-boundary comment above).
    with (workdir / "stdout.log").open("w") as fout, (workdir / "stderr.log").open("w") as ferr:
        proc_result = run_supervised(
            cmd,
            env=env,
            stdout=fout,
            stderr=ferr,
            timeout=timeout_seconds,
            trial_dir=workdir,
            attempt_id=attempt_id,
            cwd=None if trainer_cwd is None else str(trainer_cwd),
            gpu_lease_fds=gpu_lease_fds,
        )

    ctx = TrialContext(
        experiment=experiment.experiment,
        phase=phase_name,
        trial_id=trial_id,
        generation_id=generation_id,
        attempt_id=attempt_id,
        overrides_sha256=overrides_sha256,
        trial_dir=workdir,
        run_name=run_name,
        return_code=proc_result.return_code,
        duration_seconds=proc_result.duration_seconds,
    )

    return ExecutedTrial(
        ctx=ctx,
        process=proc_result,
        trainer_input=trainer_input.record(),
    )


def extract_trial_result(
    *,
    experiment: Experiment,
    executed: ExecutedTrial,
    gates: list[Gate] | None = None,
    enforce_gates: bool = True,
    deadline: float | None = None,
    trainer_timeout_is_deadline: bool = False,
) -> TrialResult:
    """Extract metrics from a completed trial. Call AFTER releasing the GPU lease.

    Failure modes that produce a `failure_reason` (and thus an Optuna FAIL via
    TrialExecutionError):
      * non-zero return code from the subprocess
      * metric extractor raised ExtractorError
      * metric extractor returned a non-finite value
      * any constraint extractor raised ExtractorError (review v0.5.2 / item 2)
      * any constraint extractor returned a non-finite value (review v0.5.2 / item 3)
      * the phase/run wallclock deadline expired before extraction completed
        (review v0.5.17 / blocker 8)

    A trial that produced a finite metric and finite constraint values but violated
    a bound is COMPLETE+infeasible — that's a valid evaluation, not an instrumentation
    failure, and should still inform the sampler. Expected extraction and gate
    failures are returned in :class:`TrialResult`; this function has no domain
    exception outcome for callers to catch.

    Args:
        experiment: Parsed config; supplies the metric and constraint extractors.
        executed: Output of :func:`launch_trial`; provides the trial context
            and the :class:`ProcessResult`.
        gates: Evidence gates that must pass for the trial to count.
        enforce_gates: If ``True``, failed gates fail the trial. If ``False``,
            gates are advisory and are recorded without changing the metric
            result.
        deadline: Optional absolute ``time.monotonic()`` phase/run deadline.
            ``timeout_seconds_per_phase`` / ``timeout_seconds_per_run`` bound
            the *whole* trial — extraction and gates included, not just the
            trainer. Enforcement is cooperative at stage boundaries (before
            the metric, each constraint, the gates, and the final result), so
            a single blocking local stage can overrun by at most its own
            duration; W&B polling additionally caps its request budget to the
            remainder.
        trainer_timeout_is_deadline: ``True`` when the caller capped the
            trainer's wallclock budget to the remaining phase/run deadline, so
            a trainer timeout is recorded as ``deadline_exhausted`` instead of
            an ordinary per-trial limit.

    Returns:
        :class:`TrialResult` with either a finite metric and feasibility flag,
        or ``metric=None`` plus a ``failure_reason``.

    """
    import time

    rc = executed.process.return_code
    duration = executed.process.duration_seconds
    failure_reason = executed.process.failure_reason

    def _deadline_failure(stage: str) -> TrialResult | None:
        """Fail the trial when the wallclock deadline expired before ``stage``.

        Args:
            stage: Human-readable name of the stage about to run, used in the
                failure reason.

        Returns:
            A deadline-failure :class:`TrialResult`, or ``None`` when no
            deadline is set or budget remains.

        """
        if deadline is None or time.monotonic() < deadline:
            return None
        return _failed_trial(
            rc=rc,
            duration=duration,
            failure_reason=(
                f"phase/run wallclock deadline exceeded before {stage}; the trainer "
                "finished but its evidence could not be evaluated within the budget"
            ),
            deadline_exhausted=True,
        )

    if rc != 0 and failure_reason is None:
        failure_reason = f"non-zero exit code {rc}"

    if failure_reason is not None:
        return _failed_trial(
            rc=rc,
            duration=duration,
            failure_reason=failure_reason,
            deadline_exhausted=trainer_timeout_is_deadline and executed.process.timed_out,
        )

    expired = _deadline_failure("metric extraction")
    if expired is not None:
        return expired

    objective_provenance: dict[str, Any] = {}
    try:
        metric_value = run_extractor(
            executed.ctx,
            experiment.metric.extractor,
            deadline=deadline,
            provenance=objective_provenance,
        )
    except DeadlineExceededError as exc:
        log.warning(
            "[%s/trial_%d] metric extraction exceeded deadline: %s",
            executed.ctx.phase,
            executed.ctx.trial_id,
            exc,
        )
        return _failed_trial(
            rc=rc,
            duration=duration,
            failure_reason=f"metric extractor: {exc}",
            deadline_exhausted=True,
        )
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
        expired = _deadline_failure(f"constraint extractor {c.name!r}")
        if expired is not None:
            expired.constraints.update(constraint_values)
            return expired
        try:
            v = run_extractor(executed.ctx, c.extractor, deadline=deadline)
        except DeadlineExceededError as exc:
            log.warning(
                "[%s/trial_%d] constraint %s extraction exceeded deadline: %s",
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
                deadline_exhausted=True,
            )
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

    gate_results = evaluate_gates(executed.ctx, gates or [], deadline=deadline)
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
            deadline_exhausted=any(gate.deadline_exhausted for gate in failed_gates),
        )

    # Final boundary check: a slow last stage must not let a trial publish a
    # "complete" evaluation past the configured wallclock bound (review
    # v0.5.17 / blocker 8 — the field names promise an end-to-end limit).
    expired = _deadline_failure("the trial result could be accepted")
    if expired is not None:
        expired.constraints.update(constraint_values)
        expired.gate_results = gate_results
        return expired

    return TrialResult(
        metric=metric_value,
        constraints=constraint_values,
        return_code=rc,
        duration_seconds=duration,
        feasible=feasible,
        failure_reason=None,
        gate_results=gate_results,
        objective_provenance=objective_provenance or None,
    )


def _json_dump_overrides(overrides: dict[str, Any], *, strict: bool) -> str:
    """Serialize resolved overrides to indented JSON for ``overrides_resolved.json``.

    Args:
        overrides: The composed (inherited + fixed + sampled) overrides dict.
        strict: When ``True`` (``yaml_file`` or ``json_file`` format), use the
            canonical strict serializer. This keeps the audit record in the
            portable value domain accepted by complete trainer YAML and makes
            it byte-faithful to the JSON compatibility wire. Load-time
            validation guarantees this succeeds. When ``False`` (a scalar/list
            CLI format), non-JSON scalars fall back through ``default=str``
            (Path, etc.) — there is no JSON wire artifact for those formats to
            diverge from.

    Returns:
        Trailing-newline-terminated, sorted, two-space-indented JSON.

    """
    from phasesweep.runtime.commands import dump_overrides_json

    if strict:
        return dump_overrides_json(overrides) + "\n"
    return json.dumps(overrides, indent=2, sort_keys=True, default=str) + "\n"
