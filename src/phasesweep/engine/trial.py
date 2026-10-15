"""Run a single trial as a supervised subprocess and extract its result.

Split into two phases:
  launch_trial  — needs GPU lease, runs subprocess
  extract_trial — no GPU needed, reads result files
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from phasesweep.config.common import check_bounds
from phasesweep.config.models import Experiment, Observation, _wandb_query
from phasesweep.engine.state import TRAINER_INPUT_SCHEMA_VERSION
from phasesweep.errors import (
    OperatorAction,
    PhaseSweepError,
)
from phasesweep.errors import (
    ProcessCleanupUncertainError as ProcessCleanupUncertainError,
)
from phasesweep.errors import (
    UnsafeProcessCleanupError as UnsafeProcessCleanupError,
)
from phasesweep.evidence.evaluation import (
    DeadlineExceededError,
    ExtractorError,
    GateResult,
    TrialContext,
    evaluate_gates,
    locate_objective_position,
    run_extractor,
)
from phasesweep.evidence.mlflow import compile_query as compile_mlflow_query
from phasesweep.evidence.models import (
    Gate,
    JsonEnvelopeExtractor,
    ObjectiveExtractor,
    compose_wandb_environment,
)
from phasesweep.runtime.commands import (
    TRAINER_INPUT_FILENAMES,
    dump_json_file_overrides,
    dump_trial_trainer_config_yaml,
    encode_overrides_json,
    merged_cli_overrides,
    render_command,
)
from phasesweep.runtime.files import atomic_write_text, private_atomic_write_text
from phasesweep.runtime.json import JsonObject
from phasesweep.runtime.process import ProcessResult, run_supervised

log = logging.getLogger("phasesweep.engine.trial")


@dataclass
class ExecutedTrial:
    """Result of launching a trial subprocess. Does NOT yet contain metrics."""

    ctx: TrialContext
    process: ProcessResult


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
    # None when metric extraction failed.
    objective_provenance: dict[str, Any] | None = None
    # True only when the phase/run deadline directly caused this result to
    # fail, rather than merely having elapsed by the time another failure was
    # observed.
    deadline_exhausted: bool = False
    # Report-only values read after acceptance, by observation name. A name
    # whose extraction failed or came out non-finite is absent.
    observed: dict[str, float] = field(default_factory=dict)


_TRIAL_BOUND_ENV = (
    "PHASESWEEP_TRIAL_DIR",
    "PHASESWEEP_TRIAL_ID",
    "PHASESWEEP_PHASE",
    "PHASESWEEP_RUN_NAME",
    "PHASESWEEP_GENERATION_ID",
    "PHASESWEEP_ATTEMPT_ID",
    "PHASESWEEP_OVERRIDES_SHA256",
    "PHASESWEEP_OBJECTIVE_PATH",
)

# Warn-once keys for :func:`_warn_dropped_cuda_visibility`, so a narrowed
# contract reports the divergence once per phase instead of once per trial.
_DROPPED_CUDA_VISIBILITY_WARNED: set[tuple[str, str, str]] = set()

# Warn-once keys for :func:`_warn_narrowed_environment_exit`, with the same
# once-per-phase-per-process scope.
_NARROWED_ENV_EXIT_WARNED: set[tuple[str, str]] = set()

# How far back from the end of ``stderr.log`` a failed trial's reason reads.
_STDERR_TAIL_BYTES = 4096

# Longest stderr quote in a failure reason, the ``...`` that marks a cut included.
_STDERR_QUOTE_CHARS = 200

# A terminal color or cursor sequence, which would otherwise leave its ``[31m`` residue.
_ANSI_SEQUENCE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# Last stderr lines that suggest the trainer lacked an environment input, so the
# narrowed-environment hint applies: a failed import (``PYTHONPATH`` dropped), a
# missing tool or file (``PATH`` dropped), an ``os.environ["NAME"]`` lookup, or
# words naming an unset variable. Any other line names its own cause and the hint
# stays out of its way.
_MISSING_ENVIRONMENT_STDERR = re.compile(
    r"ModuleNotFoundError|ImportError|No module named"
    r"|command not found|No such file or directory"
    r"|KeyError: ['\"][A-Z][A-Z0-9_]*['\"]"
    r"|(?i:environment variable|is not set)"
)


def _trainer_environment(
    experiment: Experiment,
    phase_name: str,
    *,
    require_wandb_online: bool = True,
    tracked_only: bool = False,
) -> dict[str, str]:
    """Compose the trainer environment per the execution contract.

    ``inherit_env: none`` (the default) starts from the minimal base; a list
    adds exactly the named semantic ambient variables on top of that base, and
    ``all`` copies the whole ambient environment. The pass-through names (the
    built-in ``TMPDIR`` plus ``passthrough_env``) are inherited under either
    narrowed contract. Configured
    ``experiment.env`` values apply last, except trial-bound keys that
    PhaseSweep assigns for each attempt.

    :param Experiment experiment: Parsed experiment supplying the contract.
    :param str phase_name: Phase whose remote consumers manage W&B identity.
    :param bool require_wandb_online: Reject offline/disabled W&B modes when the
        caller is preparing new remote work.
    :param bool tracked_only: Start from the ambient names the environment cohort covers even
        under ``inherit_env: all``, which gives the names :func:`_environment_identity` tracks.
    :raises PhaseSweepError: The W&B environment cannot be composed because explicit
        settings conflict or remote logging is disabled (``FIX_CONFIG``).
    :return dict[str, str]: The composed trainer environment.
    """
    admitted = experiment.execution.inherited_env_names()
    if admitted is None and not tracked_only:
        env = os.environ.copy()
    else:
        names = [
            *experiment.execution.tracked_env_names(),
            *experiment.execution.effective_passthrough_env(),
        ]
        env = {name: os.environ[name] for name in names if name in os.environ}
    env.update(experiment.env)
    # These values are assigned for each trial after the base environment is
    # recorded. Ambient copies cannot affect the trainer or its study cohort.
    for name in _TRIAL_BOUND_ENV:
        env.pop(name, None)
    phase = next(phase for phase in experiment.phases if phase.name == phase_name)
    query = _wandb_query(experiment, phase.gates)
    if query is not None:
        try:
            env = compose_wandb_environment(
                query,
                experiment.env,
                env,
                require_online=require_wandb_online,
            )
        except ValueError as exc:
            raise PhaseSweepError(str(exc), action=OperatorAction.FIX_CONFIG) from exc
    elif "WANDB_RUN_ID" not in experiment.env:
        # Without a remote evidence consumer, an explicitly configured run ID
        # remains trainer-owned. An ambient/inherited ID is not stable trial
        # semantics and must not split the persistent-study cohort.
        env.pop("WANDB_RUN_ID", None)
    if experiment.mlflow is not None:
        env.pop("MLFLOW_RUN_ID", None)
        env.pop("MLFLOW_EXPERIMENT_NAME", None)
        env["MLFLOW_TRACKING_URI"] = experiment.mlflow.tracking_uri
        if experiment.mlflow.experiment_id is not None:
            env["MLFLOW_EXPERIMENT_ID"] = experiment.mlflow.experiment_id
        else:
            env.pop("MLFLOW_EXPERIMENT_ID", None)
            env["MLFLOW_EXPERIMENT_NAME"] = str(experiment.mlflow.experiment_name)
    return env


def _preflight_trainer_environments(
    experiment: Experiment, *, from_phase: str | None = None
) -> None:
    """Validate environments for phases that a fresh artifact tree would launch.

    Existing current-format trees defer this check until recovered study state
    proves that a phase has new work. A fresh tree has no trials to replay, so
    every reached phase can be checked before any execution artifact is created.

    :param Experiment experiment: Parsed experiment supplying phase contracts.
    :param str | None from_phase: Optional first phase to execute.
    :raises PhaseSweepError: A composed runtime environment disables required
        remote evidence or conflicts with its managed identity.
    """
    for _index, phase in experiment.phases_from(from_phase):
        _trainer_environment(experiment, phase.name)


def _inherit_env_contract(experiment: Experiment) -> str | list[str]:
    """Return the ``inherit_env`` contract in its canonical persisted form.

    A list contract is a *set* of names — order carries no meaning — so it is
    sorted, matching how :func:`phasesweep.engine.fingerprints._execution_identity`
    canonicalises the same field for fingerprints.

    :param Experiment experiment: Parsed experiment supplying the contract.
    :return str | list[str]: ``"all"``, ``"none"``, or the sorted name list.
    """
    contract = experiment.execution.inherit_env
    return sorted(contract) if isinstance(contract, list) else contract


@dataclass(frozen=True)
class EnvironmentIdentity:
    """Semantic identity of the base environment a trainer receives.

    ``tracked`` is the name-to-value mapping the environment cohort covers: the base names,
    the ``inherit_env`` list names, the ``env`` entries and the keys PhaseSweep composes from
    config. It leaves out the pass-through names (the built-in ``TMPDIR`` and
    ``passthrough_env``) so credentials can rotate without contaminating a persistent study.
    Each trial records it in plain text, so the refusal for a different cohort can show both
    values. ``digest`` is the SHA-256 of that mapping, the compact key the cohort check compares.

    ``values`` is the full composed mapping used for launch and the opt-in ``environment.json``
    record. It also holds the pass-through values and, under ``inherit_env: all``, every
    ambient variable outside the cohort, so it is recorded nowhere but that owner-only file.
    """

    digest: str
    tracked: dict[str, str]
    values: dict[str, str]


def _environment_identity(
    experiment: Experiment,
    phase_name: str,
    *,
    require_wandb_online: bool = True,
) -> EnvironmentIdentity:
    """Identify the semantic part of the composed trainer environment.

    The cohort covers the names the narrowed contract composes: the base names, the
    ``inherit_env`` list names, the ``experiment.env`` entries and the W&B and MLflow keys
    PhaseSweep sets from config. Under ``inherit_env: all`` the trainer still inherits every
    ambient variable, but one outside that set is inherited untracked, so shell state such as
    ``SHLVL`` or ``OLDPWD`` cannot split a persistent study.

    Values named by the effective pass-through set (the built-in ``TMPDIR`` plus
    ``execution.passthrough_env``) are also excluded: they are scratch, credential or
    transport inputs allowed to change between invocations. Configured ``experiment.env``
    entries remain semantic even if they reuse a pass-through name, except trial bindings
    assigned by PhaseSweep, which are removed first.

    The digest is the SHA-256 of the compact JSON encoding of the tracked
    ``[[name, value], ...]`` pairs sorted by name. JSON is used rather than a
    delimiter join because ``experiment.env`` keys are arbitrary config
    strings: encoding ``{"A": "B=1"}`` and ``{"A=B": "1"}`` as ``A=B=1``
    would collide, while JSON string quoting keeps the encoding injective.
    Sorting makes the digest independent of mapping order. It is a checksum of the tracked
    mapping, not a secrecy measure: each trial records both.

    The digest is stored on every allocated trial and checked against every
    populated persistent study before another trial is allocated.

    :param Experiment experiment: Parsed experiment supplying the contract.
    :param str phase_name: Phase whose remote consumers normalize W&B defaults.
    :param bool require_wandb_online: Reject offline/disabled W&B modes when the
        caller is preparing new remote work.
    :return EnvironmentIdentity: The digest, the tracked name-to-value mapping it covers, and
        the full composed mapping the trainer receives.
    """
    env = _trainer_environment(
        experiment,
        phase_name,
        require_wandb_online=require_wandb_online,
    )
    cohort_names = set(
        _trainer_environment(
            experiment,
            phase_name,
            require_wandb_online=require_wandb_online,
            tracked_only=True,
        )
    )
    passthrough = frozenset(experiment.execution.effective_passthrough_env()) - set(experiment.env)
    tracked = {
        name: value
        for name, value in sorted(env.items())
        if name in cohort_names and name not in passthrough
    }
    encoded = json.dumps(list(tracked.items()), separators=(",", ":"), ensure_ascii=False)
    return EnvironmentIdentity(
        digest=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        tracked=tracked,
        values=env,
    )


def _record_trainer_environment(
    experiment: Experiment,
    identity: EnvironmentIdentity,
    trial_dir: Path,
) -> None:
    """Write the opt-in raw environment record for one trial.

    A no-op unless ``execution.record_env`` is set. The record holds the whole
    composed environment, pass-through values and every ambient variable under
    ``inherit_env: all`` included, so it is written owner-only (0600) through
    :func:`phasesweep.runtime.files.private_atomic_write_text` rather than the
    umask-governed artifact writer the rest of the trial directory uses. Only
    the file is hardened: the trial directory itself stays operator-trusted, so
    logs and evidence remain readable by ordinary tooling.

    The recorded mapping is the full contract-composed environment before the
    per-trial ``PHASESWEEP_*`` and GPU bindings. ``passthrough_env`` lists the
    effective pass-through names, built-in ``TMPDIR`` included, whose values are
    left out of the tracked mapping and its digest.

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
        "passthrough_env": list(experiment.execution.effective_passthrough_env()),
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
    ``CUDA_VISIBLE_DEVICES`` on purpose (see :data:`~phasesweep.config.models.BASE_INHERITED_ENV`). The
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


def _printable_line(raw: str) -> str:
    """Reduce one line of trainer output to plain single-line text.

    :param str raw: One line, without its terminator.
    :return str: The line without terminal escape sequences, control characters or
        decoding replacements, with every whitespace run collapsed to one space.
    """
    text = _ANSI_SEQUENCE.sub("", raw)
    kept = (
        " " if ch.isspace() else ch
        for ch in text
        if ch.isspace() or (ch.isprintable() and ch != "\N{REPLACEMENT CHARACTER}")
    )
    return " ".join("".join(kept).split())


def _stderr_last_line(trial_dir: Path) -> str | None:
    """Return the last non-empty line of a trial's ``stderr.log``, ready to quote.

    Reads at most :data:`_STDERR_TAIL_BYTES` from the end of the file in the
    operator-trusted trial directory. A line cut by that window is quoted from
    where the window starts. A line longer than :data:`_STDERR_QUOTE_CHARS` is cut
    and ends in ``...``. A line with nothing printable left, such as binary
    output, does not count, so the line before it is quoted.

    :param Path trial_dir: Trial directory holding ``stderr.log``.
    :return str | None: The line, or ``None`` when the file is missing, unreadable
        or has no printable line.
    """
    try:
        with (trial_dir / "stderr.log").open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - _STDERR_TAIL_BYTES))
            tail = handle.read(_STDERR_TAIL_BYTES)
    except OSError:
        return None
    for raw in reversed(tail.decode("utf-8", errors="replace").splitlines()):
        line = _printable_line(raw)
        if not line:
            continue
        if len(line) > _STDERR_QUOTE_CHARS:
            return line[: _STDERR_QUOTE_CHARS - 3].rstrip() + "..."
        return line
    return None


def _warn_narrowed_environment_exit(
    *,
    experiment: Experiment,
    phase_name: str,
    trial_id: int,
    return_code: int,
    stderr_line: str | None,
) -> None:
    """Warn once per phase when a trainer exits nonzero under a narrowed environment.

    A trainer that was started without a variable it needs, such as ``PYTHONPATH``
    or a cache path, can fail with an error that never mentions the environment.
    Name the contract and the three ways to widen it, once per phase for the
    process, so the first such failure carries the hint without repeating it for
    every later trial. A failure whose stderr line names another cause, such as an
    out-of-memory error, gets no hint and does not use up the phase's one warning.
    Without a stderr line there is nothing to rule the environment out, so the hint
    applies.

    :param Experiment experiment: Parsed experiment supplying the contract.
    :param str phase_name: Phase whose trial failed, used for the warn-once key.
    :param int trial_id: Phase-local trial number of the failed trial.
    :param int return_code: The nonzero exit code the trainer returned.
    :param str | None stderr_line: The trial's last stderr line from
        :func:`_stderr_last_line`, or ``None`` when it has none.
    """
    admitted = experiment.execution.inherited_env_names()
    if admitted is None:
        return
    if stderr_line is not None and _MISSING_ENVIRONMENT_STDERR.search(stderr_line) is None:
        return
    key = (experiment.experiment, phase_name)
    if key in _NARROWED_ENV_EXIT_WARNED:
        return
    _NARROWED_ENV_EXIT_WARNED.add(key)
    log.warning(
        "[%s/trial_%d] the trainer exited with code %d under execution.inherit_env=%r, "
        "so it received only %s plus env and passthrough_env values. If it needs another "
        "ambient variable (PYTHONPATH, a virtualenv or conda setting, a cache path, a "
        "credential), name it in execution.inherit_env: [NAME], set it under env:, or set "
        "execution.inherit_env: all.%s The trainer's own error is in stderr.log in the "
        "trial directory. This hint is shown once per phase.",
        phase_name,
        trial_id,
        return_code,
        experiment.execution.inherit_env,
        ", ".join(admitted),
        "" if stderr_line is None else f' Its last stderr line was "{stderr_line}".',
    )


def _resolved_execution_cwd(experiment: Experiment) -> Path | None:
    """Resolve the configured trainer working directory, if any.

    :param Experiment experiment: Parsed experiment supplying ``execution.cwd``.
    :raises TrialExecutionError: The configured directory does not exist.
    :return Path | None: Resolved absolute directory, or ``None`` when the
        trainer inherits the invocation working directory.
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


def _failed_trial(
    *,
    rc: int,
    duration: float,
    failure_reason: str,
    constraints: dict[str, float] | None = None,
    gate_results: list[GateResult] | None = None,
    deadline_exhausted: bool = False,
    objective_provenance: dict[str, Any] | None = None,
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
        objective_provenance: Frozen evidence already captured before a later gate failed.

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
        objective_provenance=objective_provenance,
    )


def trial_run_name(experiment: Experiment, phase_name: str, trial_id: int, attempt_id: str) -> str:
    """Name one attempt the same way for its trainer and its tracking run.

    :param Experiment experiment: Experiment whose name leads the run name.
    :param str phase_name: Running phase.
    :param int trial_id: Optuna trial number.
    :param str attempt_id: Attempt identity.
    :return str: The ``{run_name}`` placeholder and ``PHASESWEEP_RUN_NAME`` value.
    """
    return f"{experiment.experiment}-{phase_name}-{trial_id}-{attempt_id}"


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
    digest, and durable trial record. Argparse and hydra consume the flat
    ``overrides_resolved.json``, which holds the base trainer config merged
    with the overrides; YAML and JSON modes consume their generated file and
    leave ``overrides_resolved.json`` as the overrides-only audit record.

    :param Experiment experiment: Experiment supplying format and base config.
    :param str phase_name: Phase label used for runtime placeholders.
    :param int trial_id: Numeric trial identifier used for placeholders.
    :param str attempt_id: Attempt identity used in ``run_name`` placeholders.
    :param Path trial_dir: Per-trial artifact directory.
    :param dict[str, Any] overrides: Fully composed trial overrides.
    :return PreparedTrainerInput: Materialized input bytes and historical identity.
    """
    trial_dir.mkdir(parents=True, exist_ok=True)
    run_name = trial_run_name(experiment, phase_name, trial_id, attempt_id)
    substitutions = {
        "{trial_dir}": str(trial_dir),
        "{trial_id}": str(trial_id),
        "{phase}": phase_name,
        "{run_name}": run_name,
    }
    filename = TRAINER_INPUT_FILENAMES[experiment.override_format]
    if experiment.override_format == "yaml_file":
        text = dump_trial_trainer_config_yaml(
            experiment.trainer_config, overrides, substitutions=substitutions
        )
    elif experiment.override_format == "json_file":
        text = dump_json_file_overrides(
            overrides, trainer_config=experiment.trainer_config, substitutions=substitutions
        )
    else:
        text = (
            encode_overrides_json(
                merged_cli_overrides(
                    experiment.trainer_config, overrides, substitutions=substitutions
                )
            )
            + "\n"
        )

    resolved_text = (
        encode_overrides_json(overrides) + "\n"
        if experiment.override_format in ("yaml_file", "json_file")
        else text
    )
    resolved_path = trial_dir / "overrides_resolved.json"
    atomic_write_text(resolved_path, resolved_text)
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
    wallclock_deadline: float | None = None,
    gpu_id: int | str | None = None,
    gpu_lease_fds: tuple[int, ...] = (),
    prepared_input: PreparedTrainerInput,
) -> ExecutedTrial:
    """Launch the trial subprocess. Call this while holding the GPU lease.

    ``trial_dir`` is passed in (not recomputed) so the caller can persist its
    resolved absolute path as an Optuna user attribute *before* the subprocess
    starts. The stale-trial reaper then reads back that exact path on a later
    run, even if the user changed ``experiment.workdir`` or invoked phasesweep
    from a different cwd.

    Args:
        experiment: Parsed experiment config; provides ``trial_command``,
            ``trainer_config``, ``override_format``, and ``env``.
        phase_name: Name of the running phase (used in ``run_name`` and logs).
        trial_id: Optuna's numeric trial number.
        generation_id: Identity of the current engine invocation.
        attempt_id: Immutable identity of this subprocess attempt.
        trial_dir: Resolved per-trial directory; created if missing.
        overrides: Composed overrides (inherited + fixed + sampled) for this trial.
        timeout_seconds: Per-trial wall-clock budget passed to
            :func:`run_supervised`, or ``None`` for no per-trial timeout.
        wallclock_deadline: Optional absolute phase/run deadline. Passing the
            original deadline preserves time consumed by command rendering,
            artifact writes, environment recording, cwd resolution, and log
            opening before supervision begins.
        gpu_id: CUDA device token from the pool, or ``None`` for inactive pool;
            written into ``CUDA_VISIBLE_DEVICES`` if not ``None``.
        gpu_lease_fds: Host GPU-lock descriptors inherited by the trusted
            supervisor guardian (not trainer code) so exclusion lasts until
            the whole trainer process group exits, even if the orchestrator
            exits abruptly.
        prepared_input: Exact input materialized and persisted before the GPU lease.

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
    # reserved for the lock namespace, the attempt registry, and MCP state;
    # forcing trial dirs private would break normal operator/tool visibility
    # into logs and resolved overrides. See docs/runtime.md's trust-boundary
    # section.
    run_name = trial_run_name(experiment, phase_name, trial_id, attempt_id)
    trainer_input = prepared_input
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
    (workdir / "command.txt").write_text(cmd + "\n", encoding="utf-8")

    # Environment and working directory follow the explicit execution
    # contract instead of unbounded ambient inheritance. The identity is
    # computed from the same composition the subprocess receives, and the
    # opt-in raw record is written before any per-trial binding is layered
    # on, so the file stays the digest's exact preimage.
    identity = _environment_identity(experiment, phase_name)
    _record_trainer_environment(experiment, identity, workdir)
    env = dict(identity.values)
    phase = next(
        (phase for phase in experiment.phases if phase.name == phase_name), experiment.phases[0]
    )
    wandb_query = _wandb_query(experiment, phase.gates)
    if wandb_query is not None:
        env["WANDB_RUN_ID"] = attempt_id
    # Without a W&B evidence consumer, monitoring remains trainer-owned: keep
    # any configured/ambient W&B identity unchanged. Trainers can still bind
    # monitoring to this attempt explicitly through PHASESWEEP_ATTEMPT_ID.
    if experiment.mlflow is not None:
        from phasesweep.runtime.json import json_text
        from phasesweep.tracking.session import read_binding, read_record

        expected = read_record(workdir / "mlflow-expected.json")
        env.pop("MLFLOW_EXPERIMENT_NAME", None)
        env["MLFLOW_EXPERIMENT_ID"] = json_text(expected, "experiment_id")
        if experiment.mlflow.run_mode == "managed":
            env["MLFLOW_RUN_ID"] = read_binding(workdir)["run_id"]
    trainer_cwd = _resolved_execution_cwd(experiment)
    env["PHASESWEEP_TRIAL_DIR"] = str(workdir)
    env["PHASESWEEP_TRIAL_ID"] = str(trial_id)
    env["PHASESWEEP_PHASE"] = phase_name
    env["PHASESWEEP_RUN_NAME"] = run_name
    env["PHASESWEEP_GENERATION_ID"] = generation_id
    env["PHASESWEEP_ATTEMPT_ID"] = attempt_id
    env["PHASESWEEP_OVERRIDES_SHA256"] = overrides_sha256
    if isinstance(experiment.metric.extractor, JsonEnvelopeExtractor):
        env["PHASESWEEP_OBJECTIVE_PATH"] = str(workdir / experiment.metric.extractor.path)
    else:
        # This is a reserved, extractor-dependent value. Do not let an ambient
        # variable direct a log-backed trial to an unrelated path.
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
            wallclock_deadline=wallclock_deadline,
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
        wandb_environment=env if wandb_query is not None else None,
        wandb_query=wandb_query,
        mlflow_settings=None if experiment.mlflow is None else experiment.mlflow.target(),
        mlflow_query=compile_mlflow_query(experiment, phase.gates),
        mlflow_environment=env if experiment.mlflow is not None else None,
    )

    return ExecutedTrial(ctx=ctx, process=proc_result)


def extract_trial_result(
    *,
    experiment: Experiment,
    executed: ExecutedTrial,
    gates: list[Gate] | None = None,
    deadline: float | None = None,
) -> TrialResult:
    """Extract metrics from a completed trial. Call AFTER releasing the GPU lease.

    Failure modes that produce a `failure_reason` (and thus an Optuna FAIL via
    TrialExecutionError):
      * non-zero return code from the subprocess; the reason quotes the last line
        of the trial's ``stderr.log`` when it has one
      * metric extractor raised ExtractorError
      * metric extractor returned a non-finite value
      * any constraint extractor raised ExtractorError
      * any constraint extractor returned a non-finite value
      * the phase/run wallclock deadline expired before extraction completed

    A trial that produced a finite metric and finite constraint values but violated
    a bound (a constraint's, or the metric's own ``min``/``max``) is COMPLETE+infeasible
    — that's a valid evaluation, not an instrumentation failure, and should still
    inform the sampler. Expected extraction and gate
    failures are returned in :class:`TrialResult`; this function has no domain
    exception outcome for callers to catch. Report-only observations are read last,
    after acceptance; one that cannot be read warns and is left out of
    ``TrialResult.observed`` without affecting the result.

    Args:
        experiment: Parsed config; supplies the metric and constraint extractors.
        executed: Output of :func:`launch_trial`; provides the trial context
            and the :class:`ProcessResult`.
        gates: Evidence gates that must pass for the trial to count.
        deadline: Optional absolute ``time.monotonic()`` phase/run deadline.
            ``timeout_seconds_per_phase`` / ``timeout_seconds_per_run`` bound
            the *whole* trial — extraction and gates included, not just the
            trainer. Enforcement is cooperative at stage boundaries (before
            the metric, each constraint, the gates, and the final result), so
            a single blocking local stage can overrun by at most its own
            duration; W&B polling additionally caps its request budget to the
            remainder. An accepted W&B objective's evaluation position is read
            last, within what remains, and never changes the result.

    Returns:
        :class:`TrialResult` with either a finite metric and feasibility flag,
        or ``metric=None`` plus a ``failure_reason``.

    """
    import time

    executed = replace(
        executed, ctx=replace(executed.ctx, wandb_query=_wandb_query(experiment, gates or []))
    )
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
        # The one place a nonzero exit's cause is composed, so trials.csv, the
        # TRIAL line and the outcome record all carry the same stderr quote.
        stderr_line = _stderr_last_line(executed.ctx.trial_dir)
        failure_reason = f"non-zero exit code {rc}"
        if stderr_line is not None:
            failure_reason += f"; stderr: {stderr_line}"
        _warn_narrowed_environment_exit(
            experiment=experiment,
            phase_name=executed.ctx.phase,
            trial_id=executed.ctx.trial_id,
            return_code=rc,
            stderr_line=stderr_line,
        )

    if failure_reason is not None:
        return _failed_trial(
            rc=rc,
            duration=duration,
            failure_reason=failure_reason,
            deadline_exhausted=(
                executed.process.timeout_capped_by_wallclock and executed.process.timed_out
            ),
        )

    expired = _deadline_failure("metric extraction")
    if expired is not None:
        return expired

    objective_provenance: JsonObject = {}
    constraint_values: dict[str, float] = {}

    def extract_required(
        extractor: ObjectiveExtractor,
        *,
        label: str,
        log_label: str,
        provenance: JsonObject | None = None,
    ) -> float | TrialResult:
        """Read one required scalar, retaining prior constraints on failure.

        :param ObjectiveExtractor extractor: Configured scalar extractor.
        :param str label: Extractor label used in the persisted failure reason.
        :param str log_label: Metric or constraint label used in warning messages.
        :param JsonObject | None provenance: Objective-only frozen evidence sink.
        :return float | TrialResult: Finite scalar or the failed trial result.
        """
        try:
            value = run_extractor(executed.ctx, extractor, deadline=deadline, provenance=provenance)
        except ExtractorError as exc:
            exhausted = isinstance(exc, DeadlineExceededError)
            log.warning(
                "[%s/trial_%d] %s extraction %s: %s",
                executed.ctx.phase,
                executed.ctx.trial_id,
                log_label,
                "exceeded deadline" if exhausted else "failed",
                exc,
            )
            return _failed_trial(
                rc=rc,
                duration=duration,
                failure_reason=f"{label}: {exc}",
                constraints=constraint_values,
                deadline_exhausted=exhausted,
            )
        if not math.isfinite(value):
            log.warning(
                "[%s/trial_%d] %s returned non-finite value: %r",
                executed.ctx.phase,
                executed.ctx.trial_id,
                label if provenance is not None else log_label,
                value,
            )
            return _failed_trial(
                rc=rc,
                duration=duration,
                failure_reason=f"{label} returned non-finite value: {value!r}",
                constraints=constraint_values,
            )
        return value

    metric_value = extract_required(
        experiment.metric.extractor,
        label="metric extractor",
        log_label="metric",
        provenance=objective_provenance,
    )
    if isinstance(metric_value, TrialResult):
        return metric_value
    feasible = check_bounds(
        metric_value, min_value=experiment.metric.min, max_value=experiment.metric.max
    )
    for c in experiment.constraints:
        expired = _deadline_failure(f"constraint extractor {c.name!r}")
        if expired is not None:
            expired.constraints.update(constraint_values)
            return expired
        v = extract_required(
            c.extractor, label=f"constraint extractor {c.name!r}", log_label=f"constraint {c.name}"
        )
        if isinstance(v, TrialResult):
            return v
        constraint_values[c.name] = v
        if not check_bounds(v, min_value=c.min, max_value=c.max):
            feasible = False

    gate_results = evaluate_gates(executed.ctx, gates or [], deadline=deadline)
    if executed.ctx.mlflow_capture:
        objective_provenance["remote_capture"] = dict(executed.ctx.mlflow_capture)
    if executed.ctx.wandb_capture:
        objective_provenance["remote_capture"] = dict(executed.ctx.wandb_capture)
    failed_gates = [gate for gate in gate_results if not gate.passed]
    if failed_gates:
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
            objective_provenance=objective_provenance,
            constraints=constraint_values,
            gate_results=gate_results,
            deadline_exhausted=any(gate.deadline_exhausted for gate in failed_gates),
        )

    # Final boundary check: a slow last stage must not let a trial publish a
    # "complete" evaluation past the configured wallclock bound — the field
    # names promise an end-to-end limit.
    expired = _deadline_failure("the trial result could be accepted")
    if expired is not None:
        expired.constraints.update(constraint_values)
        expired.gate_results = gate_results
        return expired

    # Accepted. Where the objective was logged is advisory, so it is read
    # only now and can never spend the budget acceptance needed.
    locate_objective_position(
        executed.ctx, experiment.metric.extractor, objective_provenance, deadline=deadline
    )
    observed = _observe(executed.ctx, experiment.observe, deadline=deadline)
    return TrialResult(
        metric=metric_value,
        constraints=constraint_values,
        return_code=rc,
        duration_seconds=duration,
        feasible=feasible,
        failure_reason=None,
        gate_results=gate_results,
        objective_provenance=objective_provenance or None,
        observed=observed,
    )


def _observe(
    ctx: TrialContext, observations: list[Observation], *, deadline: float | None
) -> dict[str, float]:
    """Read the report-only observations of an accepted trial.

    An observation is advisory, so a failed or non-finite read warns and records
    nothing for that name. It never changes the trial's result and, running after
    acceptance, can never spend the budget acceptance needed.

    :param TrialContext ctx: Trial context of the accepted attempt.
    :param list[Observation] observations: Configured report-only observations.
    :param float | None deadline: Optional absolute ``time.monotonic()`` deadline.
    :return dict[str, float]: Finite value of each observation that was read.
    """
    observed: dict[str, float] = {}
    for observation in observations:
        try:
            value = run_extractor(ctx, observation.extractor, deadline=deadline)
        except ExtractorError as exc:
            log.warning(
                "[%s/trial_%d] observation %s not recorded: %s",
                ctx.phase,
                ctx.trial_id,
                observation.name,
                exc,
            )
            continue
        if not math.isfinite(value):
            log.warning(
                "[%s/trial_%d] observation %s not recorded: extractor returned non-finite value %r",
                ctx.phase,
                ctx.trial_id,
                observation.name,
                value,
            )
            continue
        observed[observation.name] = value
    return observed
