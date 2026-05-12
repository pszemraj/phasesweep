"""Top-level orchestrator: sequence phases, drive Optuna, persist winners.

Responsibilities:
  - GPU pool creation/teardown per phase.
  - Signal handler installation so orchestrator death kills child processes.
  - Stale trial reaping on study load (crash recovery).
  - Phase fingerprinting to prevent incompatible study reuse (#10).
  - GPU lease held only during subprocess, released before extraction (#2).
  - Failed trials marked FAIL in Optuna, not COMPLETE with sentinel (#4).
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna
import yaml

from phasesweep import __version__
from phasesweep._runtime import lock_dir as _lock_dir
from phasesweep.config import (
    CategoricalParam,
    Config,
    Experiment,
    FloatParam,
    Gate,
    IntParam,
    Phase,
    Sampler,
    SearchParam,
    Suite,
)
from phasesweep.gpu_pool import GpuPool
from phasesweep.runner import (
    TrialExecutionError,
    UnsafeProcessCleanupError,
    extract_trial_result,
    launch_trial,
)
from phasesweep.selector import NoFeasibleTrialError, select_winner
from phasesweep.storage_urls import canonical_storage_identity

log = logging.getLogger("phasesweep.orchestrator")


# --------------------------------------------------------------------------------------
# Winner dataclass (#9: includes effective_overrides)
# --------------------------------------------------------------------------------------


@dataclass
class Winner:
    """Phase winner: sampled params, full effective overrides, and metric value.

    ``phase_fingerprint`` is the SHA-256 of the phase's semantic execution
    context at the time the winner was selected (review v0.5.6 / blocker 3).
    Persisted into ``winner.yaml`` and re-checked when ``--from-phase`` skips
    earlier phases — without that check, editing a parent phase's search
    space, fixed overrides, env, metric, or trial command and then resuming
    would silently inherit the *old* winner against the *new* parent config.

    ``None`` only on placeholder winners produced for the dry-run skip path,
    which never get persisted.
    """

    trial_number: int
    params: dict[str, Any]  # sampled params only
    effective_overrides: dict[str, Any]  # full composed overrides (fixed + inherited + sampled)
    metric: float
    constraints: dict[str, float] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    phase_fingerprint: str | None = None


# --------------------------------------------------------------------------------------
# Stale-trial reaper (PID-reuse safe via starttime)
# --------------------------------------------------------------------------------------


def _reap_stale_trials(study: optuna.Study, experiment: Experiment, phase_name: str) -> int:
    """Mark RUNNING trials as FAIL on startup, killing orphaned process groups.

    Uses the per-trial identity files left by ``run_supervised`` (review v0.5.2 /
    blocker 7): pid + starttime for the safe path, pgid as the fallback when the
    root PID has exited but descendants are still alive.

    The trial directory is loaded from the trial's ``phasesweep_trial_dir``
    user attribute when present (review v0.5.3 / blocker 4) so the reaper
    works correctly even if the user changed ``experiment.workdir`` or invoked
    phasesweep from a different cwd. Trials created by older versions that
    didn't persist this attribute fall back to the recomputed path.

    **Fail-closed contract** (review v0.5.7 / blocker 2): if
    :func:`kill_stale_group` returns ``False`` we cannot prove the leaked
    process group is gone. Marking the trial ``FAIL`` would let new trials
    schedule onto a GPU still held by the leaked process. We raise a
    ``RuntimeError`` instead so the operator sees a loud failure and can
    investigate manually. Pre-v0.5.8 we logged the survivor and continued.

    Args:
        study: Optuna study for the phase being recovered.
        experiment: Parsed experiment (used as the fallback ``trial_dir``
            source for legacy trials without the user attribute).
        phase_name: Name of the phase being recovered.

    Returns:
        The number of RUNNING trials successfully reaped (i.e. cleanup
        confirmed AND ``study.tell(...FAIL)`` succeeded).

    Raises:
        RuntimeError: Cleanup of a stale process group could not be confirmed,
            or ``study.tell`` could not persist the FAIL state.

    """
    from phasesweep.process import kill_stale_group, read_stale_process_identity

    reaped = 0
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue

        stored = trial.user_attrs.get("phasesweep_trial_dir")
        if isinstance(stored, str) and stored:
            trial_dir = Path(stored)
        else:
            trial_dir = _trial_dir_for(experiment, phase_name, trial.number)

        identity = read_stale_process_identity(trial_dir)
        if identity.pid is not None or identity.pgid is not None:
            safe_to_fail = kill_stale_group(identity.pid, identity.starttime, pgid=identity.pgid)
            if not safe_to_fail:
                raise RuntimeError(
                    f"Refusing to mark RUNNING trial {trial.number} as FAIL: "
                    f"stale process cleanup could not prove the process group "
                    f"is gone. trial_dir={trial_dir} pid={identity.pid} "
                    f"pgid={identity.pgid}. A leaked training process may still "
                    "be holding GPU memory. Investigate (e.g. `ps -o pid,pgid,cmd "
                    f"-p {identity.pid}` and `kill -9 -- -{identity.pgid}` if "
                    "appropriate), then re-run phasesweep."
                )
            log.warning(
                "Cleared orphaned group for trial %d (pid=%s pgid=%s)",
                trial.number,
                identity.pid,
                identity.pgid,
            )

        try:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        except Exception as exc:
            raise RuntimeError(
                f"Stale process cleanup completed for RUNNING trial {trial.number}, "
                f"but Optuna state could not be updated to FAIL. Refusing to continue "
                f"with an inconsistent study. trial_dir={trial_dir}"
            ) from exc

        reaped += 1
        log.warning("Reaped stale RUNNING trial %d in study %s", trial.number, study.study_name)
    return reaped


# --------------------------------------------------------------------------------------
# Storage resolution (review v0.5.2 / blocker 6: no silent SQLite -> Journal remap)
# --------------------------------------------------------------------------------------


def _resolve_storage(url: str | None) -> Any:
    """Translate a storage URL into an Optuna storage object or pass through.

    Recognized schemes:
      * ``None`` -> in-memory study (not resumable).
      * ``journal:///path.journal`` -> Optuna ``JournalStorage(JournalFileBackend(path))``.
        Safe for parallel ``n_jobs`` on a single host.
      * Anything else (``sqlite:///``, ``postgresql://``, ``mysql://``, ...) -> passed to
        Optuna unchanged.

    Earlier versions silently rewrote ``sqlite:///x.db`` to ``x.journal`` whenever
    ``n_jobs > 1`` (review v0.5.2 / blocker 6). That broke study identity: the same
    config could resolve to two different studies depending on parallelism. Now the
    user picks the scheme, and ``Experiment._validate_phase_graph`` rejects SQLite
    with parallel ``n_jobs`` at config-load time so the failure is loud and early.

    Args:
        url: The user-supplied storage URL, or ``None`` for in-memory.

    Returns:
        ``None`` for in-memory; a configured ``JournalStorage`` for the
        ``journal:///`` scheme; the URL string unchanged otherwise (passed
        through to Optuna's RDB-aware loader).

    """
    if url is None:
        return None
    if url.startswith("journal:///"):
        path = url.removeprefix("journal:///")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        log.info("Using JournalFileStorage at %s", path)
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend

        return JournalStorage(JournalFileBackend(path))
    return url


# --------------------------------------------------------------------------------------
# Phase fingerprint — semantic, not a raw model_dump (review v0.5.2 / blocker 1)
# --------------------------------------------------------------------------------------


_RUN_CONTROL_KEYS = frozenset(
    {
        # Fields excluded from the fingerprint because they don't change trial
        # meaning. Top-up workflow (re-run with a higher n_trials) must work;
        # throughput knobs (n_jobs / gpu_ids) and circuit breakers
        # (max_consecutive_failures) likewise must not invalidate a study.
        # `comment` is operator-facing documentation — editing it is never a
        # semantic change to the experiment.
        "n_trials",
        "n_jobs",
        "gpu_ids",
        "allow_no_gpu_isolation",
        "max_consecutive_failures",
        "comment",
        "allow_unbounded_trials",
        "timeout_seconds_per_phase",
        "allow_partial_grid",
        "allow_seed_search",
    }
)


def _phase_semantic_payload(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> dict[str, Any]:
    """Return a dict capturing only fields that change *trial meaning*.

    Excludes run-control fields (review v0.5.2 / blocker 1) so that bumping
    ``n_trials`` to top up a study is a compatible operation. Includes
    ``experiment.env`` which v0.5.1 missed: env vars like ``CUBLAS_WORKSPACE_CONFIG``
    or ``MY_TRAINER_SEED`` change training behavior and must invalidate reuse.

    Args:
        experiment: The full experiment config; contributes ``trial_command``,
            ``override_format``, ``env``, metric, and constraints.
        phase: The phase being fingerprinted; ``_RUN_CONTROL_KEYS`` are stripped.
        inherited_winners: Winners loaded from parent phases; their
            ``effective_overrides`` are part of this phase's identity.

    Returns:
        A JSON-serialisable dict whose contents fully determine trial meaning
        for the phase. Stable across irrelevant config edits, varies on
        anything that would change a trial's outcome.

    """
    phase_dump = phase.model_dump(mode="json")
    semantic_phase = {k: v for k, v in phase_dump.items() if k not in _RUN_CONTROL_KEYS}
    return {
        "phasesweep_version": __version__,
        "trial_command": experiment.trial_command,
        "override_format": experiment.override_format,
        "env": dict(sorted(experiment.env.items())),
        "metric": experiment.metric.model_dump(mode="json"),
        "constraints": [c.model_dump(mode="json") for c in experiment.constraints],
        "contracts": {
            name: experiment.contracts[name].model_dump(mode="json") for name in phase.contracts
        },
        "phase": semantic_phase,
        "inherited_effective_overrides": {
            parent: inherited_winners[parent].effective_overrides for parent in phase.inherits
        },
    }


def _phase_fingerprint(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> str:
    """Hash the semantic execution context for resume-compatibility checks.

    Uses the full SHA-256 hex digest. Earlier versions truncated to 16 hex
    chars (64 bits) — defensible against accidental collision but no reason
    to leave the door open in scientific-workflow metadata.

    Args:
        experiment: The experiment config (forwarded to
            :func:`_phase_semantic_payload`).
        phase: The phase being fingerprinted.
        inherited_winners: Parent-phase winners; their effective overrides
            contribute to identity.

    Returns:
        SHA-256 hex digest (64 characters) of the canonicalised semantic
        payload. Used to detect incompatible re-runs and stamped onto
        ``winner.yaml`` files for cross-version verification.

    """
    payload = _phase_semantic_payload(experiment, phase, inherited_winners)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_fingerprint(
    study: optuna.Study,
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> None:
    """Stamp a fresh study with its fingerprint or fail on mismatch.

    Args:
        study: The Optuna study being verified or stamped.
        experiment: The current experiment config.
        phase: The phase whose fingerprint should match the stored one.
        inherited_winners: Parent-phase winners contributing to identity.

    Raises:
        RuntimeError: The study already has a fingerprint and it does not
            match the current computed value (incompatible config edit).

    """
    fp = _phase_fingerprint(experiment, phase, inherited_winners)
    existing = study.user_attrs.get("phasesweep_fingerprint")
    if existing is None:
        study.set_user_attr("phasesweep_fingerprint", fp)
    elif existing != fp:
        raise RuntimeError(
            f"Study {study.study_name!r} was created with a different phase config "
            f"(fingerprint {existing} != {fp}). Use a new experiment name, delete the "
            f"old study, or rename the phase."
        )


# --------------------------------------------------------------------------------------
# Optuna helpers
# --------------------------------------------------------------------------------------


def _build_sampler(
    cfg: Sampler, search_space: dict[str, SearchParam], n_jobs: int = 1
) -> optuna.samplers.BaseSampler:
    """Construct the Optuna sampler for a phase from its YAML ``sampler`` block.

    Args:
        cfg: Parsed sampler config (type, seed, startup-trials, etc.).
        search_space: The phase's search space, used to build the GridSampler
            grid and to defend against categorical+CmaEs combinations.
        n_jobs: Phase parallelism; enables TPE's ``constant_liar`` heuristic
            when ``n_jobs > 1``.

    Returns:
        A configured :class:`optuna.samplers.BaseSampler` subclass instance.

    Raises:
        ValueError: Sampler type incompatible with the search space (e.g. log
            scale on grid int, categorical on cmaes) or an unknown sampler type.

    """
    if cfg.type == "tpe":
        return optuna.samplers.TPESampler(
            seed=cfg.seed,
            n_startup_trials=cfg.n_startup_trials,
            constant_liar=(n_jobs > 1),
        )
    if cfg.type == "random":
        return optuna.samplers.RandomSampler(seed=cfg.seed)
    if cfg.type == "grid":
        grid: dict[str, list[Any]] = {}
        for name, p in search_space.items():
            if isinstance(p, CategoricalParam):
                grid[name] = list(p.choices)
            elif isinstance(p, IntParam):
                if p.log:
                    raise ValueError("Grid sampler does not support log-int spaces.")
                grid[name] = list(range(p.low, p.high + 1, p.step))
            elif isinstance(p, FloatParam):
                if p.step is None:
                    raise ValueError(f"Grid sampler requires a 'step' on float param {name!r}.")
                n_steps = int(round((p.high - p.low) / p.step))
                grid[name] = [round(p.low + i * p.step, 12) for i in range(n_steps + 1)]
            else:  # pragma: no cover
                raise ValueError(f"Unhandled param type for grid: {p!r}")
        return optuna.samplers.GridSampler(grid, seed=cfg.seed)
    if cfg.type == "cmaes":
        # Defense in depth. ``_validate_sampler_search_space`` already rejects
        # categorical-on-cmaes at config-load and import-checks the cmaes
        # package; we re-check categoricals here because direct callers of
        # ``_build_sampler`` (tests, future internal use) bypass that path
        # and Optuna's ``CmaEsSampler`` silently fails every trial with a
        # categorical param trying to cast strings to float.
        if any(isinstance(p, CategoricalParam) for p in search_space.values()):
            raise ValueError(
                "sampler.type='cmaes' does not support categorical parameters. "
                "Use sampler.type='tpe' or remove categorical params from this phase."
            )
        return optuna.samplers.CmaEsSampler(seed=cfg.seed)
    raise ValueError(f"Unknown sampler {cfg.type!r}")  # pragma: no cover


def _suggest(trial: optuna.Trial, name: str, p: SearchParam) -> Any:
    """Dispatch to the right Optuna ``trial.suggest_*`` based on param type.

    Args:
        trial: The active Optuna trial.
        name: Parameter name (used as the Optuna key).
        p: The concrete search parameter from the phase's ``search_space``.

    Returns:
        The sampled value. Type matches ``p`` (``float``/``int``/categorical scalar).

    """
    if isinstance(p, FloatParam):
        return trial.suggest_float(name, p.low, p.high, step=p.step, log=p.log)
    if isinstance(p, IntParam):
        return trial.suggest_int(name, p.low, p.high, step=p.step, log=p.log)
    if isinstance(p, CategoricalParam):
        return trial.suggest_categorical(name, p.choices)
    raise ValueError(f"Unhandled param: {p!r}")  # pragma: no cover


# --------------------------------------------------------------------------------------
# CSV writer (#6: stdlib, no pandas)
# --------------------------------------------------------------------------------------


def _write_trials_csv(study: optuna.Study, path: Path) -> None:
    """Snapshot every trial in ``study`` to ``path`` as a CSV (stdlib only, no pandas).

    The column set is the union of ``params`` keys and ``user_attrs`` keys seen
    across all trials, plus a fixed core (``number``, ``state``, ``value``,
    timing, ``duration``). Empty studies are a no-op.

    Args:
        study: The Optuna study to snapshot.
        path: Output CSV path; parent directories are created as needed.

    """
    trials = study.get_trials(deepcopy=False)
    if not trials:
        return
    param_names = sorted({n for t in trials for n in t.params})
    attr_names = sorted({n for t in trials for n in t.user_attrs})
    fieldnames = [
        "number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration",
        *[f"param:{n}" for n in param_names],
        *[f"user_attr:{n}" for n in attr_names],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            row: dict[str, Any] = {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "datetime_start": t.datetime_start,
                "datetime_complete": t.datetime_complete,
                "duration": t.duration,
            }
            for n in param_names:
                row[f"param:{n}"] = t.params.get(n)
            for n in attr_names:
                row[f"user_attr:{n}"] = t.user_attrs.get(n)
            writer.writerow(row)


# --------------------------------------------------------------------------------------
# Override composition (#9: uses effective_overrides from inherited winners)
# --------------------------------------------------------------------------------------


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
    """Return contract gates followed by phase-local gates."""
    gates: list[Gate] = []
    for contract_name in phase.contracts:
        gates.extend(experiment.contracts[contract_name].gates)
    gates.extend(phase.gates)
    return gates


def _gates_pass(gates: list[dict[str, Any]]) -> bool:
    """Return whether every recorded gate result passed."""
    return all(bool(gate.get("passed")) for gate in gates)


def _clone_winner_from_baseline(baseline: Winner, *, phase_fingerprint: str | None) -> Winner:
    """Clone a baseline winner for exposure under another phase/study."""
    return Winner(
        trial_number=baseline.trial_number,
        params=dict(baseline.params),
        effective_overrides=dict(baseline.effective_overrides),
        metric=baseline.metric,
        constraints=dict(baseline.constraints),
        gates=list(baseline.gates),
        phase_fingerprint=phase_fingerprint,
    )


def _metric_improvement(goal: str, candidate: Winner, baseline: Winner) -> float:
    """Return candidate improvement over baseline for a metric goal."""
    if goal == "minimize":
        return baseline.metric - candidate.metric
    return candidate.metric - baseline.metric


def _winner_summary_item(name: str, winner: Winner) -> dict[str, Any]:
    """Return the compact winner payload used in run summaries."""
    return {
        "name": name,
        "trial_number": winner.trial_number,
        "metric": winner.metric,
        "params": winner.params,
        "effective_overrides": winner.effective_overrides,
        "constraints": winner.constraints,
        "gates": winner.gates,
    }


def _trial_gate_payload(study: optuna.Study, trial_number: int) -> list[dict[str, Any]]:
    """Load persisted gate result payload for a selected trial."""
    for trial in study.get_trials(deepcopy=False):
        if trial.number != trial_number:
            continue
        raw = trial.user_attrs.get("phasesweep_gates")
        if not isinstance(raw, str) or not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]
    return []


# --------------------------------------------------------------------------------------
# Phase helpers
# --------------------------------------------------------------------------------------


def _experiment_dir(experiment: Experiment) -> Path:
    """Filesystem namespace for one experiment's artifacts.

    Was ``<workdir>`` directly through v0.5.6. As of v0.5.7 the experiment
    name is part of the path (review v0.5.6 / blocker 1) so two configs
    sharing a workdir but pointing at different Optuna namespaces no longer
    write into the same trial/winner files.

    Args:
        experiment: Parsed experiment config; supplies ``workdir`` and ``experiment``.

    Returns:
        Resolved absolute path ``<workdir>/<experiment>``.

    """
    return Path(experiment.workdir).expanduser().resolve() / experiment.experiment


def _phase_dir(experiment: Experiment, phase_name: str) -> Path:
    """Filesystem namespace for one phase's trial dirs, winner, and trials.csv.

    Args:
        experiment: Parsed experiment config.
        phase_name: Name of the phase whose directory is requested.

    Returns:
        Resolved absolute path ``<workdir>/<experiment>/<phase_name>``.

    """
    return _experiment_dir(experiment) / phase_name


def _summary_path(experiment: Experiment) -> Path:
    """Path to the experiment-wide summary written at the end of ``run_experiment``.

    Args:
        experiment: Parsed experiment config.

    Returns:
        Resolved absolute path ``<workdir>/<experiment>/summary.yaml``.

    """
    return _experiment_dir(experiment) / "summary.yaml"


def _run_log_path(experiment: Experiment) -> Path:
    """Path to the durable run log for one experiment."""
    return _experiment_dir(experiment) / "run.log"


@contextlib.contextmanager
def _file_log_handler(path: Path) -> Iterator[None]:
    """Attach a durable file handler for phasesweep logs."""
    logger = logging.getLogger("phasesweep")
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname).1s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.setLevel(logging.DEBUG)
    old_level = logger.level
    if old_level in (logging.NOTSET, 0) or old_level > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(old_level)


@contextlib.contextmanager
def _run_log_handler(experiment: Experiment) -> Iterator[None]:
    """Attach a durable file handler for one experiment run."""
    with _file_log_handler(_run_log_path(experiment)):
        yield


def _trial_dir_for(experiment: Experiment, phase_name: str, trial_number: int) -> Path:
    """Resolve the canonical on-disk trial directory.

    Used both for new launches and as the reaper's fallback when a stale trial
    has no ``phasesweep_trial_dir`` user_attr persisted (review v0.5.3 /
    blocker 4). New trials always persist the attr; this fallback only matters
    for trials created by older phasesweep versions.

    Args:
        experiment: Parsed experiment config.
        phase_name: Name of the phase.
        trial_number: Optuna's numeric trial number.

    Returns:
        Resolved absolute path
        ``<workdir>/<experiment>/<phase_name>/trial_<NNNNN>`` (zero-padded to 5 digits).

    """
    return _phase_dir(experiment, phase_name) / f"trial_{trial_number:05d}"


def _winner_path(experiment: Experiment, phase_name: str) -> Path:
    """Filesystem path to a phase's persisted ``winner.yaml`` summary.

    Args:
        experiment: Parsed experiment config.
        phase_name: Name of the phase.

    Returns:
        Resolved absolute path ``<workdir>/<experiment>/<phase_name>/winner.yaml``.

    """
    return _phase_dir(experiment, phase_name) / "winner.yaml"


def _save_winner(experiment: Experiment, phase_name: str, winner: Winner) -> None:
    """Persist a phase winner.

    The phase fingerprint is included so ``_load_winner`` can refuse stale
    winners on ``--from-phase`` resume (review v0.5.6 / blocker 3). Real
    winners always carry a fingerprint by construction in ``_run_phase_inner``;
    placeholder winners (dry-run skip) are never saved.

    Args:
        experiment: Parsed experiment config; supplies the metric name used
            in the persisted payload.
        phase_name: Name of the phase whose winner is being saved.
        winner: The winning trial. Must have ``phase_fingerprint`` set.

    Raises:
        RuntimeError: ``winner.phase_fingerprint`` is ``None``. This is an
            internal invariant; placeholder winners must not reach this path.

    """
    if winner.phase_fingerprint is None:
        # Defense in depth: should not happen — _run_phase_inner always sets
        # this before calling _save_winner. Failing loudly here prevents a
        # silently un-resumable winner from landing on disk.
        raise RuntimeError(
            f"Refusing to save winner for phase {phase_name!r} without a "
            "phase_fingerprint. This is an internal invariant — please file "
            "a bug."
        )

    path = _winner_path(experiment, phase_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase_name,
        "trial_number": winner.trial_number,
        "metric": {experiment.metric.name: winner.metric, "goal": experiment.metric.goal},
        "params": winner.params,
        "effective_overrides": winner.effective_overrides,
        "constraints": winner.constraints,
        "gates": winner.gates,
        "phase_fingerprint": winner.phase_fingerprint,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _load_winner(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> Winner:
    """Load a phase winner from disk and verify it matches the *current* config.

    ``--from-phase`` skips earlier phases by reading their persisted winners.
    Without verification, editing a parent phase's YAML between runs leaves
    the child phase silently inheriting the *old* winner against the *new*
    parent config — a correctness bug, not just a performance one.

    We re-compute the fingerprint of the current parent ``phase`` against the
    currently-resolved ``inherited_winners`` and refuse the load if either
    (a) the stored winner has no fingerprint at all (legacy or hand-edited),
    or (b) the fingerprints disagree (review v0.5.6 / blocker 3).

    Args:
        experiment: Parsed experiment config.
        phase: The phase whose winner is being loaded.
        inherited_winners: Winners loaded for phases earlier in the chain;
            contribute to the recomputed fingerprint.

    Returns:
        The reconstructed :class:`Winner` for ``phase``.

    Raises:
        FileNotFoundError: ``winner.yaml`` does not exist for the phase.
        RuntimeError: The file is unfingerprinted (legacy/hand-edited) or its
            fingerprint disagrees with the freshly computed one.

    """
    path = _winner_path(experiment, phase.name)
    if not path.is_file():
        raise FileNotFoundError(f"Winner file missing for phase {phase.name!r}: {path}")

    data = yaml.safe_load(path.read_text())

    current_fp = _phase_fingerprint(experiment, phase, inherited_winners)
    stored_fp = data.get("phase_fingerprint")

    if stored_fp is None:
        raise RuntimeError(
            f"Winner file {path} has no phase_fingerprint. Refusing to use it "
            f"for --from-phase because phasesweep cannot prove it matches the "
            f"current config for skipped phase {phase.name!r}. Re-run the "
            f"phase, or — if you know the config is unchanged — delete the "
            f"file and re-run to regenerate it with a fingerprint."
        )

    if stored_fp != current_fp:
        raise RuntimeError(
            f"Winner file {path} was produced by a different phase config "
            f"(stored fingerprint {stored_fp[:16]}... != current "
            f"{current_fp[:16]}...). Re-run phase {phase.name!r}, change the "
            f"experiment name, or restore the matching config before resuming."
        )

    return Winner(
        trial_number=int(data["trial_number"]),
        params=dict(data["params"]),
        effective_overrides=dict(data.get("effective_overrides") or data["params"]),
        metric=float(data["metric"][experiment.metric.name]),
        constraints={k: float(v) for k, v in (data.get("constraints") or {}).items()},
        gates=[item for item in (data.get("gates") or []) if isinstance(item, dict)],
        phase_fingerprint=str(stored_fp),
    )


def _reap_skipped_phase(experiment: Experiment, phase: Phase) -> None:
    """Reap stale RUNNING trials for a phase skipped by ``--from-phase``."""
    if experiment.storage is None:
        return
    direction = "minimize" if experiment.metric.goal == "minimize" else "maximize"
    study = optuna.create_study(
        study_name=f"{experiment.experiment}::{phase.name}",
        storage=_resolve_storage(experiment.storage),
        sampler=_build_sampler(phase.sampler, phase.search_space, n_jobs=phase.n_jobs),
        pruner=optuna.pruners.NopPruner(),
        direction=direction,
        load_if_exists=True,
    )
    _reap_stale_trials(study, experiment, phase.name)


def _apply_promotion(
    experiment: Experiment,
    phase: Phase,
    candidate: Winner,
    winners: dict[str, Winner],
) -> Winner | None:
    """Apply a phase promotion rule, returning the winner to expose downstream."""
    promotion = phase.promotion
    if promotion is None:
        return candidate
    if promotion.min_delta_vs not in winners:
        raise RuntimeError(
            f"Phase {phase.name!r} promotion references unknown baseline "
            f"{promotion.min_delta_vs!r}."
        )
    if promotion.requires_gates and not _gates_pass(candidate.gates):
        message = f"Phase {phase.name!r} failed promotion: required evidence gates did not pass."
        if promotion.on_fail == "stop":
            raise RuntimeError(message)
        if promotion.on_fail == "skip":
            log.warning("%s Skipping remaining dependent phases.", message)
            return None
        log.warning("%s Continuing with baseline winner.", message)
        return _clone_winner_from_baseline(
            winners[promotion.min_delta_vs],
            phase_fingerprint=candidate.phase_fingerprint,
        )

    baseline = winners[promotion.min_delta_vs]
    improvement = _metric_improvement(experiment.metric.goal, candidate, baseline)
    passed = improvement >= promotion.min_delta
    if passed:
        log.info(
            "phase=%s PROMOTED improvement=%g baseline=%s min_delta=%g",
            phase.name,
            improvement,
            promotion.min_delta_vs,
            promotion.min_delta,
        )
        return candidate

    message = (
        f"Phase {phase.name!r} failed promotion: improvement {improvement:g} "
        f"vs {promotion.min_delta_vs!r} is below min_delta {promotion.min_delta:g}."
    )
    if promotion.on_fail == "stop":
        raise RuntimeError(message)
    if promotion.on_fail == "skip":
        log.warning("%s Skipping remaining dependent phases.", message)
        return None

    log.warning("%s Continuing with baseline winner.", message)
    return _clone_winner_from_baseline(baseline, phase_fingerprint=candidate.phase_fingerprint)


# --------------------------------------------------------------------------------------
# Phase runner
# --------------------------------------------------------------------------------------


def _lock_material(experiment: Experiment, *, scope: dict[str, str]) -> dict[str, Any]:
    """Build the lock-identity dict for a given lock scope.

    For persistent storage, identity is *storage URL + scope*. For in-memory
    storage there is no shared backend, so identity falls back to
    *workdir + scope* to prevent trial-directory and ``summary.yaml`` collisions.

    Args:
        experiment: Parsed experiment config; supplies storage and workdir.
        scope: Extra key-value pairs describing the lock's scope (e.g.
            ``{"study": "exp::phase"}``).

    Returns:
        A JSON-serialisable dict identifying the lock; hashed by
        :func:`_lock_digest` into a path-safe filename.

    """
    storage_identity = canonical_storage_identity(experiment.storage)
    if storage_identity is None:
        return {
            "kind": "in_memory_workdir",
            "workdir": str(Path(experiment.workdir).expanduser().resolve()),
            **scope,
        }
    return {
        "kind": "persistent_storage",
        "storage": storage_identity,
        **scope,
    }


def _lock_digest(material: dict[str, Any]) -> str:
    """Hash a lock-material dict into a 24-char hex digest.

    Args:
        material: The output of :func:`_lock_material` (or any
            JSON-serialisable dict).

    Returns:
        First 24 hex characters of the SHA-256 of the canonicalised JSON. 24
        chars = 96 bits — well past collision risk for a same-host advisory
        lock filename.

    """
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _lock_path_from_material(experiment: Experiment, material: dict[str, str], label: str) -> Path:
    """Lock path under ``$TMPDIR/phasesweep-locks/`` named ``<exp>__<label>__<digest>.lock``.

    Args:
        experiment: Parsed experiment config; the experiment name is part of
            the filename for human readability.
        material: Lock-material dict produced by :func:`_lock_material` or
            similar; hashed into the digest segment.
        label: A short human-readable label (``"output"``, ``"storage"``,
            phase name, ...).

    Returns:
        Resolved absolute path to the lock file (the file itself is not
        created here; ``open(...).flock()`` does that lazily).

    """
    return _lock_dir() / f"{experiment.experiment}__{label}__{_lock_digest(material)}.lock"


def _output_lock_material(experiment: Experiment) -> dict[str, str]:
    """Identity for the *output namespace* lock: which directory we write to.

    Catches the case where two configs share a workdir + experiment name but
    point at different storage backends — without an output lock those would
    silently overwrite each other's ``trial_*/``, ``winner.yaml``, and
    ``summary.yaml`` (review v0.5.6 / blocker 1). Always taken regardless of
    storage backend, including in-memory storage.

    Args:
        experiment: Parsed experiment config; supplies workdir + experiment name.

    Returns:
        Lock-material dict keyed on the resolved experiment directory.

    """
    return {"kind": "output", "experiment_dir": str(_experiment_dir(experiment))}


def _storage_run_lock_material(experiment: Experiment) -> dict[str, str] | None:
    """Identity for the *Optuna storage* lock: which study namespace we write to.

    Catches the case where two configs share storage + experiment name but
    point at different workdirs. Returns ``None`` for in-memory storage —
    there is no shared backend, so the output lock alone is sufficient.

    Args:
        experiment: Parsed experiment config; supplies storage + experiment name.

    Returns:
        Lock-material dict keyed on canonical storage identity, or ``None``
        when storage is in-memory.

    """
    storage_identity = canonical_storage_identity(experiment.storage)
    if storage_identity is None:
        return None
    return {
        "kind": "persistent_storage",
        "storage": storage_identity,
        "experiment": experiment.experiment,
    }


def _run_lock_paths(experiment: Experiment) -> list[Path]:
    """All same-host locks required for one full experiment run.

    For persistent storage we take *both* locks (output + storage); for
    in-memory storage we take only the output lock. The list is sorted by
    path so acquisition order is deterministic across processes — relevant
    for clear error messages, not for deadlock avoidance (we use
    ``LOCK_NB``).

    Args:
        experiment: Parsed experiment config.

    Returns:
        Lock-file paths sorted by string order. Length is 1 (output only) for
        in-memory storage, 2 (output + storage) otherwise.

    """
    materials: list[tuple[str, dict[str, str]]] = [
        ("output", _output_lock_material(experiment)),
    ]
    storage_material = _storage_run_lock_material(experiment)
    if storage_material is not None:
        materials.append(("storage", storage_material))
    return sorted(
        (_lock_path_from_material(experiment, m, label) for label, m in materials),
        key=str,
    )


def _phase_lock_path(experiment: Experiment, phase: Phase) -> Path:
    """Compute the lock file path for a given ``experiment::phase`` study.

    Defense in depth on top of :func:`_experiment_lock`: protects direct
    internal calls into ``_run_phase`` (e.g. tests) and any future code path
    that bypasses ``run_experiment``.

    Args:
        experiment: Parsed experiment config.
        phase: The phase whose lock path is requested.

    Returns:
        Absolute path to the phase-scoped lock file.

    """
    study_name = f"{experiment.experiment}::{phase.name}"
    material = _lock_material(experiment, scope={"study": study_name})
    return _lock_dir() / f"{experiment.experiment}__{phase.name}__{_lock_digest(material)}.lock"


@contextlib.contextmanager
def _experiment_lock(experiment: Experiment) -> Iterator[None]:
    """Take all same-host locks needed for one full experiment run.

    The phase-chained pipeline has cross-phase state — parent ``winner.yaml``,
    child fingerprints, ``summary.yaml``, ``--from-phase`` semantics — so the
    consistency domain is the entire run, not a single phase study. v0.5.7
    extends this further: the consistency domain spans both the *output
    namespace* (filesystem artifacts under ``<workdir>/<experiment>/``) and
    the *Optuna storage namespace* (review v0.5.6 / blocker 1).

    Two configs can disagree on storage but share output paths, or vice
    versa; either case can corrupt skipped-phase reuse and trial logs. We
    therefore take an output lock *always*, and a storage lock additionally
    whenever storage is persistent. In-memory storage has no shared backend,
    so the output lock alone suffices.

    Both locks are *same-host advisory only*. Multi-host coordination needs
    per-trial leases + heartbeats (see ``TODO.md``).

    Args:
        experiment: Parsed experiment config.

    Yields:
        ``None``. Use as ``with _experiment_lock(exp): ...``.

    Raises:
        RuntimeError: Another phasesweep process holds one of the required
            locks (output namespace or storage identity).

    """
    import fcntl

    paths = _run_lock_paths(experiment)
    handles: list[Any] = []
    try:
        for path in paths:
            f = path.open("w")
            handles.append(f)
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"Another phasesweep process appears to be using the same "
                    f"experiment backend or output namespace for "
                    f"{experiment.experiment!r} (lock file: {path}). phasesweep "
                    f"supports one active orchestrator per experiment output "
                    f"namespace and per persistent storage identity."
                ) from exc
        yield
    finally:
        # Reverse-order release isn't required by flock semantics, but it
        # keeps the "stack" mental model intact and parallels typical
        # acquire-A-then-B / release-B-then-A discipline.
        for f in reversed(handles):
            with contextlib.suppress(OSError):
                fcntl.flock(f, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                f.close()


@contextlib.contextmanager
def _phase_lock(experiment: Experiment, phase: Phase) -> Iterator[None]:
    """Take an exclusive flock keyed by Optuna storage identity + study name.

    Defense in depth: the public entrypoint :func:`run_experiment` already
    holds an :func:`_experiment_lock` for the entire run, which subsumes this
    in normal use. ``_phase_lock`` still protects direct internal callers of
    ``_run_phase`` (tests, future code paths) from same-phase reaper collisions.

    Args:
        experiment: Parsed experiment config.
        phase: The phase whose study is being locked.

    Yields:
        ``None``. Use as ``with _phase_lock(exp, phase): ...``.

    Raises:
        RuntimeError: Another orchestrator already holds the phase lock.

    """
    import fcntl

    path = _phase_lock_path(experiment, phase)
    with path.open("w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another phasesweep process appears to be running study "
                f"{experiment.experiment!r}::{phase.name!r} on this host "
                f"(lock file: {path}). phasesweep supports one orchestrator "
                f"per study."
            ) from exc
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(f, fcntl.LOCK_UN)


def _run_phase(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    *,
    dry_run: bool = False,
    run_deadline: float | None = None,
) -> Winner:
    """Wrap :func:`_run_phase_inner` in the phase lock (skipped on ``dry_run``).

    Args:
        experiment: Parsed experiment config.
        phase: The phase to execute.
        inherited_winners: Winners loaded for phases earlier in the chain.
        dry_run: When ``True``, skip the lock and the subprocess launch; just
            render an example command for the user.
        run_deadline: Optional ``time.monotonic()`` deadline inherited from
            the experiment-level wallclock guard.

    Returns:
        The phase :class:`Winner` (real for normal runs, placeholder midpoint
        winner for dry runs).

    """
    if dry_run:
        return _run_phase_inner(
            experiment,
            phase,
            inherited_winners,
            dry_run=True,
            run_deadline=run_deadline,
        )
    with _phase_lock(experiment, phase):
        return _run_phase_inner(
            experiment,
            phase,
            inherited_winners,
            dry_run=False,
            run_deadline=run_deadline,
        )


def _run_phase_inner(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    *,
    dry_run: bool = False,
    run_deadline: float | None = None,
) -> Winner:
    """Execute one phase end-to-end (sampler, study.optimize, winner selection).

    See module docstring for the full hot-path narrative. Defines nested
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
    study_name = f"{experiment.experiment}::{phase.name}"
    sampler = _build_sampler(phase.sampler, phase.search_space, n_jobs=phase.n_jobs)

    direction = "minimize" if experiment.metric.goal == "minimize" else "maximize"
    storage = None if dry_run else _resolve_storage(experiment.storage)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        direction=direction,
        load_if_exists=True,
    )

    if not dry_run:
        # Reap first, fingerprint second (review item #7). A config-mismatch RuntimeError
        # must not leave a previous orchestrator's training process holding GPU memory.
        _reap_stale_trials(study, experiment, phase.name)
        _verify_fingerprint(study, experiment, phase, inherited_winners)

    completed = sum(1 for t in study.get_trials(deepcopy=False) if t.state.is_finished())
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
        allow_no_gpu=phase.allow_no_gpu_isolation,
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
        trial.set_user_attr("phasesweep_trial_dir", str(trial_dir))

        # GPU lease covers only subprocess lifetime, not extraction (#2).
        with gpu_pool.acquire() as gpu_id:
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

            executed = launch_trial(
                experiment=experiment,
                phase_name=phase.name,
                trial_id=trial.number,
                trial_dir=trial_dir,
                overrides=overrides,
                timeout_seconds=phase.timeout_seconds_per_trial,
                gpu_id=gpu_id,
            )

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
                    trial.set_user_attr("phasesweep_cleanup_confirmed", False)
                    trial.set_user_attr(
                        "phasesweep_failure_reason",
                        executed.process.failure_reason or "process cleanup could not be confirmed",
                    )

                raise UnsafeProcessCleanupError(message)

        # Extraction happens outside GPU lease.
        result = extract_trial_result(
            experiment=experiment,
            executed=executed,
            gates=_phase_gates(experiment, phase),
            enforce_gates=phase.promotion is None or phase.promotion.requires_gates,
        )

        trial.set_user_attr("phasesweep_feasible", result.feasible)
        trial.set_user_attr("phasesweep_return_code", result.return_code)
        trial.set_user_attr("phasesweep_duration_s", result.duration_seconds)
        trial.set_user_attr(
            "phasesweep_overrides", json.dumps(overrides, default=str, sort_keys=True)
        )
        if result.gate_results is not None:
            trial.set_user_attr(
                "phasesweep_gates",
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
            trial.set_user_attr("phasesweep_failure_reason", result.failure_reason)
            with _failure_lock:
                _consecutive_failures += 1
            raise TrialExecutionError(result.failure_reason)

        for cname, cval in result.constraints.items():
            trial.set_user_attr(f"constraint:{cname}", cval)

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
        with contextlib.suppress(Exception):
            _write_trials_csv(study, _phase_dir(experiment, phase.name) / "trials.csv")

    if remaining > 0:
        optimize_timeout = phase.timeout_seconds_per_phase
        if run_deadline is not None:
            remaining_run_seconds = max(0.0, run_deadline - time.monotonic())
            optimize_timeout = (
                remaining_run_seconds
                if optimize_timeout is None
                else min(optimize_timeout, remaining_run_seconds)
            )
        if optimize_timeout is not None and optimize_timeout <= 0.0:
            raise TimeoutError(
                f"Run wallclock deadline reached before phase {phase.name!r} could launch."
            )
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

    if abort["flag"]:
        raise NoFeasibleTrialError(
            f"Phase {phase.name!r} aborted after "
            f"{phase.max_consecutive_failures} consecutive failures. "
            f"Inspect {_phase_dir(experiment, phase.name)} for stderr logs."
        )

    # Build winner with effective_overrides (#9). Stamp it with the phase
    # fingerprint so a later --from-phase resume can detect stale parent
    # config (review v0.5.6 / blocker 3). The fingerprint is the same one
    # _verify_fingerprint stamps on the Optuna study; recomputing here keeps
    # _save_winner independent of study state.
    selected = select_winner(study, experiment)
    effective = _composed_overrides(experiment, phase, selected.params, inherited_winners)
    gate_payload = _trial_gate_payload(study, selected.trial_number)
    winner = Winner(
        trial_number=selected.trial_number,
        params=selected.params,
        effective_overrides=effective,
        metric=selected.metric,
        constraints=selected.constraints,
        gates=gate_payload,
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
    from phasesweep.overrides import render_command

    log.info("DRY RUN phase=%s would launch %d trials", phase.name, remaining)
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

    return _placeholder_winner(experiment, phase, inherited_winners)


def _midpoint_params(phase: Phase) -> dict[str, Any]:
    """Synthesize midpoint values for each search-space param (dry-run placeholder).

    Delegates per-param logic to ``config._placeholder_value_for`` to avoid
    maintaining two copies of the isinstance dispatch.

    Args:
        phase: The phase whose ``search_space`` to summarise.

    Returns:
        Dict mapping each search-space key to a deterministic placeholder
        value (interval midpoint for numeric, first choice for categorical).

    """
    from phasesweep.config import _placeholder_value_for

    return {name: _placeholder_value_for(p) for name, p in phase.search_space.items()}


def _placeholder_winner(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> Winner:
    """Synthesize a midpoint-valued placeholder winner for dry-run mode.

    Includes inherited effective_overrides so downstream dry-run previews see the
    same locked context they would at runtime (review item #10).

    Args:
        experiment: Parsed experiment; supplies named contracts.
        phase: The phase whose placeholder winner is needed.
        inherited_winners: Winners from earlier phases in the chain.

    Returns:
        A :class:`Winner` with ``trial_number=-1`` and ``metric=NaN`` so any
        accidental use in non-dry contexts surfaces obviously.

    """
    placeholder_params = _midpoint_params(phase)
    effective = _composed_overrides(experiment, phase, placeholder_params, inherited_winners)
    return Winner(
        trial_number=-1,
        params=placeholder_params,
        effective_overrides=effective,
        metric=float("nan"),
        constraints={},
        gates=[],
    )


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------


def run_experiment(
    experiment: Experiment,
    *,
    from_phase: str | None = None,
    dry_run: bool = False,
) -> dict[str, Winner]:
    """Run all phases in order, returning a map of phase name to Winner.

    If ``from_phase`` is given, prior phases are loaded from disk.
    If ``dry_run`` is True, example commands are logged but nothing launches.

    For non-dry-run invocations this acquires an :func:`_experiment_lock` for
    the entire phase sequence (review v0.5.6 / blocker 1). Phase-level state
    on disk (``winner.yaml``, fingerprints, ``summary.yaml``) is consistent
    only at run granularity, so two same-experiment processes must not
    interleave across phases.

    Signal handlers are installed here (review v0.5.7 / blocker 3) so library
    callers using the public API get the same cleanup guarantees as CLI
    callers. ``install_signal_handlers()`` is idempotent and a no-op when
    invoked from a non-main thread, so re-installation by the CLI is safe.
    Skipped on dry-run because no children will launch.

    Args:
        experiment: Parsed experiment config (result of
            :func:`phasesweep.load_experiment`).
        from_phase: Name of a phase to resume from; earlier phases are loaded
            from their persisted ``winner.yaml`` files (with fingerprint
            verification). ``None`` runs every phase from scratch.
        dry_run: If ``True``, render and log one example trial command per
            phase but launch no subprocesses; no summary is written.

    Returns:
        Mapping from phase name (in declaration order) to that phase's
        :class:`Winner`. For dry runs the winners are midpoint placeholders.

    Raises:
        NoFeasibleTrialError: A phase exhausted ``max_consecutive_failures``
            with no feasible trial.
        UnsafeProcessCleanupError: A phase hard-aborted because a trial's
            process group could not be confirmed dead (review v0.5.11).
        RuntimeError: Lock contention (another orchestrator running),
            fingerprint mismatch on ``--from-phase`` resume, or stale-reaper
            uncertainty (review v0.5.7 / blocker 2).
        FileNotFoundError: ``--from-phase`` requested but a prior phase has
            no persisted ``winner.yaml``.

    """
    if not dry_run:
        from phasesweep.process import install_signal_handlers

        install_signal_handlers()

    if dry_run:
        return _run_experiment_inner(experiment, from_phase=from_phase, dry_run=True)

    _experiment_dir(experiment).mkdir(parents=True, exist_ok=True)
    with _run_log_handler(experiment), _experiment_lock(experiment):
        return _run_experiment_inner(experiment, from_phase=from_phase, dry_run=False)


def _run_experiment_inner(
    experiment: Experiment,
    *,
    from_phase: str | None,
    dry_run: bool,
) -> dict[str, Winner]:
    """Sequential phase loop assuming locks/signal handlers are already set up.

    Args:
        experiment: Parsed experiment config.
        from_phase: Optional name of the phase to resume from; earlier phases
            are loaded from disk.
        dry_run: If ``True``, no subprocesses launch and no ``summary.yaml`` is written.

    Returns:
        Same as :func:`run_experiment`: a phase-name to :class:`Winner` mapping.

    """
    skip_until = from_phase is not None
    winners: dict[str, Winner] = {}
    run_deadline = (
        None
        if dry_run or experiment.timeout_seconds_per_run is None
        else time.monotonic() + experiment.timeout_seconds_per_run
    )

    for phase in experiment.phases:
        if run_deadline is not None and time.monotonic() >= run_deadline:
            raise TimeoutError(
                f"Run wallclock deadline reached before phase {phase.name!r} could start."
            )
        # Inherited winners must be resolved before either the skip-path winner
        # load (so we can verify its fingerprint against the *current* parent
        # context) or the actual run path. Keeping the construction in one
        # place makes the two paths symmetric.
        inherited = {p: winners[p] for p in phase.inherits}

        if skip_until and phase.name != from_phase:
            try:
                if not dry_run:
                    _reap_skipped_phase(experiment, phase)
                winners[phase.name] = _load_winner(experiment, phase, inherited)
                log.info("phase=%s SKIPPED (loaded compatible winner from disk)", phase.name)
            except FileNotFoundError:
                if not dry_run:
                    raise
                winners[phase.name] = _placeholder_winner(experiment, phase, inherited)
                log.info("phase=%s SKIPPED (DRY RUN placeholder)", phase.name)
            continue
        skip_until = False

        winner = _run_phase(
            experiment,
            phase,
            inherited,
            dry_run=dry_run,
            run_deadline=run_deadline,
        )
        if not dry_run:
            promoted = _apply_promotion(experiment, phase, winner, winners)
            if promoted is None:
                with contextlib.suppress(FileNotFoundError):
                    _winner_path(experiment, phase.name).unlink()
                break
            winner = promoted
            _save_winner(experiment, phase.name, winner)
            log.info(
                "phase=%s WINNER trial=%d metric=%g params=%s",
                phase.name,
                winner.trial_number,
                winner.metric,
                winner.params,
            )
        winners[phase.name] = winner

    if dry_run:
        log.info("DRY RUN complete. No trials launched, no summary written.")
        return winners

    summary_path = _summary_path(experiment)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": experiment.experiment,
        "metric": {"name": experiment.metric.name, "goal": experiment.metric.goal},
        "phases": [_winner_summary_item(pname, w) for pname, w in winners.items()],
    }
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    log.info("Wrote %s", summary_path)

    return winners


def _study_trial_counts(experiment: Experiment, phase: Phase) -> dict[str, int]:
    """Return Optuna trial counts by state without creating a missing study."""
    if experiment.storage is None:
        return {}
    try:
        study = optuna.load_study(
            study_name=f"{experiment.experiment}::{phase.name}",
            storage=_resolve_storage(experiment.storage),
        )
    except Exception:  # noqa: BLE001
        return {}
    counts: dict[str, int] = {}
    for trial in study.get_trials(deepcopy=False):
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
    return counts


def experiment_status(experiment: Experiment) -> dict[str, Any]:
    """Collect read-only status for one experiment config."""
    phases: list[dict[str, Any]] = []
    for phase in experiment.phases:
        winner_path = _winner_path(experiment, phase.name)
        counts = _study_trial_counts(experiment, phase)
        phases.append(
            {
                "name": phase.name,
                "winner": str(winner_path) if winner_path.is_file() else None,
                "trials": counts,
                "running": counts.get("RUNNING", 0),
            }
        )
    return {
        "kind": "experiment",
        "experiment": experiment.experiment,
        "workdir": str(_experiment_dir(experiment)),
        "phases": phases,
    }


def _suite_dir(suite: Suite) -> Path:
    """Filesystem namespace for suite-level summary/log artifacts."""
    return Path(suite.defaults.workdir).expanduser().resolve() / suite.suite


def _suite_summary_path(suite: Suite) -> Path:
    """Path to a suite-level summary."""
    return _suite_dir(suite) / "suite_summary.yaml"


def _suite_log_path(suite: Suite) -> Path:
    """Path to a suite-level run log."""
    return _suite_dir(suite) / "run.log"


@contextlib.contextmanager
def _suite_lock(suite: Suite) -> Iterator[None]:
    """Take a same-host lock for suite-level log and summary artifacts."""
    import fcntl

    material = {"kind": "suite", "suite_dir": str(_suite_dir(suite))}
    path = _lock_dir() / f"{suite.suite}__suite__{_lock_digest(material)}.lock"
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another phasesweep suite process appears to be using {suite.suite!r} "
                f"(lock file: {path})."
            ) from exc
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(handle, fcntl.LOCK_UN)


def _study_phase_winner(
    study_name: str,
    results: dict[str, dict[str, Winner]],
    selector: str,
) -> tuple[str, Winner]:
    """Resolve a suite promotion selector to a prior study winner."""
    if "." in selector:
        baseline_study, phase_name = selector.split(".", 1)
    else:
        baseline_study, phase_name = selector, ""
    if baseline_study not in results:
        raise RuntimeError(
            f"Study {study_name!r} promotion references unknown baseline study {baseline_study!r}."
        )
    study_winners = results[baseline_study]
    if not study_winners:
        raise RuntimeError(f"Baseline study {baseline_study!r} has no winners.")
    if phase_name:
        if phase_name not in study_winners:
            raise RuntimeError(
                f"Study {study_name!r} promotion references missing baseline phase {selector!r}."
            )
        return baseline_study, study_winners[phase_name]
    final_phase = next(reversed(study_winners))
    return f"{baseline_study}.{final_phase}", study_winners[final_phase]


def _apply_study_promotion(
    *,
    suite: Suite,
    study_name: str,
    experiment: Experiment,
    study_winners: dict[str, Winner],
    prior_results: dict[str, dict[str, Winner]],
) -> tuple[dict[str, Winner] | None, dict[str, Any] | None]:
    """Apply a suite study promotion rule against a prior study winner."""
    study_spec = next(study for study in suite.studies if study.name == study_name)
    promotion = study_spec.promotion
    if promotion is None:
        return study_winners, None
    if not study_winners:
        raise RuntimeError(f"Study {study_name!r} has no winner to promote.")

    baseline_label, baseline = _study_phase_winner(
        study_name,
        prior_results,
        promotion.min_delta_vs,
    )
    final_phase = next(reversed(study_winners))
    candidate = study_winners[final_phase]

    gate_passed = _gates_pass(candidate.gates)
    if promotion.requires_gates and not gate_passed:
        promoted = False
        improvement = None
    else:
        improvement = _metric_improvement(experiment.metric.goal, candidate, baseline)
        promoted = improvement >= promotion.min_delta

    decision: dict[str, Any] = {
        "study": study_name,
        "phase": final_phase,
        "baseline": baseline_label,
        "candidate_metric": candidate.metric,
        "baseline_metric": baseline.metric,
        "min_delta": promotion.min_delta,
        "improvement": improvement,
        "requires_gates": promotion.requires_gates,
        "gates_passed": gate_passed,
        "promoted": promoted,
        "on_fail": promotion.on_fail,
    }
    if promoted:
        log.info(
            "suite=%s study=%s PROMOTED improvement=%s baseline=%s min_delta=%g",
            suite.suite,
            study_name,
            improvement,
            baseline_label,
            promotion.min_delta,
        )
        return study_winners, decision

    message = (
        f"Study {study_name!r} failed promotion against {baseline_label!r}: "
        f"improvement {improvement!r}, min_delta {promotion.min_delta:g}, "
        f"gates_passed={gate_passed}."
    )
    if promotion.on_fail == "stop":
        raise RuntimeError(message)
    if promotion.on_fail == "skip":
        log.warning("%s Skipping this study for downstream dependencies.", message)
        return None, decision

    log.warning("%s Continuing with baseline winner.", message)
    exposed = dict(study_winners)
    exposed[final_phase] = _clone_winner_from_baseline(
        baseline,
        phase_fingerprint=candidate.phase_fingerprint,
    )
    return exposed, decision


def run_suite(suite: Suite, *, dry_run: bool = False) -> dict[str, dict[str, Winner]]:
    """Run every study in a suite in dependency order."""
    results: dict[str, dict[str, Winner]] = {}
    promotion_decisions: dict[str, dict[str, Any]] = {}
    if dry_run:
        for study_spec in suite.studies:
            experiment = suite.experiment_for_study(study_spec)
            results[study_spec.name] = run_experiment(experiment, dry_run=True)
        return results

    _suite_dir(suite).mkdir(parents=True, exist_ok=True)
    with _suite_lock(suite), _file_log_handler(_suite_log_path(suite)):
        for study_spec in suite.studies:
            for dep in study_spec.depends_on:
                if dep not in results:
                    raise RuntimeError(
                        f"Study {study_spec.name!r} dependency {dep!r} did not complete."
                    )
            experiment = suite.experiment_for_study(study_spec)
            log.info("suite=%s study=%s START", suite.suite, study_spec.name)
            study_winners = run_experiment(experiment, dry_run=False)
            exposed_winners, decision = _apply_study_promotion(
                suite=suite,
                study_name=study_spec.name,
                experiment=experiment,
                study_winners=study_winners,
                prior_results=results,
            )
            if decision is not None:
                promotion_decisions[study_spec.name] = decision
            if exposed_winners is not None:
                results[study_spec.name] = exposed_winners
            log.info("suite=%s study=%s COMPLETE", suite.suite, study_spec.name)

        summary = {
            "suite": suite.suite,
            "promotion_decisions": list(promotion_decisions.values()),
            "studies": [
                {
                    "name": study_name,
                    "promotion": promotion_decisions.get(study_name),
                    "phases": [
                        _winner_summary_item(phase_name, winner)
                        for phase_name, winner in study_winners.items()
                    ],
                }
                for study_name, study_winners in results.items()
            ],
        }
        _suite_summary_path(suite).write_text(yaml.safe_dump(summary, sort_keys=False))
    return results


def run_config(
    config: Config,
    *,
    from_phase: str | None = None,
    dry_run: bool = False,
) -> dict[str, Winner] | dict[str, dict[str, Winner]]:
    """Run an experiment or suite config."""
    if isinstance(config, Suite):
        if from_phase is not None:
            raise RuntimeError("--from-phase is only supported for single experiment configs.")
        return run_suite(config, dry_run=dry_run)
    return run_experiment(config, from_phase=from_phase, dry_run=dry_run)


def config_status(config: Config) -> dict[str, Any]:
    """Collect read-only status for an experiment or suite config."""
    if isinstance(config, Suite):
        return {
            "kind": "suite",
            "suite": config.suite,
            "workdir": str(_suite_dir(config)),
            "studies": [
                {
                    "name": study.name,
                    "depends_on": study.depends_on,
                    "status": experiment_status(config.experiment_for_study(study)),
                }
                for study in config.studies
            ],
        }
    return experiment_status(config)


__all__ = [
    "NoFeasibleTrialError",
    "Winner",
    "config_status",
    "experiment_status",
    "run_config",
    "run_experiment",
    "run_suite",
]
