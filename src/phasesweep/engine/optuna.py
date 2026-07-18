"""Optuna sampler, storage, and study helpers."""

from __future__ import annotations

import logging
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never

import optuna
from optuna.exceptions import ExperimentalWarning

from phasesweep.config import (
    CategoricalParam,
    Experiment,
    FloatParam,
    IntParam,
    Phase,
    Sampler,
    SearchParam,
    grid_search_space,
)
from phasesweep.runtime.files import file_url_path, sqlite_readonly_uri, storage_backend


@dataclass(frozen=True)
class _PhaseTrialStats:
    """One read-only storage snapshot for phase counts."""

    counts: dict[str, int]
    available: bool


def _build_sampler(
    cfg: Sampler, search_space: dict[str, SearchParam], n_jobs: int = 1
) -> optuna.samplers.BaseSampler:
    """Construct the Optuna sampler for a phase from its YAML ``sampler`` block.

    Args:
        cfg: Parsed sampler config (type, seed, startup-trials, etc.).
        search_space: The phase's validated search space, used to build the
            ``GridSampler`` grid.
        n_jobs: Phase parallelism; enables TPE's ``constant_liar`` heuristic
            when ``n_jobs > 1``.

    Returns:
        A configured :class:`optuna.samplers.BaseSampler` subclass instance.

    Raises:
        ValueError: The validated grid search space cannot be enumerated.

    """
    if cfg.type == "tpe":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=ExperimentalWarning,
                message=r"Argument ``constant_liar`` is an experimental feature.*",
            )
            return optuna.samplers.TPESampler(
                seed=cfg.seed,
                n_startup_trials=cfg.n_startup_trials,
                constant_liar=(n_jobs > 1),
            )
    if cfg.type == "random":
        return optuna.samplers.RandomSampler(seed=cfg.seed)
    if cfg.type == "grid":
        return optuna.samplers.GridSampler(grid_search_space(search_space), seed=cfg.seed)
    if cfg.type == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=cfg.seed)
    assert_never(cfg.type)


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
    assert_never(p)


log = logging.getLogger("phasesweep.engine.optuna")


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
    if storage_backend(url) == "journal":
        path = Path(file_url_path(url)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Using JournalFileStorage at %s", path)
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend

        return JournalStorage(JournalFileBackend(str(path)))
    return url


def _phase_study_name(experiment: Experiment, phase: Phase) -> str:
    """Return the stable Optuna study name for a phase.

    :param Experiment experiment: Parsed experiment config supplying the experiment name.
    :param Phase phase: Phase whose name is appended to the study namespace.
    :return str: Stable Optuna study name for the experiment/phase pair.
    """
    return f"{experiment.experiment}::{phase.name}"


def _study_direction(experiment: Experiment) -> str:
    """Return the Optuna direction for the experiment metric goal.

    :param Experiment experiment: Parsed experiment config containing the metric goal.
    :return str: ``"minimize"`` or ``"maximize"`` for Optuna.
    """
    return "minimize" if experiment.metric.goal == "minimize" else "maximize"


def _create_phase_study(
    experiment: Experiment,
    phase: Phase,
    *,
    dry_run: bool = False,
) -> optuna.Study:
    """Create or load the Optuna study for a phase.

    :param Experiment experiment: Parsed experiment config containing storage and metric settings.
    :param Phase phase: Phase whose sampler, search space, and study name are used.
    :param bool dry_run: If ``True``, force in-memory storage for the preview study.
    :return optuna.Study: Created or loaded Optuna study for the phase.
    """
    return optuna.create_study(
        study_name=_phase_study_name(experiment, phase),
        storage=None if dry_run else _resolve_storage(experiment.storage),
        sampler=_build_sampler(phase.sampler, phase.search_space, n_jobs=phase.n_jobs),
        pruner=optuna.pruners.NopPruner(),
        direction=_study_direction(experiment),
        load_if_exists=True,
    )


def _load_phase_study(experiment: Experiment, phase: Phase) -> optuna.Study:
    """Load an existing persistent Optuna study for a phase.

    :param Experiment experiment: Parsed experiment config containing storage settings.
    :param Phase phase: Phase whose stable study name is loaded.
    :return optuna.Study: Existing Optuna study for the phase.
    """
    return optuna.load_study(
        study_name=_phase_study_name(experiment, phase),
        storage=_resolve_storage(experiment.storage),
    )


def _sqlite_study_exists(experiment: Experiment, phase: Phase) -> bool:
    """Return whether a SQLite storage already contains the phase study.

    :param Experiment experiment: Parsed experiment config with SQLite storage.
    :param Phase phase: Phase whose stable study name should be checked.
    :return bool: ``True`` only when the backing database is readable and contains the study.
    """
    assert experiment.storage is not None
    uri = sqlite_readonly_uri(experiment.storage)
    if uri is None:
        return False
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=0.1)
        try:
            row = conn.execute(
                "SELECT 1 FROM studies WHERE study_name = ? LIMIT 1",
                (_phase_study_name(experiment, phase),),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return row is not None


def _load_existing_phase_study(experiment: Experiment, phase: Phase) -> optuna.Study | None:
    """Load a phase study only if it already exists.

    Recovery and read-like paths must not call ``create_study(load_if_exists=True)`` because
    that can create an empty study and make missing evidence look safe. This helper checks
    file-backed storage before delegating to Optuna's loader, then treats an absent study as
    ``None``.

    :param Experiment experiment: Parsed experiment config containing storage settings.
    :param Phase phase: Phase whose stable study name should be loaded.
    :return optuna.Study | None: Existing study, or ``None`` when no durable study exists.
    """
    if experiment.storage is None:
        return None
    backend = storage_backend(experiment.storage)
    if backend == "sqlite" and not _sqlite_study_exists(experiment, phase):
        return None
    if backend == "journal" and not Path(file_url_path(experiment.storage)).expanduser().exists():
        return None
    try:
        return _load_phase_study(experiment, phase)
    except KeyError:
        return None


def _sqlite_phase_trial_stats(experiment: Experiment, phase: Phase) -> _PhaseTrialStats:
    """Return trial-state counts in one SQLite read.

    Status polling must be read-only. Passing a fresh SQLite URL through
    Optuna's storage constructor can create the database/schema and race the
    runner's first ``create_study`` call. Opening the file in SQLite read-only
    mode avoids both side effects: a missing, locked, or still-initializing DB
    simply reports no counts for now.

    :param Experiment experiment: Parsed experiment config containing the SQLite storage URL.
    :param Phase phase: Phase whose stable Optuna study name is counted.
    :return _PhaseTrialStats: Counts and an availability flag; counts are empty
        and availability false when the DB cannot be read safely.
    """
    assert experiment.storage is not None
    uri = sqlite_readonly_uri(experiment.storage)
    if uri is None:
        return _PhaseTrialStats({}, False)
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=0.1)
        try:
            rows = conn.execute(
                """
                SELECT trials.state, COUNT(*)
                FROM trials
                JOIN studies ON trials.study_id = studies.study_id
                WHERE studies.study_name = ?
                GROUP BY trials.state
                """,
                (_phase_study_name(experiment, phase),),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return _PhaseTrialStats({}, False)
    return _PhaseTrialStats({str(state): int(count) for state, count in rows}, True)


def _phase_trial_stats(experiment: Experiment, phase: Phase) -> _PhaseTrialStats:
    """Read counts without creating a missing study.

    :param Experiment experiment: Parsed experiment config containing storage settings.
    :param Phase phase: Phase whose existing study is inspected.
    :return _PhaseTrialStats: One permissive storage snapshot with explicit availability.
    """
    if experiment.storage is None:
        return _PhaseTrialStats({}, False)
    backend = storage_backend(experiment.storage)
    if backend == "sqlite":
        return _sqlite_phase_trial_stats(experiment, phase)
    if backend == "journal" and not Path(file_url_path(experiment.storage)).expanduser().exists():
        return _PhaseTrialStats({}, False)
    try:
        study = _load_phase_study(experiment, phase)
        trials = study.get_trials(deepcopy=False)
    except Exception:  # noqa: BLE001
        return _PhaseTrialStats({}, False)
    counts: dict[str, int] = {}
    for trial in trials:
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
    return _PhaseTrialStats(counts, True)


def _phase_trial_counts(experiment: Experiment, phase: Phase) -> dict[str, int]:
    """Return Optuna trial counts by state without creating a missing study.

    :param Experiment experiment: Parsed experiment config containing storage settings.
    :param Phase phase: Phase whose existing study is inspected.
    :return dict[str, int]: Counts keyed by Optuna trial-state name.
    """
    return _phase_trial_stats(experiment, phase).counts
