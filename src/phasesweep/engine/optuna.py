"""Optuna sampler, storage, and study helpers."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import optuna

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
        return optuna.samplers.GridSampler(grid_search_space(search_space), seed=cfg.seed)
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


def _sqlite_trial_counts(experiment: Experiment, phase: Phase) -> dict[str, int]:
    """Return trial-state counts from a SQLite Optuna DB without creating schema.

    Status polling must be read-only. Passing a fresh SQLite URL through
    Optuna's storage constructor can create the database/schema and race the
    runner's first ``create_study`` call. Opening the file in SQLite read-only
    mode avoids both side effects: a missing, locked, or still-initializing DB
    simply reports no counts for now.

    :param Experiment experiment: Parsed experiment config containing the SQLite storage URL.
    :param Phase phase: Phase whose stable Optuna study name is counted.
    :return dict[str, int]: Trial counts keyed by Optuna state name, or an empty dict when the backing DB cannot be read safely.
    """
    assert experiment.storage is not None
    uri = sqlite_readonly_uri(experiment.storage)
    if uri is None:
        return {}
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
        return {}
    return {str(state): int(count) for state, count in rows}


def _phase_trial_counts(experiment: Experiment, phase: Phase) -> dict[str, int]:
    """Return Optuna trial counts by state without creating a missing study.

    :param Experiment experiment: Parsed experiment config containing storage settings.
    :param Phase phase: Phase whose existing study is inspected.
    :return dict[str, int]: Counts keyed by Optuna trial-state name.
    """
    if experiment.storage is None:
        return {}
    if storage_backend(experiment.storage) == "sqlite":
        return _sqlite_trial_counts(experiment, phase)
    try:
        study = _load_phase_study(experiment, phase)
    except Exception:  # noqa: BLE001
        return {}
    counts: dict[str, int] = {}
    for trial in study.get_trials(deepcopy=False):
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
    return counts
