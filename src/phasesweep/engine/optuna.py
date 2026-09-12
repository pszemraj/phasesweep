"""Optuna sampler, storage, and study helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, assert_never

import optuna
import sqlalchemy
from optuna.exceptions import ExperimentalWarning
from optuna.storages import JournalStorage
from optuna.storages.journal import BaseJournalBackend

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
from phasesweep.engine.errors import StudyStorageUnavailableError
from phasesweep.engine.state import ATTEMPT_ID_ATTR, GENERATION_ID_ATTR
from phasesweep.runtime.files import (
    file_url_path,
    sqlite_database_path,
    sqlite_readonly_uri,
    storage_backend,
)


@dataclass(frozen=True)
class _TrialRef:
    """Identity of one trial recorded in a publication or storage snapshot.

    Reconciles a row against published evidence or external cleanup evidence:
    which trial it is, and which generation/attempt claimed it.
    Values are ``None`` when the trial never recorded that attribute (e.g. a
    row written by an older PhaseSweep). Published references additionally
    carry their terminal and complete count boundary. Those count fields are
    not part of a trial-row identity comparison.
    """

    trial_number: int
    generation_id: str | None
    attempt_id: str | None
    finished_trials: int | None = field(default=None, compare=False)
    completed_trials: int | None = field(default=None, compare=False)


@dataclass(frozen=True)
class _PhaseTrialStats:
    """One read-only storage snapshot for phase counts.

    ``running_attempts`` is part of the *same* snapshot as ``counts``, not a
    second read: it lists the RUNNING trials those counts describe, and is
    ``None`` exactly when ``available`` is ``False`` (the counts are unknown).
    A confirmed absent study has known zero counts and an empty attempt list.
    ``published_trial_available`` checks the requested published trial identity
    in that same snapshot; false also covers unreadable or absent storage.
    """

    counts: dict[str, int]
    available: bool
    generation_counts: dict[str, dict[str, int]]
    running_attempts: list[_TrialRef] | None
    published_trial_available: bool = False


def _published_phase_trial_refs(
    summary: Mapping[str, Any] | None,
) -> dict[str, _TrialRef | None]:
    """Return each published phase's local winning or promotion-candidate identity.

    A continued baseline belongs to an earlier phase. Its promotion record
    identifies the local candidate whose history must survive in this phase.

    :param Mapping[str, Any] | None summary: Already-resolved publication summary.
    :return dict[str, _TrialRef | None]: Local trial identities by phase, with
        their known terminal-count boundary; None when a published phase does
        not record a complete identity.
    """
    refs: dict[str, _TrialRef | None] = {}
    for item in (summary or {}).get("phases", ()):
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            continue
        promotion = item.get("promotion")
        source = promotion if isinstance(promotion, Mapping) else item
        prefix = "candidate_" if isinstance(promotion, Mapping) else ""
        number = source.get(f"{prefix}trial_number")
        generation = source.get(f"{prefix}generation_id")
        attempt = source.get(f"{prefix}attempt_id")
        completion = item.get("completion")
        finished_trials = (
            completion.get("finished_trials")
            if isinstance(completion, Mapping)
            and type(completion.get("finished_trials")) is int
            and completion["finished_trials"] >= 0
            else None
        )
        completed_trials = (
            completion.get("completed_trials")
            if isinstance(completion, Mapping)
            and type(completion.get("completed_trials")) is int
            and completion["completed_trials"] >= 0
            else None
        )
        refs[item["name"]] = (
            _TrialRef(number, generation, attempt, finished_trials, completed_trials)
            if isinstance(number, int)
            and not isinstance(number, bool)
            and isinstance(generation, str)
            and generation
            and isinstance(attempt, str)
            and attempt
            else None
        )
    return refs


def _published_trial_history_available(
    stats: _PhaseTrialStats, published_trial: _TrialRef | None
) -> bool:
    """Return whether one storage snapshot retains a published trial history.

    :param _PhaseTrialStats stats: Counts and identity result from one storage read.
    :param _TrialRef | None published_trial: Published trial and count boundary.
    :return bool: ``True`` when identity and every recorded count boundary survive.
    """
    if not stats.available or not stats.published_trial_available or published_trial is None:
        return False
    finished = sum(stats.counts.get(state, 0) for state in ("COMPLETE", "PRUNED", "FAIL"))
    completed = stats.counts.get("COMPLETE", 0)
    return (
        published_trial.finished_trials is None or finished >= published_trial.finished_trials
    ) and (
        published_trial.completed_trials is None or completed >= published_trial.completed_trials
    )


def _published_trial_matches(trial: optuna.trial.FrozenTrial, expected: _TrialRef) -> bool:
    """Return whether a completed trial has the published identity.

    :param optuna.trial.FrozenTrial trial: Ledger trial to compare.
    :param _TrialRef expected: Published trial number, generation, and attempt identity.
    :return bool: ``True`` when the trial is complete and all identity fields match.
    """
    return (
        trial.state == optuna.trial.TrialState.COMPLETE
        and trial.number == expected.trial_number
        and trial.user_attrs.get(GENERATION_ID_ATTR) == expected.generation_id
        and trial.user_attrs.get(ATTEMPT_ID_ATTR) == expected.attempt_id
    )


class _TrialNumberRandomSampler(optuna.samplers.RandomSampler):
    """Seeded random sampler whose draws survive process-local RNG restarts."""

    def __init__(self, *, seed: int) -> None:
        """Store the caller's seed without handing it to the base sampler.

        The base ``RandomSampler`` is constructed with ``seed=None`` because
        this sampler never draws through it directly; ``seed`` is instead
        mixed into a fresh per-draw seed by :meth:`sample_independent`, so
        continuation across process restarts stays deterministic.

        Args:
            seed: Base seed supplied by the phase's sampler config.

        """
        super().__init__(seed=None)
        self._base_seed = seed

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        """Sample from a deterministic stream position owned by the durable trial.

        Derives a fresh per-parameter seed from the base seed, study name,
        trial number, and parameter name, so the same (trial, param) pair
        always draws the same value regardless of process restarts or call
        ordering.

        Args:
            study: The active Optuna study (contributes ``study_name`` to the
                derived seed).
            trial: The trial being sampled for (contributes ``trial.number``).
            param_name: Name of the parameter being sampled.
            param_distribution: The parameter's Optuna distribution.

        Returns:
            The sampled value, drawn from a fresh :class:`optuna.samplers.RandomSampler`
            seeded deterministically from ``(base_seed, study_name, trial.number, param_name)``.

        """
        material = json.dumps(
            [self._base_seed, study.study_name, trial.number, param_name],
            separators=(",", ":"),
        ).encode()
        derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
        sampler = optuna.samplers.RandomSampler(seed=derived_seed)
        return sampler.sample_independent(study, trial, param_name, param_distribution)


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
        if cfg.seed is None:
            return optuna.samplers.RandomSampler()
        return _TrialNumberRandomSampler(seed=cfg.seed)
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

    ``Experiment.resolved_storage`` selects the URL for ``storage: auto`` before
    this helper is called. Explicit SQLite URLs are never rewritten; validation
    rejects those with parallel jobs. Journal URLs also accept escaped
    ``file:`` filenames with ``uri=true``, as generated by auto storage.

    Args:
        url: The resolved storage URL, or ``None`` for in-memory.

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
        storage=None if dry_run else _resolve_storage(experiment.resolved_storage),
        sampler=_build_sampler(phase.sampler, phase.search_space, n_jobs=phase.n_jobs),
        pruner=optuna.pruners.NopPruner(),
        direction=experiment.metric.goal,
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
        storage=_resolve_storage(experiment.resolved_storage),
    )


def _sqlite_study_exists(experiment: Experiment, phase: Phase) -> bool:
    """Return whether a SQLite storage already contains the phase study.

    Strict and tri-state on purpose (PR #5 review / reviewer 2, issue 1):
    callers use this verdict to decide whether root-binding and recovery
    guards apply, so "the database could not be read" must never collapse
    into "the study does not exist" -- a briefly locked file would then skip
    the artifact-root check for a study that becomes readable one call later.
    Only two conditions report absence: the database file does not exist, or
    it exists without Optuna's schema (nothing ever created a study in it).
    Every other read failure raises. Observational polling keeps its tolerant
    reader (:func:`_sqlite_phase_trial_stats`), where ``available: false`` is
    part of the contract.

    :param Experiment experiment: Parsed experiment config with SQLite storage.
    :param Phase phase: Phase whose stable study name should be checked.
    :return bool: ``True`` when the database contains the study, ``False``
        when the database or its schema does not exist.
    :raises StudyStorageUnavailableError: The database file exists but could
        not be read (locked, corrupt, permission-denied, ...), so whether the
        study exists cannot be determined.
    """
    assert experiment.resolved_storage is not None
    uri = sqlite_readonly_uri(experiment.resolved_storage)
    database = sqlite_database_path(experiment.resolved_storage)
    if uri is None or database is None or not database.exists():
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
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            # The file exists but holds no Optuna schema: nothing ever
            # created a study in it, which is genuine absence.
            return False
        raise StudyStorageUnavailableError(
            f"SQLite storage {database} exists but could not be read while checking for "
            f"study {_phase_study_name(experiment, phase)!r}."
        ) from exc
    except sqlite3.Error as exc:
        raise StudyStorageUnavailableError(
            f"SQLite storage {database} exists but could not be read while checking for "
            f"study {_phase_study_name(experiment, phase)!r}."
        ) from exc
    return row is not None


def _rdb_study_exists(experiment: Experiment, phase: Phase) -> bool:
    """Return whether external RDB storage contains a phase study.

    :param Experiment experiment: Parsed experiment with external RDB storage.
    :param Phase phase: Phase whose stable study name should be checked.
    :return bool: ``True`` when the storage contains the named study, ``False`` when its
        Optuna ``studies`` table does not exist or contains no matching row.
    :raises StudyStorageUnavailableError: The storage or its study table could not be read.
    """
    assert experiment.resolved_storage is not None
    try:
        engine = sqlalchemy.create_engine(experiment.resolved_storage)
        try:
            if not sqlalchemy.inspect(engine).has_table("studies"):
                return False
            with engine.connect() as connection:
                row = connection.execute(
                    sqlalchemy.text("SELECT 1 FROM studies WHERE study_name = :study_name"),
                    {"study_name": _phase_study_name(experiment, phase)},
                ).fetchone()
        finally:
            engine.dispose()
    except Exception as exc:  # noqa: BLE001 - mutating preflight must fail closed
        raise StudyStorageUnavailableError(
            f"External RDB storage could not be read while checking for study "
            f"{_phase_study_name(experiment, phase)!r}."
        ) from exc
    return row is not None


@dataclass
class _JournalSnapshot(BaseJournalBackend):
    """Replay one captured journal without rereading or writing the live file."""

    logs: list[dict[str, Any]]

    def read_logs(self, log_number_from: int) -> Iterable[dict[str, Any]]:
        """Return snapshot records starting at the requested log number.

        :param int log_number_from: First record to replay.
        :return Iterable[dict[str, Any]]: Captured records from that position.
        """
        return self.logs[log_number_from:]

    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        """Reject writes through an observational journal snapshot.

        :param list[dict[str, Any]] logs: Records the caller attempted to append.
        :raises RuntimeError: Always; the snapshot is read-only.
        """
        raise RuntimeError("A journal snapshot is read-only.")


def _load_journal_study_snapshot(storage_url: str, study_name: str) -> optuna.Study | None:
    """Replay a complete journal snapshot before deciding whether a study exists.

    Optuna tolerates an undecodable final record while another writer appends.
    Such a snapshot is incomplete, so inspection cannot claim known counts or
    confirmed absence. Replay also must finish before a missing-study KeyError
    can be distinguished from a malformed journal operation.

    :param str storage_url: Journal storage URL to inspect.
    :param str study_name: Named study to load from the captured records.
    :return optuna.Study | None: Read-only snapshot study, or confirmed absence.
    :raises StudyStorageUnavailableError: The snapshot is unreadable or incomplete.
    """
    path = Path(file_url_path(storage_url)).expanduser()
    try:
        try:
            with path.open("rb") as source:
                size = os.fstat(source.fileno()).st_size
                data = source.read(size)
        except FileNotFoundError:
            return None
        if len(data) != size:
            raise ValueError("The journal was truncated while its snapshot was being read.")
        if data and not data.endswith(b"\n"):
            raise ValueError("The journal ends with an incomplete record.")
        records = [json.loads(line) for line in data.split(b"\n")[:-1]]
        storage = JournalStorage(_JournalSnapshot(records))
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Journal storage {path} could not be completely read while checking for "
            f"study {study_name!r}."
        ) from exc
    try:
        return optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        return None


def _load_existing_phase_study(experiment: Experiment, phase: Phase) -> optuna.Study | None:
    """Load a phase study only if it already exists.

    Recovery and read-like paths must not call ``create_study(load_if_exists=True)`` because
    that can create an empty study and make missing evidence look safe. This helper checks
    persistent storage before delegating to Optuna's loader, then treats an absent study as
    ``None``.

    :param Experiment experiment: Parsed experiment config containing storage settings.
    :param Phase phase: Phase whose stable study name should be loaded.
    :return optuna.Study | None: Existing study, or ``None`` when no durable study exists.
    :raises StudyStorageUnavailableError: Persistent storage exists but
        could not be read, so whether the study exists cannot be determined;
        callers on mutating paths must abort rather than treat this as absence.
    """
    if experiment.resolved_storage is None:
        return None
    backend = storage_backend(experiment.resolved_storage)
    if backend == "sqlite":
        if not _sqlite_study_exists(experiment, phase):
            return None
    elif backend == "journal":
        if (
            _load_journal_study_snapshot(
                experiment.resolved_storage, _phase_study_name(experiment, phase)
            )
            is None
        ):
            return None
        # Preflight verified a complete snapshot. Mutating callers still use
        # Optuna's normal backend, including its concurrent-append semantics.
        return _load_phase_study(experiment, phase)
    elif not _rdb_study_exists(experiment, phase):
        # Optuna initializes its schema even when load_study finds no study.
        return None
    try:
        return _load_phase_study(experiment, phase)
    except KeyError:
        return None


def _decoded_string_attr(value_json: object) -> str | None:
    """Decode one JSON-encoded trial user attribute as a string.

    :param object value_json: Raw ``trial_user_attributes.value_json`` cell.
    :return str | None: The decoded value when it is a string; ``None`` for a
        missing, unparsable, or non-string attribute.
    """
    if not isinstance(value_json, str):
        return None
    try:
        value = json.loads(value_json)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, str) else None


_PHASE_TRIAL_STATS_SQL = """
    WITH phase_trials AS (
        SELECT trials.number,
               trials.state,
               generation.value_json AS generation_json,
               attempt.value_json AS attempt_json
        FROM trials
        JOIN studies ON trials.study_id = studies.study_id
        LEFT JOIN trial_user_attributes AS generation
          ON trials.trial_id = generation.trial_id AND generation.key = :generation_key
        LEFT JOIN trial_user_attributes AS attempt
          ON trials.trial_id = attempt.trial_id AND attempt.key = :attempt_key
        WHERE studies.study_name = :study_name
    )
    SELECT 'count', NULL, state, generation_json, NULL, COUNT(*)
    FROM phase_trials
    GROUP BY state, generation_json
    UNION ALL
    SELECT 'running', number, state, generation_json, attempt_json, 1
    FROM phase_trials
    WHERE state = 'RUNNING'
    UNION ALL
    SELECT 'published', number, state, generation_json, attempt_json, 1
    FROM phase_trials
    WHERE number = :published_trial_number
"""


def _phase_trial_stats_params(
    experiment: Experiment, phase: Phase, published_trial: _TrialRef | None
) -> dict[str, str | int]:
    """Return bind parameters for the observational phase-trial SQL query.

    :param Experiment experiment: Config providing the phase's stable study name.
    :param Phase phase: Phase whose study is queried.
    :param _TrialRef | None published_trial: Published local trial whose number is included,
        or ``None`` to bind ``published_trial_number`` to ``-1``.
    :return dict[str, str | int]: Attribute keys, study name, and published trial number;
        only the trial number from ``published_trial`` is used.
    """
    return {
        "generation_key": GENERATION_ID_ATTR,
        "attempt_key": ATTEMPT_ID_ATTR,
        "study_name": _phase_study_name(experiment, phase),
        "published_trial_number": published_trial.trial_number if published_trial else -1,
    }


def _unavailable_phase_trial_stats(
    experiment: Experiment, phase: Phase, exc: BaseException
) -> _PhaseTrialStats:
    """Log a storage-read failure and return an unavailable phase-trial snapshot.

    External RDB URLs and driver messages can contain credentials, so those
    warnings report only the backend and exception type.

    :param Experiment experiment: Config whose storage backend or local path is reported.
    :param Phase phase: Phase whose status read failed.
    :param BaseException exc: Read exception whose direct cause is reported when present.
    :return _PhaseTrialStats: Empty count maps and ``running_attempts=None`` with
        ``available=False``.
    """
    cause = exc.__cause__ or exc
    backend = storage_backend(experiment.resolved_storage)
    if backend in {"sqlite", "journal"}:
        log.warning(
            "could not read status trial data from storage %s for phase %s: %s: %s",
            experiment.resolved_storage,
            phase.name,
            type(cause).__name__,
            cause,
        )
    else:
        log.warning(
            "could not read status trial data from %s storage for phase %s: %s",
            backend,
            phase.name,
            type(cause).__name__,
        )
    return _PhaseTrialStats({}, False, {}, None)


def _sqlite_phase_trial_stats(
    experiment: Experiment, phase: Phase, published_trial: _TrialRef | None = None
) -> _PhaseTrialStats:
    """Return trial-state counts and RUNNING identities in one SQLite read.

    Status polling must be read-only. Passing a fresh SQLite URL through
    Optuna's storage constructor can create the database/schema and race the
    runner's first ``create_study`` call. Opening the file in SQLite read-only
    mode avoids both side effects: a missing, locked, or still-initializing DB
    simply reports no counts for now. A confirmed missing file has known zero
    counts; a failed read, including an unreadable schema, reports unavailable counts.

    Counts and RUNNING identities come from one CTE-backed statement, so one
    storage snapshot. SQL aggregates terminal rows by state and generation,
    while the ``UNION ALL`` arms return RUNNING identities and the requested
    published trial. This
    avoids transferring every historical trial on every status poll without
    splitting the read into two snapshots that could disagree during a live
    write (PR #5 review / reviewer 2, blocker 6).

    :param Experiment experiment: Parsed experiment config containing the SQLite storage URL.
    :param Phase phase: Phase whose stable Optuna study name is counted.
    :param _TrialRef | None published_trial: Published local trial to verify in this snapshot.
    :return _PhaseTrialStats: Counts, RUNNING identities, and an availability
        flag; counts are empty, running attempts ``None``, and availability
        false when the DB cannot be read safely.
    """
    assert experiment.resolved_storage is not None
    uri = sqlite_readonly_uri(experiment.resolved_storage)
    if uri is None:
        return _PhaseTrialStats({}, False, {}, None)
    database = sqlite_database_path(experiment.resolved_storage)
    assert database is not None
    try:
        database.stat()
    except FileNotFoundError:
        return _PhaseTrialStats({}, True, {}, [])
    except OSError as exc:
        return _unavailable_phase_trial_stats(experiment, phase, exc)
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=0.1)
        try:
            rows = conn.execute(
                _PHASE_TRIAL_STATS_SQL,
                _phase_trial_stats_params(experiment, phase, published_trial),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return _unavailable_phase_trial_stats(experiment, phase, exc)
    return _trial_stats_from_rows(
        rows, study_name=_phase_study_name(experiment, phase), published_trial=published_trial
    )


def _rdb_phase_trial_stats(
    experiment: Experiment, phase: Phase, published_trial: _TrialRef | None = None
) -> _PhaseTrialStats:
    """Inspect external SQL storage without Optuna's schema-initializing loader.

    :param Experiment experiment: Config containing the external storage URL.
    :param Phase phase: Phase whose trial counts and running identities are read.
    :param _TrialRef | None published_trial: Published local trial to verify in this snapshot.
    :return _PhaseTrialStats: One snapshot, or an unavailable observation on read failure.
    """
    assert experiment.resolved_storage is not None
    try:
        engine = sqlalchemy.create_engine(experiment.resolved_storage)
        try:
            with engine.connect() as connection:
                rows = connection.execute(
                    sqlalchemy.text(_PHASE_TRIAL_STATS_SQL),
                    _phase_trial_stats_params(experiment, phase, published_trial),
                ).fetchall()
        finally:
            engine.dispose()
    except Exception as exc:  # noqa: BLE001 - status reports unavailable on any connection/read failure
        return _unavailable_phase_trial_stats(experiment, phase, exc)
    return _trial_stats_from_rows(
        rows, study_name=_phase_study_name(experiment, phase), published_trial=published_trial
    )


def _trial_stats_from_rows(
    rows: Iterable[Sequence[Any]], *, study_name: str, published_trial: _TrialRef | None = None
) -> _PhaseTrialStats:
    """Decode aggregated counts and RUNNING identities from a SQL snapshot.

    :param Iterable[Sequence[Any]] rows: Rows from the observational SQL query.
    :param str study_name: Study name used in damaged-row diagnostics.
    :param _TrialRef | None published_trial: Expected local trial from the publication.
    :return _PhaseTrialStats: Counts and running identities from the supplied rows.
    """
    counts: dict[str, int] = {}
    generation_counts: dict[str, dict[str, int]] = {}
    running_attempts: list[_TrialRef] = []
    published_trial_available = False
    for row_kind, number, state, generation_json, attempt_json, tally in rows:
        state_name = str(state)
        generation_id = _decoded_string_attr(generation_json)
        if row_kind == "count":
            count = int(tally)
            counts[state_name] = counts.get(state_name, 0) + count
            if generation_id is not None:
                states = generation_counts.setdefault(generation_id, {})
                states[state_name] = states.get(state_name, 0) + count
            continue
        if row_kind == "published":
            published_trial_available = (
                state_name == "COMPLETE"
                and _TrialRef(number, generation_id, _decoded_string_attr(attempt_json))
                == published_trial
            )
            continue
        if row_kind != "running":
            continue
        if not isinstance(number, int):
            # Optuna assigns ``number`` in the same INSERT that creates the
            # row, so a missing one is a damaged row rather than a race with
            # a live writer. It still counts as RUNNING; it just has no
            # identity to report.
            log.warning(
                "study %s has a RUNNING trial with no trial number; "
                "omitting it from the reported running attempts",
                study_name,
            )
            continue
        running_attempts.append(
            _TrialRef(
                trial_number=number,
                generation_id=generation_id,
                attempt_id=_decoded_string_attr(attempt_json),
            )
        )
    return _PhaseTrialStats(
        counts, True, generation_counts, running_attempts, published_trial_available
    )


def _phase_trial_stats(
    experiment: Experiment, phase: Phase, published_trial: _TrialRef | None = None
) -> _PhaseTrialStats:
    """Read counts and RUNNING identities without creating a missing study.

    SQL backends use one SELECT statement; journal storage uses one trial list.
    Counts, RUNNING identities, and published trial verification therefore
    describe the same snapshot.
    Confirmed absence reports available zero counts; read failures report
    unavailable counts.

    :param Experiment experiment: Parsed experiment config containing storage settings.
    :param Phase phase: Phase whose existing study is inspected.
    :param _TrialRef | None published_trial: Published local trial to verify in this snapshot.
    :return _PhaseTrialStats: One permissive storage snapshot with explicit availability.
    """
    if experiment.resolved_storage is None:
        return _PhaseTrialStats({}, False, {}, None)
    backend = storage_backend(experiment.resolved_storage)
    if backend == "sqlite":
        return _sqlite_phase_trial_stats(experiment, phase, published_trial)
    if backend != "journal":
        return _rdb_phase_trial_stats(experiment, phase, published_trial)
    try:
        study = _load_journal_study_snapshot(
            experiment.resolved_storage, _phase_study_name(experiment, phase)
        )
        if study is None:
            return _PhaseTrialStats({}, True, {}, [])
        trials = study.get_trials(deepcopy=False)
    except Exception as exc:  # noqa: BLE001
        return _unavailable_phase_trial_stats(experiment, phase, exc)
    counts: dict[str, int] = {}
    generation_counts: dict[str, dict[str, int]] = {}
    running_attempts: list[_TrialRef] = []
    published_trial_available = False
    for trial in trials:
        if published_trial is not None and _published_trial_matches(trial, published_trial):
            published_trial_available = True
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
        generation_id = trial.user_attrs.get(GENERATION_ID_ATTR)
        if isinstance(generation_id, str):
            states = generation_counts.setdefault(generation_id, {})
            states[trial.state.name] = states.get(trial.state.name, 0) + 1
        if trial.state.name != "RUNNING":
            continue
        attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
        running_attempts.append(
            _TrialRef(
                trial_number=trial.number,
                generation_id=generation_id if isinstance(generation_id, str) else None,
                attempt_id=attempt_id if isinstance(attempt_id, str) else None,
            )
        )
    return _PhaseTrialStats(
        counts, True, generation_counts, running_attempts, published_trial_available
    )
