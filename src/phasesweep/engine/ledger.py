"""The single storage chokepoint: every Optuna/SQLite storage object is built here.

No other module in ``phasesweep`` may name ``optuna.create_study``,
``optuna.load_study``, ``RDBStorage``, ``JournalFileBackend``, or
``sqlite3.connect``; ``tests/test_ledger_contract.py`` enforces that statically.
Concentrating the constructors here is what makes the durability invariants
checkable, because every path that touches a ledger has to come through this
module and therefore through its fixed order:

1. Take the experiment lock before touching any durable state.
2. Validate read-only before writing anything: the artifact-root binding check
   first (no writes, no storage), then the ledger format scan, then the tree
   binding is written, then existing studies are claimed, and only then is live
   file-backed storage opened.
3. Pure read paths stop after step 2's scan: they never create a study and
   never construct file-backed storage.

Read-only inspection uses SQLite ``mode=ro`` URIs and replayed journal
snapshots (:class:`_JournalSnapshot`), never a live backend, so a status poll
can never initialize, stamp, or recover a ledger it was only meant to observe.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import BaseJournalBackend

from phasesweep.config import Experiment, Phase
from phasesweep.engine.artifact_roots import (
    BindingState,
    _artifact_root_binding_applies,
    _artifact_root_claim_needed,
    _artifact_root_identity,
    _bind_study_artifact_root,
    _check_artifact_root_binding,
    _check_published_phase_studies,
    _claim_study_artifact_root,
    _write_artifact_root_binding,
)
from phasesweep.engine.errors import (
    ArtifactRootConflictError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
)
from phasesweep.engine.optuna import (
    _build_sampler,
    _phase_study_name,
    _PhaseTrialStats,
    _published_trial_matches,
    _TrialRef,
)
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GENERATION_ID_ATTR,
    STUDY_SCHEMA_ATTR,
    STUDY_SCHEMA_VERSION,
)
from phasesweep.runtime.files import (
    file_url_path,
    sqlite_database_path,
    sqlite_readonly_uri,
    storage_backend,
    storage_is_in_memory,
)

__all__ = [
    "ClaimedLedger",
    "ValidatedLedger",
    "claim_ledger",
    "open_existing_study",
    "open_phase_study",
    "open_preview_study",
    "open_registry_study",
    "read_phase_trial_stats",
    "validate_ledger",
]

log = logging.getLogger(__name__)

#: Which durable form a ledger takes. ``"memory"`` is deliberate no-ledger
#: execution, so nothing about it is on disk to validate, bind, or reopen.
Backend = Literal["memory", "sqlite", "journal"]


@dataclass(frozen=True, slots=True)
class ValidatedLedger:
    """A ledger whose binding and format passed, in that order, writing nothing.

    Holding one of these is the proof that step two of the fixed order ran:
    :func:`validate_ledger` checked the artifact-root binding first and only
    then scanned the ledger's format, and it did both without creating a study,
    constructing file-backed storage, or writing a byte. Read paths need
    nothing more, so this handle is all they take.

    It is deliberately *not* enough to open a live study for writing. That
    needs a bound tree and root-claimed studies, which only a
    :class:`ClaimedLedger` records; the type split is what stops a read path
    from reaching an opener that mutates.
    """

    #: The exact config this handle was validated for. Study names, samplers,
    #: and refusal messages all derive from it, so carrying it is what lets every
    #: opener take the handle alone: no caller can pair a validated ledger with
    #: a different config than the one that passed.
    experiment: Experiment
    experiment_name: str
    storage_url: str | None
    backend: Backend
    ledger_path: Path | None
    artifact_root: str
    binding_state: BindingState


@dataclass(frozen=True, slots=True)
class ClaimedLedger(ValidatedLedger):
    """A validated ledger whose tree is bound and whose existing studies are claimed.

    Holding one of these is the proof that the fixed order ran through its
    last write: :func:`claim_ledger` took a :class:`ValidatedLedger`, found
    every existing phase study in one strict pass, wrote the tree's ownership
    record, and only then claimed each study's artifact root. It is the only
    handle :func:`open_phase_study` accepts, so live storage cannot be opened
    for writing before that order completes.
    """

    #: Every existing declared phase study, keyed by phase name, from the one
    #: strict discovery pass :func:`claim_ledger` made (phases with no durable
    #: study yet are absent). Preflight reaps and validates exactly these
    #: objects, so a second, luckier read can never hand recovery a study whose
    #: root was never checked.
    studies: Mapping[str, optuna.Study]


def _resolve_storage(url: str | None) -> Any:
    """Translate a storage URL into an Optuna storage object or pass through.

    Recognized schemes:
      * Any URL recognized by :func:`storage_is_in_memory` -> in-memory study
        (not resumable).
      * ``journal:///path.journal`` -> Optuna ``JournalStorage(JournalFileBackend(path))``.
        Safe for parallel ``n_jobs`` on a single host.
      * ``sqlite:///path.db`` -> passed to Optuna unchanged.

    ``Experiment.resolved_storage`` selects the URL for ``storage: auto`` before
    this helper is called. Explicit SQLite URLs are never rewritten; validation
    rejects those with parallel jobs. Journal URLs also accept escaped
    ``file:`` filenames with ``uri=true``, as generated by auto storage.

    The journal's parent directory is *not* created here. Resolving a URL is
    something read paths do too, and a read must never bring a ledger
    directory into existence; only the live opener
    (:func:`open_phase_study`) prepares the directory first.

    Args:
        url: The resolved storage URL, or an in-memory sentinel.

    Returns:
        ``None`` for in-memory; a configured ``JournalStorage`` for the
        ``journal:///`` scheme; the SQLite URL unchanged otherwise.

    """
    if url is None or storage_is_in_memory(url):
        return None
    if storage_backend(url) == "journal":
        path = Path(file_url_path(url)).expanduser()
        log.info("Using JournalFileStorage at %s", path)
        from optuna.storages.journal import JournalFileBackend

        return JournalStorage(JournalFileBackend(str(path)))
    return url


def _build_phase_study(experiment: Experiment, phase: Phase, storage: Any) -> optuna.Study:
    """Create or load a phase's Optuna study in already-resolved storage.

    The one ``create_study`` call both openers share, so a live study and its
    dry-run preview can never disagree on sampler, pruner, or direction.

    :param Experiment experiment: Parsed experiment config supplying the metric direction.
    :param Phase phase: Phase whose sampler, search space, and study name are used.
    :param Any storage: Resolved Optuna storage, or ``None`` for in-memory.
    :return optuna.Study: Created or loaded Optuna study for the phase.
    """
    return optuna.create_study(
        study_name=_phase_study_name(experiment, phase),
        storage=storage,
        sampler=_build_sampler(phase.sampler, phase.search_space, n_jobs=phase.n_jobs),
        pruner=optuna.pruners.NopPruner(),
        direction=experiment.metric.goal,
        load_if_exists=True,
    )


def _sqlite_study_exists(experiment: Experiment, phase: Phase | str) -> bool:
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
    :param Phase | str phase: Phase or historical phase name to check.
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


def _journal_snapshot_storage(storage_url: str, label: str) -> JournalStorage | None:
    """Capture a complete journal as a read-only in-memory storage.

    Optuna tolerates an undecodable final record while another writer appends.
    Such a snapshot is incomplete, so inspection cannot claim known counts or
    confirmed absence.

    :param str storage_url: Journal storage URL to inspect.
    :param str label: Study or experiment named in an inspection failure.
    :return JournalStorage | None: Snapshot storage, or ``None`` when absent.
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
        return JournalStorage(_JournalSnapshot(records))
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Journal storage {path} could not be completely read while checking for {label}."
        ) from exc


def _load_journal_study_snapshot(storage_url: str, study_name: str) -> optuna.Study | None:
    """Replay a complete journal snapshot before deciding whether a study exists.

    Replay must finish before a missing-study KeyError can be distinguished
    from a malformed journal operation.

    :param str storage_url: Journal storage URL to inspect.
    :param str study_name: Named study to load from the captured records.
    :return optuna.Study | None: Read-only snapshot study, or confirmed absence.
    :raises StudyStorageUnavailableError: The snapshot is unreadable or incomplete.
    """
    storage = _journal_snapshot_storage(storage_url, f"study {study_name!r}")
    if storage is None:
        return None
    try:
        return optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        return None


def _scan_ledger_format(experiment: Experiment) -> None:
    """Reject pre-cutover PhaseSweep studies without mutating local storage.

    The check covers every PhaseSweep-shaped study in the selected local
    ledger, not only studies belonging to the current experiment name. This
    prevents a new experiment name or output directory from treating a
    populated pre-cutover ledger as fresh. SQLite is opened read-only and a
    journal is replayed from an observational snapshot before Optuna can
    initialize, stamp, recover, or otherwise mutate the live backend.

    :param Experiment experiment: Experiment whose resolved local ledger is inspected.
    :raises StudyStorageUnavailableError: An existing local ledger cannot be
        inspected without mutation.
    :raises StudySchemaMismatchError: A populated PhaseSweep study is unmarked,
        or a PhaseSweep study uses a pre-cutover/unsupported schema.
    """
    storage = experiment.resolved_storage
    if storage_is_in_memory(storage):
        return
    assert storage is not None
    backend = storage_backend(storage)
    versions: list[tuple[str, object, bool]] = []
    if backend == "sqlite":
        database = sqlite_database_path(storage)
        uri = sqlite_readonly_uri(storage)
        if database is None or uri is None or not database.exists():
            return
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=0.1)
            try:
                tables = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                if "studies" not in tables:
                    return
                studies = [
                    (int(study_id), str(name))
                    for study_id, name in conn.execute(
                        "SELECT study_id, study_name FROM studies"
                    ).fetchall()
                    if "::" in str(name)
                ]
                if not studies:
                    return
                attrs: dict[str, object] = {}
                if "study_user_attributes" in tables:
                    rows = conn.execute(
                        """
                        SELECT studies.study_name, study_user_attributes.value_json
                        FROM studies
                        JOIN study_user_attributes
                          ON studies.study_id = study_user_attributes.study_id
                        WHERE study_user_attributes.key = ?
                        """,
                        (STUDY_SCHEMA_ATTR,),
                    ).fetchall()
                    for study_name, value_json in rows:
                        try:
                            attrs[str(study_name)] = json.loads(value_json)
                        except (TypeError, json.JSONDecodeError):
                            attrs[str(study_name)] = value_json
                populated_study_ids = (
                    {
                        int(study_id)
                        for (study_id,) in conn.execute(
                            "SELECT DISTINCT study_id FROM trials"
                        ).fetchall()
                    }
                    if "trials" in tables
                    else set()
                )
                versions = [
                    (name, attrs.get(name), study_id in populated_study_ids)
                    for study_id, name in studies
                ]
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise StudyStorageUnavailableError(
                f"SQLite storage {database} could not be inspected for its PhaseSweep "
                "format without mutation."
            ) from exc
    elif backend == "journal":
        snapshot = _journal_snapshot_storage(storage, "the PhaseSweep format boundary")
        if snapshot is None:
            return
        try:
            versions = [
                (
                    study.study_name,
                    study.user_attrs.get(STUDY_SCHEMA_ATTR),
                    bool(snapshot.get_all_trials(study._study_id, deepcopy=False)),
                )
                for study in snapshot.get_all_studies()
                if "::" in study.study_name
            ]
        except Exception as exc:
            raise StudyStorageUnavailableError(
                "Journal storage could not be replayed while checking its PhaseSweep format."
            ) from exc
    else:
        raise ValueError(f"Unsupported local storage backend: {backend!r}.")

    _validate_storage_versions(versions)


def _validate_storage_versions(versions: Iterable[tuple[str, object, bool]]) -> None:
    """Reject unsupported PhaseSweep study versions from one storage snapshot.

    :param Iterable[tuple[str, object, bool]] versions: Study name, decoded schema
        version, and whether the study contains trials.
    :raises StudySchemaMismatchError: A populated study is unmarked or any study
        carries an unsupported explicit schema version.
    """
    # Match _validate_study_schema: create_study can survive an interruption
    # before the first schema stamp, and an empty unmarked study is claimable.
    unsupported = [
        (name, version)
        for name, version, has_trials in versions
        if (type(version) is not int or version != STUDY_SCHEMA_VERSION)
        and (version is not None or has_trials)
    ]
    if unsupported:
        detail = ", ".join(f"{name!r} ({version!r})" for name, version in unsupported)
        raise StudySchemaMismatchError(
            "The selected local storage ledger contains pre-cutover or unsupported "
            f"PhaseSweep study state: {detail}. Use a fresh local storage ledger and "
            "artifact root with this PhaseSweep release, or use the preserved PhaseSweep "
            "0.3.1 environment to operate the existing state. Nothing was written."
        )


def _load_existing_phase_study(experiment: Experiment, phase: Phase | str) -> optuna.Study | None:
    """Load a phase study only if it already exists.

    Recovery and read-like paths must not call ``create_study(load_if_exists=True)`` because
    that can create an empty study and make missing evidence look safe. This helper checks
    persistent storage before delegating to Optuna's loader, then treats an absent study as
    ``None``.

    :param Experiment experiment: Parsed experiment config containing storage settings.
    :param Phase | str phase: Phase or historical phase name whose study is loaded.
    :return optuna.Study | None: Existing study, or ``None`` when no durable study exists.
    :raises StudyStorageUnavailableError: Persistent storage exists but
        could not be read, so whether the study exists cannot be determined;
        callers on mutating paths must abort rather than treat this as absence.
    """
    storage = experiment.resolved_storage
    if storage_is_in_memory(storage):
        return None
    assert storage is not None
    backend = storage_backend(storage)
    study_name = _phase_study_name(experiment, phase)
    if backend == "sqlite":
        if not _sqlite_study_exists(experiment, phase):
            return None
    elif backend == "journal":
        if _load_journal_study_snapshot(storage, study_name) is None:
            return None
        # Preflight verified a complete snapshot. Mutating callers still use
        # Optuna's normal backend, including its concurrent-append semantics.
        return optuna.load_study(study_name=study_name, storage=_resolve_storage(storage))
    else:
        raise ValueError(f"Unsupported local storage backend: {backend!r}.")
    try:
        return optuna.load_study(study_name=study_name, storage=_resolve_storage(storage))
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
               attempt.value_json AS attempt_json,
               schema.value_json AS schema_json
        FROM trials
        JOIN studies ON trials.study_id = studies.study_id
        LEFT JOIN trial_user_attributes AS generation
          ON trials.trial_id = generation.trial_id AND generation.key = :generation_key
        LEFT JOIN trial_user_attributes AS attempt
          ON trials.trial_id = attempt.trial_id AND attempt.key = :attempt_key
        LEFT JOIN study_user_attributes AS schema
          ON studies.study_id = schema.study_id AND schema.key = :study_schema_key
        WHERE studies.study_name = :study_name
    )
    SELECT 'count', NULL, state, generation_json, NULL, COUNT(*), schema_json
    FROM phase_trials
    GROUP BY state, generation_json, schema_json
    UNION ALL
    SELECT 'running', number, state, generation_json, attempt_json, 1, schema_json
    FROM phase_trials
    WHERE state = 'RUNNING'
    UNION ALL
    SELECT 'published', number, state, generation_json, attempt_json, 1, schema_json
    FROM phase_trials
    WHERE number = :published_trial_number
"""


def _phase_trial_stats_params(
    experiment: Experiment,
    phase: Phase,
    published_trial: _TrialRef | None,
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
        "study_schema_key": STUDY_SCHEMA_ATTR,
        "study_name": _phase_study_name(experiment, phase),
        "published_trial_number": published_trial.trial_number if published_trial else -1,
    }


def _unavailable_phase_trial_stats(
    experiment: Experiment, phase: Phase, exc: BaseException
) -> _PhaseTrialStats:
    """Log a storage-read failure and return an unavailable phase-trial snapshot.

    :param Experiment experiment: Config whose storage backend or local path is reported.
    :param Phase phase: Phase whose status read failed.
    :param BaseException exc: Read exception whose direct cause is reported when present.
    :return _PhaseTrialStats: Empty count maps and ``running_attempts=None`` with
        ``available=False``.
    """
    cause = exc.__cause__ or exc
    log.warning(
        "could not read status trial data from storage %s for phase %s: %s: %s",
        experiment.resolved_storage,
        phase.name,
        type(cause).__name__,
        cause,
    )
    return _PhaseTrialStats({}, False, {}, None)


def _sqlite_phase_trial_stats(
    experiment: Experiment,
    phase: Phase,
    published_trial: _TrialRef | None = None,
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
    study_schema: object = None
    for row_kind, number, state, generation_json, attempt_json, tally, schema_json in rows:
        try:
            decoded_schema = (
                json.loads(schema_json) if isinstance(schema_json, str) else schema_json
            )
        except (TypeError, json.JSONDecodeError):
            decoded_schema = schema_json
        state_name = str(state)
        if study_schema is None:
            study_schema = decoded_schema
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
    # The phase's own study answers to the same cutover rule as the ledger-wide
    # scan. Its stamp is only visible here through its trial rows, so an empty
    # study reads as unstamped and passes; judging an empty study's stamp is
    # the job of the scan validate_ledger already ran.
    _validate_storage_versions([(study_name, study_schema, bool(counts))])
    return _PhaseTrialStats(
        counts, True, generation_counts, running_attempts, published_trial_available
    )


def _phase_trial_stats(
    experiment: Experiment,
    phase: Phase,
    published_trial: _TrialRef | None = None,
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
        raise ValueError(f"Unsupported local storage backend: {backend!r}.")
    try:
        snapshot = _journal_snapshot_storage(
            experiment.resolved_storage,
            f"study {_phase_study_name(experiment, phase)!r}",
        )
        if snapshot is None:
            return _PhaseTrialStats({}, True, {}, [])
        try:
            study = optuna.load_study(
                study_name=_phase_study_name(experiment, phase),
                storage=snapshot,
            )
        except KeyError:
            return _PhaseTrialStats({}, True, {}, [])
        trials = study.get_trials(deepcopy=False)
        _validate_storage_versions(
            [(study.study_name, study.user_attrs.get(STUDY_SCHEMA_ATTR), bool(trials))]
        )
    except StudySchemaMismatchError:
        raise
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


def _describe_ledger(experiment: Experiment, binding_state: BindingState) -> ValidatedLedger:
    """Build the handle that names one experiment's selected ledger.

    :param Experiment experiment: Experiment whose resolved storage is described.
    :param BindingState binding_state: Verdict :func:`_check_artifact_root_binding` reached.
    :return ValidatedLedger: Handle recording the ledger's backend and location.
    :raises ValueError: The resolved storage names an unsupported local backend.
    """
    url = experiment.resolved_storage
    backend: Backend
    ledger_path: Path | None = None
    if url is None or storage_is_in_memory(url):
        backend = "memory"
    else:
        named = storage_backend(url)
        if named == "sqlite":
            backend = "sqlite"
            ledger_path = sqlite_database_path(url)
        elif named == "journal":
            backend = "journal"
            ledger_path = Path(file_url_path(url)).expanduser()
        else:
            raise ValueError(f"Unsupported local storage backend: {named!r}.")
    return ValidatedLedger(
        experiment=experiment,
        experiment_name=experiment.experiment,
        storage_url=url,
        backend=backend,
        ledger_path=ledger_path,
        artifact_root=_artifact_root_identity(experiment),
        binding_state=binding_state,
    )


def validate_ledger(experiment: Experiment) -> ValidatedLedger:
    """Check the artifact-root binding, then the ledger format, writing nothing.

    This is the one door every read path goes through, and it performs the
    fixed order's first two steps in that order. The binding check comes first
    because a tree bound to a different ledger has to be refused before this
    process reads a single row of the ledger it was offered. The format scan
    comes second because a pre-cutover ledger has to be refused before anything
    opens it. Neither step creates a study, constructs file-backed storage, or
    writes a byte, so a refusal leaves both the tree and the ledger exactly as
    they were.

    An unreadable ledger is fatal while the tree is still ``"unbound"``: with
    no recorded owner, "the ledger is unavailable" and "the ledger is empty and
    fresh" are different facts and must not collapse into one. Once the tree is
    ``"bound"`` its record already proves which ledger owns it, so a transient
    read failure is left to the caller's own tolerant snapshot to report, and
    mutating callers fail in their strict discovery pass instead. A
    :class:`StudySchemaMismatchError` always propagates.

    :param Experiment experiment: Experiment whose tree and ledger must agree.
    :return ValidatedLedger: Handle proving both checks ran, in that order.
    :raises ArtifactRootConflictError: The tree records a different owner, is
        unreadable, or holds unmarked pre-cutover PhaseSweep state.
    :raises StudySchemaMismatchError: The ledger holds pre-cutover or otherwise
        unsupported PhaseSweep study state.
    :raises StudyStorageUnavailableError: An unbound tree's ledger exists but
        cannot be inspected without mutating it.
    """
    binding_state = _check_artifact_root_binding(experiment)
    ledger = _describe_ledger(experiment, binding_state)
    if ledger.backend != "memory":
        try:
            _scan_ledger_format(experiment)
        except StudyStorageUnavailableError:
            if binding_state == "unbound":
                raise
    return ledger


def open_existing_study(ledger: ValidatedLedger, phase: Phase | str) -> optuna.Study | None:
    """Open a phase's durable study through a validated ledger, never creating one.

    Taking the handle rather than a bare config is the point: reaching this
    opener at all proves the binding check and the format scan already ran, in
    order, so no caller can load a study out of a ledger this release refuses.

    :param ValidatedLedger ledger: Handle from :func:`validate_ledger`.
    :param Phase | str phase: Phase or historical phase name whose study is opened.
    :return optuna.Study | None: Existing study, or ``None`` when none exists.
    :raises StudyStorageUnavailableError: Persistent storage exists but could
        not be read, so whether the study exists cannot be determined.
    """
    return _load_existing_phase_study(ledger.experiment, phase)


def read_phase_trial_stats(
    ledger: ValidatedLedger,
    phase: Phase,
    published_trial: _TrialRef | None = None,
) -> _PhaseTrialStats:
    """Read one phase's counts and RUNNING identities from a validated ledger.

    The snapshot is permissive by design -- an unreadable ledger reports
    ``available=False`` rather than raising -- because the ledger's *format* was
    already settled by :func:`validate_ledger` before this handle existed. That
    is why no format argument appears here: the scan is not this read's job.
    The phase's own study still answers to the same cutover rule, because on a
    bound tree an unreadable scan is tolerated and this snapshot may be the
    first read to see that study's rows.

    :param ValidatedLedger ledger: Handle from :func:`validate_ledger`.
    :param Phase phase: Phase whose existing study is inspected.
    :param _TrialRef | None published_trial: Published local trial to verify in
        this same snapshot.
    :return _PhaseTrialStats: One permissive storage snapshot with explicit
        availability.
    :raises StudySchemaMismatchError: The phase's own study is populated under
        a pre-cutover or unsupported schema.
    """
    return _phase_trial_stats(ledger.experiment, phase, published_trial)


def claim_ledger(ledger: ValidatedLedger, *, from_phase: str | None = None) -> ClaimedLedger:
    """Discover every existing phase study once, then bind the tree and claim them.

    This performs the fixed order's writes, and only after both ownership
    directions are known to agree. Discovery is strict and tri-state (PR #5
    review / reviewer 2, issue 1): a phase's study is *absent* (storage read
    fine, no such study), *present* (loaded and kept), or *unavailable* -- and
    unavailable raises before any claim, reaping, or registry recovery runs.
    Swallowing a read failure and letting preflight read again would be unsafe:
    a transient failure (for example, a brief SQLite lock) could succeed on the
    second read and hand stale-trial reaping a study whose artifact-root
    binding was never checked. The returned handle's ``studies`` is therefore
    the *only* discovery pass: the objects that passed the root check here are
    the exact objects preflight goes on to reap and validate.

    Every declared phase is checked before anything is written (re-review
    v0.5.19 / blocker B3), so a refused multi-phase run leaves neither the tree
    nor any study bound to the rejected root. Phases whose study does not exist
    yet are bound by :func:`open_phase_study` when the phase creates them. The
    exception is a phase with a winner in the current published generation:
    that publication proves the phase previously had trial rows with a specific
    generation and attempt identity, so missing or replaced published trials
    mean durable evidence was lost rather than that the phase is new. Skipped
    phases load their saved winners instead.

    The caller holds :func:`phasesweep.engine.locking._experiment_lock` across
    :func:`validate_ledger` and this call, so no cooperating process can bind
    the tree in between. The binding is still read once more before the first
    write, because that re-read is cheap and a tree modified outside the lock
    must be refused rather than overwritten.

    :param ValidatedLedger ledger: Handle from :func:`validate_ledger`.
    :param str | None from_phase: Resume point; earlier phases will not execute.
    :return ClaimedLedger: Bound handle carrying every existing phase study.
    :raises StudyStorageUnavailableError: A phase's persistent storage could
        not be inspected.
    :raises PublishedStudyMissingError: A phase to execute has a published
        result whose local trial identity is missing from durable storage.
    :raises ArtifactRootConflictError: A phase study is already bound to a
        different artifact root, carries a binding that is not a string, or the
        tree's ownership record changed after validation.
    """
    experiment = ledger.experiment
    loaded: dict[str, optuna.Study] = {}
    for phase in experiment.phases:
        try:
            study = open_existing_study(ledger, phase)
        except Exception as exc:
            unavailable = StudyStorageUnavailableError(
                f"Could not inspect persistent study storage for phase {phase.name!r}."
            )
            raise unavailable from exc
        if study is not None:
            loaded[phase.name] = study
    claimable: list[optuna.Study] = []
    if _artifact_root_binding_applies(experiment):
        _check_published_phase_studies(experiment, loaded, from_phase=from_phase)
        claimable = [
            study for study in loaded.values() if _artifact_root_claim_needed(study, experiment)
        ]
    offered = _artifact_root_identity(experiment)
    binding_state = _check_artifact_root_binding(experiment)
    if binding_state != ledger.binding_state or offered != ledger.artifact_root:
        raise ArtifactRootConflictError(
            f"Artifact root {ledger.artifact_root!r} changed while its ledger was being "
            f"claimed: validation found it {ledger.binding_state}, but it is now "
            f"{binding_state} at {offered!r}. Another process modified this tree without "
            "holding the experiment lock. Stop that process before retrying. Nothing was "
            "written."
        )
    # Both directions are now known-compatible. Claim the tree first, then
    # empty studies: a crash cannot leave a study pointing at a tree that does
    # not itself name the same ledger. Crucially, neither claim occurs when a
    # symlink-retargeted leaf exposed a study still bound to the old target.
    if binding_state == "unbound":
        _write_artifact_root_binding(experiment)
    for study in claimable:
        _claim_study_artifact_root(study, offered)
    return ClaimedLedger(
        experiment=experiment,
        experiment_name=ledger.experiment_name,
        storage_url=ledger.storage_url,
        backend=ledger.backend,
        ledger_path=ledger.ledger_path,
        artifact_root=ledger.artifact_root,
        binding_state="bound",
        studies=MappingProxyType(loaded),
    )


def open_phase_study(ledger: ClaimedLedger, phase: Phase) -> optuna.Study:
    """Open a phase's live study for writing, creating it if it does not exist.

    This is the only opener that may create a study in file-backed storage,
    so it is also the only place a journal's parent directory is brought into
    existence. Requiring a :class:`ClaimedLedger` is what makes the fixed order
    checkable: that handle exists only once :func:`validate_ledger` and
    :func:`claim_ledger` have run, so no caller can create a study in a ledger
    this release refuses or in a tree that does not yet name this ledger. The
    type is enforced at runtime as well, because an untyped caller could
    otherwise pass a bare :class:`ValidatedLedger`, which carries every field
    this function reads.

    A study this call creates was invisible to :func:`claim_ledger`'s
    discovery, so it claims its publication root here, before any inspection,
    reaping, or trial work can run against it (review v0.5.19 / finding F5);
    a study that already existed only re-confirms the root it was claimed for.

    :param ClaimedLedger ledger: Handle from :func:`claim_ledger`.
    :param Phase phase: Phase whose sampler, search space, and study name are used.
    :return optuna.Study: The phase's live study, bound to the claimed artifact root.
    :raises TypeError: ``ledger`` is not a :class:`ClaimedLedger`.
    :raises ArtifactRootConflictError: The study is already bound to a different
        artifact root, or holds trials but records no root.
    """
    if not isinstance(ledger, ClaimedLedger):
        raise TypeError(
            "open_phase_study requires the ClaimedLedger returned by claim_ledger, not "
            f"{type(ledger).__name__}."
        )
    experiment = ledger.experiment
    if ledger.backend == "journal":
        assert ledger.ledger_path is not None
        ledger.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    study = _build_phase_study(experiment, phase, _resolve_storage(ledger.storage_url))
    _bind_study_artifact_root(study, experiment)
    return study


def open_preview_study(experiment: Experiment, phase: Phase) -> optuna.Study:
    """Create the throwaway in-memory study a dry run renders its preview from.

    A preview must leave the configured ledger and tree exactly as they were,
    so it takes no handle and never resolves the configured storage: the study
    it returns lives only in this process's memory.

    :param Experiment experiment: Parsed experiment supplying direction and study name.
    :param Phase phase: Phase whose sampler and search space the preview uses.
    :return optuna.Study: Fresh in-memory study for the phase.
    """
    return _build_phase_study(experiment, phase, None)


def open_registry_study(locator: str, study_name: str) -> optuna.Study:
    """Open a study named by an attempt-registry entry, on any ledger it names.

    The attempt registry can point at a ledger this process never configured -
    a run that moved its storage, or a foreign locator recorded by an earlier
    orchestrator - so this is the one opener that takes a bare locator instead
    of a validated handle. It keeps the registry recovery semantics it was
    extracted from: a missing study surfaces as :class:`KeyError`, which the
    caller resolves differently for SQLite and journal ledgers.

    :param str locator: Storage URL recorded in (or recovered for) the entry.
    :param str study_name: Study the registry entry names.
    :return optuna.Study: The study recorded under ``study_name``.
    :raises KeyError: ``locator`` holds no study called ``study_name``.
    """
    return optuna.load_study(study_name=study_name, storage=_resolve_storage(locator))
