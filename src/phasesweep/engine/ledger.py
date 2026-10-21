"""Optuna storage construction, read-only snapshots, and ledger claims.

This module owns storage construction; journals are the only durable backend.
Mutations require the experiment lock. Reads and previews use captured snapshots.
``validate_ledger`` checks tree ownership before scanning; a bound tree's unreadable
scan may still serve its frozen publication. ``claim_ledger`` requires a complete
scan, repairs a partial final record and validates existing studies before writing.
Execution requires a ClaimedLedger.

MCP recovery may open live studies from a ValidatedLedger, checking their artifact
ownership before reaping. Registry recovery validates its separately recorded ledger.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import stat
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any, Literal

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import BaseJournalBackend
from optuna.storages.journal._file import BaseJournalFileLock

from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import sampler_search_space
from phasesweep.engine.artifact_roots import (
    BindingState,
    _artifact_root_binding_applies,
    _artifact_root_claim_needed,
    _bind_study_artifact_root,
    _check_artifact_root_binding,
    _check_published_phase_studies,
    _claim_study_artifact_root,
    _write_artifact_root_binding,
)
from phasesweep.engine.errors import (
    ArtifactRootConflictError,
    IncompleteJournalRecordError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
)
from phasesweep.engine.optuna import (
    _build_sampler,
    _phase_study_name,
    _PhaseTrialStats,
    _published_trial_matches,
    _trial_count_state,
    _TrialRef,
)
from phasesweep.engine.paths import _artifact_root_identity
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GENERATION_ID_ATTR,
    STUDY_SCHEMA_ATTR,
    _outcome_sequence_counts,
    _recovered_trial_state,
    _study_schema_supported,
)
from phasesweep.errors import OperatorAction, PhaseSweepError
from phasesweep.runtime.files import fsync_directory, journal_file_path, try_flock, unlock_file
from phasesweep.runtime.shutdown import defer_shutdown_signals

__all__ = [
    "ClaimedLedger",
    "PhaseTrials",
    "ValidatedLedger",
    "claim_ledger",
    "log_storage_read_failure",
    "open_existing_study",
    "open_phase_study",
    "open_preview_study",
    "open_registry_study",
    "read_phase_studies",
    "read_phase_trials",
    "read_trial_stats",
    "repair_incomplete_journal_record",
    "resolved_ledger_path",
    "validate_ledger",
]

log = logging.getLogger(__name__)

#: Which durable form a ledger takes. ``"memory"`` is deliberate no-ledger
#: execution, so nothing about it is on disk to validate, bind, or reopen.
Backend = Literal["memory", "journal"]


@dataclass(frozen=True, slots=True)
class ValidatedLedger:
    """A ledger's ownership and format-scan verdict, produced without writes.

    ``format_verified=False`` records an unreadable scan tolerated on a bound tree.
    Trial-data reads then report unavailable; live openers refuse the handle, and
    :func:`claim_ledger` must rescan it successfully before mutation.

    This handle permits read-only snapshots and opening existing live studies.
    Callers using live studies must check their artifact-root ownership themselves;
    trial execution requires a :class:`ClaimedLedger`.
    """

    #: The exact config whose ownership and format this handle records.
    experiment: Experiment
    experiment_name: str
    storage_url: str | None
    backend: Backend
    ledger_path: Path | None
    artifact_root: str
    binding_state: BindingState
    #: Why the ledger-wide format scan did not complete, or ``None`` when it
    #: did. Only a bound tree tolerates an unreadable scan, so publication data
    #: stays readable. Nothing that depends on the ledger's format may treat
    #: such a handle as scanned (see :attr:`format_verified`).
    format_scan_failure: StudyStorageUnavailableError | None

    @property
    def format_verified(self) -> bool:
        """Whether the ledger-wide format scan completed for this handle.

        :return bool: ``True`` when the scan ran to completion and found no
            refused state; ``False`` when a bound tree tolerated a scan that
            could not read the ledger.
        """
        return self.format_scan_failure is None


@dataclass(frozen=True, slots=True)
class ClaimedLedger(ValidatedLedger):
    """A validated ledger whose tree and existing phase studies are bound.

    :func:`claim_ledger` checks every study before binding the tree, then claims each
    empty study. Only this handle permits :func:`open_phase_study` for trial execution.
    """

    #: Existing phase studies from the ownership-checked discovery pass.
    #: Preflight reaps and validates these same objects; absent studies are omitted.
    studies: Mapping[str, optuna.Study]
    #: The one storage every phase of a ``"memory"`` ledger shares, so the
    #: invocation can still read its own trials after a phase fails; ``None``
    #: for a journal ledger.
    memory_storage: optuna.storages.InMemoryStorage | None


def _resolve_storage(url: str) -> JournalStorage:
    """Construct journal storage from its resolved URL.

    ``auto`` must already be resolved. Directory creation belongs to :func:`claim_ledger`.

    :param str url: Resolved journal URL, including escaped ``file:``
        filenames with ``uri=true``.
    :return JournalStorage: Configured journal storage.
    :raises ValueError: ``url`` selects an unsupported local storage backend.
    """
    path = journal_file_path(url)
    log.info("Using JournalFileStorage at %s", path)
    from optuna.storages.journal import JournalFileBackend

    return JournalStorage(JournalFileBackend(str(path), lock_obj=_JournalLock(path)))


def _build_phase_study(experiment: Experiment, phase: Phase, storage: Any) -> optuna.Study:
    """Create or load a phase's Optuna study in already-resolved storage.

    :param Experiment experiment: Parsed experiment config supplying the metric direction.
    :param Phase phase: Phase whose sampler, search space, and study name are used.
    :param Any storage: Resolved Optuna storage, or ``None`` for in-memory.
    :return optuna.Study: Created or loaded Optuna study for the phase.
    """
    return optuna.create_study(
        study_name=_phase_study_name(experiment, phase),
        storage=storage,
        sampler=_build_sampler(
            phase.sampler,
            sampler_search_space(phase),
            n_jobs=phase.n_jobs,
            repeats=phase.repeats,
        ),
        pruner=optuna.pruners.NopPruner(),
        direction=experiment.metric.goal,
        load_if_exists=True,
    )


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


def _capture_journal(path: Path) -> bytes | None:
    """Read a journal's bytes as of one moment, without writing anything.

    :param Path path: Journal file to read.
    :return bytes | None: The file's bytes at the size it had when opened, or
        ``None`` when it does not exist, including when a component of its
        directory is not a directory, so no journal can exist there.
    :raises OSError: The journal exists but could not be read.
    :raises ValueError: The journal shrank while it was being read.
    """
    try:
        with path.open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            data = source.read(size)
    except (FileNotFoundError, NotADirectoryError):
        # A path under a regular file holds no ledger; claim_ledger's directory
        # creation then refuses the path as unusable, with its remedy.
        return None
    if len(data) != size:
        raise ValueError("The journal was truncated while its snapshot was being read.")
    return data


def _journal_records(data: bytes) -> tuple[list[dict[str, Any]], int]:
    """Decode captured journal bytes using Optuna's incomplete-final-line policy.

    Skip a final line lacking a newline or valid JSON; reject an incomplete or
    undecodable line followed by another record.

    :param bytes data: Journal bytes from :func:`_capture_journal`.
    :return tuple[list[dict[str, Any]], int]: Every complete record, and the
        byte offset just past the last of them, which is ``len(data)`` unless
        a final line was skipped.
    :raises ValueError: A line that another line follows is incomplete or
        undecodable, or a line is not UTF-8.
    """
    lines = data.split(b"\n")
    records: list[dict[str, Any]] = []
    end = 0
    for index, line in enumerate(lines[:-1]):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 2 and not lines[-1]:
                return records, end
            raise
        end += len(line) + 1
    return records, end


def _journal_snapshot_storage(storage_url: str, label: str) -> JournalStorage | None:
    """Capture a journal's complete records as read-only in-memory storage.

    An incomplete final line is skipped by :func:`_journal_records`. Mutating paths
    remove it through :func:`repair_incomplete_journal_record` before appending.

    :param str storage_url: Journal storage URL to inspect.
    :param str label: Study or experiment named in an inspection failure.
    :return JournalStorage | None: Snapshot storage, or ``None`` when absent.
    :raises StudyStorageUnavailableError: The journal is unreadable, changed
        while it was read, or holds a malformed record before its last line.
    """
    path = journal_file_path(storage_url)
    try:
        data = _capture_journal(path)
        if data is None:
            return None
        records, _end = _journal_records(data)
        return JournalStorage(_JournalSnapshot(records))
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Journal storage {path} could not be completely read while checking for {label}."
        ) from exc


def _scan_journal_before_write(path: Path) -> tuple[bytes, int] | None:
    """Capture a journal a caller is about to write through, and find its complete prefix.

    :param Path path: Journal file to scan.
    :return tuple[bytes, int] | None: The captured bytes and the offset just
        past their last complete record, or ``None`` when no journal exists.
    :raises StudyStorageUnavailableError: The journal is unreadable or corrupt
        before its final record; only an incomplete final append is repairable.
    """
    try:
        data = _capture_journal(path)
    except (OSError, ValueError) as exc:
        raise StudyStorageUnavailableError(
            f"Journal storage {path} could not be read before writing to it: {exc}. "
            "Nothing was written."
        ) from exc
    if data is None:
        return None
    try:
        _records, end = _journal_records(data)
    except ValueError as exc:
        raise StudyStorageUnavailableError(
            f"Journal storage {path} holds a malformed record that other records follow. "
            f"Repair backups, if present, are beside it as `{path.name}.*.bak`. "
            "Nothing was written."
        ) from exc
    return data, end


def _truncate_incomplete_journal_record(lock: _JournalLock) -> tuple[Path, int] | None:
    """Back up a journal, then cut its incomplete final record; the caller holds its lock.

    :param _JournalLock lock: The journal inode lock held by the caller.
    :return tuple[Path, int] | None: The backup's path and the number of bytes
        removed, or ``None`` when the final record had completed by the time
        the lock was taken, so nothing needed repair.
    :raises IncompleteJournalRecordError: The backup or truncation failed, or
        the journal pathname or size changed while the lock was held.
    :raises StudyStorageUnavailableError: The journal could not be read, holds
        a malformed record before its last, or does not replay once repaired.
    """
    path = lock.path
    scanned = _scan_journal_before_write(path)
    if scanned is None or scanned[1] == len(scanned[0]):
        return None
    data, end = scanned
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.name}.{stamp}.bak")
    try:
        journal = lock._handle
        assert journal is not None
        source = os.fstat(journal.fileno())
        # The backup carries the journal's permissions, narrowed by the umask.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        with os.fdopen(os.open(backup, flags, stat.S_IMODE(source.st_mode)), "wb") as copy:
            copy.write(data)
            copy.flush()
            os.fsync(copy.fileno())
        fsync_directory(path.parent)
        if not lock.held():
            raise IncompleteJournalRecordError(
                f"Journal storage {path}: another process removed or replaced its pathname "
                "during the repair, so its incomplete final record was not truncated. "
                f"The journal as it was is saved at {backup}. "
                "Nothing was appended.",
                action=OperatorAction.RETRY,
            )
        if os.fstat(journal.fileno()).st_size != len(data):
            raise IncompleteJournalRecordError(
                f"Journal storage {path} changed while its journal lock was held for "
                "repair, so it was not truncated. The journal as it was is saved at "
                f"{backup}. Nothing was appended.",
                action=OperatorAction.RETRY,
            )
        journal.truncate(end)
        os.fsync(journal.fileno())
    except OSError as exc:
        raise IncompleteJournalRecordError(
            f"Journal storage {path} ends with an incomplete record that could not be "
            f"repaired: {exc}. Nothing was appended."
        ) from exc
    try:
        repaired = _capture_journal(path)
        if repaired is None or len(repaired) != end:
            raise ValueError("the repaired journal does not end at its last complete record")
        records, repaired_end = _journal_records(repaired)
        if repaired_end != end:
            raise ValueError("the repaired journal still ends with an incomplete record")
        JournalStorage(_JournalSnapshot(records))
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Journal storage {path} does not replay after its incomplete final record was "
            f"removed. The journal as it was is saved at {backup}. Nothing was appended."
        ) from exc
    return backup, len(data) - end


#: Maximum wait for the journal inode lock during repair.
_JOURNAL_LOCK_WAIT_SECONDS = 30.0


class _JournalLock(BaseJournalFileLock):
    """Flock the journal inode for every append and repair.

    Writers wait without a deadline; repair supplies a bounded timeout. The
    descriptor pins ownership until release, including after pathname replacement.
    """

    def __init__(self, path: Path, *, timeout: float | None = None) -> None:
        """Name the existing journal and the caller's acquisition deadline.

        :param Path path: Canonical journal path.
        :param float | None timeout: Repair wait budget, or unbounded for writers.
        """
        self.path = path
        self._timeout = timeout
        self._handle: IO[str] | None = None

    def acquire(self) -> bool:
        """Lock the journal, refusing a timed-out repair or replaced pathname.

        :raises IncompleteJournalRecordError: The journal could not be locked.
        :return bool: ``True`` once acquired.
        """
        deadline = None if self._timeout is None else time.monotonic() + self._timeout
        delay = 0.001
        handle: IO[str] | None = None
        try:
            with defer_shutdown_signals():
                handle = self.path.open("r+", encoding="utf-8")
            while True:
                with defer_shutdown_signals():
                    acquired = try_flock(handle)
                    if acquired:
                        self._handle = handle
                        if not self.held():
                            raise IncompleteJournalRecordError(
                                f"Journal storage {self.path} was removed or replaced while "
                                "its journal lock was being taken. Nothing was written.",
                                action=OperatorAction.RETRY,
                            )
                if acquired:
                    return True
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise IncompleteJournalRecordError(
                        f"Journal storage {self.path} ends with an incomplete record, and "
                        f"its journal lock was still held after {self._timeout:g} seconds, "
                        "so the record was not repaired. Wait for the writer to finish, "
                        "then retry. Nothing was written."
                    )
                time.sleep(delay if remaining is None else min(delay, remaining))
                delay = min(delay * 2, 1.0)
        except BaseException as exc:
            with defer_shutdown_signals():
                self._handle = None
                if handle is not None:
                    unlock_file(handle)
            if isinstance(exc, OSError):
                raise IncompleteJournalRecordError(
                    f"Journal storage {self.path} could not be repaired or appended: "
                    f"its journal lock could not be taken: {exc}. Nothing was written."
                ) from exc
            raise

    def held(self) -> bool:
        """Return whether the pathname still names the locked journal inode.

        :return bool: ``False`` when unheld, removed, or replaced, including by a symlink.
        """
        if self._handle is None:
            return False
        try:
            current, held = self.path.lstat(), os.fstat(self._handle.fileno())
        except OSError:
            return False
        return (current.st_dev, current.st_ino) == (held.st_dev, held.st_ino)

    def release(self) -> None:
        """Close the held descriptor before servicing deferred shutdown."""
        with defer_shutdown_signals():
            handle, self._handle = self._handle, None
            if handle is not None:
                unlock_file(handle)


def _repair_incomplete_journal_record(storage_url: str) -> None:
    """Back up and truncate an incomplete final journal record under its lock.

    A complete journal needs no lock or repair. Otherwise acquire :class:`_JournalLock`
    and rescan; a final append that completed while waiting is left intact. Copy the
    captured bytes to ``<journal>.<UTC stamp>.bak`` with permissions no broader than
    the journal's, truncate to the complete prefix, fsync, and verify replay before
    releasing the lock. Never reuse the offset scanned before acquisition.

    If the pathname is replaced, leave the replacement untouched.
    Malformed records before the final line are refused without automatic repair.

    :param str storage_url: Journal storage URL a caller is about to write through.
    :raises IncompleteJournalRecordError: The record could not be repaired,
        the journal lock stayed held past the wait, or another process
        removed or replaced the journal pathname during the repair.
    :raises StudyStorageUnavailableError: The journal could not be read, holds
        a malformed record before its last, or did not replay once repaired.
    """
    path = journal_file_path(storage_url)
    scanned = _scan_journal_before_write(path)
    if scanned is None or scanned[1] == len(scanned[0]):
        return
    log.warning(
        "Journal storage %s ends with an incomplete record; taking its journal lock to repair it.",
        path,
    )
    lock = _JournalLock(path, timeout=_JOURNAL_LOCK_WAIT_SECONDS)
    lock.acquire()
    try:
        repaired = _truncate_incomplete_journal_record(lock)
        if not lock.held():
            raise IncompleteJournalRecordError(
                f"Journal storage {path} was removed or replaced while its incomplete final "
                "record was being repaired. The replacement was left in place. "
                "Retry to check the journal again.",
                action=OperatorAction.RETRY,
            )
    finally:
        lock.release()
    if repaired is not None:
        backup, dropped = repaired
        log.warning(
            "Repaired journal storage %s: removed an incomplete final record of %d bytes "
            "that no reader could see. The journal as it was is saved at %s.",
            path,
            dropped,
            backup,
        )


def _load_snapshot_study(
    storage: JournalStorage | optuna.storages.InMemoryStorage, study_name: str
) -> optuna.Study | None:
    """Load and schema-check a snapshot study, returning only confirmed absence.

    :param JournalStorage | optuna.storages.InMemoryStorage storage: Captured or invocation-local storage.
    :param str study_name: Named study to inspect.
    :return optuna.Study | None: Existing study, or ``None`` when its name is absent.
    :raises StudyStorageUnavailableError: The study or its trials could not be read.
    :raises StudySchemaMismatchError: The study carries an unsupported schema.
    """
    try:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except KeyError:
            return None
        trials = study.get_trials(deepcopy=False)
    except Exception as exc:
        raise StudyStorageUnavailableError(
            f"Snapshot storage could not be replayed while reading study {study_name!r}."
        ) from exc
    _check_study_schema(study, trials)
    return study


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
    return _load_snapshot_study(storage, study_name)


def _scan_ledger_format(storage: str | None) -> None:
    """Reject unsupported PhaseSweep studies without mutating local storage.

    Replay a captured journal and check every PhaseSweep-shaped study, including
    those outside the current experiment namespace.

    :param str | None storage: Resolved storage URL of the ledger to inspect,
        or ``None`` for in-memory storage.
    :raises StudyStorageUnavailableError: An existing local ledger cannot be
        inspected without mutation.
    :raises StudySchemaMismatchError: A populated PhaseSweep study is unmarked,
        or a PhaseSweep study uses an unsupported schema.
    """
    if storage is None:
        return
    versions: list[tuple[str, object, bool]] = []
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

    _validate_storage_versions(versions)


def _validate_storage_versions(versions: Iterable[tuple[str, object, bool]]) -> None:
    """Reject unsupported PhaseSweep study versions from one storage snapshot.

    :param Iterable[tuple[str, object, bool]] versions: Study name, decoded schema
        version, and whether the study contains trials.
    :raises StudySchemaMismatchError: A populated study is unmarked or any study
        carries an unsupported explicit schema version.
    """
    unsupported = [
        (name, version)
        for name, version, has_trials in versions
        if not _study_schema_supported(version, has_trials)
    ]
    if unsupported:
        detail = ", ".join(f"{name!r} ({version!r})" for name, version in unsupported)
        raise StudySchemaMismatchError(
            "The selected local storage ledger contains unsupported PhaseSweep study "
            f"state: {detail}. Nothing was written.",
        )


def _require_scanned(ledger: ValidatedLedger) -> None:
    """Refuse an unreadable format scan with its original exception type and remedy.

    :param ValidatedLedger ledger: Handle whose format boundary must be complete.
    :raises StudyStorageUnavailableError: The handle records a failed format scan.
    """
    failure = ledger.format_scan_failure
    if failure is not None:
        raise type(failure).rewrap(failure, str(failure)) from failure


def _check_study_schema(study: optuna.Study, trials: list[optuna.trial.FrozenTrial]) -> None:
    """Validate a captured study's schema, allowing an empty, unstamped allocation.

    :param optuna.Study study: Study whose recorded schema is checked.
    :param list[optuna.trial.FrozenTrial] trials: Trials from the same snapshot.
    :raises StudySchemaMismatchError: The study carries unsupported populated state.
    """
    _validate_storage_versions(
        [(study.study_name, study.user_attrs.get(STUDY_SCHEMA_ATTR), bool(trials))]
    )


def log_storage_read_failure(
    ledger: ValidatedLedger,
    phase: Phase,
    exc: BaseException,
    *,
    status: bool = False,
    logger: logging.Logger = log,
) -> None:
    """Report a failed observational read using its direct cause when present.

    :param ValidatedLedger ledger: Handle whose storage could not be read.
    :param Phase phase: Phase whose trial data is unavailable.
    :param BaseException exc: Read failure, possibly wrapping the underlying cause.
    :param bool status: Identify a status-count read in the diagnostic.
    :param logging.Logger logger: Reader's diagnostic channel.
    """
    cause = exc.__cause__ or exc
    logger.warning(
        "could not read %strial data from storage %s for phase %s: %s: %s",
        "status " if status else "",
        ledger.storage_url,
        phase.name,
        type(cause).__name__,
        cause,
    )


def _unavailable_phase_trial_stats(
    ledger: ValidatedLedger, phase: Phase, exc: BaseException
) -> _PhaseTrialStats:
    """Log a storage-read failure and return an unavailable phase-trial snapshot.

    :param ValidatedLedger ledger: Handle whose storage is reported.
    :param Phase phase: Phase whose status read failed.
    :param BaseException exc: Read exception whose direct cause is reported when present.
    :return _PhaseTrialStats: Empty count maps and ``running_attempts=None`` with
        ``available=False``.
    """
    log_storage_read_failure(ledger, phase, exc, status=True)
    return _PhaseTrialStats({}, False, {}, None)


def _phase_trial_stats(
    snapshot: JournalStorage, study_name: str, published_trial: _TrialRef | None
) -> _PhaseTrialStats:
    """Read one study's counts and RUNNING identities from a captured snapshot.

    Counts, RUNNING identities, and published trial verification all come
    from ``snapshot``, so they describe the same moment. An absent study
    reports available zero counts.

    :param JournalStorage snapshot: Captured journal replay to read.
    :param str study_name: Ledger study name of the phase inspected.
    :param _TrialRef | None published_trial: Published local trial to verify in this snapshot.
    :return _PhaseTrialStats: The study's available counts and identities.
    :raises StudySchemaMismatchError: The study is populated under an
        unsupported schema.
    """
    study = _load_snapshot_study(snapshot, study_name)
    if study is None:
        return _PhaseTrialStats({}, True, {}, [])
    trials = study.get_trials(deepcopy=False)
    counts: dict[str, int] = {}
    generation_counts: dict[str, dict[str, int]] = {}
    running_attempts: list[_TrialRef] = []
    published_trial_available = False
    sequence_counts = _outcome_sequence_counts(trials)
    for trial in trials:
        if published_trial is not None and _published_trial_matches(trial, published_trial):
            published_trial_available = True
        state = _trial_count_state(trial)
        counts[state] = counts.get(state, 0) + 1
        generation_id = trial.user_attrs.get(GENERATION_ID_ATTR)
        if isinstance(generation_id, str):
            states = generation_counts.setdefault(generation_id, {})
            states[state] = states.get(state, 0) + 1
        if trial.state.name != "RUNNING":
            continue
        attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
        running_attempts.append(
            _TrialRef(
                trial_number=trial.number,
                generation_id=generation_id if isinstance(generation_id, str) else None,
                attempt_id=attempt_id if isinstance(attempt_id, str) else None,
                recovers_as=_recovered_trial_state(trial, sequence_counts),
            )
        )
    return _PhaseTrialStats(
        counts, True, generation_counts, running_attempts, published_trial_available
    )


def resolved_ledger_path(experiment: Experiment) -> Path | None:
    """Return the resolved journal path reported by status for external inspection.

    :param Experiment experiment: Experiment whose resolved storage is named.
    :return Path | None: The journal file, or ``None`` for in-memory storage.
    """
    url = experiment.resolved_storage
    if url is None:
        return None
    return journal_file_path(url)


def _describe_ledger(experiment: Experiment, binding_state: BindingState) -> ValidatedLedger:
    """Build the handle that names one experiment's selected ledger.

    :param Experiment experiment: Experiment whose resolved storage is described.
    :param BindingState binding_state: Verdict :func:`_check_artifact_root_binding` reached.
    :return ValidatedLedger: Handle recording the ledger's backend and location.
    """
    url = experiment.resolved_storage
    backend: Backend = "memory" if url is None else "journal"
    ledger_path = None if url is None else journal_file_path(url)
    return ValidatedLedger(
        experiment=experiment,
        experiment_name=experiment.experiment,
        storage_url=url,
        backend=backend,
        ledger_path=ledger_path,
        artifact_root=_artifact_root_identity(experiment),
        binding_state=binding_state,
        format_scan_failure=None,
    )


def validate_ledger(experiment: Experiment) -> ValidatedLedger:
    """Check artifact-root ownership, then ledger format, without writes.

    Neither check creates a study or constructs file-backed storage. An unreadable
    ledger is fatal on an unbound tree. On a bound tree it is recorded on the handle,
    allowing frozen publication reads while trial-data reads report unavailable.
    Live opening refuses an incomplete scan; :func:`claim_ledger` retries it strictly.
    :class:`StudySchemaMismatchError` always propagates.

    :param Experiment experiment: Experiment whose tree and ledger must agree.
    :return ValidatedLedger: Handle proving both checks ran, in that order, and
        recording whether the format scan completed.
    :raises ArtifactRootConflictError: The tree records a different owner, is
        unreadable, or holds unmarked PhaseSweep state.
    :raises StudySchemaMismatchError: The ledger holds unsupported PhaseSweep
        study state.
    :raises StudyStorageUnavailableError: An unbound tree's ledger exists but
        cannot be inspected without mutating it.
    """
    binding_state = _check_artifact_root_binding(experiment)
    ledger = _describe_ledger(experiment, binding_state)
    if ledger.backend == "memory":
        return ledger
    try:
        _scan_ledger_format(ledger.storage_url)
    except StudyStorageUnavailableError as exc:
        if binding_state == "unbound":
            raise
        cause = exc.__cause__ or exc
        log.info(
            "format scan of storage %s did not complete; its trial data is reported "
            "unavailable and any claim rescans first: %s: %s",
            ledger.storage_url,
            type(cause).__name__,
            cause,
        )
        return replace(ledger, format_scan_failure=exc)
    return ledger


def open_existing_study(ledger: ValidatedLedger, phase: Phase | str) -> optuna.Study | None:
    """Open an existing live study through a completely validated ledger.

    Confirm existence in a read-only snapshot before constructing live storage;
    absence creates neither a study nor a ledger file. The live replay must retain
    the snapshot's trial identities. The returned study is writable, so observational
    callers use :func:`read_trial_stats` or :func:`read_phase_trials` instead.

    This opener does not check the study's artifact root. The claiming or recovery
    caller must do that before writing or reaping.

    :param ValidatedLedger ledger: Handle from :func:`validate_ledger`.
    :param Phase | str phase: Phase or historical phase name whose study is opened.
    :return optuna.Study | None: Existing study, or ``None`` when none exists.
    :raises StudyStorageUnavailableError: The handle's format scan did not
        complete, or persistent storage exists but could not be read, so
        whether the study exists cannot be determined.
    """
    _require_scanned(ledger)
    if ledger.backend == "memory":
        return None
    assert ledger.storage_url is not None
    return _open_existing_journal_study(
        ledger.storage_url, _phase_study_name(ledger.experiment, phase)
    )


def repair_incomplete_journal_record(ledger: ValidatedLedger) -> None:
    """Repair a crashed writer's partial final record before live mutation.

    The caller holds the experiment lock. Pure reads skip the incomplete record;
    recovery inspection leaves repair to the confirmed operation. An incomplete
    format scan is left to the live openers to refuse.

    :param ValidatedLedger ledger: Handle from :func:`validate_ledger`.
    :raises IncompleteJournalRecordError: The record could not be repaired, or
        the journal lock was lost during the repair.
    :raises StudyStorageUnavailableError: The journal could not be read, holds
        a malformed record before its last, or did not replay once repaired.
    """
    if ledger.backend != "journal" or not ledger.format_verified:
        return
    assert ledger.storage_url is not None
    _repair_incomplete_journal_record(ledger.storage_url)


def read_trial_stats(
    ledger: ValidatedLedger, published_trials: Mapping[str, _TrialRef | None]
) -> dict[str, _PhaseTrialStats]:
    """Read all phase counts and RUNNING identities from one ledger snapshot.

    An unreadable or unverified ledger reports ``available=False`` for each phase.
    Study schemas are checked in the captured replay. Capture follows publication
    pointer resolution, so published trial references can be checked against it.

    :param ValidatedLedger ledger: Handle from :func:`validate_ledger`.
    :param Mapping[str, _TrialRef | None] published_trials: Published local trial to
        verify in the same snapshot, by phase name.
    :return dict[str, _PhaseTrialStats]: One permissive result per configured
        phase, by phase name, with explicit availability.
    :raises StudySchemaMismatchError: A phase's own study is populated under an
        unsupported schema.
    """
    if ledger.format_scan_failure is not None:
        return {
            phase.name: _unavailable_phase_trial_stats(ledger, phase, ledger.format_scan_failure)
            for phase in ledger.experiment.phases
        }
    phases = ledger.experiment.phases
    if ledger.backend == "memory":
        return {phase.name: _PhaseTrialStats({}, False, {}, None) for phase in phases}
    assert ledger.storage_url is not None
    try:
        snapshot = _journal_snapshot_storage(
            ledger.storage_url, f"experiment {ledger.experiment_name!r} trial data"
        )
    except Exception as exc:  # noqa: BLE001
        return {phase.name: _unavailable_phase_trial_stats(ledger, phase, exc) for phase in phases}
    stats: dict[str, _PhaseTrialStats] = {}
    for phase in phases:
        if snapshot is None:
            stats[phase.name] = _PhaseTrialStats({}, True, {}, [])
            continue
        try:
            stats[phase.name] = _phase_trial_stats(
                snapshot,
                _phase_study_name(ledger.experiment, phase),
                published_trials.get(phase.name),
            )
        except StudySchemaMismatchError:
            raise
        except Exception as exc:  # noqa: BLE001
            stats[phase.name] = _unavailable_phase_trial_stats(ledger, phase, exc)
    return stats


@dataclass(frozen=True, slots=True)
class PhaseTrials:
    """One phase study's trials and its confirmed cleanup recoveries, from one snapshot.

    Trial numbers are study-local, so :attr:`cleanup_recovered` answers only
    for the trials beside it, never for another phase's study.
    """

    #: The study's trials, with their historical attributes exactly as recorded.
    trials: list[optuna.trial.FrozenTrial]
    #: Trial numbers whose cleanup was confirmed by the study-level recovery ledger.
    #: Recovery leaves the trial attributes unchanged.
    cleanup_recovered: frozenset[int]


def _read_phase_study(ledger: ValidatedLedger, phase: Phase | str) -> optuna.Study | None:
    """Read one phase study as a read-only snapshot without creating storage.

    The snapshot supports trial and attribute reads but rejects journal writes.
    In-memory trials are available only through the invocation's :class:`ClaimedLedger`.

    :param ValidatedLedger ledger: Handle from :func:`validate_ledger` or :func:`claim_ledger`.
    :param Phase | str phase: Phase or historical phase name whose study is read.
    :return optuna.Study | None: The snapshot study, or ``None`` when no such
        study exists or this handle cannot reach it.
    :raises StudyStorageUnavailableError: The handle's format scan did not
        complete, or the journal could not be completely read.
    :raises StudySchemaMismatchError: The study is populated under an unsupported schema.
    """
    _require_scanned(ledger)
    study_name = _phase_study_name(ledger.experiment, phase)
    if ledger.backend == "memory":
        if not isinstance(ledger, ClaimedLedger) or ledger.memory_storage is None:
            return None
        return _load_snapshot_study(ledger.memory_storage, study_name)
    assert ledger.storage_url is not None
    return _load_journal_study_snapshot(ledger.storage_url, study_name)


def read_phase_studies(ledger: ValidatedLedger) -> dict[str, optuna.Study | None]:
    """Read every configured phase study from one journal capture, without writes.

    All studies describe the same replay, including empty studies whose accepted
    trial targets still bind a launch.

    :param ValidatedLedger ledger: Handle from :func:`validate_ledger` or :func:`claim_ledger`.
    :return dict[str, optuna.Study | None]: One entry per configured phase, by
        phase name; ``None`` for a phase with no such study, or one this
        handle cannot reach.
    :raises StudyStorageUnavailableError: The handle's format scan did not
        complete, or the journal could not be completely read or replayed.
    :raises StudySchemaMismatchError: A study is populated under an unsupported schema.
    """
    _require_scanned(ledger)
    phases = ledger.experiment.phases
    if ledger.backend == "memory":
        return {phase.name: _read_phase_study(ledger, phase) for phase in phases}
    assert ledger.storage_url is not None
    snapshot = _journal_snapshot_storage(
        ledger.storage_url, f"experiment {ledger.experiment_name!r} trial data"
    )
    studies: dict[str, optuna.Study | None] = {}
    for phase in phases:
        study_name = _phase_study_name(ledger.experiment, phase)
        if snapshot is None:
            studies[phase.name] = None
            continue
        studies[phase.name] = _load_snapshot_study(snapshot, study_name)
    return studies


def read_phase_trials(ledger: ValidatedLedger, phase: Phase | str) -> PhaseTrials | None:
    """Read one phase's trials from a validated ledger, writing nothing.

    The study comes from :func:`_read_phase_study`, so the trials and the
    study's cleanup recovery ledger come from the same snapshot and describe
    the same moment.

    :param ValidatedLedger ledger: Handle from :func:`validate_ledger` or :func:`claim_ledger`.
    :param Phase | str phase: Phase or historical phase name whose study is read.
    :return PhaseTrials | None: The study's trials and confirmed cleanup
        recoveries, or ``None`` when no such study exists or this handle cannot reach it.
    :raises StudyStorageUnavailableError: The handle's format scan did not
        complete, or the journal could not be completely read.
    :raises StudySchemaMismatchError: The study is populated under an unsupported schema.
    """
    from phasesweep.engine.attempts import _cleanup_recovered_trial_numbers

    study = _read_phase_study(ledger, phase)
    if study is None:
        return None
    return PhaseTrials(
        trials=study.get_trials(deepcopy=False),
        cleanup_recovered=frozenset(_cleanup_recovered_trial_numbers(study)),
    )


# Creating a directory fails with these when the configured path itself is
# unusable, which no permission repair fixes.
_UNUSABLE_PATH_ERRNOS = frozenset({errno.EEXIST, errno.ENOTDIR, errno.ENAMETOOLONG, errno.ELOOP})


def _create_ledger_directory(directory: Path) -> None:
    """Create the directory a file-backed ledger is written in.

    Nothing exists there yet, so there is no ledger to restore: either the
    configured path is wrong or its parent does not let this user write.

    :param Path directory: The ledger file's parent directory.
    :raises PhaseSweepError: The path names something that cannot be a
        directory, or the directory could not be created; nothing was written.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno in _UNUSABLE_PATH_ERRNOS:
            raise PhaseSweepError(
                f"Storage ledger directory {directory} cannot be created at that path: "
                f"{exc}. Correct the storage path, or the workdir for auto storage, so it "
                "names a directory PhaseSweep can create. Nothing was written.",
                action=OperatorAction.FIX_CONFIG,
            ) from exc
        raise PhaseSweepError(
            f"Storage ledger directory {directory} could not be created: {exc}. "
            "Nothing was written.",
            action=OperatorAction.RESTORE_TREE,
        ) from exc


def claim_ledger(ledger: ValidatedLedger, *, from_phase: str | None = None) -> ClaimedLedger:
    """Validate all phases and claim their ledger under the experiment lock.

    Repair a partial final record before opening live storage. Read or ownership
    failures abort before claims or recovery. Reached published phases require their
    recorded trial history; skipped phases use saved winners. Recheck the tree binding,
    create the ledger directory, bind the tree, then claim empty studies. Absent studies
    are created by :func:`open_phase_study`; existing studies are returned for preflight.

    :param ValidatedLedger ledger: Handle from :func:`validate_ledger`.
    :param str | None from_phase: Resume point; earlier phases will not execute.
    :return ClaimedLedger: Bound handle carrying every existing phase study.
    :raises IncompleteJournalRecordError: A journal's partial final record
        could not be repaired.
    :raises StudyStorageUnavailableError: The ledger's format could still not
        be scanned, the journal holds a malformed record before its last, or
        a phase's persistent storage could not be inspected.
    :raises StudySchemaMismatchError: The rescan found unsupported PhaseSweep
        study state.
    :raises PublishedStudyMissingError: A phase to execute has a published
        result whose local trial identity is missing from durable storage.
    :raises ArtifactRootConflictError: A phase study is already bound to a
        different artifact root, carries a binding that is not a string, or the
        tree's ownership record changed after validation.
    :raises PhaseSweepError: The ledger's directory could not be created;
        nothing was bound or claimed.
    """
    if not ledger.format_verified:
        _scan_ledger_format(ledger.storage_url)
        ledger = replace(ledger, format_scan_failure=None)
    repair_incomplete_journal_record(ledger)
    experiment = ledger.experiment
    loaded: dict[str, optuna.Study] = {}
    for phase in experiment.phases:
        try:
            study = open_existing_study(ledger, phase)
        except Exception as exc:
            unavailable = StudyStorageUnavailableError.rewrap(
                exc, f"Could not inspect persistent study storage for phase {phase.name!r}."
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
            "written.",
            action=OperatorAction.INSPECT_LOGS,
        )
    # Refuse unusable ledger paths before binding the artifact tree.
    if ledger.ledger_path is not None:
        _create_ledger_directory(ledger.ledger_path.parent)
    # Bind the tree before studies so a crash cannot leave a one-way ownership claim.
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
        format_scan_failure=None,
        studies=MappingProxyType(loaded),
        memory_storage=optuna.storages.InMemoryStorage() if ledger.backend == "memory" else None,
    )


def open_phase_study(ledger: ClaimedLedger, phase: Phase) -> optuna.Study:
    """Open or create a phase's live study for writing on a claimed ledger.

    The handle must be a :class:`ClaimedLedger` at runtime as well as in its type.
    Bind a newly created study to the claimed artifact root before returning it;
    reconfirm an existing study's root.

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
    storage: optuna.storages.BaseStorage
    if ledger.memory_storage is not None:
        storage = ledger.memory_storage
    else:
        assert ledger.storage_url is not None
        storage = _resolve_storage(ledger.storage_url)
    study = _build_phase_study(experiment, phase, storage)
    _bind_study_artifact_root(study, experiment)
    return study


def open_preview_study(experiment: Experiment, phase: Phase) -> optuna.Study:
    """Create a fresh in-memory study for a dry run without resolving configured storage.

    :param Experiment experiment: Parsed experiment supplying direction and study name.
    :param Phase phase: Phase whose sampler and search space the preview uses.
    :return optuna.Study: Fresh in-memory study for the phase.
    """
    return _build_phase_study(experiment, phase, None)


def _require_replay_matches_snapshot(
    snapshot: optuna.Study, live: optuna.Study, locator: str
) -> None:
    """Require a live replay to retain every captured trial's identity.

    Appends are allowed; missing or replaced trial identities indicate the journal
    changed between snapshot inspection and live opening.

    :param optuna.Study snapshot: Study replayed from the complete snapshot.
    :param optuna.Study live: Same study replayed from the live journal.
    :param str locator: Journal storage URL, named in the refusal.
    :raises StudyStorageUnavailableError: A snapshot trial is missing from the
        live replay, or carries a different trial id, generation, or attempt.
    """
    current = {trial.number: trial for trial in live.get_trials(deepcopy=False)}
    for captured in snapshot.get_trials(deepcopy=False):
        replayed = current.get(captured.number)
        if (
            replayed is None
            or replayed._trial_id != captured._trial_id
            or any(
                captured.user_attrs.get(key) is not None
                and replayed.user_attrs.get(key) != captured.user_attrs[key]
                for key in (GENERATION_ID_ATTR, ATTEMPT_ID_ATTR)
            )
        ):
            raise StudyStorageUnavailableError(
                f"Journal storage {journal_file_path(locator)} changed while "
                f"study {snapshot.study_name!r} was being opened: trial {captured.number} "
                "no longer matches the complete snapshot read first."
            )


def _open_existing_journal_study(storage_url: str, study_name: str) -> optuna.Study | None:
    """Open a journal study live only after a complete snapshot proves it exists.

    This avoids Optuna creating a missing journal during lookup. The live replay
    must retain the captured study and trial identities.

    :param str storage_url: Journal storage URL holding the study.
    :param str study_name: Study to open.
    :return optuna.Study | None: The live study, or ``None`` when the complete
        snapshot holds no such study.
    :raises StudyStorageUnavailableError: The snapshot is unreadable or
        incomplete, or the live replay disagrees with it.
    """
    snapshot = _load_journal_study_snapshot(storage_url, study_name)
    if snapshot is None:
        return None
    try:
        live = optuna.load_study(study_name=study_name, storage=_resolve_storage(storage_url))
    except KeyError as exc:
        raise StudyStorageUnavailableError(
            f"Journal storage {journal_file_path(storage_url)} lost study "
            f"{study_name!r} between its complete snapshot and the live replay."
        ) from exc
    _require_replay_matches_snapshot(snapshot, live, storage_url)
    return live


def open_registry_study(locator: str, study_name: str) -> optuna.Study | None:
    """Validate and open the live study named by a recovery-registry locator.

    The caller holds the experiment lock. Scan the locator's whole ledger before
    opening any study, repair a partial final record, then use
    :func:`_open_existing_journal_study` to confirm absence or matching live replay.
    Missing journals and studies return ``None`` without creating storage.

    :param str locator: Storage URL recorded in (or recovered for) the entry.
    :param str study_name: Study the registry entry names.
    :return optuna.Study | None: The live study, or ``None`` when the ledger
        confirms it does not hold one called ``study_name``.
    :raises StudySchemaMismatchError: The ledger holds unsupported PhaseSweep
        study state.
    :raises IncompleteJournalRecordError: The journal's partial final record
        could not be repaired.
    :raises StudyStorageUnavailableError: The ledger exists but could not be
        read completely, or a journal changed while the study was being opened.
    :raises ValueError: ``locator`` selects an unsupported local storage backend.
    """
    _scan_ledger_format(locator)
    _repair_incomplete_journal_record(locator)
    return _open_existing_journal_study(locator, study_name)
