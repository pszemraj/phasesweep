"""Breaking-format boundary coverage for artifact roots and local ledgers."""

from __future__ import annotations

import inspect
import json
import os
import stat
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import optuna
import pytest
from optuna.storages.journal import JournalFileSymlinkLock

from phasesweep import run_experiment
from phasesweep.config import Experiment
from phasesweep.engine import (
    ArtifactRootConflictError,
    IncompleteJournalRecordError,
    ProcessCleanupUncertainError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    read_status,
)
from phasesweep.engine import ledger as engine_ledger
from phasesweep.engine.artifact_roots import (
    ARTIFACT_ROOT_BINDING_SCHEMA_VERSION,
    _artifact_root_binding_payload,
)
from phasesweep.engine.ledger import (
    ClaimedLedger,
    ValidatedLedger,
    _resolve_storage,
    claim_ledger,
    open_existing_study,
    open_phase_study,
    open_registry_study,
    repair_incomplete_journal_record,
    validate_ledger,
)
from phasesweep.engine.locking import _experiment_lock
from phasesweep.engine.paths import _artifact_root_binding_path, _experiment_dir
from phasesweep.engine.state import ARTIFACT_ROOT_ATTR, STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION
from phasesweep.errors import OperatorAction, PhaseSweepError
from phasesweep.mcp.recovery import (
    _load_recovery_studies,
)
from tests.conftest import (
    make_experiment,
    mark_current_format,
    requires_nonroot,
    temporary_umask,
    write_constant_trainer,
)
from tests.ledger_fixtures import (
    ledger_file,
    materialize,
    tree_snapshot,
)
from tests.recovery_helpers import load_only_recovery_needs


def _experiment(tmp_path: Path, *, storage: str | None) -> Experiment:
    """Build one real-trial experiment for format-boundary tests."""
    trainer = write_constant_trainer(tmp_path)
    return make_experiment(workdir=tmp_path / "runs", storage=storage, trainer=trainer, n_trials=1)


@pytest.mark.parametrize("durable_entry", ["generations", "attempts"])
def test_pre_cutover_memory_output_is_refused_before_mutation(
    tmp_path: Path, durable_entry: str
) -> None:
    """Completed and interrupted unmarked memory-backed trees are never adopted."""
    experiment = _experiment(tmp_path, storage=None)
    root = _experiment_dir(experiment)
    state = root / durable_entry
    state.mkdir(parents=True)
    (state / "old-state").write_text("0.3.1\n", encoding="utf-8")
    before = tree_snapshot(root)

    with pytest.raises(
        ArtifactRootConflictError,
        match=r"pre-cutover.*fresh artifact root.*0\.3\.1.*Nothing was written",
    ):
        run_experiment(experiment)

    assert tree_snapshot(root) == before
    assert not _artifact_root_binding_path(experiment).exists()


def test_pre_cutover_artifact_binding_version_is_refused_before_mutation(
    tmp_path: Path,
) -> None:
    """An old binding marker cannot authorize a current-format artifact tree."""
    experiment = _experiment(tmp_path, storage=None)
    root = _experiment_dir(experiment)
    binding_path = _artifact_root_binding_path(experiment)
    binding_path.parent.mkdir(parents=True)
    binding_path.write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_ROOT_BINDING_SCHEMA_VERSION - 1,
                "experiment": experiment.experiment,
                "artifact_root": str(root.resolve()),
                "storage_key": None,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    before = tree_snapshot(root)

    with pytest.raises(
        ArtifactRootConflictError,
        match=r"unsupported pre-cutover.*format 2.*fresh artifact root",
    ):
        run_experiment(experiment)

    assert tree_snapshot(root) == before


@pytest.mark.parametrize("schema_version", [2, None], ids=["old-version", "unmarked"])
def test_old_populated_ledger_is_refused_with_a_fresh_output_root(
    tmp_path: Path, schema_version: int | None
) -> None:
    """Changing experiment/workdir cannot reinterpret an old populated ledger as fresh."""
    ledger = tmp_path / "old.journal"
    storage = f"journal:///{ledger}"
    old = optuna.create_study(
        study_name="old_experiment::removed_phase", storage=_resolve_storage(storage)
    )
    if schema_version is not None:
        old.set_user_attr(STUDY_SCHEMA_ATTR, schema_version)
    trial = old.ask()
    old.tell(trial, 1.0)
    ledger_before = {
        path.name: path.read_bytes()
        for path in sorted(tmp_path.glob(f"{ledger.name}*"))
        if path.is_file()
    }
    experiment = _experiment(tmp_path, storage=storage).model_copy(
        update={"experiment": "fresh_name", "workdir": str(tmp_path / "fresh-runs")}
    )

    with pytest.raises(
        StudySchemaMismatchError,
        match=r"old_experiment::removed_phase.*fresh local storage ledger.*0\.3\.1",
    ):
        run_experiment(experiment)

    assert not _experiment_dir(experiment).exists()
    assert {
        path.name: path.read_bytes()
        for path in sorted(tmp_path.glob(f"{ledger.name}*"))
        if path.is_file()
    } == ledger_before


@pytest.mark.parametrize("leftover_study", ["t::retired", "other_experiment::phase"])
@pytest.mark.integration
def test_bound_root_status_refuses_empty_stamped_legacy_studies(
    tmp_path: Path, leftover_study: str
) -> None:
    """Status rejects empty retired or foreign studies with an old format stamp."""
    storage = f"journal:///{tmp_path / 'current.journal'}"
    experiment = _experiment(tmp_path, storage=storage)
    run_experiment(experiment)
    leftover = optuna.create_study(study_name=leftover_study, storage=_resolve_storage(storage))
    leftover.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION - 1)

    with pytest.raises(StudySchemaMismatchError, match="pre-cutover or unsupported"):
        read_status(experiment)


def test_validate_ledger_on_a_fresh_root_creates_nothing(tmp_path: Path) -> None:
    """Validating a never-run experiment reports an unbound tree and writes nothing.

    The handle is what every read path takes, so obtaining one must not be the
    thing that brings the tree or the ledger into existence: a status poll on a
    config that has never run has to stay observable and leave no trace.
    """
    ledger_path = tmp_path / "fresh" / "study.journal"
    experiment = _experiment(tmp_path, storage=f"journal:///{ledger_path}")

    ledger = validate_ledger(experiment)

    assert ledger.binding_state == "unbound"
    assert ledger.backend == "journal"
    assert ledger.experiment_name == experiment.experiment
    assert ledger.artifact_root == str(_experiment_dir(experiment).resolve())
    assert not ledger_path.exists()
    assert not ledger_path.parent.exists()
    assert not _artifact_root_binding_path(experiment).exists()
    assert not _experiment_dir(experiment).exists()


def test_claim_ledger_creates_ledger_parent_but_validate_does_not(tmp_path: Path) -> None:
    """Claiming brings the ledger's directory into existence before it binds the tree.

    Every read path validates, so validating must never materialize a ledger
    directory. The claim is the first step that writes, and the directory is
    its first write: a path that cannot hold a ledger then fails before the
    tree names that ledger, while the path can still be corrected. Neither
    step creates the ledger file itself; the first live open does. The journal
    backend creates only that file, so a claim that skipped the directory
    would bind the tree to a ledger its first open cannot create.
    """
    missing = tmp_path / "not-created-by-a-read"
    ledger_file = missing / "study.journal"
    experiment = _experiment(tmp_path, storage=f"journal:///{ledger_file}")

    ledger = validate_ledger(experiment)

    assert ledger.backend == "journal"
    assert ledger.ledger_path == ledger_file
    assert not missing.exists()

    claimed = claim_ledger(ledger)

    assert missing.is_dir()
    assert _artifact_root_binding_path(experiment).is_file()
    assert not ledger_file.exists()

    open_phase_study(claimed, experiment.phases[0])

    assert ledger_file.is_file()


def test_unwritable_ledger_directory_fails_before_the_tree_is_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ledger directory that cannot be created leaves the tree unbound and correctable."""
    experiment = _experiment(
        tmp_path, storage=f"journal:///{tmp_path / 'missing' / 'study.journal'}"
    )
    real_mkdir = Path.mkdir

    def refuse_ledger_directory(path: Path, *args: object, **kwargs: object) -> None:
        if path == tmp_path / "missing":
            raise PermissionError(13, "Permission denied", str(path))
        real_mkdir(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", refuse_ledger_directory)

    with pytest.raises(PhaseSweepError, match="could not be created") as refused:
        claim_ledger(validate_ledger(experiment))

    assert isinstance(refused.value.__cause__, PermissionError)
    assert not _artifact_root_binding_path(experiment).exists()


@pytest.mark.parametrize("offered", ["storage-url", "validated-ledger"])
def test_live_openers_reject_unvalidated_storage(tmp_path: Path, offered: str) -> None:
    """Nothing weaker than a claimed handle reaches the live opener.

    The annotation already refuses both at type-check time; this pins the
    runtime guard for callers the checker never sees. A bare validated handle
    is the dangerous case: it carries every field the opener reads, and the
    only thing it lacks is the proof that the tree and studies were claimed.
    The refusal lands before storage is resolved, so nothing is created.
    """
    missing = tmp_path / "ledger"
    storage = f"journal:///{missing / 'study.journal'}"
    experiment = _experiment(tmp_path, storage=storage)
    handle: object = storage if offered == "storage-url" else validate_ledger(experiment)

    with pytest.raises(TypeError, match="requires the ClaimedLedger"):
        open_phase_study(handle, experiment.phases[0])  # type: ignore[arg-type]

    assert not missing.exists()
    assert not _artifact_root_binding_path(experiment).exists()


def test_claim_ledger_binds_tree_then_returns_bound_handle(tmp_path: Path) -> None:
    """Claiming a fresh tree records its owner and returns the handle that opens studies.

    The validated handle keeps saying what validation saw ("unbound"); only the
    claimed handle records that the fixed order's writes happened, and a study
    opened through it carries the claimed root. Reclaiming the bound tree finds
    that study in its one discovery pass.
    """
    experiment = _experiment(tmp_path, storage=f"journal:///{tmp_path / 'study.journal'}")
    validated = validate_ledger(experiment)

    claimed = claim_ledger(validated)

    assert validated.binding_state == "unbound"
    assert isinstance(claimed, ClaimedLedger)
    assert claimed.binding_state == "bound"
    assert claimed.artifact_root == validated.artifact_root
    assert dict(claimed.studies) == {}
    binding = json.loads(_artifact_root_binding_path(experiment).read_text(encoding="utf-8"))
    assert binding == _artifact_root_binding_payload(experiment)
    assert validate_ledger(experiment).binding_state == "bound"

    study = open_phase_study(claimed, experiment.phases[0])

    assert study.user_attrs[ARTIFACT_ROOT_ATTR] == claimed.artifact_root
    reclaimed = claim_ledger(validate_ledger(experiment))
    assert list(reclaimed.studies) == [experiment.phases[0].name]
    with pytest.raises(TypeError):
        reclaimed.studies["other"] = study  # type: ignore[index]


def test_claim_ledger_writes_tree_binding_before_claiming_studies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between the two claims leaves a bound tree and unclaimed studies.

    The tree records its ledger first and the studies record their root second.
    A process that dies in between leaves state the next run claims normally.
    The opposite order could leave a study pointing at a tree that does not
    name the same ledger, which no later run could tell apart from a conflict.
    """
    import phasesweep.engine.ledger as ledger_module

    storage = f"journal:///{tmp_path / 'study.journal'}"
    base = _experiment(tmp_path, storage=storage)
    experiment = base.model_copy(
        update={"phases": [base.phases[0], base.phases[0].model_copy(update={"name": "q"})]}
    )
    for phase in experiment.phases:
        optuna.create_study(study_name=f"t::{phase.name}", storage=_resolve_storage(storage))

    class CrashBeforeStudyClaim(Exception):
        pass

    def crash(*_args: object, **_kwargs: object) -> None:
        assert _artifact_root_binding_path(experiment).is_file()
        raise CrashBeforeStudyClaim

    with monkeypatch.context() as patched:
        patched.setattr(ledger_module, "_claim_study_artifact_root", crash)
        with pytest.raises(CrashBeforeStudyClaim):
            claim_ledger(validate_ledger(experiment))

    binding = json.loads(_artifact_root_binding_path(experiment).read_text(encoding="utf-8"))
    assert binding == _artifact_root_binding_payload(experiment)
    for phase in experiment.phases:
        study = optuna.load_study(study_name=f"t::{phase.name}", storage=_resolve_storage(storage))
        assert ARTIFACT_ROOT_ATTR not in study.user_attrs

    reclaimed = claim_ledger(validate_ledger(experiment))

    assert sorted(reclaimed.studies) == ["p", "q"]
    for study in reclaimed.studies.values():
        assert study.user_attrs[ARTIFACT_ROOT_ATTR] == reclaimed.artifact_root


def _bound_tree_with_legacy_study(tmp_path: Path, *, legacy: bool = True) -> Experiment:
    """Build a bound current-format tree whose ledger may also hold pre-cutover state.

    The phase study is current-format and root-claimed, so every refusal these
    tests observe comes from the ledger-wide format scan and its handling, not
    from the phase study itself.

    :param Path tmp_path: Test-owned directory for the tree and the ledger.
    :param bool legacy: Add a populated, unmarked ``legacy::p`` study, which
        only a completed format scan can see.
    :return Experiment: Experiment whose tree is bound to that ledger.
    """
    storage = f"journal:///{tmp_path / 'current.journal'}"
    experiment = _experiment(tmp_path, storage=storage)
    study = optuna.create_study(study_name="t::p", storage=_resolve_storage(storage))
    study.add_trial(optuna.trial.create_trial(value=0.5, state=optuna.trial.TrialState.COMPLETE))
    study.set_user_attr(ARTIFACT_ROOT_ATTR, str(_experiment_dir(experiment).resolve()))
    mark_current_format(experiment, study)
    if legacy:
        old = optuna.create_study(study_name="legacy::p", storage=_resolve_storage(storage))
        old.add_trial(optuna.trial.create_trial(value=0.5, state=optuna.trial.TrialState.COMPLETE))
    return experiment


def _scan_fails(monkeypatch: pytest.MonkeyPatch, *, times: int) -> list[str | None]:
    """Make the ledger-wide format scan fail as if the ledger were briefly locked.

    :param pytest.MonkeyPatch monkeypatch: Fixture that owns the patch.
    :param int times: Number of leading scans that fail before scans run for real.
    :return list[str | None]: Storage URL of every scan attempted, in order.
    """
    import phasesweep.engine.ledger as ledger_module

    real_scan = ledger_module._scan_ledger_format
    attempts: list[str | None] = []

    def flaky_scan(storage: str | None) -> None:
        attempts.append(storage)
        if len(attempts) <= times:
            raise StudyStorageUnavailableError("injected: the ledger was briefly locked")
        real_scan(storage)

    monkeypatch.setattr(ledger_module, "_scan_ledger_format", flaky_scan)
    return attempts


def test_inconclusive_scan_on_a_bound_tree_never_reports_unchecked_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Status reports trial data unavailable rather than counts no scan checked.

    A bound tree tolerates an unreadable scan so publication data stays
    readable, but the ledger may hold state this release refuses, and only a
    completed scan can tell. Counts read around that gap would present the
    ledger as current-format when nothing verified it.
    """
    experiment = _bound_tree_with_legacy_study(tmp_path)
    with pytest.raises(StudySchemaMismatchError, match="'legacy::p'"):
        read_status(experiment)
    _scan_fails(monkeypatch, times=1)

    status = read_status(experiment)

    phase = status["phases"][0]
    assert phase["trial_data_available"] is False
    assert phase["running_attempts"] is None
    assert not any(phase["trials"].values())


def test_unverified_handle_records_the_gap_and_opens_nothing_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tolerated scan failure is on the handle, and no live opener accepts it.

    Recovery opens existing studies through the validated handle, so an
    unverified handle must not open one: it would read and then reap studies
    in a ledger whose format nothing checked.
    """
    experiment = _bound_tree_with_legacy_study(tmp_path)
    _scan_fails(monkeypatch, times=1)

    ledger = validate_ledger(experiment)

    assert ledger.binding_state == "bound"
    assert ledger.format_verified is False
    with pytest.raises(StudyStorageUnavailableError, match="briefly locked"):
        open_existing_study(ledger, experiment.phases[0])


@pytest.mark.parametrize("second_scan", ["completes", "fails-again"])
def test_claim_rescans_an_unverified_ledger_before_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second_scan: str
) -> None:
    """A run never proceeds on a ledger whose format scan did not complete.

    Claiming mutates the tree and studies, so the scan validation tolerated has
    to complete before discovery: the pre-cutover study then refuses the run,
    and a second unreadable scan stops it as unavailable storage. Nothing is
    written either way.
    """
    experiment = _bound_tree_with_legacy_study(tmp_path)
    ledger_file = tmp_path / "current.journal"
    before = ledger_file.read_bytes()
    scans = _scan_fails(monkeypatch, times=1 if second_scan == "completes" else 2)

    if second_scan == "completes":
        with pytest.raises(StudySchemaMismatchError, match="'legacy::p'"):
            run_experiment(experiment)
    else:
        with pytest.raises(ProcessCleanupUncertainError) as exc_info:
            run_experiment(experiment)
        assert isinstance(exc_info.value.__cause__, StudyStorageUnavailableError)

    assert len(scans) == 2
    assert ledger_file.read_bytes() == before


def test_verified_handle_reads_and_claims_after_one_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed scan is trusted as before: counts are real and claim scans no more."""
    experiment = _bound_tree_with_legacy_study(tmp_path, legacy=False)
    scans = _scan_fails(monkeypatch, times=0)

    ledger = validate_ledger(experiment)
    status = read_status(experiment)
    claimed = claim_ledger(ledger)

    assert ledger.format_verified is True
    assert claimed.format_verified is True
    assert status["phases"][0]["trial_data_available"] is True
    assert status["phases"][0]["trials"]["COMPLETE"] == 1
    assert len(scans) == 2  # one for this handle, one for read_status's own; none in claim


@pytest.mark.integration
def test_startup_scans_the_ledger_format_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Root preflight scans once before Optuna loads any declared study."""
    import phasesweep.engine.ledger as ledger

    storage = f"journal:///{tmp_path / 'current.journal'}"
    experiment = _experiment(tmp_path, storage=storage)
    real_validate = ledger._scan_ledger_format
    real_load = ledger._load_existing_phase_study
    calls = 0

    def counted_validate(candidate: str | None) -> None:
        nonlocal calls
        calls += 1
        real_validate(candidate)

    def checked_load(candidate: ValidatedLedger, phase):
        assert calls == 1
        return real_load(candidate, phase)

    monkeypatch.setattr(ledger, "_scan_ledger_format", counted_validate)
    monkeypatch.setattr(ledger, "_load_existing_phase_study", checked_load)

    run_experiment(experiment)

    assert calls == 1


@pytest.mark.integration
def test_empty_unstamped_phase_study_can_resume_in_current_ledger(tmp_path: Path) -> None:
    """An interrupted new phase is stamped and resumed rather than classified as old state."""
    ledger = tmp_path / "current.journal"
    storage = f"journal:///{ledger}"
    experiment = _experiment(tmp_path, storage=storage)
    run_experiment(experiment)

    resumed_phase = experiment.phases[0].model_copy(update={"name": "q"})
    resumed = experiment.model_copy(update={"phases": [experiment.phases[0], resumed_phase]})
    interrupted = optuna.create_study(
        study_name="t::q", storage=_resolve_storage(resumed.resolved_storage)
    )
    assert interrupted.user_attrs.get(STUDY_SCHEMA_ATTR) is None

    run_experiment(resumed)

    study = optuna.load_study(study_name="t::q", storage=_resolve_storage(resumed.resolved_storage))
    assert study.user_attrs[STUDY_SCHEMA_ATTR] == STUDY_SCHEMA_VERSION
    assert len(study.trials) == 1


_PARTIAL_FINAL_RECORDS = {
    "torn": b'{"op_code": 5, "worker_id": "crashed',
    "undecodable": b"not a journal record\n",
}


@pytest.mark.parametrize("damage", sorted(_PARTIAL_FINAL_RECORDS))
def test_writers_repair_a_partial_final_journal_record(tmp_path: Path, damage: str) -> None:
    """Every path that may write repairs a bad last journal line before it proceeds.

    Reads skip that line as Optuna does, so the format scan passes. A write no
    longer refuses on sight: under Optuna's own journal lock it backs the
    damaged bytes up beside the journal and truncates back to the last
    complete record, for ``claim_ledger``, ``open_registry_study``, and a
    confirmed recovery alike. Unconfirmed inspection no longer checks the tail
    at all, so it neither raises nor changes a byte.
    """

    def freshly_damaged(label: str) -> tuple[Experiment, Path, bytes, bytes]:
        # Each writer gets its own copy of the fixture: a repair mutates the
        # journal it runs against, so the four scenarios cannot share one.
        materialized = materialize("current-journal", tmp_path / label, mode="tree")
        experiment = materialized.experiment
        journal = ledger_file(materialized, "journal")
        complete = journal.read_bytes()
        damaged = complete + _PARTIAL_FINAL_RECORDS[damage]
        journal.write_bytes(damaged)
        return experiment, journal, complete, damaged

    def assert_repaired(journal: Path, complete: bytes, damaged: bytes) -> None:
        assert journal.read_bytes() == complete
        backups = sorted(journal.parent.glob(f"{journal.name}.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == damaged
        assert not journal.with_name(f"{journal.name}.lock").exists()
        assert not _staged_locks(journal)

    experiment, journal, complete, damaged = freshly_damaged("claim")
    claim_ledger(validate_ledger(experiment))
    assert_repaired(journal, complete, damaged)

    experiment, journal, complete, damaged = freshly_damaged("registry")
    assert experiment.resolved_storage is not None
    assert open_registry_study(experiment.resolved_storage, "t::p") is not None
    assert_repaired(journal, complete, damaged)

    experiment, journal, complete, damaged = freshly_damaged("confirmed")
    with _experiment_lock(experiment):
        _load_recovery_studies(experiment, load_only_recovery_needs(), confirm=True)
    assert_repaired(journal, complete, damaged)

    experiment, journal, complete, damaged = freshly_damaged("inspect")
    _load_recovery_studies(experiment, load_only_recovery_needs())
    assert journal.read_bytes() == damaged
    assert not list(journal.parent.glob(f"{journal.name}.*.bak"))


def test_repair_leaves_an_append_that_finished_while_it_awaited_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repair that finds the record complete once it holds the lock writes nothing.

    The scan before the lock may see a partial final line a concurrent append
    is still writing. Optuna's own journal lock excludes every writer, so if
    that append settles the journal back to fully complete before this
    repair's rescan runs under the lock, no offset from before the lock is
    ever used: nothing is truncated, and no backup is written.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    experiment = materialized.experiment
    journal = ledger_file(materialized, "journal")
    complete = journal.read_bytes()
    journal.write_bytes(complete + _PARTIAL_FINAL_RECORDS["torn"])

    real_acquire = engine_ledger._JournalLock.acquire

    def finish_append_then_acquire(lock: engine_ledger._JournalLock) -> None:
        journal.write_bytes(complete)
        real_acquire(lock)

    monkeypatch.setattr(engine_ledger._JournalLock, "acquire", finish_append_then_acquire)

    repair_incomplete_journal_record(validate_ledger(experiment))

    assert journal.read_bytes() == complete
    assert not list(journal.parent.glob(f"{journal.name}.*.bak"))


def _manual_lock_clock(
    monkeypatch: pytest.MonkeyPatch, on_pause: Callable[[], None] | None = None
) -> dict[str, float]:
    """Replace the repair's lock-wait clock with a manual one each pause advances."""
    clock = {"now": 0.0, "pauses": 0.0}

    def pause(seconds: float) -> None:
        clock["now"] += seconds
        clock["pauses"] += 1
        if on_pause is not None:
            on_pause()

    monkeypatch.setattr(
        engine_ledger, "time", SimpleNamespace(monotonic=lambda: clock["now"], sleep=pause)
    )
    return clock


def _staged_locks(journal: Path) -> list[Path]:
    """List the private names a repair's journal lock was staged under and left behind."""
    return list(journal.parent.glob(f".{journal.name}.lock.*.tmp"))


def _take_over_journal_lock(journal: Path) -> os.stat_result:
    """Remove a journal's lock and take a new one, as a waiting Optuna writer does to a stale lock.

    :return os.stat_result: The new lock symlink's own status.
    """
    JournalFileSymlinkLock(str(journal)).release()
    JournalFileSymlinkLock(str(journal)).acquire()
    return journal.with_name(f"{journal.name}.lock").lstat()


def test_repair_waits_for_a_journal_lock_another_writer_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A journal lock held when the repair starts is awaited, then taken, not refused on sight."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = ledger_file(materialized, "journal")
    complete = journal.read_bytes()
    journal.write_bytes(complete + _PARTIAL_FINAL_RECORDS["torn"])
    lock_path = journal.with_name(f"{journal.name}.lock")
    lock_path.symlink_to(journal)
    ledger = validate_ledger(materialized.experiment)
    clock = _manual_lock_clock(monkeypatch, on_pause=lock_path.unlink)

    repair_incomplete_journal_record(ledger)

    assert clock["pauses"] == 1
    assert journal.read_bytes() == complete
    assert len(list(journal.parent.glob(f"{journal.name}.*.bak"))) == 1
    assert not lock_path.exists()


@pytest.mark.parametrize("lock_target", ["journal", "dangling"])
def test_repair_refuses_a_journal_lock_that_is_never_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_target: str
) -> None:
    """A journal lock a killed writer left behind ends the repair in a bounded refusal.

    Optuna 4.0 and 4.1 wait on such a lock forever, and later releases retry
    forever when its symlink dangles, as a lock taken through a nested
    relative journal path does. The repair gives up once its wait has run
    out, leaves the lock and the journal as they were, and names the lock to
    remove. It runs in a thread so a wait that never ends fails the test
    instead of hanging the suite.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = ledger_file(materialized, "journal")
    damaged = journal.read_bytes() + _PARTIAL_FINAL_RECORDS["torn"]
    journal.write_bytes(damaged)
    lock_path = journal.with_name(f"{journal.name}.lock")
    target = str(journal) if lock_target == "journal" else f"ledger/{journal.name}"
    lock_path.symlink_to(target)
    ledger = validate_ledger(materialized.experiment)
    clock = _manual_lock_clock(monkeypatch)

    raised: list[BaseException] = []

    def repair() -> None:
        try:
            repair_incomplete_journal_record(ledger)
        except BaseException as exc:  # noqa: BLE001 - handed to the assertions below
            raised.append(exc)

    worker = threading.Thread(target=repair, daemon=True)
    worker.start()
    worker.join(timeout=10.0)

    assert not worker.is_alive()
    assert clock["now"] == pytest.approx(engine_ledger._JOURNAL_LOCK_WAIT_SECONDS)
    assert len(raised) == 1
    assert isinstance(raised[0], IncompleteJournalRecordError)
    assert raised[0].action == OperatorAction.RESTORE_LEDGER
    assert f"remove {lock_path}" in str(raised[0])
    assert journal.read_bytes() == damaged
    assert os.readlink(lock_path) == target
    assert not list(journal.parent.glob(f"{journal.name}.*.bak"))
    assert not _staged_locks(journal)


@pytest.mark.parametrize("spelling", ["nested-relative", "symlink-alias"])
def test_journal_writers_and_repair_lock_the_journals_real_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    """Every journal lock is ``<real journal>.lock``, a symlink to the journal's real path.

    Optuna's lock is a symlink named after, and pointing at, the path its
    backend was given. A nested relative URL such as
    ``journal:///ledger/study.journal`` would leave that symlink dangling, so
    a stale lock could never be released. A file symlink to the journal
    would name a second lock, so an append through one spelling and a repair
    through the other would not exclude each other. Optuna's appends and the
    repair both lock the resolved path instead.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ledger").mkdir()
    journal = tmp_path.resolve() / "ledger" / "study.journal"
    if spelling == "nested-relative":
        url = "journal:///ledger/study.journal"
    else:
        (tmp_path / "alias.journal").symlink_to("ledger/study.journal")
        url = f"journal:///{tmp_path}/alias.journal"
    locks: set[tuple[str, str]] = set()
    real_symlink = os.symlink
    real_link = os.link

    def recording_symlink(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        if os.fspath(dst).endswith(".lock"):
            locks.add((os.fspath(dst), os.fspath(src)))
        real_symlink(src, dst, *args, **kwargs)

    def recording_link(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        # The repair links its staged symlink into place as the lock.
        if os.fspath(dst).endswith(".lock"):
            locks.add((os.fspath(dst), os.readlink(src)))
        real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "symlink", recording_symlink)
    monkeypatch.setattr(os, "link", recording_link)

    optuna.create_study(study_name="s", storage=engine_ledger._resolve_storage(url))
    complete = journal.read_bytes()
    journal.write_bytes(complete + _PARTIAL_FINAL_RECORDS["torn"])
    engine_ledger._repair_incomplete_journal_record(url)

    assert locks == {(f"{journal}.lock", str(journal))}
    assert journal.read_bytes() == complete


def test_repair_through_a_symlink_alias_is_excluded_by_the_real_names_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repair reaching a journal through a symlink never cuts a writer's in-flight record.

    Two experiments may share one journal, one naming it through a file
    symlink. An append still in flight through the real name looks exactly
    like a crashed writer's partial record. The repair takes the same lock
    that writer holds, so it waits and refuses, and the bytes stay whole.
    """
    real = tmp_path.resolve() / "real.journal"
    alias = tmp_path / "alias.journal"
    optuna.create_study(
        study_name="s", storage=engine_ledger._resolve_storage(f"journal:///{real}")
    )
    alias.symlink_to(real)
    in_flight = real.read_bytes() + _PARTIAL_FINAL_RECORDS["torn"]
    real.write_bytes(in_flight)
    real.with_name(f"{real.name}.lock").symlink_to(real)  # the appending writer's lock
    _manual_lock_clock(monkeypatch)

    with pytest.raises(IncompleteJournalRecordError, match="still held"):
        engine_ledger._repair_incomplete_journal_record(f"journal:///{alias}")

    assert real.read_bytes() == in_flight
    assert not alias.with_name(f"{alias.name}.lock").exists()


def test_repair_truncates_only_while_holding_the_journal_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The truncation step only ever runs while this process holds Optuna's own journal lock."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    experiment = materialized.experiment
    journal = ledger_file(materialized, "journal")
    complete = journal.read_bytes()
    journal.write_bytes(complete + _PARTIAL_FINAL_RECORDS["torn"])
    lock_path = journal.with_name(f"{journal.name}.lock")

    real_truncate = engine_ledger._truncate_incomplete_journal_record

    def truncate_while_locked(lock: engine_ledger._JournalLock) -> tuple[Path, int] | None:
        assert lock_path.is_symlink()
        assert lock.held()
        return real_truncate(lock)

    monkeypatch.setattr(engine_ledger, "_truncate_incomplete_journal_record", truncate_while_locked)

    repair_incomplete_journal_record(validate_ledger(experiment))

    assert not lock_path.exists()
    assert journal.read_bytes() == complete


def test_journal_lock_appears_only_after_the_journal_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The journal's modification time changes before the repair's lock appears.

    An Optuna writer whose grace period ran out on a stale lock removes it,
    then checks the next lock it meets against the modification time it
    last saw. Had the journal not changed since, it would remove the
    repair's lock too.
    """
    journal = tmp_path / "study.journal"
    journal.write_bytes(b"")
    os.utime(journal, (0, 0))
    seen: list[float] = []
    real_link = os.link

    def recording_link(src: str, dst: str, *args: Any, **kwargs: Any) -> None:
        seen.append(journal.stat().st_mtime)
        real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", recording_link)
    lock = engine_ledger._JournalLock(journal)
    lock.acquire()

    assert lock.release()
    assert seen
    assert all(mtime > 0 for mtime in seen)


@pytest.mark.parametrize("taken_over", ["before truncation", "after truncation"])
def test_repair_never_truncates_under_or_releases_a_lock_another_process_took_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, taken_over: str
) -> None:
    """A repair whose journal lock was taken over stops, and leaves the new lock in place.

    Optuna's lock records no owner, so a process that removes it and takes
    its own leaves a lock indistinguishable by name. Found before
    truncation, the repair refuses with the journal untouched. Found at
    release, the truncation already ran under the repair's own lock, and
    the repair still reports the takeover. Either way the successor's lock
    survives, so the writer holding it stays excluded.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = ledger_file(materialized, "journal")
    complete = journal.read_bytes()
    damaged = complete + _PARTIAL_FINAL_RECORDS["torn"]
    journal.write_bytes(damaged)
    lock_path = journal.with_name(f"{journal.name}.lock")
    successor: list[os.stat_result] = []

    if taken_over == "before truncation":
        real_fsync_directory = engine_ledger.fsync_directory

        def fsync_then_take_over(path: Path) -> None:
            real_fsync_directory(path)
            successor.append(_take_over_journal_lock(journal))

        monkeypatch.setattr(engine_ledger, "fsync_directory", fsync_then_take_over)
    else:
        real_truncate = engine_ledger._truncate_incomplete_journal_record

        def truncate_then_take_over(lock: engine_ledger._JournalLock) -> tuple[Path, int] | None:
            repaired = real_truncate(lock)
            successor.append(_take_over_journal_lock(journal))
            return repaired

        monkeypatch.setattr(
            engine_ledger, "_truncate_incomplete_journal_record", truncate_then_take_over
        )

    with pytest.raises(IncompleteJournalRecordError, match="removed or replaced") as excinfo:
        repair_incomplete_journal_record(validate_ledger(materialized.experiment))

    assert excinfo.value.action == OperatorAction.RETRY
    assert journal.read_bytes() == (damaged if taken_over == "before truncation" else complete)
    assert lock_path.lstat().st_ino == successor[0].st_ino
    assert len(list(journal.parent.glob(f"{journal.name}.*.bak"))) == 1
    assert not _staged_locks(journal)


@pytest.mark.integration
def test_waiting_optuna_writer_never_takes_over_a_long_repairs_journal_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer waiting out its grace period gets the lock only once the repair releases it.

    Optuna 4.2 and later remove a lock whose journal a waiting writer has
    seen unchanged for its grace period, 30 seconds by default, and a
    repair of a large journal can hold the lock longer. The repair
    refreshes the journal's modification time while it holds the lock, so
    a real Optuna writer with a one-second grace period, waiting through
    three seconds of the repair, never takes it over.
    """
    if "grace_period" not in inspect.signature(JournalFileSymlinkLock).parameters:
        pytest.skip("Optuna before 4.2 never takes a journal lock over")
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = ledger_file(materialized, "journal")
    complete = journal.read_bytes()
    journal.write_bytes(complete + _PARTIAL_FINAL_RECORDS["torn"])
    monkeypatch.setattr(engine_ledger, "_JOURNAL_LOCK_REFRESH_SECONDS", 0.1)
    with pytest.warns(UserWarning, match="grace_period"):
        waiter_lock = JournalFileSymlinkLock(str(journal), grace_period=1)
    waiter_holds = threading.Event()

    def wait_for_lock() -> None:
        waiter_lock.acquire()
        waiter_holds.set()

    waiter = threading.Thread(target=wait_for_lock, daemon=True)
    real_fsync_directory = engine_ledger.fsync_directory

    def hold_past_the_grace_period(path: Path) -> None:
        real_fsync_directory(path)
        waiter.start()
        assert not waiter_holds.wait(3.0)

    monkeypatch.setattr(engine_ledger, "fsync_directory", hold_past_the_grace_period)

    repair_incomplete_journal_record(validate_ledger(materialized.experiment))

    assert journal.read_bytes() == complete
    assert waiter_holds.wait(timeout=10.0)
    waiter_lock.release()


def test_repair_backup_is_no_more_permissive_than_the_journal(tmp_path: Path) -> None:
    """A private journal's backup stays private under a permissive umask.

    The backup holds every study and trial attribute the journal does, so it
    is created with the journal's mode, which the umask can only narrow.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = ledger_file(materialized, "journal")
    journal.write_bytes(journal.read_bytes() + _PARTIAL_FINAL_RECORDS["torn"])
    journal.chmod(0o600)

    with temporary_umask(0o022):
        repair_incomplete_journal_record(validate_ledger(materialized.experiment))

    (backup,) = journal.parent.glob(f"{journal.name}.*.bak")
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600


def test_repair_refuses_a_journal_with_another_hard_linked_name(tmp_path: Path) -> None:
    """A journal reachable by a second hard-linked name is never truncated.

    An append through the other name takes that name's own journal lock, so
    holding this name's lock proves nothing about it. The repair refuses
    before writing a backup, and the journal keeps every byte.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = ledger_file(materialized, "journal")
    damaged = journal.read_bytes() + _PARTIAL_FINAL_RECORDS["torn"]
    journal.write_bytes(damaged)
    os.link(journal, tmp_path / "other-name.journal")

    with pytest.raises(IncompleteJournalRecordError) as excinfo:
        repair_incomplete_journal_record(validate_ledger(materialized.experiment))

    assert excinfo.value.action == OperatorAction.RESTORE_LEDGER
    assert "2 hard links" in str(excinfo.value)
    assert journal.read_bytes() == damaged
    assert not list(journal.parent.glob(f"{journal.name}.*.bak"))
    assert not journal.with_name(f"{journal.name}.lock").exists()


def test_repair_refuses_a_journal_that_grew_under_its_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A journal that changed size while this repair held its lock is never truncated.

    A writer that ignored the lock is exactly what the size check guards
    against: the repair still backs the damaged bytes up, but it refuses to
    guess which of the old or new bytes past that point are safe to keep.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    experiment = materialized.experiment
    journal = ledger_file(materialized, "journal")
    complete = journal.read_bytes()
    damaged = complete + _PARTIAL_FINAL_RECORDS["torn"]
    journal.write_bytes(damaged)
    appended = b'{"op_code": 0}\n'

    real_fsync_directory = engine_ledger.fsync_directory

    def fsync_then_append(path: Path) -> None:
        real_fsync_directory(path)
        with journal.open("ab") as target:
            target.write(appended)

    monkeypatch.setattr(engine_ledger, "fsync_directory", fsync_then_append)

    with pytest.raises(IncompleteJournalRecordError) as excinfo:
        repair_incomplete_journal_record(validate_ledger(experiment))

    assert excinfo.value.action == OperatorAction.RETRY
    assert journal.read_bytes() == damaged + appended
    backups = sorted(journal.parent.glob(f"{journal.name}.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == damaged


@requires_nonroot
@pytest.mark.parametrize("step", ["lock", "backup"])
def test_repair_that_cannot_write_beside_the_journal_leaves_it_unchanged(
    tmp_path: Path, step: str
) -> None:
    """A repair refused its lock or its backup leaves the journal exactly as it was.

    A read-only journal directory refuses Optuna's lock symlink before
    anything else. The backup is reached only by a caller that holds the lock
    already, so that step takes the lock before the directory turns read-only
    and is driven directly.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = ledger_file(materialized, "journal")
    complete = journal.read_bytes()
    damaged = complete + _PARTIAL_FINAL_RECORDS["torn"]
    journal.write_bytes(damaged)
    ledger = validate_ledger(materialized.experiment)
    lock = engine_ledger._JournalLock(journal)
    if step == "backup":
        lock.acquire()
    mode = journal.parent.stat().st_mode
    journal.parent.chmod(0o555)
    try:
        with pytest.raises(IncompleteJournalRecordError) as excinfo:
            if step == "lock":
                repair_incomplete_journal_record(ledger)
            else:
                engine_ledger._truncate_incomplete_journal_record(lock)
    finally:
        journal.parent.chmod(mode)
        lock.release()

    assert excinfo.value.action == OperatorAction.RESTORE_LEDGER
    assert "could not be repaired" in str(excinfo.value)
    assert journal.read_bytes() == damaged
    assert not list(journal.parent.glob(f"{journal.name}.*.bak"))
    assert not journal.with_name(f"{journal.name}.lock").exists()
    assert not _staged_locks(journal)
