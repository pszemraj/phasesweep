"""Breaking-format boundary coverage for artifact roots and local ledgers."""

from __future__ import annotations

import json
import os
import select
import signal
import stat
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import IO

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment
from phasesweep.engine import ledger as engine_ledger
from phasesweep.engine.artifact_roots import (
    ARTIFACT_ROOT_BINDING_SCHEMA_VERSION,
    _artifact_root_binding_payload,
)
from phasesweep.engine.errors import (
    ArtifactRootConflictError,
    IncompleteJournalRecordError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
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
from phasesweep.engine.read import read_status
from phasesweep.engine.state import ARTIFACT_ROOT_ATTR, STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION
from phasesweep.engine.study_policy import _initialize_study_schema, _validate_study_schema
from phasesweep.errors import OperatorAction, PhaseSweepError, ProcessCleanupUncertainError
from phasesweep.mcp.recovery import (
    _load_recovery_studies,
)
from phasesweep.runtime.shutdown import PhaseSweepShutdown, signal_handler_scope
from tests.conftest import (
    make_experiment,
    mark_current_format,
    requires_nonroot,
    temporary_umask,
    write_constant_trainer,
)
from tests.ledger_fixtures import (
    append_journal_tail,
    materialize,
    tree_snapshot,
)
from tests.recovery_helpers import load_only_recovery_needs


def _experiment(tmp_path: Path, *, storage: str | None) -> Experiment:
    """Build one real-trial experiment for format-boundary tests."""
    trainer = write_constant_trainer(tmp_path)
    return make_experiment(workdir=tmp_path / "runs", storage=storage, trainer=trainer, n_trials=1)


@pytest.mark.parametrize("durable_entry", ["generations", "attempts"])
def test_unmarked_output_root_is_refused_before_mutation(
    tmp_path: Path, durable_entry: str
) -> None:
    """Completed and interrupted unmarked memory-backed trees are never adopted."""
    experiment = _experiment(tmp_path, storage=None)
    root = _experiment_dir(experiment)
    state = root / durable_entry
    state.mkdir(parents=True)
    (state / "old-state").write_text("stale\n", encoding="utf-8")
    before = tree_snapshot(root)

    with pytest.raises(
        ArtifactRootConflictError,
        match=r"unmarked PhaseSweep state.*Nothing was written",
    ) as excinfo:
        run_experiment(experiment)

    assert excinfo.value.action is OperatorAction.FRESH_NAMESPACE
    assert tree_snapshot(root) == before
    assert not _artifact_root_binding_path(experiment).exists()


def test_unsupported_artifact_binding_version_is_refused_before_mutation(
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
        match=rf"unsupported PhaseSweep format {ARTIFACT_ROOT_BINDING_SCHEMA_VERSION - 1}",
    ) as excinfo:
        run_experiment(experiment)

    assert excinfo.value.action is OperatorAction.FRESH_NAMESPACE
    assert tree_snapshot(root) == before


@pytest.mark.parametrize(
    "schema_version",
    [2, None, float(STUDY_SCHEMA_VERSION), True],
    ids=["old-version", "unmarked", "float-version", "bool-version"],
)
def test_unsupported_populated_ledger_is_refused_with_a_fresh_output_root(
    tmp_path: Path, schema_version: int | float | None
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
        match=r"old_experiment::removed_phase.*Nothing was written",
    ) as excinfo:
        run_experiment(experiment)

    assert excinfo.value.action is OperatorAction.FRESH_NAMESPACE
    assert not _experiment_dir(experiment).exists()
    assert {
        path.name: path.read_bytes()
        for path in sorted(tmp_path.glob(f"{ledger.name}*"))
        if path.is_file()
    } == ledger_before


def test_float_schema_is_refused_by_live_and_snapshot_checks(tmp_path: Path) -> None:
    """A numerically equal float stamp is unsupported through every schema check."""
    storage = f"journal:///{tmp_path / 'float.journal'}"
    study = optuna.create_study(study_name="t::p", storage=_resolve_storage(storage))
    study.set_user_attr(STUDY_SCHEMA_ATTR, float(STUDY_SCHEMA_VERSION))
    study.ask()
    trials = study.get_trials(deepcopy=False)
    before = tree_snapshot(tmp_path)

    with pytest.raises(StudySchemaMismatchError, match=r"t::p.*Affected trial numbers: \[0\]"):
        _initialize_study_schema(study)
    with pytest.raises(StudySchemaMismatchError, match="t::p.*Nothing was written"):
        engine_ledger._check_study_schema(study, trials)
    with pytest.raises(StudySchemaMismatchError, match="t::p.*Nothing was written"):
        engine_ledger._scan_ledger_format(storage)

    assert tree_snapshot(tmp_path) == before


def test_empty_unmarked_schema_check_is_read_only_and_live_initializer_stamps() -> None:
    """Only the live initializer stamps a valid empty allocation, using an integer."""
    study = optuna.create_study()
    _validate_study_schema(study)
    assert STUDY_SCHEMA_ATTR not in study.user_attrs

    _initialize_study_schema(study)
    assert type(study.user_attrs[STUDY_SCHEMA_ATTR]) is int
    assert study.user_attrs[STUDY_SCHEMA_ATTR] == STUDY_SCHEMA_VERSION


@pytest.mark.parametrize("leftover_study", ["t::retired", "other_experiment::phase"])
@pytest.mark.integration
def test_bound_root_status_refuses_empty_stamped_unsupported_studies(
    tmp_path: Path, leftover_study: str
) -> None:
    """Status rejects empty retired or foreign studies with an old format stamp."""
    storage = f"journal:///{tmp_path / 'current.journal'}"
    experiment = _experiment(tmp_path, storage=storage)
    run_experiment(experiment)
    leftover = optuna.create_study(study_name=leftover_study, storage=_resolve_storage(storage))
    leftover.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION - 1)

    with pytest.raises(StudySchemaMismatchError, match="unsupported PhaseSweep study state"):
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


@pytest.mark.parametrize(
    "denial", ["injected", pytest.param("permissions", marks=requires_nonroot)]
)
def test_unwritable_ledger_directory_fails_before_the_tree_is_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, denial: str
) -> None:
    """A ledger directory that cannot be created leaves the tree unbound and correctable."""
    parent = tmp_path / "read-only" if denial == "permissions" else tmp_path
    missing = parent / "missing"
    experiment = _experiment(tmp_path, storage=f"journal:///{missing / 'study.journal'}")
    real_mkdir = Path.mkdir

    def refuse_ledger_directory(path: Path, *args: object, **kwargs: object) -> None:
        if path == missing:
            raise PermissionError(13, "Permission denied", str(path))
        real_mkdir(path, *args, **kwargs)  # type: ignore[arg-type]

    if denial == "permissions":
        parent.mkdir()
        parent.chmod(0o555)
    else:
        monkeypatch.setattr(Path, "mkdir", refuse_ledger_directory)

    try:
        with pytest.raises(PhaseSweepError, match="could not be created") as refused:
            claim_ledger(validate_ledger(experiment))
    finally:
        if denial == "permissions":
            parent.chmod(0o755)

    assert type(refused.value) is PhaseSweepError
    assert refused.value.action is OperatorAction.RESTORE_TREE
    assert type(refused.value.__cause__) is PermissionError
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


def _bound_tree_with_foreign_study(tmp_path: Path, *, foreign: bool = True) -> Experiment:
    """Build a bound current-format tree whose ledger may also hold unsupported state.

    The phase study is current-format and root-claimed, so every refusal these
    tests observe comes from the ledger-wide format scan and its handling, not
    from the phase study itself.

    :param Path tmp_path: Test-owned directory for the tree and the ledger.
    :param bool foreign: Add a populated, unmarked ``legacy::p`` study, which
        only a completed format scan can see.
    :return Experiment: Experiment whose tree is bound to that ledger.
    """
    storage = f"journal:///{tmp_path / 'current.journal'}"
    experiment = _experiment(tmp_path, storage=storage)
    study = optuna.create_study(study_name="t::p", storage=_resolve_storage(storage))
    study.add_trial(optuna.trial.create_trial(value=0.5, state=optuna.trial.TrialState.COMPLETE))
    study.set_user_attr(ARTIFACT_ROOT_ATTR, str(_experiment_dir(experiment).resolve()))
    mark_current_format(experiment, study)
    if foreign:
        other = optuna.create_study(study_name="legacy::p", storage=_resolve_storage(storage))
        other.add_trial(
            optuna.trial.create_trial(value=0.5, state=optuna.trial.TrialState.COMPLETE)
        )
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
    readable, but the ledger may hold unsupported state, and only a
    completed scan can tell. Counts read around that gap would present the
    ledger as current-format when nothing verified it.
    """
    experiment = _bound_tree_with_foreign_study(tmp_path)
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
    experiment = _bound_tree_with_foreign_study(tmp_path)
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
    to complete before discovery: the unsupported study then refuses the run,
    and a second unreadable scan stops it as unavailable storage. Nothing is
    written either way.
    """
    experiment = _bound_tree_with_foreign_study(tmp_path)
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
    """A completed scan is trusted: counts are real and claim scans no more."""
    experiment = _bound_tree_with_foreign_study(tmp_path, foreign=False)
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
    real_load = ledger.open_existing_study
    calls = 0

    def counted_validate(candidate: str | None) -> None:
        nonlocal calls
        calls += 1
        real_validate(candidate)

    def checked_load(candidate: ValidatedLedger, phase):
        assert calls == 1
        return real_load(candidate, phase)

    monkeypatch.setattr(ledger, "_scan_ledger_format", counted_validate)
    monkeypatch.setattr(ledger, "open_existing_study", checked_load)

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

    Reads skip that line as Optuna does, so the format scan passes. A write
    does not refuse on sight: under the shared journal inode lock it backs the
    damaged bytes up beside the journal and truncates back to the last
    complete record, for ``claim_ledger``, ``open_registry_study``, and a
    confirmed recovery alike. Unconfirmed inspection does not check the tail
    at all, so it neither raises nor changes a byte.
    """

    def freshly_damaged(label: str) -> tuple[Experiment, Path, bytes, bytes]:
        # Each writer gets its own copy of the fixture: a repair mutates the
        # journal it runs against, so the four scenarios cannot share one.
        materialized = materialize("current-journal", tmp_path / label, mode="tree")
        experiment = materialized.experiment
        journal = materialized.ledger_path
        complete, damaged = append_journal_tail(journal, _PARTIAL_FINAL_RECORDS[damage])
        return experiment, journal, complete, damaged

    def assert_repaired(journal: Path, complete: bytes, damaged: bytes) -> None:
        assert journal.read_bytes() == complete
        backups = sorted(journal.parent.glob(f"{journal.name}.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == damaged
        _assert_journal_unlocked(journal)

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
    is still writing. The inode lock excludes every PhaseSweep writer, so if
    that append settles the journal back to fully complete before this
    repair's rescan runs under the lock, no offset from before the lock is
    ever used: nothing is truncated, and no backup is written.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    experiment = materialized.experiment
    journal = materialized.ledger_path
    complete, _ = append_journal_tail(journal, _PARTIAL_FINAL_RECORDS["torn"])

    real_acquire = engine_ledger._JournalLock.acquire

    def finish_append_then_acquire(lock: engine_ledger._JournalLock) -> bool:
        journal.write_bytes(complete)
        return real_acquire(lock)

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


def _assert_journal_unlocked(journal: Path) -> None:
    """Confirm another descriptor can acquire the journal inode immediately."""
    lock = engine_ledger._JournalLock(journal, timeout=0)
    try:
        assert lock.acquire()
        assert lock.held()
    finally:
        lock.release()
    assert not lock.held()


def _replace_locked_journal(journal: Path) -> engine_ledger._JournalLock:
    """Replace the journal pathname and hold the replacement's distinct inode."""
    replacement = journal.with_suffix(".replacement")
    replacement.write_bytes(journal.read_bytes())
    replacement.replace(journal)
    successor = engine_ledger._JournalLock(journal, timeout=0)
    successor.acquire()
    return successor


def test_repair_waits_for_a_journal_lock_another_writer_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A journal lock held when the repair starts is awaited, then taken, not refused on sight."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = materialized.ledger_path
    complete, _ = append_journal_tail(journal, _PARTIAL_FINAL_RECORDS["torn"])
    holder = engine_ledger._JournalLock(journal)
    holder.acquire()
    ledger = validate_ledger(materialized.experiment)
    clock = _manual_lock_clock(monkeypatch, on_pause=holder.release)
    try:
        repair_incomplete_journal_record(ledger)
    finally:
        holder.release()

    assert clock["pauses"] == 1
    assert journal.read_bytes() == complete
    assert len(list(journal.parent.glob(f"{journal.name}.*.bak"))) == 1
    _assert_journal_unlocked(journal)


def test_repair_refuses_a_journal_lock_that_is_never_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repair refuses bounded contention without releasing the writer's inode lock."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = materialized.ledger_path
    _, damaged = append_journal_tail(journal, _PARTIAL_FINAL_RECORDS["torn"])
    holder = engine_ledger._JournalLock(journal)
    holder.acquire()
    ledger = validate_ledger(materialized.experiment)
    clock = _manual_lock_clock(monkeypatch)

    raised: list[BaseException] = []

    def repair() -> None:
        try:
            repair_incomplete_journal_record(ledger)
        except BaseException as exc:  # noqa: BLE001 - handed to the assertions below
            raised.append(exc)

    worker = threading.Thread(target=repair, daemon=True)
    try:
        worker.start()
        worker.join(timeout=10.0)
        assert not worker.is_alive()
        assert clock["now"] == pytest.approx(engine_ledger._JOURNAL_LOCK_WAIT_SECONDS)
        assert len(raised) == 1
        assert isinstance(raised[0], IncompleteJournalRecordError)
        assert raised[0].action == OperatorAction.RESTORE_LEDGER
        assert "Wait for the writer to finish" in str(raised[0])
        assert journal.read_bytes() == damaged
        assert holder.held()
        assert not list(journal.parent.glob(f"{journal.name}.*.bak"))
    finally:
        holder.release()
    _assert_journal_unlocked(journal)


@pytest.mark.parametrize("spelling", ["nested-relative", "symlink-alias"])
def test_journal_writers_and_repair_lock_the_journals_real_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    """Live backend appends and repair both acquire the canonical journal inode."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ledger").mkdir()
    journal = tmp_path.resolve() / "ledger" / "study.journal"
    if spelling == "nested-relative":
        url = "journal:///ledger/study.journal"
    else:
        (tmp_path / "alias.journal").symlink_to("ledger/study.journal")
        url = f"journal:///{tmp_path}/alias.journal"
    acquired: list[tuple[Path, float | None]] = []
    real_acquire = engine_ledger._JournalLock.acquire

    def recording_acquire(lock: engine_ledger._JournalLock) -> bool:
        result = real_acquire(lock)
        assert lock.held()
        acquired.append((lock.path, lock._timeout))
        return result

    monkeypatch.setattr(engine_ledger._JournalLock, "acquire", recording_acquire)

    optuna.create_study(study_name="s", storage=engine_ledger._resolve_storage(url))
    complete, _ = append_journal_tail(journal, _PARTIAL_FINAL_RECORDS["torn"])
    engine_ledger._repair_incomplete_journal_record(url)

    assert acquired == [(journal, None), (journal, engine_ledger._JOURNAL_LOCK_WAIT_SECONDS)]
    assert journal.read_bytes() == complete
    _assert_journal_unlocked(journal)


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
    _, in_flight = append_journal_tail(real, _PARTIAL_FINAL_RECORDS["torn"])
    holder = engine_ledger._JournalLock(real)
    holder.acquire()
    _manual_lock_clock(monkeypatch)
    try:
        with pytest.raises(IncompleteJournalRecordError, match="still held"):
            engine_ledger._repair_incomplete_journal_record(f"journal:///{alias}")
        assert holder.held()
    finally:
        holder.release()

    assert real.read_bytes() == in_flight
    _assert_journal_unlocked(real)


def test_repair_truncates_only_while_holding_the_journal_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The truncation step runs only while this process holds the shared journal inode lock."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    experiment = materialized.experiment
    journal = materialized.ledger_path
    complete, _ = append_journal_tail(journal, _PARTIAL_FINAL_RECORDS["torn"])
    real_truncate = engine_ledger._truncate_incomplete_journal_record

    def truncate_while_locked(lock: engine_ledger._JournalLock) -> tuple[Path, int] | None:
        assert lock.held()
        with pytest.raises(IncompleteJournalRecordError, match="still held"):
            engine_ledger._JournalLock(journal, timeout=0).acquire()
        return real_truncate(lock)

    monkeypatch.setattr(engine_ledger, "_truncate_incomplete_journal_record", truncate_while_locked)

    repair_incomplete_journal_record(validate_ledger(experiment))

    _assert_journal_unlocked(journal)
    assert journal.read_bytes() == complete


def test_journal_writer_wait_is_unbounded_and_does_not_change_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live append waits past the repair deadline without changing the journal while waiting."""
    journal = tmp_path / "study.journal"
    journal.write_bytes(b"")
    os.utime(journal, (0, 0))
    holder = engine_ledger._JournalLock(journal)
    holder.acquire()

    def finish_writer() -> None:
        assert journal.stat().st_mtime == 0
        if clock["now"] > engine_ledger._JOURNAL_LOCK_WAIT_SECONDS:
            holder.release()

    clock = _manual_lock_clock(monkeypatch, on_pause=finish_writer)
    try:
        optuna.create_study(study_name="s", storage=_resolve_storage(f"journal:///{journal}"))
    finally:
        holder.release()
    assert clock["now"] > engine_ledger._JOURNAL_LOCK_WAIT_SECONDS
    _assert_journal_unlocked(journal)


@pytest.mark.parametrize("taken_over", ["before truncation", "after truncation"])
def test_repair_never_truncates_under_or_releases_a_lock_another_process_took_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, taken_over: str
) -> None:
    """Repair detects pathname replacement and never releases the successor's inode lock."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = materialized.ledger_path
    complete, damaged = append_journal_tail(journal, _PARTIAL_FINAL_RECORDS["torn"])
    successor: list[engine_ledger._JournalLock] = []

    if taken_over == "before truncation":
        real_fsync_directory = engine_ledger.fsync_directory

        def fsync_then_take_over(path: Path) -> None:
            real_fsync_directory(path)
            successor.append(_replace_locked_journal(journal))

        monkeypatch.setattr(engine_ledger, "fsync_directory", fsync_then_take_over)
    else:
        real_truncate = engine_ledger._truncate_incomplete_journal_record

        def truncate_then_take_over(lock: engine_ledger._JournalLock) -> tuple[Path, int] | None:
            repaired = real_truncate(lock)
            successor.append(_replace_locked_journal(journal))
            return repaired

        monkeypatch.setattr(
            engine_ledger, "_truncate_incomplete_journal_record", truncate_then_take_over
        )

    try:
        with pytest.raises(IncompleteJournalRecordError, match="removed or replaced") as excinfo:
            repair_incomplete_journal_record(validate_ledger(materialized.experiment))
        assert excinfo.value.action == OperatorAction.RETRY
        assert journal.read_bytes() == (damaged if taken_over == "before truncation" else complete)
        assert successor[0].held()
        with pytest.raises(IncompleteJournalRecordError, match="still held"):
            engine_ledger._JournalLock(journal, timeout=0).acquire()
        assert len(list(journal.parent.glob(f"{journal.name}.*.bak"))) == 1
    finally:
        for lock in successor:
            lock.release()
    _assert_journal_unlocked(journal)


@pytest.mark.integration
def test_waiting_optuna_writer_never_takes_over_a_long_repairs_journal_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer has no timeout or stale takeover while a repair holds the journal inode."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = materialized.ledger_path
    complete, _ = append_journal_tail(journal, _PARTIAL_FINAL_RECORDS["torn"])
    monkeypatch.setattr(engine_ledger, "_JOURNAL_LOCK_WAIT_SECONDS", 0.01)
    waiter_lock = engine_ledger._JournalLock(journal)
    waiter_holds = threading.Event()

    def wait_for_lock() -> None:
        waiter_lock.acquire()
        waiter_holds.set()

    waiter = threading.Thread(target=wait_for_lock, daemon=True)
    real_fsync_directory = engine_ledger.fsync_directory

    def hold_past_the_grace_period(path: Path) -> None:
        real_fsync_directory(path)
        waiter.start()
        assert not waiter_holds.wait(0.1)

    monkeypatch.setattr(engine_ledger, "fsync_directory", hold_past_the_grace_period)

    try:
        repair_incomplete_journal_record(validate_ledger(materialized.experiment))
        assert journal.read_bytes() == complete
        assert waiter_holds.wait(timeout=10.0)
    finally:
        waiter.join(timeout=10.0)
        waiter_lock.release()
    _assert_journal_unlocked(journal)


@pytest.mark.integration
def test_journal_kernel_lock_is_released_when_its_owner_dies(tmp_path: Path) -> None:
    """A killed writer leaves repair free to acquire its journal without stale-lock cleanup."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = materialized.ledger_path
    complete, _ = append_journal_tail(journal, _PARTIAL_FINAL_RECORDS["torn"])
    worker = (
        "import signal,sys; from pathlib import Path; "
        "from phasesweep.engine.ledger import _JournalLock; "
        "lock=_JournalLock(Path(sys.argv[1])); lock.acquire(); "
        "print('locked',flush=True); signal.pause()"
    )
    with subprocess.Popen(
        [sys.executable, "-c", worker, str(journal)], stdout=subprocess.PIPE, text=True
    ) as owner:
        try:
            assert owner.stdout is not None
            assert select.select([owner.stdout], [], [], 10.0)[0]
            assert owner.stdout.readline().strip() == "locked"
            with pytest.raises(IncompleteJournalRecordError, match="still held"):
                engine_ledger._JournalLock(journal, timeout=0).acquire()
            owner.kill()
            owner.wait(timeout=10.0)
            repair_incomplete_journal_record(validate_ledger(materialized.experiment))
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait(timeout=10.0)
    assert journal.read_bytes() == complete
    _assert_journal_unlocked(journal)


@pytest.mark.integration
@pytest.mark.signals_own_pid
@pytest.mark.parametrize("stage", ["acquire", "release"])
def test_journal_lock_shutdown_closes_the_acquired_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Shutdown arriving just after acquisition or during release leaves no held descriptor."""
    journal = tmp_path / "study.journal"
    journal.write_bytes(b"")
    lock = engine_ledger._JournalLock(journal)
    original_acquire = engine_ledger.try_flock
    original_release = engine_ledger.unlock_file

    def acquire_and_signal(handle: IO[str]) -> bool:
        result = original_acquire(handle)
        signal.raise_signal(signal.SIGTERM)
        return result

    def release_and_signal(handle: IO[str]) -> None:
        original_release(handle)
        assert handle.closed
        signal.raise_signal(signal.SIGTERM)

    target = "try_flock" if stage == "acquire" else "unlock_file"
    callback = acquire_and_signal if stage == "acquire" else release_and_signal
    monkeypatch.setattr(engine_ledger, target, callback)
    with signal_handler_scope(), pytest.raises(PhaseSweepShutdown) as shutdown:
        lock.acquire()
        lock.release()
    assert not lock.held(), str(shutdown.traceback)
    monkeypatch.setattr(engine_ledger, "try_flock", original_acquire)
    monkeypatch.setattr(engine_ledger, "unlock_file", original_release)
    _assert_journal_unlocked(journal)


def test_journal_lock_detects_a_symlink_replacing_its_path(tmp_path: Path) -> None:
    """A symlink to the held inode still replaces the canonical journal pathname."""
    journal = tmp_path / "study.journal"
    retained = tmp_path / "retained.journal"
    journal.write_bytes(b"")
    lock = engine_ledger._JournalLock(journal)
    lock.acquire()
    try:
        journal.rename(retained)
        journal.symlink_to(retained)
        assert not lock.held()
    finally:
        lock.release()
    _assert_journal_unlocked(retained)


def test_repair_backup_is_no_more_permissive_than_the_journal(tmp_path: Path) -> None:
    """A private journal's backup stays private under a permissive umask.

    The backup holds every study and trial attribute the journal does, so it
    is created with the journal's mode, which the umask can only narrow.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = materialized.ledger_path
    journal.write_bytes(journal.read_bytes() + _PARTIAL_FINAL_RECORDS["torn"])
    journal.chmod(0o600)

    with temporary_umask(0o022):
        repair_incomplete_journal_record(validate_ledger(materialized.experiment))

    (backup,) = journal.parent.glob(f"{journal.name}.*.bak")
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600


def test_journal_hard_link_names_share_lock_and_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hard-linked journal names contend for the same inode lock."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = materialized.ledger_path
    complete, damaged = append_journal_tail(journal, _PARTIAL_FINAL_RECORDS["torn"])
    alias = tmp_path / "other-name.journal"
    os.link(journal, alias)
    holder = engine_ledger._JournalLock(alias)
    holder.acquire()
    _manual_lock_clock(monkeypatch)
    try:
        with pytest.raises(IncompleteJournalRecordError, match="still held"):
            repair_incomplete_journal_record(validate_ledger(materialized.experiment))
        assert journal.read_bytes() == damaged
        assert not list(journal.parent.glob(f"{journal.name}.*.bak"))
    finally:
        holder.release()
    repair_incomplete_journal_record(validate_ledger(materialized.experiment))
    assert journal.read_bytes() == alias.read_bytes() == complete
    assert len(list(journal.parent.glob(f"{journal.name}.*.bak"))) == 1
    _assert_journal_unlocked(journal)


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
    journal = materialized.ledger_path
    _, damaged = append_journal_tail(journal, _PARTIAL_FINAL_RECORDS["torn"])
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
    """Failure to open the journal for locking or create its backup preserves every byte."""
    materialized = materialize("current-journal", tmp_path, mode="tree")
    journal = materialized.ledger_path
    _, damaged = append_journal_tail(journal, _PARTIAL_FINAL_RECORDS["torn"])
    ledger = validate_ledger(materialized.experiment)
    lock = engine_ledger._JournalLock(journal)
    if step == "backup":
        lock.acquire()
    denied = journal if step == "lock" else journal.parent
    mode = denied.stat().st_mode
    denied.chmod(0o400 if step == "lock" else 0o555)
    try:
        with pytest.raises(IncompleteJournalRecordError) as excinfo:
            if step == "lock":
                repair_incomplete_journal_record(ledger)
            else:
                engine_ledger._truncate_incomplete_journal_record(lock)
    finally:
        denied.chmod(mode)
        lock.release()

    assert excinfo.value.action == OperatorAction.RESTORE_LEDGER
    assert "could not be repaired" in str(excinfo.value)
    assert journal.read_bytes() == damaged
    assert not list(journal.parent.glob(f"{journal.name}.*.bak"))
    _assert_journal_unlocked(journal)
