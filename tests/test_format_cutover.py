"""Breaking-format boundary coverage for artifact roots and local ledgers."""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import sqlite3
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config import Experiment
from phasesweep.engine import (
    ArtifactRootConflictError,
    IncompleteJournalRecordError,
    LedgerTransactionInterruptedError,
    ProcessCleanupUncertainError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    read_status,
)
from phasesweep.engine.artifact_roots import (
    ARTIFACT_ROOT_BINDING_SCHEMA_VERSION,
    _artifact_root_binding_payload,
)
from phasesweep.engine.ledger import (
    ClaimedLedger,
    _resolve_storage,
    claim_ledger,
    open_existing_study,
    open_phase_study,
    open_registry_study,
    roll_back_interrupted_transaction,
    validate_ledger,
)
from phasesweep.engine.locking import _experiment_lock
from phasesweep.engine.paths import _artifact_root_binding_path, _experiment_dir
from phasesweep.engine.state import ARTIFACT_ROOT_ATTR, STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION
from phasesweep.errors import PhaseSweepError
from phasesweep.mcp.recovery import (
    RunRecoveryError,
    _load_recovery_studies,
    recover_run,
)
from phasesweep.mcp.runs import RunStore
from tests.conftest import make_experiment, mark_current_format, write_constant_trainer
from tests.ledger_fixtures import (
    leave_hot_journal,
    ledger_file,
    materialize,
    rollback_journal,
    tree_snapshot,
)
from tests.mcp_helpers import stage_dead_run
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


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
@pytest.mark.parametrize("schema_version", [2, None], ids=["old-version", "unmarked"])
def test_old_populated_ledger_is_refused_with_a_fresh_output_root(
    tmp_path: Path, backend: str, schema_version: int | None
) -> None:
    """Changing experiment/workdir cannot reinterpret an old populated ledger as fresh."""
    ledger = tmp_path / f"old.{backend}"
    storage = f"{backend}:///{ledger}"
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


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
@pytest.mark.parametrize("leftover_study", ["t::retired", "other_experiment::phase"])
@pytest.mark.integration
def test_bound_root_status_refuses_empty_stamped_legacy_studies(
    tmp_path: Path, backend: str, leftover_study: str
) -> None:
    """Status rejects empty retired or foreign studies with an old format stamp."""
    storage = f"{backend}:///{tmp_path / f'current.{backend}'}"
    experiment = _experiment(tmp_path, storage=storage)
    run_experiment(experiment)
    leftover = optuna.create_study(study_name=leftover_study, storage=_resolve_storage(storage))
    leftover.set_user_attr(STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION - 1)

    with pytest.raises(StudySchemaMismatchError, match="pre-cutover or unsupported"):
        read_status(experiment)


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
def test_validate_ledger_on_a_fresh_root_creates_nothing(tmp_path: Path, backend: str) -> None:
    """Validating a never-run experiment reports an unbound tree and writes nothing.

    The handle is what every read path takes, so obtaining one must not be the
    thing that brings the tree or the ledger into existence: a status poll on a
    config that has never run has to stay observable and leave no trace.
    """
    ledger_path = tmp_path / "fresh" / f"study.{backend}"
    experiment = _experiment(tmp_path, storage=f"{backend}:///{ledger_path}")

    ledger = validate_ledger(experiment)

    assert ledger.binding_state == "unbound"
    assert ledger.backend == backend
    assert ledger.experiment_name == experiment.experiment
    assert ledger.artifact_root == str(_experiment_dir(experiment).resolve())
    assert not ledger_path.exists()
    assert not ledger_path.parent.exists()
    assert not _artifact_root_binding_path(experiment).exists()
    assert not _experiment_dir(experiment).exists()


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
def test_claim_ledger_creates_ledger_parent_but_validate_does_not(
    tmp_path: Path, backend: str
) -> None:
    """Claiming brings the ledger's directory into existence before it binds the tree.

    Every read path validates, so validating must never materialize a ledger
    directory. The claim is the first step that writes, and the directory is
    its first write: a path that cannot hold a ledger then fails before the
    tree names that ledger, while the path can still be corrected. Neither
    step creates the ledger file itself; the first live open does. SQLite
    creates only that file, so a claim that skipped the directory would bind
    the tree to a ledger its first open cannot create.
    """
    missing = tmp_path / "not-created-by-a-read"
    ledger_file = missing / f"study.{backend}"
    experiment = _experiment(tmp_path, storage=f"{backend}:///{ledger_file}")

    ledger = validate_ledger(experiment)

    assert ledger.backend == backend
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
    experiment = _experiment(tmp_path, storage=f"sqlite:///{tmp_path / 'missing' / 'study.db'}")
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


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
def test_claim_ledger_binds_tree_then_returns_bound_handle(tmp_path: Path, backend: str) -> None:
    """Claiming a fresh tree records its owner and returns the handle that opens studies.

    The validated handle keeps saying what validation saw ("unbound"); only the
    claimed handle records that the fixed order's writes happened, and a study
    opened through it carries the claimed root. Reclaiming the bound tree finds
    that study in its one discovery pass.
    """
    experiment = _experiment(tmp_path, storage=f"{backend}:///{tmp_path / f'study.{backend}'}")
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


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
def test_claim_ledger_writes_tree_binding_before_claiming_studies(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between the two claims leaves a bound tree and unclaimed studies.

    The tree records its ledger first and the studies record their root second.
    A process that dies in between leaves state the next run claims normally.
    The opposite order could leave a study pointing at a tree that does not
    name the same ledger, which no later run could tell apart from a conflict.
    """
    import phasesweep.engine.ledger as ledger_module

    storage = f"{backend}:///{tmp_path / f'study.{backend}'}"
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
    storage = f"sqlite:///{tmp_path / 'current.db'}"
    experiment = _experiment(tmp_path, storage=storage)
    study = optuna.create_study(study_name="t::p", storage=storage)
    study.add_trial(optuna.trial.create_trial(value=0.5, state=optuna.trial.TrialState.COMPLETE))
    study.set_user_attr(ARTIFACT_ROOT_ATTR, str(_experiment_dir(experiment).resolve()))
    mark_current_format(experiment, study)
    if legacy:
        old = optuna.create_study(study_name="legacy::p", storage=storage)
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
    ledger_file = tmp_path / "current.db"
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


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
@pytest.mark.integration
def test_startup_scans_the_ledger_format_once(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Root preflight scans once before Optuna loads any declared study."""
    import phasesweep.engine.ledger as ledger

    storage = f"{backend}:///{tmp_path / f'current.{backend}'}"
    experiment = _experiment(tmp_path, storage=storage)
    real_validate = ledger._scan_ledger_format
    real_load = ledger._load_existing_phase_study
    calls = 0

    def counted_validate(candidate: str | None) -> None:
        nonlocal calls
        calls += 1
        real_validate(candidate)

    def checked_load(candidate: Experiment, phase):
        assert calls == 1
        return real_load(candidate, phase)

    monkeypatch.setattr(ledger, "_scan_ledger_format", counted_validate)
    monkeypatch.setattr(ledger, "_load_existing_phase_study", checked_load)

    run_experiment(experiment)

    assert calls == 1


@pytest.mark.parametrize("backend", ["sqlite", "journal"])
@pytest.mark.integration
def test_empty_unstamped_phase_study_can_resume_in_current_ledger(
    tmp_path: Path, backend: str
) -> None:
    """An interrupted new phase is stamped and resumed rather than classified as old state."""
    ledger = tmp_path / f"current.{backend}"
    storage = f"{backend}:///{ledger}"
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


def _committed_trials(database: Path) -> list[tuple[int, str]]:
    """Return every committed trial's number and state, reading a snapshot copy.

    The copy is taken without its journal, so the read never touches, or
    recovers, the ledger under test.
    """
    snapshot = database.with_name(f"{database.name}.snapshot")
    shutil.copyfile(database, snapshot)
    try:
        with contextlib.closing(sqlite3.connect(snapshot)) as conn:
            return conn.execute("SELECT number, state FROM trials ORDER BY trial_id").fetchall()
    finally:
        snapshot.unlink()


_INTERRUPTED = "holds a transaction that a crash interrupted"


@pytest.mark.parametrize("mode", ["tree", "ledger-only"])
def test_reads_report_an_interrupted_transaction_and_leave_it_for_a_locked_path(
    tmp_path: Path, mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Status and recovery inspection name the interrupted transaction and roll nothing back.

    A ``mode=ro`` reader cannot finish SQLite's crash recovery, and no read may
    write, so every read reports the ledger unavailable for that reason and
    points at the commands that hold the experiment lock. The ledger and its
    journal stay byte-identical. An unbound tree records the gap too, since
    refusing there would also stop the run that rolls it back.
    """
    materialized = materialize("current-sqlite", tmp_path, mode=mode)
    experiment = materialized.experiment
    leave_hot_journal(ledger_file(materialized, "sqlite"))
    before = tree_snapshot(materialized.root)

    ledger = validate_ledger(experiment)
    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.ledger"):
        status = read_status(experiment)
    with pytest.raises(LedgerTransactionInterruptedError) as opened:
        open_existing_study(ledger, experiment.phases[0])
    with pytest.raises(RunRecoveryError) as inspected:
        _load_recovery_studies(experiment, load_only_recovery_needs())

    assert ledger.binding_state == ("bound" if mode == "tree" else "unbound")
    assert isinstance(ledger.format_scan_failure, LedgerTransactionInterruptedError)
    assert status["phases"][0]["trial_data_available"] is False
    assert _INTERRUPTED in caplog.text
    for refusal in (opened.value, inspected.value):
        assert _INTERRUPTED in str(refusal)
    assert tree_snapshot(materialized.root) == before


def test_confirmed_recovery_rolls_back_an_interrupted_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A confirmed ``recover-run`` holds the lock, so SQLite finishes its own rollback.

    Recovery then reads the committed trials exactly as they were before the
    crash, and the uncommitted rows never appear.
    """
    materialized = materialize("current-sqlite", tmp_path, mode="tree")
    database = ledger_file(materialized, "sqlite")
    committed = _committed_trials(database)
    leave_hot_journal(database)
    state_dir = tmp_path / "mcp-state"
    stage_dead_run(
        RunStore(state_dir),
        "interrupted",
        materialized.config_path,
        materialized.experiment.experiment,
        cleanup_uncertain=True,
    )
    monkeypatch.setattr("phasesweep.mcp.recovery.kill_stale_group", lambda *_a, **_k: True)
    messages: list[str] = []

    recover_run(state_dir, "interrupted", confirm=True, emit=messages.append)

    assert not rollback_journal(database).exists()
    assert _committed_trials(database) == committed
    assert any("Cleared cleanup uncertainty for interrupted" in message for message in messages)
    with contextlib.closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as conn:
        uncommitted = conn.execute(
            "SELECT COUNT(*) FROM study_user_attributes WHERE key LIKE 'uncommitted-%'"
        ).fetchone()
    assert uncommitted == (0,)


def test_claim_on_an_unbound_tree_rolls_back_a_shared_ledger(tmp_path: Path) -> None:
    """A new experiment sharing a ledger clears another run's interrupted commit.

    The tree is unbound, so the read that found the journal could not rely on
    a recorded owner; the claim still reaches the rollback, then rescans the
    whole ledger strictly and binds the new tree.
    """
    owner = _bound_tree_with_legacy_study(tmp_path, legacy=False)
    database = tmp_path / "current.db"
    committed = _committed_trials(database)
    leave_hot_journal(database)
    newcomer = owner.model_copy(update={"experiment": "u", "workdir": str(tmp_path / "other")})

    with _experiment_lock(newcomer):
        claimed = claim_ledger(validate_ledger(newcomer))

    assert claimed.format_verified is True
    assert not rollback_journal(database).exists()
    assert _committed_trials(database) == committed
    assert _artifact_root_binding_path(newcomer).is_file()


def test_registry_opener_rolls_back_an_interrupted_transaction(tmp_path: Path) -> None:
    """Attempt-registry recovery, reached only under a lock, clears the journal too.

    Its locator can name a ledger no claim in this run touched, so the opener
    cannot rely on ``claim_ledger`` having rolled that ledger back first.
    """
    _bound_tree_with_legacy_study(tmp_path, legacy=False)
    database = tmp_path / "current.db"
    committed = _committed_trials(database)
    leave_hot_journal(database)

    study = open_registry_study(f"sqlite:///{database}", "t::p")

    assert study is not None
    assert not rollback_journal(database).exists()
    assert [(trial.number, trial.state.name) for trial in study.get_trials()] == committed


def test_rollback_open_never_creates_a_missing_ledger(tmp_path: Path) -> None:
    """The one read-write open refuses a ledger that vanished instead of creating it."""
    experiment = _bound_tree_with_legacy_study(tmp_path, legacy=False)
    database = tmp_path / "current.db"
    leave_hot_journal(database)
    ledger = validate_ledger(experiment)
    database.unlink()

    with pytest.raises(StudyStorageUnavailableError, match="could not roll it back") as refused:
        roll_back_interrupted_transaction(ledger)

    # Still the interrupted transaction's refusal.
    assert type(refused.value) is LedgerTransactionInterruptedError
    assert not database.exists()


_PARTIAL_FINAL_RECORDS = {
    "torn": b'{"op_code": 5, "worker_id": "crashed',
    "undecodable": b"not a journal record\n",
}


@pytest.mark.parametrize("damage", sorted(_PARTIAL_FINAL_RECORDS))
def test_writers_refuse_a_partial_final_journal_record_and_leave_it(
    tmp_path: Path, damage: str
) -> None:
    """Every path that may write refuses a bad last journal line, names the repair, writes nothing.

    Reads skip that line as Optuna does, so the format scan passes. Optuna's
    next append would be glued onto it, though, and the line may be another
    experiment's append still in flight, so nothing truncates it. Recovery
    inspection previews the confirmed write and refuses in the same words.
    """
    materialized = materialize("current-journal", tmp_path, mode="tree")
    experiment = materialized.experiment
    journal = ledger_file(materialized, "journal")
    complete = journal.read_bytes()
    journal.write_bytes(complete + _PARTIAL_FINAL_RECORDS[damage])
    before = tree_snapshot(materialized.root)
    assert experiment.resolved_storage is not None

    assert validate_ledger(experiment).format_verified is True
    with pytest.raises(IncompleteJournalRecordError) as claimed:
        claim_ledger(validate_ledger(experiment))
    with pytest.raises(IncompleteJournalRecordError) as registry:
        open_registry_study(experiment.resolved_storage, "t::p")
    with pytest.raises(RunRecoveryError) as inspected:
        _load_recovery_studies(experiment, load_only_recovery_needs())
    with _experiment_lock(experiment), pytest.raises(RunRecoveryError) as confirmed:
        _load_recovery_studies(experiment, load_only_recovery_needs(), confirm=True)

    assert str(inspected.value) == str(confirmed.value) == str(claimed.value)
    for refusal in (claimed.value, registry.value, inspected.value, confirmed.value):
        assert "ends with an incomplete record" in str(refusal)
        assert f"truncate -s {len(complete)} {journal}" in str(refusal)
    assert tree_snapshot(materialized.root) == before


@pytest.mark.integration
def test_run_rolls_back_an_interrupted_transaction_and_continues(tmp_path: Path) -> None:
    """The next run lets SQLite restore the committed ledger, then tops it up.

    Before the fix every path scanned ``mode=ro`` first, so the run refused
    with a remedy to restore a ledger that was intact, and nothing ever
    cleared the journal.
    """
    database = tmp_path / "current.db"
    experiment = _experiment(tmp_path, storage=f"sqlite:///{database}")
    run_experiment(experiment)
    committed = _committed_trials(database)
    leave_hot_journal(database)
    topped_up = experiment.model_copy(
        update={"phases": [experiment.phases[0].model_copy(update={"n_trials": 2})]}
    )

    run_experiment(topped_up)

    assert not rollback_journal(database).exists()
    trials = _committed_trials(database)
    assert trials[: len(committed)] == committed
    assert [state for _number, state in trials] == ["COMPLETE", "COMPLETE"]
