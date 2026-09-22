"""Breaking-format boundary coverage for artifact roots and local ledgers."""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config import Experiment
from phasesweep.engine import ArtifactRootConflictError, StudySchemaMismatchError, read_status
from phasesweep.engine.artifact_roots import ARTIFACT_ROOT_BINDING_SCHEMA_VERSION
from phasesweep.engine.ledger import _resolve_storage, validate_ledger
from phasesweep.engine.paths import _artifact_root_binding_path, _experiment_dir
from phasesweep.engine.state import STUDY_SCHEMA_ATTR, STUDY_SCHEMA_VERSION
from tests.conftest import make_experiment, write_constant_trainer
from tests.ledger_fixtures import _tree_bytes


def _experiment(tmp_path: Path, *, storage: str | None) -> Experiment:
    """Build one real-trial experiment for format-boundary tests."""
    trainer = write_constant_trainer(tmp_path)
    return make_experiment(
        workdir=tmp_path / "runs",
        storage=storage,
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        override_format="argparse",
        n_trials=1,
    )


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
    before = _tree_bytes(root)

    with pytest.raises(
        ArtifactRootConflictError,
        match=r"pre-cutover.*fresh artifact root.*0\.3\.1.*Nothing was written",
    ):
        run_experiment(experiment)

    assert _tree_bytes(root) == before
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
    before = _tree_bytes(root)

    with pytest.raises(
        ArtifactRootConflictError,
        match=r"unsupported pre-cutover.*format 2.*fresh artifact root",
    ):
        run_experiment(experiment)

    assert _tree_bytes(root) == before


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


@pytest.mark.integration
def test_current_memory_output_records_deliberate_no_ledger_identity(tmp_path: Path) -> None:
    """Fresh memory runs receive the same output-format marker and can continue."""
    experiment = _experiment(tmp_path, storage=None)

    run_experiment(experiment)
    run_experiment(experiment)

    payload = json.loads(_artifact_root_binding_path(experiment).read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": ARTIFACT_ROOT_BINDING_SCHEMA_VERSION,
        "experiment": experiment.experiment,
        "artifact_root": str(_experiment_dir(experiment).resolve()),
        "storage_key": None,
    }


@pytest.mark.integration
def test_current_sqlite_format_continues_after_top_up(tmp_path: Path) -> None:
    """The early compatibility precheck preserves supported continuation."""
    storage = f"sqlite:///{tmp_path / 'current.db'}"
    experiment = _experiment(tmp_path, storage=storage)
    run_experiment(experiment)
    topped_up = experiment.model_copy(
        update={"phases": [experiment.phases[0].model_copy(update={"n_trials": 2})]}
    )

    run_experiment(topped_up)

    study = optuna.load_study(study_name="t::p", storage=storage)
    assert study.user_attrs[STUDY_SCHEMA_ATTR] == STUDY_SCHEMA_VERSION
    assert len(study.trials) == 2
    binding = json.loads(_artifact_root_binding_path(experiment).read_text(encoding="utf-8"))
    assert binding["schema_version"] == ARTIFACT_ROOT_BINDING_SCHEMA_VERSION


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


def test_validate_ledger_never_creates_a_journal_parent_directory(tmp_path: Path) -> None:
    """A journal URL under a missing directory is classified without creating it.

    ``_resolve_storage`` used to create the parent as a side effect of merely
    translating a URL, which put a read path one call away from materializing a
    ledger directory. Only the create path may do that now.
    """
    missing = tmp_path / "not-created-by-a-read"
    experiment = _experiment(tmp_path, storage=f"journal:///{missing / 'study.journal'}")

    ledger = validate_ledger(experiment)

    assert ledger.backend == "journal"
    assert ledger.ledger_path == missing / "study.journal"
    assert not missing.exists()


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

    def counted_validate(candidate: Experiment) -> None:
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
