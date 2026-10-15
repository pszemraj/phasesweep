"""Relative config paths resolve against the config file's directory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from phasesweep import load_experiment, run_experiment
from phasesweep.config import ExecutionContext, Experiment
from phasesweep.config.io import load_experiment_bytes
from phasesweep.engine.paths import _artifact_root_binding_path, _experiment_dir
from phasesweep.mcp.runs import RunStore
from phasesweep.runtime.files import file_url_path
from tests.conftest import make_experiment
from tests.mcp_helpers import claim_runner_handle, make_run_handle, runner_argv, runner_main


def _write_config(path: Path, **fields: object) -> Path:
    """Write a one-trial echo-trainer experiment to ``path`` with ``fields`` merged in.

    :param Path path: Destination YAML file; its parent must exist.
    :param fields: Top-level experiment keys added to, or replacing, the base config.
    :return Path: ``path``.
    """
    payload: dict[str, object] = {
        "experiment": "anchored",
        "trial_command": "echo x=0.5 {overrides}",
        "override_format": "argparse",
        "metric": {
            "extractor": {"type": "log_regex", "pattern": r"x=(?P<value>[0-9.eE+-]+)"},
        },
        "phases": [
            {
                "name": "p",
                "n_trials": 1,
                "sampler": {"type": "random", "seed": 0},
                "search_space": {"x": {"type": "int", "low": 0, "high": 3}},
            }
        ],
        **fields,
    }
    if payload.get("storage") is not None:
        payload.setdefault("provenance", {"revision": "test-fixture-v1"})
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _project(tmp_path: Path) -> Path:
    """Create and return the resolved directory that holds the config file."""
    project = tmp_path / "project"
    project.mkdir()
    return project.resolve()


def test_file_loaded_relative_paths_resolve_against_config_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    explicit = load_experiment(
        _write_config(
            project / "explicit.yaml",
            workdir="runs",
            storage="journal:///./ledgers/study.journal",
            execution={"cwd": "trainer"},
        )
    )
    defaulted = load_experiment(_write_config(project / "defaulted.yaml", workdir="."))
    omitted = load_experiment(_write_config(project / "omitted.yaml"))

    assert explicit.workdir == str(project / "runs")
    assert explicit.storage == f"journal:///{project}/ledgers/study.journal"
    assert explicit.execution.cwd == str(project / "trainer")
    assert defaulted.workdir == str(project)
    assert omitted.workdir == str(project / "runs")
    # An unset execution.cwd keeps meaning the invocation directory.
    assert omitted.execution.cwd is None


def test_python_built_experiment_keeps_cwd_relative_paths() -> None:
    experiment = make_experiment(
        workdir="runs",
        storage="journal:///ledger.journal",
        execution=ExecutionContext(cwd="trainer"),
    )
    defaulted = make_experiment()

    assert experiment.workdir == "runs"
    assert experiment.storage == "journal:///ledger.journal"
    assert experiment.execution.cwd == "trainer"
    assert defaulted.workdir == "./runs"


def test_bytes_without_a_config_directory_keep_cwd_relative_paths(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "exp.yaml", workdir="runs")

    assert load_experiment_bytes(config.read_bytes()).workdir == "runs"
    with pytest.raises(ValueError, match="config_dir must be absolute"):
        load_experiment_bytes(config.read_bytes(), config_dir=Path("relative"))


def test_absolute_and_home_paths_are_not_anchored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = _project(tmp_path)
    absolute = tmp_path / "absolute-runs"

    experiment = load_experiment(
        _write_config(
            project / "exp.yaml",
            workdir=str(absolute),
            storage="journal:///~/study.journal",
            execution={"cwd": "~/trainer"},
        )
    )

    assert experiment.workdir == str(absolute)
    assert experiment.storage == "journal:///~/study.journal"
    assert experiment.execution.cwd == "~/trainer"


def test_auto_storage_follows_anchored_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    monkeypatch.chdir(tmp_path)

    experiment = load_experiment(
        _write_config(project / "exp.yaml", workdir="runs", storage="auto")
    )

    assert experiment.resolved_storage is not None
    assert file_url_path(experiment.resolved_storage) == str(
        project / "runs" / "anchored" / "study.journal"
    )


def test_relative_journal_refused_under_config_dir_with_query_char(tmp_path: Path) -> None:
    project = tmp_path / "a?b"
    project.mkdir()
    config = _write_config(project / "exp.yaml", storage="journal:///study.journal")

    with pytest.raises(ValidationError, match=r"contains \? or #"):
        load_experiment(config)
    # auto escapes the directory name, so the same location stays usable.
    load_experiment(_write_config(project / "auto.yaml", storage="auto"))


def test_revalidation_without_context_keeps_anchored_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    loaded = load_experiment(
        _write_config(
            project / "exp.yaml",
            workdir="runs",
            storage="journal:///study.journal",
            execution={"cwd": "trainer"},
        )
    )
    monkeypatch.chdir(tmp_path)

    for revalidated in (
        Experiment.model_validate(loaded),
        Experiment.model_validate(loaded.model_dump()),
    ):
        assert revalidated.workdir == loaded.workdir
        assert revalidated.storage == loaded.storage
        assert revalidated.execution.cwd == loaded.execution.cwd


def test_file_loaded_relative_storage_is_bound_to_the_config_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running one config from two directories writes one artifact tree and one ledger.

    ``execution.cwd: .`` pins the trainer to the config directory too; left
    unset it would be each invocation directory, which the fingerprint binds.
    """
    project = _project(tmp_path)
    config = _write_config(
        project / "exp.yaml",
        workdir="runs",
        storage="journal:///study.journal",
        execution={"cwd": "."},
    )
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()

    monkeypatch.chdir(first_cwd)
    first = load_experiment(config)
    run_experiment(first)
    binding = _artifact_root_binding_path(first).read_bytes()

    monkeypatch.chdir(second_cwd)
    second = load_experiment(config)
    run_experiment(second)

    assert second == first
    assert _experiment_dir(second) == project / "runs" / "anchored"
    assert (project / "study.journal").is_file()
    assert _artifact_root_binding_path(second).read_bytes() == binding
    assert not any(first_cwd.iterdir())
    assert not any(second_cwd.iterdir())


@pytest.mark.integration
def test_runner_snapshot_resolves_against_recorded_config_dir(tmp_path: Path) -> None:
    """The runner anchors its state-directory snapshot to the recorded config directory."""
    project = _project(tmp_path)
    (project / "trainer").mkdir()
    config = _write_config(
        project / "exp.yaml",
        workdir="runs",
        storage="journal:///study.journal",
        execution={"cwd": "trainer"},
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    store = RunStore(tmp_path / "state")
    run_id = "anchored-run"
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    started_at = "2026-06-24T00:00:00Z"
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        config_dir=project,
        started_at=started_at,
    )
    snapshot = store.config_snapshot_path(run_id)
    snapshot.write_bytes(config.read_bytes())

    exit_code = runner_main(
        runner_argv(
            store,
            run_id=run_id,
            config=snapshot,
            config_sha256=config_sha256,
            experiment_id="srv",
            started_at=started_at,
        ),
        cwd=elsewhere,
    )

    assert exit_code == 0
    assert json.loads(store.status_path(run_id).read_text())["returncode"] == 0
    assert (project / "study.journal").is_file()
    assert (project / "runs" / "anchored").is_dir()
    assert not any(elsewhere.iterdir())
    assert not (snapshot.parent / "runs").exists()
    trial_dirs = list((project / "runs" / "anchored" / "p").glob("trial_*"))
    assert len(trial_dirs) == 1
    assert "x=0.5" in (trial_dirs[0] / "stdout.log").read_text()


def test_handle_config_dir_is_immutable(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    pending = make_run_handle(
        run_id="r-dir", launch_state="launching", config_dir=str(tmp_path / "a")
    )
    store.create(pending)
    spawned = make_run_handle(
        run_id="r-dir", started_at=pending.started_at, config_dir=str(tmp_path / "b")
    )

    with pytest.raises(ValueError, match=r"immutable field\(s\): \['config_dir'\]"):
        store.update(spawned)
    store.update(replace(spawned, config_dir=pending.config_dir))

    persisted = store.get("r-dir")
    assert persisted is not None
    assert persisted.config_dir == str(tmp_path / "a")


@pytest.mark.parametrize("config_dir", ["relative/dir", "", "/nul\0dir"])
def test_handle_without_an_absolute_config_dir_is_malformed(
    tmp_path: Path, config_dir: str
) -> None:
    store = RunStore(tmp_path / "state")
    store.create(make_run_handle(run_id="r-bad", config_dir=config_dir))

    assert store.get("r-bad") is None
