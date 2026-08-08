"""Installed-package starter configuration workflow."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from optuna import create_study

from phasesweep import load_experiment
from phasesweep.cli import _starter_experiment_text
from phasesweep.cli import cli as cli_main
from phasesweep.mcp.registry import Registry
from phasesweep.runtime.files import sqlite_uri_filename_path


def test_init_creates_runnable_starter_and_catalog(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli_main, ["init"])
        assert result.exit_code == 0, result.output
        assert "phasesweep validate experiment.yaml" in result.output
        assert "phasesweep run experiment.yaml --dry-run" in result.output
        assert "phasesweep mcp init-catalog --from experiment.yaml" in result.output

        config_path = Path("experiment.yaml").resolve()
        experiment = load_experiment(config_path)
        assert experiment.experiment == "phasesweep_starter"
        assert Path(experiment.workdir).is_absolute()
        assert experiment.storage == (
            f"sqlite:///file:{config_path.parent / 'runs' / 'phases.db'}?uri=true"
        )
        assert [phase.name for phase in experiment.phases] == ["depth", "learning_rate"]
        assert sum(phase.n_trials for phase in experiment.phases) == 4
        assert experiment.phases[1].inherits == ["depth"]
        assert "phasesweep.examples.fake_train" in experiment.trial_command

        validated = runner.invoke(cli_main, ["validate", "experiment.yaml"])
        assert validated.exit_code == 0, validated.output
        dry_run = runner.invoke(cli_main, ["run", "experiment.yaml", "--dry-run"])
        assert dry_run.exit_code == 0, dry_run.output
        scaffolded = runner.invoke(
            cli_main,
            ["mcp", "init-catalog", "--from", "experiment.yaml", "-o", "catalog.yaml"],
        )
        assert scaffolded.exit_code == 0, scaffolded.output
        assert Registry.load(Path("catalog.yaml")).get("experiment").allow_launch is False


def test_init_creates_parent_directories_for_custom_output(tmp_path: Path) -> None:
    output = tmp_path / "project" / "configs" / "starter.yaml"

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert output.is_file()
    experiment = load_experiment(output)
    assert Path(experiment.workdir) == output.parent / "runs"


def test_init_round_trips_non_bmp_paths(tmp_path: Path) -> None:
    """Rendered paths must survive YAML decoding byte-for-byte.

    ``json.dumps`` with the default ``ensure_ascii=True`` escapes a non-BMP
    character as a UTF-16 surrogate pair, which a YAML double-quoted scalar
    decodes into lone surrogates; every later filesystem call on the decoded
    workdir then fails with ``UnicodeEncodeError``.
    """
    project = tmp_path / "\U0001f680 sweeps"
    project.mkdir()
    output = project / "experiment.yaml"

    rendered = yaml.safe_load(_starter_experiment_text(output))
    runs_dir = output.parent / "runs"
    assert rendered["workdir"] == str(runs_dir)
    assert sqlite_uri_filename_path(rendered["storage"]) == str(runs_dir / "phases.db")

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 0, result.output
    experiment = load_experiment(output)
    # Lone surrogates raise UnicodeEncodeError here; a real path does not.
    Path(experiment.workdir).mkdir(parents=True)
    assert Path(experiment.workdir) == runs_dir


def test_init_renders_destinations_containing_placeholder_literals(tmp_path: Path) -> None:
    """Placeholder substitution must never rescan text it already inserted.

    Replacing the placeholders one after another lets a destination path that
    contains a placeholder literal be rewritten by a later replacement, which
    corrupts a config that ``init`` still reports as written.
    """
    project = tmp_path / "__PHASESWEEP_STORAGE__"
    output = project / "experiment.yaml"
    runs_dir = project / "runs"

    rendered = yaml.safe_load(_starter_experiment_text(output))

    assert rendered["workdir"] == str(runs_dir)
    assert sqlite_uri_filename_path(rendered["storage"]) == str(runs_dir / "phases.db")

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 0, result.output
    experiment = load_experiment(output)
    assert Path(experiment.workdir) == runs_dir


def test_init_preserves_question_mark_in_sqlite_path(tmp_path: Path) -> None:
    project = tmp_path / "local?sweeps"
    output = project / "experiment.yaml"

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 0, result.output
    experiment = load_experiment(output)
    database = project / "runs" / "phases.db"
    assert sqlite_uri_filename_path(experiment.storage) == str(database)

    database.parent.mkdir()
    create_study(storage=experiment.storage, study_name="path_check")
    assert database.is_file()
    assert not (tmp_path / "local").exists()


def test_init_prints_next_commands_with_the_expanded_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Printed commands must name the written file, not the raw ``-o`` value.

    Shell quoting suppresses ``~`` expansion, so echoing the unexpanded option
    value produces commands that address a different, nonexistent path.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli_main, ["init", "-o", "~/experiments/starter.yaml"])

    assert result.exit_code == 0, result.output
    written = tmp_path / "experiments" / "starter.yaml"
    assert written.is_file()
    assert f"phasesweep validate {written}" in result.output
    assert f"phasesweep run {written} --dry-run" in result.output
    assert f"phasesweep mcp init-catalog --from {written}" in result.output
    assert "~" not in result.output


def test_init_refuses_to_replace_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "experiment.yaml"
    output.write_text("keep me\n")

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output
    assert output.read_text() == "keep me\n"


def test_init_refuses_broken_symlink(tmp_path: Path) -> None:
    output = tmp_path / "experiment.yaml"
    output.symlink_to(tmp_path / "missing-target.yaml")

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output
    assert output.is_symlink()


def test_init_write_failure_does_not_claim_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "experiment.yaml"

    def fail_fsync(_fd: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr("phasesweep.cli.os.fsync", fail_fsync)

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 1
    assert f"phasesweep init: cannot write {output}: fsync failed" in result.output
    assert not output.exists()
    assert not list(tmp_path.glob(".experiment.yaml.*.tmp"))


def test_init_reports_publish_errors_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hard-link failure is a user-facing error, not a traceback.

    ``init`` is the first command a new user runs, and ``os.link`` fails with
    ``EPERM`` on filesystems without hard links, not only with
    ``FileExistsError``.
    """
    output = tmp_path / "experiment.yaml"

    def deny_hard_link(_source: Path, _destination: Path) -> None:
        raise PermissionError("hard links are not supported")

    monkeypatch.setattr("phasesweep.cli.os.link", deny_hard_link)

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert f"phasesweep init: cannot write {output}: hard links are not supported" in result.output
    assert "Traceback" not in result.output
    assert not output.exists()
    assert not list(tmp_path.glob(".experiment.yaml.*.tmp"))


def test_init_losing_publish_race_preserves_other_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "experiment.yaml"

    def lose_publish_race(_source: Path, destination: Path) -> None:
        destination.write_text("other process\n")
        raise FileExistsError(destination)

    monkeypatch.setattr("phasesweep.cli.os.link", lose_publish_race)

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output
    assert output.read_text() == "other process\n"
    assert not list(tmp_path.glob(".experiment.yaml.*.tmp"))
