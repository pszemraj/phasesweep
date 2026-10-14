"""Installed-package starter configuration workflow."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from optuna import create_study
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from phasesweep import load_experiment
from phasesweep.cli import cli as cli_main
from phasesweep.mcp.registry import Registry
from phasesweep.runtime.files import file_url_path


def test_init_creates_runnable_starter_and_catalog(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli_main, ["init"])
        assert result.exit_code == 0, result.output
        assert "phasesweep validate experiment.yaml" in result.output
        assert "phasesweep run experiment.yaml --dry-run" in result.output
        assert "phasesweep mcp init-catalog --from experiment.yaml" in result.output

        config_path = Path("experiment.yaml").resolve()
        # The template is written verbatim; its relative workdir anchors at load.
        assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["workdir"] == "./runs"
        experiment = load_experiment(config_path)
        assert experiment.experiment == "phasesweep_starter"
        assert experiment.workdir == str(config_path.parent / "runs")
        assert experiment.storage == "auto"
        assert file_url_path(experiment.resolved_storage) == str(
            config_path.parent / "runs" / experiment.experiment / "study.journal"
        )
        assert [phase.name for phase in experiment.phases] == ["depth", "learning_rate"]
        assert sum(phase.n_trials for phase in experiment.phases) == 4
        assert experiment.phases[1].inherits == ["depth"]
        assert "phasesweep.examples.fake_train" in experiment.trial_command
        assert experiment.override_format == "yaml_file"
        assert experiment.execution.inherit_env == "none"
        assert "{config_path}" in experiment.trial_command
        assert experiment.trainer_config["model"]["n_layers"] == 8
        assert "--out" not in experiment.trial_command

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


def test_init_round_trips_non_bmp_destination_path(tmp_path: Path) -> None:
    """A non-BMP character in the destination directory survives config load.

    The starter's relative workdir resolves against the config file's
    directory, so the loaded workdir must name that directory's ``runs`` with
    every character intact; a lone surrogate would fail every later
    filesystem call with ``UnicodeEncodeError``.
    """
    project = tmp_path / "\U0001f680 sweeps"
    project.mkdir()
    output = project / "experiment.yaml"

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 0, result.output
    workdir = Path(load_experiment(output).workdir)
    workdir.mkdir(parents=True)
    assert workdir == project / "runs"


def test_init_preserves_question_mark_in_journal_path(tmp_path: Path) -> None:
    project = tmp_path / "local?sweeps"
    output = project / "experiment.yaml"

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 0, result.output
    experiment = load_experiment(output)
    database = project / "runs" / experiment.experiment / "study.journal"
    resolved_path = file_url_path(experiment.resolved_storage)
    assert resolved_path == str(database)

    database.parent.mkdir(parents=True)
    create_study(storage=JournalStorage(JournalFileBackend(resolved_path)), study_name="path_check")
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


@pytest.mark.parametrize(
    ("patched", "error"),
    [
        pytest.param("fsync", OSError("fsync failed"), id="fsync_failure"),
        pytest.param("link", PermissionError("hard links are not supported"), id="eperm_on_link"),
    ],
)
def test_init_write_failure_is_a_user_facing_error_that_does_not_claim_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched: str,
    error: OSError,
) -> None:
    """A write or publish failure is a user-facing error, not a traceback.

    ``init`` is the first command a new user runs, and ``os.link`` fails with
    ``EPERM`` on filesystems without hard links, not only with
    ``FileExistsError``. Whichever step fails, the destination is never claimed
    and no temporary file is left behind.
    """
    output = tmp_path / "experiment.yaml"

    def fail(*_args: object) -> None:
        raise error

    monkeypatch.setattr(f"phasesweep.runtime.files.os.{patched}", fail)

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert f"phasesweep init: cannot write {output}: {error}" in result.output
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

    monkeypatch.setattr("phasesweep.runtime.files.os.link", lose_publish_race)

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output
    assert output.read_text() == "other process\n"
    assert not list(tmp_path.glob(".experiment.yaml.*.tmp"))
