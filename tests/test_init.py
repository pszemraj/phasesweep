"""Installed-package template configuration workflow."""

from __future__ import annotations

import logging
import os
import sys
from importlib import resources
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from optuna import create_study
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import phasesweep.config.reference as reference
from phasesweep import load_experiment
from phasesweep.cli import _template_names, _template_purpose
from phasesweep.cli import cli as cli_main
from phasesweep.engine.run import run_experiment
from phasesweep.engine.state import Winner
from phasesweep.mcp.registry import Registry
from phasesweep.runtime.files import file_url_path

TEMPLATES = _template_names()


def _template_bytes(template: str) -> bytes:
    """Read one packaged template file as shipped.

    :param str template: Template name.
    :return bytes: The file's exact bytes.
    """
    return resources.files("phasesweep").joinpath("templates", f"{template}.yaml").read_bytes()


@pytest.fixture
def trainer_python_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the test interpreter the ``python`` the templates' trial command finds.

    Templates run ``python -m phasesweep.examples.fake_train`` with the default
    ``inherit_env: none``, so the trainer gets only the ambient ``PATH``.
    """
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    )


def _init_and_run(tmp_path: Path, template: str) -> dict[str, Winner]:
    """Write a template in a scratch directory and run it on the toy trainer.

    Asserts that every phase published a winner that ``show-winners`` can read.

    :param Path tmp_path: Directory that holds the scratch directory.
    :param str template: Template name.
    :return dict[str, Winner]: Winners by phase name.
    """
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        written = runner.invoke(cli_main, ["init", "--template", template])
        assert written.exit_code == 0, written.output
        config = f"{template}.yaml"
        experiment = load_experiment(Path(config).resolve())
        winners = run_experiment(experiment)
        assert list(winners) == [phase.name for phase in experiment.phases]
        shown = runner.invoke(cli_main, ["show-winners", config])
        assert shown.exit_code == 0, shown.output
    return winners


def test_init_list_prints_every_template_with_a_purpose(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli_main, ["init", "--list"])

        assert result.exit_code == 0, result.output
        assert list(Path().iterdir()) == []
    listed = [line.split(maxsplit=1) for line in result.output.splitlines()]
    assert [name for name, _purpose in listed] == TEMPLATES
    assert TEMPLATES[0] == "starter"
    for name, purpose in listed:
        assert purpose == _template_purpose(name)
        assert purpose.strip()


def test_init_default_writes_starter_yaml_in_cwd(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli_main, ["init"])

        assert result.exit_code == 0, result.output
        assert [path.name for path in Path().iterdir()] == ["starter.yaml"]


def test_init_into_directory_writes_template_named_file(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    created = tmp_path / "new" / "configs"

    into_existing = CliRunner().invoke(
        cli_main, ["init", "--template", "noisy", "-o", str(existing)]
    )
    into_created = CliRunner().invoke(
        cli_main, ["init", "--template", "chained", "-o", f"{created}{os.sep}"]
    )

    assert into_existing.exit_code == 0, into_existing.output
    assert (existing / "noisy.yaml").is_file()
    assert into_created.exit_code == 0, into_created.output
    assert (created / "chained.yaml").is_file()
    assert f"phasesweep validate {existing / 'noisy.yaml'}" in into_existing.output


def test_init_unknown_template_exits_2(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli_main, ["init", "--template", "bogus"])

        assert result.exit_code == 2
        assert "'bogus' is not one of" in result.output
        assert list(Path().iterdir()) == []


@pytest.mark.parametrize("template", TEMPLATES)
def test_template_headers_name_explain(template: str) -> None:
    first, second, *_rest = _template_bytes(template).decode("utf-8").splitlines()

    assert first.startswith("# phasesweep template: ")
    assert _template_purpose(template)
    assert second.startswith("# Details on any key: phasesweep explain ")
    assert "phasesweep explain 'phases[].sampler'" in second


@pytest.mark.parametrize("template", TEMPLATES)
def test_init_preserves_template_except_documentation_links(tmp_path: Path, template: str) -> None:
    """Generated templates retain their content while pinning project documentation links."""
    output = tmp_path / "out" / "chosen.yaml"

    result = CliRunner().invoke(cli_main, ["init", "--template", template, "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8") == reference.pin_documentation_links(
        _template_bytes(template).decode("utf-8")
    )


@pytest.mark.parametrize(
    ("version", "editable", "ref"),
    [
        ("0.3.11", False, "v0.3.11"),
        ("0.3.11", True, "main"),
        ("0.3.12.dev4+g123abcd.d20261023", False, "main"),
        ("0.3.13.dev0", False, "main"),
        ("0.0.0+unknown", False, "main"),
    ],
)
def test_init_pins_template_documentation_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str, editable: bool, ref: str
) -> None:
    """A generated template links to documentation for its installed source."""
    monkeypatch.setattr(reference, "__version__", version)
    monkeypatch.setattr(reference, "_editable_install", lambda: editable)
    output = tmp_path / "noisy.yaml"

    result = CliRunner().invoke(cli_main, ["init", "--template", "noisy", "-o", str(output)])

    assert result.exit_code == 0, result.output
    text = output.read_text(encoding="utf-8")
    assert (
        f"https://github.com/pszemraj/phasesweep/blob/{ref}/"
        "docs/config.md#noisy-objectives-and-small-budgets"
    ) in text
    assert ("/blob/main/" in text) == (ref == "main")


@pytest.mark.parametrize("template", TEMPLATES)
def test_init_template_loads_validates_and_scaffolds_a_catalog(
    tmp_path: Path, template: str
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        config = f"{template}.yaml"
        result = runner.invoke(cli_main, ["init", "--template", template])
        assert result.exit_code == 0, result.output
        assert f"Wrote the {template} template to" in result.output
        assert f"phasesweep validate {config}" in result.output
        assert f"phasesweep run {config} --dry-run" in result.output
        assert f"phasesweep status {config} --trials" in result.output
        assert f"phasesweep mcp init-catalog --from {config}" in result.output
        assert "phasesweep explain" in result.output

        config_path = Path(config).resolve()
        experiment = load_experiment(config_path)
        # Config-relative paths and a pinned trainer directory: the same file
        # names the same study from any directory.
        assert experiment.workdir == str(config_path.parent / "runs")
        assert experiment.execution.cwd == str(config_path.parent)
        assert experiment.storage == "auto"
        if template == "mlflow":
            assert experiment.trial_command == "phasesweep-mlflow-train --config {config_path}"
        else:
            assert "phasesweep.examples.fake_train" in experiment.trial_command

        validated = runner.invoke(cli_main, ["validate", config])
        assert validated.exit_code == 0, validated.output
        dry_run = runner.invoke(cli_main, ["run", config, "--dry-run"])
        assert dry_run.exit_code == 0, dry_run.output
        scaffolded = runner.invoke(
            cli_main, ["mcp", "init-catalog", "--from", config, "-o", "catalog.yaml"]
        )
        assert scaffolded.exit_code == 0, scaffolded.output
        assert Registry.load(Path("catalog.yaml")).get(template).allow_launch is False


def test_starter_template_shape(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        assert runner.invoke(cli_main, ["init"]).exit_code == 0

        config_path = Path("starter.yaml").resolve()
        # The template is written verbatim; its relative workdir anchors at load.
        assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["workdir"] == "./runs"
        experiment = load_experiment(config_path)
        assert experiment.experiment == "phasesweep_starter"
        assert file_url_path(experiment.resolved_storage) == str(
            config_path.parent / "runs" / experiment.experiment / "study.journal"
        )
        assert [phase.name for phase in experiment.phases] == ["depth", "learning_rate"]
        assert sum(phase.n_trials for phase in experiment.phases) == 4
        assert experiment.phases[1].inherits == ["depth"]
        assert experiment.override_format == "yaml_file"
        assert experiment.execution.inherit_env == "none"
        assert "{config_path}" in experiment.trial_command
        assert experiment.trainer_config["model"]["n_layers"] == 8
        assert "--out" not in experiment.trial_command


def test_argparse_template_renders_trainer_config_leaves_as_flags(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        assert runner.invoke(cli_main, ["init", "--template", "argparse"]).exit_code == 0
        experiment = load_experiment(Path("argparse.yaml").resolve())

        with caplog.at_level(logging.INFO):
            run_experiment(experiment, dry_run=True)

    assert experiment.override_format == "argparse"
    assert "{overrides}" in experiment.trial_command
    assert "--model.n_layers=8" in caplog.text
    assert "--optimizer.weight_decay=0.05" in caplog.text
    assert "--optimizer.lr=" in caplog.text


@pytest.mark.integration
@pytest.mark.usefixtures("trainer_python_on_path")
@pytest.mark.parametrize(
    "template", [name for name in TEMPLATES if name not in {"noisy", "mlflow"}]
)
def test_init_template_runs_to_a_winner_per_phase_on_the_toy_trainer(
    tmp_path: Path, template: str
) -> None:
    """Local templates publish winners without an external tracking server."""
    assert _init_and_run(tmp_path, template)


@pytest.mark.integration
@pytest.mark.usefixtures("trainer_python_on_path")
def test_noisy_template_winner_comes_from_single_trials_and_records_the_observation(
    tmp_path: Path,
) -> None:
    winners = _init_and_run(tmp_path, "noisy")

    assert list(winners) == ["learning_rate"]
    assert winners["learning_rate"].repeats is None
    assert winners["learning_rate"].observed == {"param_bytes": 8_800_000.0}


def test_noisy_template_is_a_dense_single_seed_learning_rate_grid(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        assert runner.invoke(cli_main, ["init", "--template", "noisy"]).exit_code == 0
        experiment = load_experiment(Path("noisy.yaml").resolve())

    (phase,) = experiment.phases
    assert phase.repeats is None
    assert phase.n_trials == 8
    assert phase.sampler.type == "grid"
    assert list(phase.search_space) == ["optimizer.lr"]
    assert experiment.trainer_config["seed"] == 1


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
    output = project / "starter.yaml"

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 0, result.output
    workdir = Path(load_experiment(output).workdir)
    workdir.mkdir(parents=True)
    assert workdir == project / "runs"


def test_init_preserves_question_mark_in_journal_path(tmp_path: Path) -> None:
    project = tmp_path / "local?sweeps"
    output = project / "starter.yaml"

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
    output = tmp_path / "starter.yaml"
    output.write_text("keep me\n")

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output
    assert output.read_text() == "keep me\n"


def test_init_refuses_broken_symlink(tmp_path: Path) -> None:
    output = tmp_path / "starter.yaml"
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
    output = tmp_path / "starter.yaml"

    def fail(*_args: object) -> None:
        raise error

    monkeypatch.setattr(f"phasesweep.runtime.files.os.{patched}", fail)

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert f"phasesweep init: cannot write {output}: {error}" in result.output
    assert "Traceback" not in result.output
    assert not output.exists()
    assert not list(tmp_path.glob(".starter.yaml.*.tmp"))


def test_init_losing_publish_race_preserves_other_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "starter.yaml"

    def lose_publish_race(_source: Path, destination: Path) -> None:
        destination.write_text("other process\n")
        raise FileExistsError(destination)

    monkeypatch.setattr("phasesweep.runtime.files.os.link", lose_publish_race)

    result = CliRunner().invoke(cli_main, ["init", "-o", str(output)])

    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output
    assert output.read_text() == "other process\n"
    assert not list(tmp_path.glob(".starter.yaml.*.tmp"))


def test_init_points_to_template_prerequisites_and_adaptation(tmp_path: Path) -> None:
    """The generated header owns setup and adaptation advice before launch commands."""
    result = CliRunner().invoke(cli_main, ["init", "--template", "mlflow", "-o", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "read the template header for prerequisites and trainer adaptation" in result.output
    assert result.output.index("template header") < result.output.index("phasesweep validate")
    assert "Before running:" not in result.output
    config = (tmp_path / "mlflow.yaml").read_text(encoding="utf-8")
    assert "Remove or replace the demo cost constraint and mlflow_required gates" in config
    assert "first mlflow.start_run() or mlflow.log_* resumes MLFLOW_RUN_ID" in config
    assert "mlflow.active_run().info.run_id" in config


def test_init_help_identifies_mlflow_prerequisites() -> None:
    """MLflow help names its server and experiment prerequisites."""
    result = CliRunner().invoke(cli_main, ["init", "--help"])
    assert result.exit_code == 0
    text = " ".join(result.output.split())
    assert "mlflow needs a tracking server and an existing experiment" in text
    assert "Every one runs as written" not in text
