"""Installed-package starter configuration workflow."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from phasesweep import load_experiment
from phasesweep.cli import main as cli_main
from phasesweep.mcp.registry import Registry


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
        assert experiment.storage == f"sqlite:///{config_path.parent / 'runs' / 'phases.db'}"
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
