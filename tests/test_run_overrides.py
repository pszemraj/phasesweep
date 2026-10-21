"""Derive a run from one config without editing or copying that config."""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import phasesweep.cli as cli_module
from phasesweep.cli import cli
from phasesweep.config.models import Experiment
from phasesweep.engine.state import Winner
from tests.experiments import phase_payload, write_experiment


def test_run_overrides_are_typed_validated_and_leave_yaml_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated dotted assignments update phase lists and top-level fields in memory."""
    path = write_experiment(
        tmp_path / "sweep.yaml",
        storage="auto",
        workdir="original",
        execution={"cwd": "."},
        phases=[phase_payload(n_trials=0)],
    )
    original = path.read_bytes()
    received: list[Experiment] = []
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)

    def capture(
        experiment: Experiment, *, from_phase: str | None, dry_run: bool
    ) -> dict[str, Winner]:
        assert dry_run
        assert from_phase is None
        received.append(experiment)
        return {}

    monkeypatch.setattr(cli_module, "run_experiment", capture)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(path),
            "--dry-run",
            "--workdir",
            "derived",
            "--set",
            "workdir=ignored",
            "--set",
            "experiment=smoke",
            "--set",
            "phases.0.n_trials=2",
            "--set",
            "phases.0.fixed_overrides={seed: 9, enabled: true}",
            "--set",
            "env.NOTE='a=b'",
        ],
    )

    assert result.exit_code == 0, result.output
    (experiment,) = received
    assert experiment.experiment == "smoke"
    assert Path(experiment.workdir) == caller / "derived"
    assert experiment.execution.cwd == str(tmp_path)
    assert experiment.phases[0].n_trials == 2
    assert experiment.phases[0].fixed_overrides == {"seed": 9, "enabled": True}
    assert experiment.env == {"NOTE": "a=b"}
    assert path.read_bytes() == original
    assert not (caller / "derived").exists()


def test_derived_run_writes_its_overrides_to_the_config_snapshot(tmp_path: Path) -> None:
    """A real one-trial run uses the derived namespace and records its effective config."""
    path = write_experiment(
        tmp_path / "sweep.yaml",
        storage="auto",
        workdir="original",
        trial_command="echo x=0.5 {overrides}",
        metric={"extractor": {"type": "log_regex", "pattern": r"x=(?P<value>[0-9.]+)"}},
    )
    original = path.read_bytes()
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(path),
            "--workdir",
            str(tmp_path / "derived"),
            "--set",
            "experiment=smoke",
            "--set",
            "phases.0.gpu_policy=none",
        ],
    )

    assert result.exit_code == 0, result.output
    snapshots = list(
        (tmp_path / "derived" / "smoke" / "generations").glob("*/config.snapshot.yaml")
    )
    assert len(snapshots) == 1
    recorded = yaml.safe_load(snapshots[0].read_text(encoding="utf-8"))
    assert recorded["experiment"] == "smoke"
    assert recorded["workdir"] == str(tmp_path / "derived")
    assert path.read_bytes() == original
    assert not (tmp_path / "original").exists()


@pytest.mark.parametrize(
    ("assignment", "message"),
    [
        ("phases.0.n_trials=zero", "n_trials"),
        ("phases.9.n_trials=1", "phases.9"),
        ("unknown=1", "Extra inputs are not permitted"),
        ("nonsense", "KEY=VALUE"),
    ],
)
def test_run_override_errors_refuse_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    assignment: str,
    message: str,
) -> None:
    """Malformed or invalid overrides remain ordinary configuration errors."""
    from tests.conftest import invoke_cli_boundary

    path = write_experiment(tmp_path / "sweep.yaml", workdir="original")
    result = invoke_cli_boundary(["run", str(path), "--dry-run", "--set", assignment], monkeypatch)

    assert result == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "original").exists()
