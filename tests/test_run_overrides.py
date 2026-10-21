"""Derive a run from one config without editing or copying that config."""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import phasesweep.cli as cli_module
from phasesweep.cli import cli
from phasesweep.config.models import Experiment
from phasesweep.engine.state import Winner
from tests.conftest import invoke_cli_boundary
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
    path = write_experiment(tmp_path / "sweep.yaml", workdir="original")
    result = invoke_cli_boundary(["run", str(path), "--dry-run", "--set", assignment], monkeypatch)

    assert result == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "original").exists()


def _two_phase_config(tmp_path: Path) -> Path:
    """Write a config with phases ``lr`` and ``wd``, a dotted search-space key and a trainer_config."""
    return write_experiment(
        tmp_path / "sweep.yaml",
        workdir="original",
        provenance={"revision": "r1"},
        trainer_config={"optimizer": {"lr": 0.1}},
        phases=[
            phase_payload(
                name="lr",
                search_space={"model.n_layers": {"type": "categorical", "choices": [4, 5]}},
            ),
            phase_payload(name="wd"),
        ],
    )


def _derived(path: Path, monkeypatch: pytest.MonkeyPatch, *settings: str) -> Experiment:
    """Run ``run --dry-run`` with the settings and return the experiment it would run."""
    received: list[Experiment] = []

    def capture(
        experiment: Experiment, *, from_phase: str | None, dry_run: bool
    ) -> dict[str, Winner]:
        received.append(experiment)
        return {}

    monkeypatch.setattr(cli_module, "run_experiment", capture)
    arguments = ["run", str(path), "--dry-run"]
    for setting in settings:
        arguments += ["--set", setting]
    result = CliRunner().invoke(cli, arguments)
    assert result.exit_code == 0, result.output
    (experiment,) = received
    return experiment


def _refused(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    setting: str,
) -> str:
    """Run ``run --dry-run --set SETTING`` through the process boundary and return its stderr."""
    assert invoke_cli_boundary(["run", str(path), "--dry-run", "--set", setting], monkeypatch) == 2
    return capsys.readouterr().err


@pytest.mark.parametrize("key", ["phases.1.n_trials", "phases[1].n_trials", "phases[wd].n_trials"])
def test_set_addresses_a_phase_by_dotted_index_bracket_index_or_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """The three spellings of the second phase change that phase and no other."""
    experiment = _derived(_two_phase_config(tmp_path), monkeypatch, f"{key}=3")

    assert [phase.n_trials for phase in experiment.phases] == [1, 3]
    assert [phase.name for phase in experiment.phases] == ["lr", "wd"]


def test_set_bracket_addresses_a_search_space_key_that_contains_dots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``[model.n_layers]`` is one mapping key, not a nested ``model`` mapping."""
    experiment = _derived(
        _two_phase_config(tmp_path),
        monkeypatch,
        "phases[lr].search_space[model.n_layers].choices=[6, 8]",
    )

    space = experiment.phases[0].search_space
    assert list(space) == ["model.n_layers"]
    assert space["model.n_layers"].model_dump()["choices"] == [6, 8]


@pytest.mark.parametrize(
    ("written", "key"),
    [
        ("a.b", "a.b"),
        ("a=b", "a=b"),
        (r"a\]b", "a]b"),
        (r"a\\b", "a\\b"),
        (r"a[0\]", "a[0]"),
        (" spaced ", " spaced "),
    ],
)
def test_set_bracket_addresses_any_mapping_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, written: str, key: str
) -> None:
    """Dots, equals signs, brackets and backslashes in a key are written inside ``[...]``."""
    experiment = _derived(_two_phase_config(tmp_path), monkeypatch, f"provenance[{written}]=v")

    assert experiment.provenance == {"revision": "r1", key: "v"}


def test_set_can_add_keys_to_the_free_form_trainer_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new key, nested or not, is accepted under ``trainer_config`` and keeps existing keys."""
    experiment = _derived(
        _two_phase_config(tmp_path),
        monkeypatch,
        "trainer_config.optimizer.warmup=100",
        "trainer_config.extra.depth=4",
    )

    assert experiment.trainer_config == {
        "optimizer": {"lr": 0.1, "warmup": 100},
        "extra": {"depth": 4},
    }


def test_set_refuses_an_unknown_phase_name_and_lists_the_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown phase name stops at ``--set`` with the argument, segment and syntax named."""
    message = _refused(_two_phase_config(tmp_path), monkeypatch, capsys, "phases[wd2].n_trials=2")

    assert "'phases[wd2].n_trials=2'" in message
    assert "'[wd2]'" in message
    assert "lr, wd" in message
    assert "Accepted syntax" in message
    assert "validation error" not in message
    assert not (tmp_path / "original").exists()


@pytest.mark.parametrize(
    ("setting", "created"),
    [
        ("phases[0].n_trial=2", "phases[0].n_trial"),
        ("phases.0.serach_space.x.low=1", "phases.0.serach_space"),
        ("phases[lr].sampler.sed=3", "phases[lr].sampler.sed"),
        ("unknown=1", "unknown"),
    ],
)
def test_set_names_the_argument_that_created_a_key_validation_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    setting: str,
    created: str,
) -> None:
    """A schema typo is reported against the ``--set`` that created the key."""
    message = _refused(_two_phase_config(tmp_path), monkeypatch, capsys, setting)

    assert "Extra inputs are not permitted" in message
    assert f"--set {setting!r} created {created!r}" in message
    assert not (tmp_path / "original").exists()


def test_set_does_not_blame_an_override_for_an_error_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rejected value at an existing key carries no creation note."""
    message = _refused(_two_phase_config(tmp_path), monkeypatch, capsys, "phases[0].n_trials=zero")

    assert "n_trials" in message
    assert "created" not in message


@pytest.mark.parametrize(
    ("setting", "fragment"),
    [
        ("phases[9].n_trials=1", "index 9 is out of range"),
        ("phases.lr.n_trials=1", "'lr' is not a list index"),
        ("phases[].n_trials=1", "is named ''"),
        ("phases[0.n_trials=1", "no closing ']'"),
        ("phases0].n_trials=1", "unmatched ']'"),
        ("phases[0]x=1", "must follow a '.'"),
        ("phases..n_trials=1", "empty path segment"),
        (".phases=1", "empty path segment"),
        ("phases.=1", "empty path segment"),
        ("=1", "nonempty KEY"),
        ("[0]=1", "no name before it"),
        ("phases[0].n_trials.x=1", "descend into the int value at 'phases[0].n_trials'"),
        ("phases[lr].sampler.type.x=1", "descend into the str value"),
    ],
)
def test_set_path_errors_name_the_argument_the_segment_and_the_syntax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    setting: str,
    fragment: str,
) -> None:
    """Every refused path quotes its argument, says what failed and repeats the syntax."""
    message = _refused(_two_phase_config(tmp_path), monkeypatch, capsys, setting)

    assert repr(setting) in message
    assert fragment in message
    assert "Accepted syntax" in message
