"""Concise repeated winners expose ranking and point to complete checkpoint evidence."""

import csv
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from phasesweep import run_experiment
from phasesweep.cli import cli
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import Repeats, Sampler
from phasesweep.engine.paths import _generation_winner_path, _phase_dir
from phasesweep.engine.repeats import ReduceName
from tests.conftest import make_experiment, write_experiment_config
from tests.ledger_fixtures import published_generation_id


def _publish_winners(
    tmp_path: Path, reducer: ReduceName = "mean"
) -> tuple[Experiment, Path, dict[str, bytes]]:
    experiment = make_experiment(
        persistent=tmp_path,
        trial_command="echo x=0.5 {overrides}",
        phases=[
            Phase(
                name="repeated",
                n_trials=3,
                sampler=Sampler(type="random", seed=0),
                repeats=Repeats(key="seed", values=[11, 22, 33], reduce=reducer),
            ),
            Phase(name="scalar", n_trials=1, sampler=Sampler(type="random", seed=0)),
        ],
    )
    run_experiment(experiment)
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    contents = {
        phase.name: _generation_winner_path(experiment, generation_id, phase.name).read_bytes()
        for phase in experiment.phases
    }
    config_path = tmp_path / "experiment.yaml"
    write_experiment_config(config_path, experiment)
    return experiment, config_path, contents


def _assert_member_directories(experiment: Experiment, saved: bytes) -> None:
    repeats = yaml.safe_load(saved)["repeats"]
    with (_phase_dir(experiment, "repeated") / "trials.csv").open(encoding="utf-8") as handle:
        rows = {int(row["number"]): row for row in csv.DictReader(handle)}
    for member in repeats["members"]:
        row = rows[member["trial_number"]]
        directory = Path(row["user_attr:phasesweep_trial_dir"])
        assert directory.is_dir()
        assert (directory / "stdout.log").is_file()
        assert row["param:seed"] == str(member["repeat"])
        assert member["generation_id"] in directory.name
        assert member["attempt_id"] in directory.name


def _assert_full_record(experiment: Experiment, phase: str, saved: bytes, output: str) -> None:
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    path = _generation_winner_path(experiment, generation_id, phase)
    assert f"# full record: {path}" in output.splitlines()
    assert path.read_bytes() == saved


def _assert_repeat_summary(saved: bytes, output: str) -> None:
    full = yaml.safe_load(saved)
    compact = yaml.safe_load(output)
    repeats = full["repeats"]
    assert compact["trial_number"] == full["trial_number"]
    assert compact["metric"] == {"objective": full["metric"]["objective"]}
    assert compact["params"] == full["params"]
    assert compact["effective_overrides"] == full["effective_overrides"]
    assert {
        key: value for key, value in compact["repeats"].items() if key not in {"key", "members"}
    } == {
        "reduce": repeats["reduce"],
        **repeats["spread"],
        "member_trials": [member["trial_number"] for member in repeats["members"]],
    }
    assert compact["repeats"]["key"] == repeats["key"]
    for expected, member in zip(repeats["members"], compact["repeats"]["members"], strict=True):
        assert member["trial_number"] == expected["trial_number"]
        assert member["repeat"] == expected["repeat"]
        directory = Path(member["trial_dir"])
        assert directory.is_dir()
        assert expected["generation_id"] in directory.name
        assert expected["attempt_id"] in directory.name
    assert (
        f"# metric is the {repeats['reduce']} over {repeats['spread']['n']} repeats."
    ) in output.splitlines()
    assert (
        f"# trial_number {full['trial_number']} names the group's last member."
    ) in output.splitlines()
    assert (
        "# Choose a checkpoint explicitly from repeats.members below; each member includes its trial_dir."
    ) in output.splitlines()


@pytest.mark.parametrize("reducer", ["mean", "median"])
def test_show_winners_explains_repeat_metric_and_checkpoint_choice(
    tmp_path: Path, reducer: ReduceName
) -> None:
    experiment, config_path, contents = _publish_winners(tmp_path, reducer)

    result = CliRunner().invoke(cli, ["show-winners", str(config_path)])

    assert result.exit_code == 0, result.output
    repeated = result.output.split("=== repeated ===\n", 1)[1].split("=== scalar ===", 1)[0]
    _assert_repeat_summary(contents["repeated"], repeated)
    _assert_member_directories(experiment, contents["repeated"])
    _assert_full_record(experiment, "repeated", contents["repeated"], repeated)
    scalar = result.output.split("=== scalar ===\n", 1)[1]
    scalar_full = yaml.safe_load(contents["scalar"])
    scalar_compact = yaml.safe_load(scalar)
    assert scalar_compact["trial_number"] == scalar_full["trial_number"]
    assert scalar_compact["metric"] == {"objective": scalar_full["metric"]["objective"]}
    assert scalar_compact["repeats"] is None
    _assert_full_record(experiment, "scalar", contents["scalar"], scalar)


def test_show_winners_uses_published_repeat_metadata_after_config_drift(tmp_path: Path) -> None:
    experiment, config_path, contents = _publish_winners(tmp_path, "median")
    changed = experiment.model_dump(mode="json")
    changed["phases"][0]["repeats"] = {"key": "fold", "values": [1, 2], "reduce": "mean"}
    changed["phases"][0]["n_trials"] = 2
    changed["metric"]["name"] = "changed_objective"
    config_path.write_text(yaml.safe_dump(changed), encoding="utf-8")

    result = CliRunner().invoke(cli, ["show-winners", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "# Historical experiment result:" in result.output
    repeated = result.output.split("=== repeated ===\n", 1)[1].split("=== scalar ===", 1)[0]
    _assert_repeat_summary(contents["repeated"], repeated)
    _assert_member_directories(experiment, contents["repeated"])
    _assert_full_record(experiment, "repeated", contents["repeated"], repeated)
    compact = yaml.safe_load(repeated)
    assert compact["repeats"]["reduce"] == "median"
    assert compact["repeats"]["n"] == 3
    assert "changed_objective" not in compact["metric"]
