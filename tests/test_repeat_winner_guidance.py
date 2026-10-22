"""Explain repeated winners beside the CLI's unchanged publication YAML."""

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


def _assert_member_directories(experiment: Experiment, saved: bytes, guidance: str) -> None:
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
        assert (
            f"# seed={member['repeat']!r} trial_number={member['trial_number']} "
            f"trial_dir={directory}"
        ) in guidance.splitlines()


@pytest.mark.parametrize("reducer", ["mean", "median"])
def test_show_winners_explains_repeat_metric_and_checkpoint_choice(
    tmp_path: Path, reducer: ReduceName
) -> None:
    experiment, config_path, contents = _publish_winners(tmp_path, reducer)

    result = CliRunner().invoke(cli, ["show-winners", str(config_path)])

    assert result.exit_code == 0, result.output
    repeated = result.output.split("=== repeated ===\n", 1)[1].split("=== scalar ===", 1)[0]
    saved = contents["repeated"].decode("utf-8")
    guidance, rendered = repeated.split(saved, 1)
    assert f"# metric is the {reducer} over 3 values of repeat key 'seed'." in guidance
    assert "# trial_number 2 names the group's last member." in guidance
    assert "# Choose a checkpoint explicitly from repeats.members." in guidance
    _assert_member_directories(experiment, contents["repeated"], guidance)
    assert all(line.startswith("# ") for line in guidance.splitlines())
    assert rendered == "\n"
    assert yaml.safe_load(repeated) == yaml.safe_load(saved)
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    assert (
        _generation_winner_path(experiment, generation_id, "repeated").read_bytes()
        == contents["repeated"]
    )


def test_show_winners_leaves_scalar_winner_unannotated(tmp_path: Path) -> None:
    _, config_path, contents = _publish_winners(tmp_path)

    result = CliRunner().invoke(cli, ["show-winners", str(config_path)])

    assert result.exit_code == 0, result.output
    scalar = result.output.split("=== scalar ===\n", 1)[1]
    assert scalar == contents["scalar"].decode("utf-8") + "\n"


def test_show_winners_uses_published_repeat_metadata_after_config_drift(tmp_path: Path) -> None:
    experiment, config_path, contents = _publish_winners(tmp_path, "median")
    changed = experiment.model_dump(mode="json")
    changed["phases"][0]["repeats"] = {"key": "fold", "values": [1, 2], "reduce": "mean"}
    changed["phases"][0]["n_trials"] = 2
    config_path.write_text(yaml.safe_dump(changed), encoding="utf-8")

    result = CliRunner().invoke(cli, ["show-winners", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "# Historical experiment result:" in result.output
    repeated = result.output.split("=== repeated ===\n", 1)[1].split("=== scalar ===", 1)[0]
    guidance = repeated.split(contents["repeated"].decode("utf-8"), 1)[0]
    assert "# metric is the median over 3 values of repeat key 'seed'." in guidance
    assert "# trial_number 2 names the group's last member." in guidance
    assert "# Choose a checkpoint explicitly from repeats.members." in guidance
    _assert_member_directories(experiment, contents["repeated"], guidance)
    assert "fold" not in guidance
    assert "mean" not in guidance
