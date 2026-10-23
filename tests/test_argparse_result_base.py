"""Argparse results expose the producing trainer base independently of phase overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from phasesweep import run_experiment
from phasesweep.cli import cli
from phasesweep.config.models import Phase
from phasesweep.engine.paths import _generation_summary_path
from phasesweep.runtime.json import JsonObject
from tests.conftest import make_experiment, write_constant_trainer
from tests.ledger_fixtures import published_generation_id


@pytest.mark.parametrize("surface", ["show-winners", "show-winners-format-edit", "summary"])
def test_argparse_result_reports_producing_base_after_current_config_edit(
    tmp_path: Path, surface: str
) -> None:
    """Historical output keeps the recorded base while phase inheritance keeps only overrides."""
    recorded_base: JsonObject = {"model": {"name": "recorded-base"}, "epochs": 3}
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_constant_trainer(tmp_path),
        trainer_config=recorded_base,
        phases=[Phase(name="p", n_trials=1, fixed_overrides={"epochs": 4})],
    )
    winners = run_experiment(experiment)
    assert winners["p"].effective_overrides == {"epochs": 4}
    config = experiment.model_dump(mode="json")
    config["trainer_config"] = {"model": {"name": "edited-base"}, "epochs": 99}
    if surface == "show-winners-format-edit":
        config["override_format"] = "hydra"
    config_path = tmp_path / "edited.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    if surface.startswith("show-winners"):
        result = CliRunner().invoke(cli, ["show-winners", str(config_path)])
        assert result.exit_code == 0, result.output
        producing = yaml.safe_load(result.output.split("=== p ===", 1)[0])
        assert producing["trainer_config"] == recorded_base
        assert "edited-base" not in result.output
    else:
        summary = yaml.safe_load(
            _generation_summary_path(experiment, published_generation_id(experiment)).read_text(
                encoding="utf-8"
            )
        )
        assert summary["trainer_config"] == recorded_base
        assert summary["phases"][0]["effective_overrides"] == {"epochs": 4}
