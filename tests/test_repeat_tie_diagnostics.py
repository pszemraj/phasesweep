"""Repeat-group tie warnings show the measurements behind their noise comparison."""

from __future__ import annotations

import json
import logging

import optuna
import pytest

from phasesweep.engine.repeats import RepeatAxis
from phasesweep.engine.selection import select_winner
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
)
from tests.conftest import log_regex_provenance, make_experiment


@pytest.mark.parametrize("other_x", [1, 2])
def test_repeat_tie_warning_names_params_gap_and_combined_standard_error(
    caplog: pytest.LogCaptureFixture,
    other_x: int,
) -> None:
    """A near tie gives an interpretable comparison while distant groups stay out."""
    study = optuna.create_study(direction="minimize")
    for x, metrics in [(1, [0.5, 0.6]), (other_x, [0.52, 0.62]), (3, [0.9, 0.91])]:
        for seed, value in enumerate(metrics):
            number = len(study.trials)
            study.add_trial(
                optuna.trial.create_trial(
                    value=value,
                    params={"x": x, "seed": seed},
                    distributions={
                        "x": optuna.distributions.IntDistribution(0, 10),
                        "seed": optuna.distributions.IntDistribution(0, 1),
                    },
                    user_attrs={
                        FEASIBLE_ATTR: True,
                        GENERATION_ID_ATTR: "generation-a",
                        ATTEMPT_ID_ATTR: f"attempt-{number}",
                        OBJECTIVE_PROVENANCE_ATTR: json.dumps(log_regex_provenance()),
                        TRAINER_ENV_DIGEST_ATTR: "a" * 64,
                        TRAINER_INPUT_ATTR: {"schema_version": 1},
                    },
                )
            )

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        winner = select_winner(
            study,
            make_experiment(n_trials=6),
            phase_name="p",
            repeats=RepeatAxis("seed", (0, 1), "mean", 2),
        )

    messages = [record.getMessage() for record in caplog.records if "tie:" in record.message]
    assert winner.params == {"x": 1}
    if other_x == 1:
        assert messages == []
        return
    (message,) = messages
    assert "gap=0.02" in message and "combined_se=0.0707107" in message
    assert "'x': 1" in message and f"'x': {other_x}" in message and "'x': 3" not in message
