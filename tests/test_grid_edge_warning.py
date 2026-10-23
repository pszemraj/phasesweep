"""Advisory warnings when a numeric grid selects an endpoint."""

from __future__ import annotations

import json
import logging

import optuna
import pytest

from phasesweep.config.models import Phase
from phasesweep.config.search import (
    CategoricalParam,
    FloatParam,
    IntParam,
    Sampler,
    SearchParam,
    grid_search_space,
)
from phasesweep.engine.selection import select_winner
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
)
from phasesweep.runtime.json import JsonValue
from tests.conftest import log_regex_provenance, make_experiment


@pytest.mark.parametrize("sampler_type", ["grid", "random"])
@pytest.mark.parametrize(
    ("param", "value", "edge"),
    [
        (IntParam(type="int", low=0, high=9, step=3), 0, "lower"),
        (IntParam(type="int", low=0, high=9, step=3), 9, "upper"),
        (FloatParam(type="float", low=0.1, high=0.3, step=0.1), 0.3, "upper"),
        (CategoricalParam(type="categorical", choices=[0.3, 0.1, 0.2]), 0.1, "lower"),
        (CategoricalParam(type="categorical", choices=[0.3, 0.1, 0.2]), 0.3, "upper"),
        (IntParam(type="int", low=0, high=2), 1, None),
        (CategoricalParam(type="categorical", choices=["small", "large"]), "small", None),
        (CategoricalParam(type="categorical", choices=[True, False]), True, None),
        (CategoricalParam(type="categorical", choices=[1, "large"]), 1, None),
        (CategoricalParam(type="categorical", choices=[0.1]), 0.1, None),
    ],
)
def test_grid_endpoint_warning_uses_the_concrete_numeric_axis(
    caplog: pytest.LogCaptureFixture,
    param: SearchParam,
    value: JsonValue,
    edge: str | None,
    sampler_type: str,
) -> None:
    """Only ordered numeric grid axes have edges, including stepped axes."""
    grid = grid_search_space({"x": param})
    phase = Phase(
        name="p",
        n_trials=len(grid["x"]),
        sampler=Sampler.model_validate({"type": sampler_type, "seed": 0}),
        search_space={"x": param},
    )
    experiment = make_experiment(phases=[phase])
    study = optuna.create_study(direction="minimize")
    study.add_trial(
        optuna.trial.create_trial(
            value=0.5,
            params={"x": value},
            distributions={"x": optuna.distributions.CategoricalDistribution(grid["x"])},
            user_attrs={
                FEASIBLE_ATTR: True,
                GENERATION_ID_ATTR: "generation-a",
                ATTEMPT_ID_ATTR: "attempt-a",
                OBJECTIVE_PROVENANCE_ATTR: json.dumps(log_regex_provenance()),
                TRAINER_ENV_DIGEST_ATTR: "a" * 64,
                TRAINER_INPUT_ATTR: {"schema_version": 1},
            },
        )
    )

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        selected = select_winner(study, experiment, phase_name="p")

    warnings = [
        record.getMessage() for record in caplog.records if "grid boundary" in record.message
    ]
    assert selected.trial_number == 0 and selected.metric == 0.5
    if edge is None or sampler_type != "grid":
        assert warnings == []
    else:
        assert len(warnings) == 1
        assert "[p]" in warnings[0] and "x" in warnings[0] and edge in warnings[0]
        assert str(value) in warnings[0] and "expand" in warnings[0].lower()
