"""Advisory warnings when a numeric grid selects an endpoint."""

from __future__ import annotations

import json
import logging

import optuna
import pytest

from phasesweep.config.models import Phase, Repeats
from phasesweep.config.search import (
    CategoricalParam,
    FloatParam,
    IntParam,
    Sampler,
    SearchParam,
    grid_search_space,
    sampler_warnings,
)
from phasesweep.engine.repeats import ranked_group
from phasesweep.engine.selection import _warn_cheaper_within_error, select_winner
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
        (CategoricalParam(type="categorical", choices=[8, 16]), 16, None),
        (IntParam(type="int", low=8, high=16, step=8), 16, None),
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


@pytest.mark.parametrize(("gap", "advised"), [(0.005, True), (0.05, False)])
def test_cost_advisory_uses_one_combined_error_and_keeps_the_winner(
    caplog: pytest.LogCaptureFixture, gap: float, advised: bool
) -> None:
    """A cheaper group between one and two errors gets no one-error recommendation."""
    phase = Phase(
        name="epochs",
        n_trials=4,
        repeats=Repeats(key="seed", values=[1, 2]),
        cost_key="epochs",
        sampler=Sampler(type="grid"),
        search_space={"epochs": CategoricalParam(type="categorical", choices=[8, 16])},
    )
    winner = ranked_group([(0, 0.48), (1, 0.52)], {"epochs": 16}, "mean")
    cheaper = ranked_group([(2, 0.48 + gap), (3, 0.52 + gap)], {"epochs": 8}, "mean")
    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        _warn_cheaper_within_error(phase, winner, [winner, cheaper])
    warnings = [record.message for record in caplog.records if "cost advisory" in record.message]
    assert bool(warnings) == advised
    assert winner.params == {"epochs": 16}
    assert winner.aggregate == 0.5
    if advised:
        assert "epochs=8" in warnings[0]
        assert "selected winner is unchanged" in warnings[0]


@pytest.mark.parametrize("sampler", ["grid", "random"])
@pytest.mark.parametrize("sampled", [False, True])
def test_one_configuration_repeat_phase_has_no_failure_streak_advice(
    sampler: str, sampled: bool
) -> None:
    """An unrankable single group cannot benefit from more failed repeat attempts."""
    phase = Phase(
        name="audit",
        n_trials=3,
        max_consecutive_failures=2,
        repeats=Repeats(key="seed", values=[1, 2, 3]),
        sampler=Sampler(type=sampler, seed=0),
        search_space={"epochs": CategoricalParam(type="categorical", choices=[16])}
        if sampled
        else {},
    )
    assert not any("max_consecutive_failures" in warning for warning in sampler_warnings(phase))
