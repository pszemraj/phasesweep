"""Advisory warnings when a numeric grid selects an endpoint."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import optuna
import pytest
from click.testing import CliRunner

from phasesweep import run_experiment
from phasesweep.cli import cli
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
from phasesweep.engine.paths import _generation_winner_path
from phasesweep.engine.read import read_result
from phasesweep.engine.repeats import ranked_group
from phasesweep.engine.selection import _cost_advisory, select_winner
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
)
from phasesweep.mcp.snapshots import capture_result_snapshot
from phasesweep.runtime.json import JsonValue
from tests.conftest import (
    log_regex_provenance,
    make_experiment,
    write_experiment_config,
    write_trainer,
)
from tests.ledger_fixtures import published_generation_id


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


@pytest.mark.parametrize(
    ("gap", "half_range", "advised"),
    [(0.005, 0.02, True), (0.025, 0.02, False), (0.05, 0.1, False)],
)
def test_cost_advisory_uses_winner_error_and_keeps_the_winner(
    gap: float, half_range: float, advised: bool
) -> None:
    """Candidate noise cannot widen the winner's one-standard-error window."""
    phase = Phase(
        name="epochs",
        n_trials=4,
        repeats=Repeats(key="seed", values=[1, 2]),
        cost_key="epochs",
        sampler=Sampler(type="grid"),
        search_space={"epochs": CategoricalParam(type="categorical", choices=[8, 16])},
    )
    winner = ranked_group([(0, 0.48), (1, 0.52)], {"epochs": 16}, "mean")
    cheaper = ranked_group(
        [(2, 0.5 - half_range + gap), (3, 0.5 + half_range + gap)], {"epochs": 8}, "mean"
    )
    advisory = _cost_advisory(phase, winner, [winner, cheaper], goal="minimize")
    assert (advisory is not None) == advised
    assert winner.params == {"epochs": 16}
    assert winner.aggregate == 0.5
    if advised:
        assert advisory is not None
        assert "epochs=8" in advisory
        assert "winner_se=0.02" in advisory
        assert "selected winner is unchanged" in advisory


@pytest.mark.parametrize("maximize", [False, True])
def test_cost_advisory_breaks_cost_ties_by_objective_then_trial_order(maximize: bool) -> None:
    """Equally cheap alternatives prefer better objectives before their trial numbers."""
    phase = Phase(
        name="epochs",
        n_trials=8,
        repeats=Repeats(key="seed", values=[1, 2]),
        cost_key="epochs",
        sampler=Sampler(type="grid"),
        search_space={
            "epochs": CategoricalParam(type="categorical", choices=[8, 16]),
            "lr": CategoricalParam(type="categorical", choices=[0.1, 0.2, 0.3]),
        },
    )
    winner = ranked_group([(0, 0.48), (1, 0.52)], {"epochs": 16, "lr": 0.1}, "mean")
    sign = -1 if maximize else 1
    groups = [winner]
    for number, lr, gap in [(2, 0.1, 0.015), (4, 0.2, 0.005), (6, 0.3, 0.005)]:
        groups.append(
            ranked_group(
                [(number, 0.48 + sign * gap), (number + 1, 0.52 + sign * gap)],
                {"epochs": 8, "lr": lr},
                "mean",
            )
        )
    advisory = _cost_advisory(phase, winner, groups, goal="maximize" if maximize else "minimize")
    assert advisory is not None
    assert "'lr': 0.2" in advisory
    assert (
        _cost_advisory(phase.model_copy(update={"cost_key": None}), winner, groups, goal="minimize")
        is None
    )


@pytest.mark.integration
@pytest.mark.parametrize("configured", [False, True])
def test_cost_key_added_after_publication_is_visible_on_reads_and_replay(
    tmp_path: Path, configured: bool
) -> None:
    """Read-time advice uses retained groups without replacing the scientific winner."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--seed', type=int)
        args, _ = parser.parse_known_args()
        value = (0.48 if args.seed == 1 else 0.52) + (0.005 if args.epochs == 8 else 0)
        print(f'x={value}')
    """,
    )
    phase = Phase(
        name="epochs",
        n_trials=4,
        gpu_policy="none",
        repeats=Repeats(key="seed", values=[1, 2]),
        sampler=Sampler(type="grid", seed=0),
        search_space={"epochs": CategoricalParam(type="categorical", choices=[8, 16])},
        cost_key="epochs" if configured else None,
    )
    original = make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase])
    winner = run_experiment(original)["epochs"]
    generation = published_generation_id(original)
    assert generation is not None
    path = _generation_winner_path(original, generation, "epochs")
    saved = path.read_bytes()
    _, initial_views = read_result(original)
    assert initial_views[0].cost_advisory_status == (
        "available" if configured else "not_configured"
    )
    changed = original.model_copy(
        update={"phases": [phase.model_copy(update={"cost_key": "epochs"})]}
    )
    _, views = read_result(changed)
    assert views[0].cost_advisory is not None
    assert "epochs=8" in views[0].cost_advisory
    assert views[0].cost_advisory_status == "available"
    comparisons = views[0].tie_groups
    assert comparisons is not None and len(comparisons) == 1
    assert comparisons[0]["params"] == {"epochs": 8}
    assert comparisons[0]["gap"] == pytest.approx(0.005)
    assert comparisons[0]["threshold"] == 2 * comparisons[0]["combined_se"]
    drifted = changed.model_copy(
        update={"metric": changed.metric.model_copy(update={"goal": "maximize"})}
    )
    _, historical = read_result(drifted)
    assert historical[0].config_matches_current is False
    assert historical[0].cost_advisory is None
    assert historical[0].cost_advisory_status == "unavailable"
    removed_key = drifted.model_copy(
        update={"phases": [phase.model_copy(update={"cost_key": None})]}
    )
    _, removed_views = read_result(removed_key)
    assert removed_views[0].cost_advisory_status == (
        "unavailable" if configured else "not_configured"
    )
    config_path = tmp_path / "experiment.yaml"
    write_experiment_config(config_path, changed)
    displayed = CliRunner().invoke(cli, ["show-winners", str(config_path)])
    assert displayed.exit_code == 0, displayed.exception
    assert "one-standard-error cost advisory" in displayed.output
    assert "tie_groups:" in displayed.output
    assert "cost_advisory_status: available" in displayed.output
    replayed = run_experiment(changed)["epochs"]
    assert replayed.params == winner.params and replayed.metric == winner.metric
    snapshot = capture_result_snapshot(
        changed, generation_id=generation, engine_winners={"epochs": replayed}
    )
    assert "epochs=8" in snapshot["winners"][0]["cost_advisory"]
    assert snapshot["schema_version"] == 1
    assert snapshot["winners"][0]["tie_groups"][0]["params"] == {"epochs": 8}
    assert snapshot["winners"][0]["cost_advisory_status"] == "available"
    assert path.read_bytes() == saved


@pytest.mark.integration
def test_repeat_overlap_details_keep_the_selected_population_after_top_up(tmp_path: Path) -> None:
    """An older winner compares only the repeat groups it ranked before a top-up."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int)
        args, _ = parser.parse_known_args()
        print(f'x={0.39 if args.seed == 1 else 0.41}')
        """,
    )
    phase = Phase(
        name="epochs",
        n_trials=4,
        gpu_policy="none",
        repeats=Repeats(key="seed", values=[1, 2]),
        sampler=Sampler(type="random", seed=0),
        search_space={"epochs": FloatParam(type="float", low=8, high=32)},
    )
    original = make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase])
    run_experiment(original)
    generation = published_generation_id(original)
    assert generation is not None
    extended = original.model_copy(update={"phases": [phase.model_copy(update={"n_trials": 6})]})
    run_experiment(extended)
    _, historical = read_result(extended, generation_id=generation)
    _, current = read_result(extended)
    assert historical[0].tie_groups is not None and len(historical[0].tie_groups) == 1
    assert current[0].tie_groups is not None and len(current[0].tie_groups) == 2


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
