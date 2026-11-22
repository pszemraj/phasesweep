"""Selection advisories: a winner no trial ran beyond, and a cheaper configuration within noise."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import optuna
import pytest
import yaml
from click.testing import CliRunner

from phasesweep import run_experiment
from phasesweep.cli import cli
from phasesweep.config.models import Metric, Phase, Repeats
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
from phasesweep.engine.read import read_result, read_trial_tables
from phasesweep.engine.repeats import ranked_group, repeat_axis
from phasesweep.engine.selection import (
    NoFeasibleTrialError,
    _cost_advisory,
    _rank_groups,
    select_winner,
)
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
    TRIAL_OUTCOME_ATTR,
)
from phasesweep.evidence.models import JsonEnvelopeExtractor, LogRegexExtractor
from phasesweep.mcp.snapshots import RESULT_SNAPSHOT_SCHEMA_VERSION, capture_result_snapshot
from phasesweep.runtime.json import JsonValue
from phasesweep.runtime.terminal import significant
from tests.conftest import (
    log_regex_provenance,
    make_experiment,
    phase_study,
    phase_trials,
    trial_outcome,
    write_experiment_config,
    write_trainer,
)
from tests.ledger_fixtures import published_generation_id

Row = tuple[dict[str, JsonValue], float | str]
"""One trial: its params, and its metric or the state ``fail``, ``infeasible`` or ``interrupted``."""


def _edge_warnings(
    caplog: pytest.LogCaptureFixture,
    search_space: dict[str, SearchParam],
    rows: list[Row],
    *,
    sampler_type: str = "random",
    repeats: Repeats | None = None,
) -> list[str]:
    """Select a winner over ``rows`` in phase ``p`` and return its untried-beyond warnings."""
    phase = Phase(
        name="p",
        n_trials=len(rows),
        sampler=Sampler.model_validate({"type": sampler_type, "seed": 0}),
        search_space=search_space,
        repeats=repeats,
        allow_partial_grid=sampler_type == "grid" and repeats is None,
    )
    experiment = make_experiment(phases=[phase])
    study = optuna.create_study(direction="minimize")
    for number, (params, outcome) in enumerate(rows):
        distributions = {
            name: optuna.distributions.CategoricalDistribution([value])
            for name, value in params.items()
        }
        attrs: dict[str, JsonValue] = {
            FEASIBLE_ATTR: outcome != "infeasible",
            GENERATION_ID_ATTR: "generation-a",
            ATTEMPT_ID_ATTR: f"attempt-{number}",
            OBJECTIVE_PROVENANCE_ATTR: json.dumps(log_regex_provenance()),
            TRAINER_ENV_DIGEST_ATTR: "a" * 64,
            TRAINER_INPUT_ATTR: {"schema_version": 1},
        }
        if isinstance(outcome, float) or outcome == "infeasible":
            value = outcome if isinstance(outcome, float) else 9.0
            study.add_trial(
                optuna.trial.create_trial(
                    value=value, params=params, distributions=distributions, user_attrs=attrs
                )
            )
            continue
        failure = {TRIAL_OUTCOME_ATTR: trial_outcome(number + 1, "interrupted")}
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.FAIL,
                params=params,
                distributions=distributions,
                user_attrs=failure if outcome == "interrupted" else {},
            )
        )
    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        select_winner(study, experiment, phase_name="p", repeats=repeat_axis(phase))
    return [record.getMessage() for record in caplog.records if "no trial ran" in record.message]


@pytest.mark.parametrize("sampler_type", ["grid", "random", "tpe"])
@pytest.mark.parametrize(
    ("param", "value", "warning"),
    [
        (
            IntParam(type="int", low=0, high=9, step=3),
            0,
            "below the winner's x=0 (tried 0..9, at the configured lower bound)",
        ),
        (
            IntParam(type="int", low=0, high=9, step=3),
            9,
            "above the winner's x=9 (tried 0..9, at the configured upper bound)",
        ),
        (
            FloatParam(type="float", low=0.1, high=0.3, step=0.1),
            0.3,
            "above the winner's x=0.3 (tried 0.1..0.3, at the configured upper bound)",
        ),
        (
            CategoricalParam(type="categorical", choices=[0.3, 0.1, 0.2]),
            0.1,
            "below the winner's x=0.1 (tried 0.1..0.3, at the configured lower bound)",
        ),
        (
            CategoricalParam(type="categorical", choices=[0.3, 0.1, 0.2]),
            0.3,
            "above the winner's x=0.3 (tried 0.1..0.3, at the configured upper bound)",
        ),
        (IntParam(type="int", low=0, high=2), 1, None),
        (CategoricalParam(type="categorical", choices=["small", "large", "huge"]), "small", None),
        (CategoricalParam(type="categorical", choices=[True, False]), True, None),
        (CategoricalParam(type="categorical", choices=[1, 2, "large"]), 1, None),
        (CategoricalParam(type="categorical", choices=[0.1]), 0.1, None),
        (CategoricalParam(type="categorical", choices=[8, 16]), 16, None),
        (IntParam(type="int", low=8, high=16, step=8), 16, None),
    ],
)
def test_edge_warning_uses_the_ordered_numeric_values_tried(
    caplog: pytest.LogCaptureFixture,
    param: SearchParam,
    value: JsonValue,
    warning: str | None,
    sampler_type: str,
) -> None:
    """Every sampler gets the one rule; only numeric axes with three values tried have edges."""
    values = grid_search_space({"x": param})["x"]
    rows: list[Row] = [({"x": item}, 0.1 if item == value else 0.5) for item in values]
    warnings = _edge_warnings(caplog, {"x": param}, rows, sampler_type=sampler_type)
    assert warnings == ([] if warning is None else [f"[p] no trial ran {warning}"])


@pytest.mark.parametrize("sampler_type", ["random", "tpe", "grid"])
def test_edge_warning_says_the_configured_range_extends_past_an_untried_edge(
    caplog: pytest.LogCaptureFixture, sampler_type: str
) -> None:
    """A random or TPE best at its highest draw, or a grid whose edge cell never ran, can be beaten inside the range."""
    param: SearchParam = (
        CategoricalParam(type="categorical", choices=[1, 2, 3, 4, 5])
        if sampler_type == "grid"
        else IntParam(type="int", low=1, high=5)
    )
    rows: list[Row] = [({"x": 1}, 0.9), ({"x": 2}, 0.7), ({"x": 4}, 0.5), ({"x": 3}, 0.6)]
    assert _edge_warnings(caplog, {"x": param}, rows, sampler_type=sampler_type) == [
        "[p] no trial ran above the winner's x=4 (tried 1..4, configured range extends to 5)"
    ]


def test_edge_warning_reads_log_float_ranges_and_names_every_parameter_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A continuous float reaches its bound only by equalling it; one warning covers both params."""
    space: dict[str, SearchParam] = {
        "lr": FloatParam(type="float", low=1e-4, high=1e-2, log=True),
        "width": IntParam(type="int", low=64, high=512, step=64),
        "depth": IntParam(type="int", low=1, high=64, log=True),
    }
    rows: list[Row] = [
        ({"lr": 0.0002, "width": 512, "depth": 8}, 0.9),
        ({"lr": 0.001, "width": 256, "depth": 2}, 0.8),
        ({"lr": 0.0081, "width": 64, "depth": 4}, 0.1),
    ]
    assert _edge_warnings(caplog, space, rows) == [
        "[p] no trial ran above the winner's lr=0.0081 (tried 0.0002..0.0081, configured range "
        "extends to 0.01) or below the winner's width=64 (tried 64..512, at the configured lower "
        "bound)"
    ]


@pytest.mark.parametrize(
    ("beyond", "silenced"),
    [("fail", True), ("infeasible", True), ("interrupted", False)],
)
def test_a_finished_trial_beyond_the_winner_counts_as_tried(
    caplog: pytest.LogCaptureFixture, beyond: str, silenced: bool
) -> None:
    """A diverged run above the best value is a worse value on that side; an interrupted one tried nothing."""
    param = FloatParam(type="float", low=1e-4, high=1e-1, log=True)
    rows: list[Row] = [
        ({"lr": 0.001}, 0.9),
        ({"lr": 0.002}, 0.7),
        ({"lr": 0.004}, 0.5),
        ({"lr": 0.008}, beyond),
    ]
    warnings = _edge_warnings(caplog, {"lr": param}, rows)
    assert warnings == (
        []
        if silenced
        else [
            "[p] no trial ran above the winner's lr=0.004 (tried 0.001..0.004, configured range "
            "extends to 0.1)"
        ]
    )


def test_fewer_than_three_distinct_values_tried_have_no_edge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    param = IntParam(type="int", low=0, high=10)
    rows: list[Row] = [({"x": 1}, 0.9), ({"x": 2}, 0.1), ({"x": 2}, 0.3), ({"x": 1}, 0.8)]
    assert _edge_warnings(caplog, {"x": param}, rows) == []


def test_repeats_edge_warning_uses_the_winning_group_and_skips_the_repeat_axis(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The winning configuration is a group; its seeds are no searched axis."""
    param = CategoricalParam(type="categorical", choices=[1, 2, 3])
    rows: list[Row] = [
        ({"x": x, "seed": seed}, 0.1 * x + 0.01 * seed) for x in (1, 2, 3) for seed in (0, 1, 2)
    ]
    assert _edge_warnings(
        caplog,
        {"x": param},
        rows,
        sampler_type="grid",
        repeats=Repeats(key="seed", values=[0, 1, 2]),
    ) == ["[p] no trial ran below the winner's x=1 (tried 1..3, at the configured lower bound)"]


@pytest.mark.integration
def test_edge_advisory_is_recomputed_at_read_over_the_trials_selection_saw(
    tmp_path: Path,
) -> None:
    """show-winners and the MCP snapshot carry the edge warning, which nothing stores.

    It is judged over the finished trials the phase held at selection, so a trial a later run
    adds below the winner does not enter it, and a config that no longer matches the producing
    one leaves it unavailable.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--x', type=int)
        args, _ = parser.parse_known_args()
        print(f'x={args.x}')
    """,
    )
    phase = Phase(
        name="p",
        n_trials=3,
        gpu_policy="none",
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[2, 3, 4])},
    )
    experiment = make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase])
    winner = run_experiment(experiment)["p"]
    expected = "no trial ran below the winner's x=2 (tried 2..4, at the configured lower bound)"
    path = _generation_winner_path(experiment, winner.generation_id, "p")
    assert "no trial ran" not in path.read_text(encoding="utf-8")

    _, views = read_result(experiment)
    assert (views[0].edge_advisory, views[0].edge_advisory_status) == (expected, "available")
    generation = published_generation_id(experiment)
    engine = capture_result_snapshot(
        experiment, generation_id=generation, engine_winners={"p": winner}
    )
    read = capture_result_snapshot(experiment)
    for snapshot in (engine, read):
        assert snapshot["winners"][0]["edge_advisory"] == expected
        assert snapshot["winners"][0]["edge_advisory_status"] == "available"
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    shown = CliRunner().invoke(cli, ["show-winners", str(config)])
    assert shown.exit_code == 0, shown.exception
    payload = yaml.safe_load(shown.output.split("=== p ===", 1)[1])
    assert (payload["edge_advisory"], payload["edge_advisory_status"]) == (expected, "available")

    phase_study(experiment).add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.FAIL,
            params={"x": 1},
            distributions={"x": optuna.distributions.CategoricalDistribution([1])},
        )
    )
    _, later = read_result(experiment)
    assert (later[0].edge_advisory, later[0].edge_advisory_status) == (expected, "available")

    drifted = experiment.model_copy(
        update={"metric": experiment.metric.model_copy(update={"goal": "maximize"})}
    )
    _, historical = read_result(drifted)
    assert historical[0].config_matches_current is False
    assert (historical[0].edge_advisory, historical[0].edge_advisory_status) == (
        None,
        "unavailable",
    )


@pytest.mark.parametrize(
    ("gap", "half_range", "standard_error"),
    [(0.005, 0.02, "0.02"), (0.025, 0.02, None), (0.05, 0.1, "0.072111")],
)
def test_cost_advisory_uses_the_pooled_standard_error_and_keeps_the_winner(
    gap: float, half_range: float, standard_error: str | None
) -> None:
    """The window is the phase's pooled sd over the root of the winner's group size.

    Both groups' spread sets it, as it sets the tie band: variances 0.0008 and 0.0008 pool to
    sd 0.0282843 and a window of 0.02, while a candidate with variance 0.02 widens the pool to
    sd 0.10198 and the window to 0.072111.
    """
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
    assert winner.params == {"epochs": 16}
    assert winner.aggregate == 0.5
    if standard_error is None:
        assert advisory is None
    else:
        assert advisory == (
            "cost advisory: epochs=8 is cheaper than the winner's epochs=16, with a mean within "
            f"one standard error of the winner's (gap {gap:g}, standard error {standard_error}); "
            "the selected winner is unchanged"
        )


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
    assert advisory.startswith("cost advisory: epochs=8 lr=0.2 is cheaper than")
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
    # Two groups of two seeds pool to df 2, whose two-sided 5% t value is 4.302653.
    assert comparisons[0]["threshold"] == pytest.approx(4.302653 * comparisons[0]["combined_se"])
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
    shown = yaml.safe_load(displayed.output.split("=== epochs ===", 1)[1])
    assert shown["cost_advisory"].startswith(
        "cost advisory: epochs=8 is cheaper than the winner's epochs=16"
    )
    assert "tie_groups:" in displayed.output
    assert "cost_advisory_status: available" in displayed.output
    replayed = run_experiment(changed)["epochs"]
    assert replayed.params == winner.params and replayed.metric == winner.metric
    snapshot = capture_result_snapshot(
        changed, generation_id=generation, engine_winners={"epochs": replayed}
    )
    assert "epochs=8" in snapshot["winners"][0]["cost_advisory"]
    assert snapshot["schema_version"] == RESULT_SNAPSHOT_SCHEMA_VERSION
    assert snapshot["winners"][0]["tie_groups"][0]["params"] == {"epochs": 8}
    assert snapshot["winners"][0]["cost_advisory_status"] == "available"
    assert path.read_bytes() == saved


@pytest.mark.integration
def test_repeat_advisories_keep_the_selected_population_after_failed_top_up(tmp_path: Path) -> None:
    """A new cheap group in an aborted generation cannot enter a published comparison."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse
        import sys
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument('--width', type=int)
        parser.add_argument('--seed', type=int)
        args, _ = parser.parse_known_args()
        counter = Path(__file__).with_suffix('.count')
        if counter.exists():
            count = int(counter.read_text())
            counter.write_text(str(count + 1))
            if count >= 3:
                sys.exit('injected failure')
        values = {
            1: (0.515, 0.525, 0.535), 3: (0.50, 0.52, 0.54),
            5: (0.53, 0.55, 0.57),
        }
        print(f'x={values.get(args.width, (0.60, 0.62, 0.64))[args.seed-1]}')
        """,
    )
    phase = Phase(
        name="p1",
        n_trials=6,
        gpu_policy="none",
        repeats=Repeats(key="seed", values=[1, 2, 3]),
        sampler=Sampler(type="random", seed=0),
        search_space={"width": CategoricalParam(type="categorical", choices=[1, 2, 3, 4, 5, 6])},
        cost_key="width",
    )
    original = make_experiment(
        experiment="x8", persistent=tmp_path, trainer=trainer, phases=[phase]
    )
    run_experiment(original)
    generation = published_generation_id(original)
    assert generation is not None
    _, before = read_result(original)
    assert before[0].params == {"width": 3}
    assert before[0].cost_advisory_status == "no_cheaper_configuration"
    trainer.with_suffix(".count").write_text("0", encoding="utf-8")
    extended = original.model_copy(
        update={
            "phases": [phase.model_copy(update={"n_trials": 30, "max_consecutive_failures": 2})]
        }
    )
    with pytest.raises(NoFeasibleTrialError, match="aborted"):
        run_experiment(extended)
    trials = phase_trials(extended, "p1")
    assert [
        trial.params["width"]
        for trial in trials[6:]
        if trial.state == optuna.trial.TrialState.COMPLETE
    ] == [1, 1, 1]
    axis = repeat_axis(phase)
    assert axis is not None
    live_groups = _rank_groups(trials, extended, axis).groups
    represented = next(group for group in live_groups if dict(group.params) == {"width": 3})
    live_advice = _cost_advisory(phase, represented, live_groups, goal="minimize")
    assert live_advice is not None and "width=1" in live_advice
    _, historical = read_result(extended)
    assert historical[0].cost_advisory is None
    assert historical[0].cost_advisory_status == "no_cheaper_configuration"
    assert historical[0].tie_groups == before[0].tie_groups
    snapshot = capture_result_snapshot(extended, generation_id=generation)
    assert snapshot["winners"][0]["cost_advisory"] is None
    assert snapshot["winners"][0]["tie_groups"] == before[0].tie_groups


@pytest.mark.integration
@pytest.mark.parametrize(
    ("value", "displayed"),
    [
        (4938268.0, 4938268.0),
        (1.2345678901234567, 1.23456789012),
        (0.10000000012345678, 0.100000000123),
        (1.23456789e-9, 1.23456789e-9),
        (0.39999999999999997, 0.4),
        (0.0299999999999999, 0.03),
    ],
)
def test_float_displays_share_precision_without_changing_recorded_values(
    tmp_path: Path, value: float, displayed: float
) -> None:
    """Winner YAML keeps twelve digits and the trial table shortens the objective, params and durations; stored values stay exact."""
    trainer = write_trainer(tmp_path / "trainer.py", f"print('x={value!r}')")
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        n_trials=1,
        gpu_policy="none",
        metric=Metric(
            name="x",
            goal="minimize",
            extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>\S+)"),
        ),
        trainer_config={"precise": value},
        search_space={"lr": FloatParam(type="float", low=0.0001, high=0.001)},
    )
    winner = run_experiment(experiment)["p"]
    sampled = phase_trials(experiment)[0].params["lr"]
    assert f"{sampled:.4g}" != f"{sampled:.12g}"
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    result = CliRunner().invoke(cli, ["show-winners", str(config)])
    assert result.exit_code == 0, result.exception
    base_text, phase_text = result.output.split("=== p ===", 1)
    payload = yaml.safe_load(phase_text)
    assert payload["metric"]["x"] == displayed
    assert payload["params"]["lr"] == float(f"{sampled:.12g}")
    assert yaml.safe_load(base_text)["trainer_config"]["precise"] == displayed
    assert winner.metric == value
    assert capture_result_snapshot(experiment)["winners"][0]["metric"] == value
    table = CliRunner().invoke(cli, ["status", str(config), "--trials"])
    assert table.exit_code == 0, table.output
    cells = next(line for line in table.output.splitlines() if "COMPLETE" in line).split()
    assert cells[-4:-1] == [significant(value, 6), f"{sampled:.4g}", "COMPLETE"]
    assert f"{sampled:.12g}" not in table.output
    duration = read_trial_tables(experiment, latest=True).phases[0].rows[0]["duration_s"]
    assert isinstance(duration, float) and f"{duration:.12g}" != f"{duration:.1f}"
    assert cells[-1] == f"{duration:.1f}s"
    assert f"{duration:.12g}" not in table.output


@pytest.mark.integration
def test_overlap_warning_keeps_threshold_with_groups_and_leaves_the_remedy_to_show_winners(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The pooled threshold stays beside the group's gap, and the evaluation guidance is in show-winners.

    Both groups have sd 0.02 over three seeds, so the band is sd 0.02 on 4 df and the threshold
    2.776445 * 0.02 * sqrt(2/3) = 0.0453392 holds the 0.03 gap.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import argparse
        from phasesweep.reporting import report_objective
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--seed', type=int)
        args, _ = parser.parse_known_args()
        value = (0.50, 0.52, 0.54)[args.seed-1] + (0.03 if args.epochs == 8 else 0)
        report_objective(value, name='x', split='validation', policy='synthetic',
                         checkpoint='test', step=args.epochs)
        """,
    )
    phase = Phase(
        name="epochs",
        n_trials=6,
        gpu_policy="none",
        repeats=Repeats(key="seed", values=[1, 2, 3]),
        sampler=Sampler(type="grid", seed=0),
        search_space={"epochs": CategoricalParam(type="categorical", choices=[8, 16])},
        cost_key="epochs",
    )
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        phases=[phase],
        metric=Metric(
            name="x",
            goal="minimize",
            extractor=JsonEnvelopeExtractor(
                type="json_envelope",
                path="r.json",
                objective_name="x",
                split="validation",
                policy="synthetic",
                checkpoint="test",
            ),
        ),
    )
    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        run_experiment(experiment)
    warning = next(
        record.getMessage() for record in caplog.records if "tie: " in record.getMessage()
    )
    assert re.search(
        r"tie: 1 group is within noise of the winner, threshold 0\.0453392 \(sd 0\.02 pooled "
        r"over 2 groups, df 4\): group \d+ at gap 0\.03$",
        warning,
    )
    assert "report progress=" not in warning
    _, views = read_result(experiment)
    assert views[0].cost_advisory_status == "no_cheaper_configuration"
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    result = CliRunner().invoke(cli, ["show-winners", str(config)])
    assert result.exit_code == 0, result.exception
    payload = yaml.safe_load(result.output.split("=== epochs ===", 1)[1])
    assert payload["repeats"]["sd"] == 0.02
    assert payload["tie_groups"][0]["gap"] == 0.03
    assert payload["confound"]["evaluation_point"]["remedy"].startswith(
        "For a phase that searches training length, report progress="
    )


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
