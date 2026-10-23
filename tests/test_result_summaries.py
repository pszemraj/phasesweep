"""Read-only statistics for every repeat configuration, including incomplete ones."""

from __future__ import annotations

import statistics
from pathlib import Path

import optuna
import pytest

from phasesweep.cli import _render_trial_tables
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import IntParam, Repeats, Sampler
from phasesweep.engine.read import read_trial_tables
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GENERATION_ID_ATTR,
    RERUN_OF_ATTR,
    TRIAL_OUTCOME_ATTR,
)
from phasesweep.runtime.json import JsonObject
from tests.conftest import make_experiment, trial_outcome
from tests.recovery_helpers import new_phase_study


def _experiment(tmp_path: Path, *, repeats: Repeats, n_trials: int) -> Experiment:
    """Build a persistent random phase whose configuration slots can repeat."""
    return make_experiment(
        persistent=tmp_path,
        phases=[
            Phase(
                name="p",
                n_trials=n_trials,
                sampler=Sampler(type="random", seed=0),
                search_space={"x": IntParam(type="int", low=0, high=10)},
                repeats=repeats,
            )
        ],
    )


def _add_trial(
    study: optuna.Study,
    *,
    x: int,
    seed: int,
    value: float | None,
    feasible: bool = True,
    attrs: JsonObject | None = None,
) -> None:
    """Record one measured or failed member without launching a trainer."""
    number = len(study.trials)
    study.add_trial(
        optuna.trial.create_trial(
            params={"x": x, "seed": seed},
            distributions={
                "x": optuna.distributions.IntDistribution(0, 10),
                "seed": optuna.distributions.IntDistribution(0, 2),
            },
            value=value,
            state=(
                optuna.trial.TrialState.COMPLETE
                if value is not None
                else optuna.trial.TrialState.FAIL
            ),
            user_attrs={
                FEASIBLE_ATTR: feasible,
                GENERATION_ID_ATTR: "generation-a",
                ATTEMPT_ID_ATTR: f"attempt-{number}",
                **(attrs or {}),
            },
        )
    )


@pytest.mark.parametrize("reduce", ["mean", "median"])
def test_repeat_summary_includes_losing_infeasible_and_partial_groups(
    tmp_path: Path, reduce: str
) -> None:
    """Partial statistics stay visible, while only complete eligible groups rank."""
    experiment = _experiment(
        tmp_path,
        repeats=Repeats.model_validate({"key": "seed", "values": [0, 1, 2], "reduce": reduce}),
        n_trials=9,
    )
    study = new_phase_study(experiment, "p")
    for seed, value in enumerate([1.0, 2.0, 6.0]):
        _add_trial(study, x=1, seed=seed, value=value)
    for seed, value in enumerate([3.0, 4.0, 5.0]):
        _add_trial(study, x=2, seed=seed, value=value, feasible=seed != 2)
    _add_trial(study, x=3, seed=0, value=7.0)
    _add_trial(study, x=3, seed=1, value=None)

    tables = read_trial_tables(experiment)
    (table,) = tables.phases
    first, infeasible, partial = table.groups

    assert first.group == 2 and first.params == {"x": 1}
    assert first.n == first.expected == 3
    assert first.reduce == reduce
    assert first.aggregate == (3.0 if reduce == "mean" else 2.0)
    assert first.mean == 3.0 and first.sd == statistics.stdev([1.0, 2.0, 6.0])
    assert (first.min, first.max) == (1.0, 6.0)
    assert first.eligible and not first.winner
    assert infeasible.group == 5 and infeasible.n == 3
    assert infeasible.mean == 4.0 and infeasible.sd == 1.0
    assert not infeasible.eligible and infeasible.aggregate is None
    assert partial.group == 7 and partial.n == 1 and partial.expected == 3
    assert partial.mean == partial.min == partial.max == 7.0
    assert partial.sd is None and partial.aggregate is None and not partial.eligible
    rendered = _render_trial_tables(tables, persisted=True)
    assert "Repeat groups (statistics include infeasible measured members):" in rendered
    assert "n/expected" in rendered and "aggregate" in rendered and "eligible" in rendered
    assert "3/3" in rendered and "1/3" in rendered


def test_repeat_table_explains_group_numbers_while_members_arrive(tmp_path: Path) -> None:
    """An incomplete group's displayed number tracks its highest recorded member."""
    experiment = _experiment(tmp_path, repeats=Repeats(key="seed", values=[0, 1]), n_trials=2)
    study = new_phase_study(experiment, "p")

    for seed in [0, 1]:
        _add_trial(study, x=1, seed=seed, value=1.0 + seed)
        tables = read_trial_tables(experiment)
        (table,) = tables.phases
        (group,) = table.groups
        assert group.params == {"x": 1} and group.group == seed
        assert all(row["repeat_group"] == seed for row in table.rows)
        rendered = _render_trial_tables(tables, persisted=True)
        assert f"{seed + 1}/2" in rendered
        assert "highest member trial number so far" in rendered
        assert "can change as members arrive" in rendered


@pytest.mark.parametrize("outcome", ["failure", "interrupted"])
def test_repeat_summary_counts_a_rerun_once_and_keeps_random_slots_distinct(
    tmp_path: Path, outcome: str
) -> None:
    """A replacement completes its original slot; equal later draws stay separate."""
    experiment = _experiment(tmp_path, repeats=Repeats(key="seed", values=[0, 1]), n_trials=4)
    study = new_phase_study(experiment, "p")
    _add_trial(
        study,
        x=1,
        seed=0,
        value=None,
        attrs={TRIAL_OUTCOME_ATTR: trial_outcome(1, outcome)},
    )
    _add_trial(study, x=1, seed=1, value=2.0)
    _add_trial(study, x=1, seed=0, value=4.0, attrs={RERUN_OF_ATTR: 0})
    _add_trial(study, x=1, seed=0, value=6.0)
    _add_trial(study, x=1, seed=1, value=8.0)

    (table,) = read_trial_tables(experiment).phases

    assert [group.group for group in table.groups] == [2, 4]
    assert [group.n for group in table.groups] == [2, 2]
    assert [group.aggregate for group in table.groups] == [3.0, 7.0]
    assert all(group.params == {"x": 1} and group.eligible for group in table.groups)


def test_repeat_summary_keeps_an_interrupted_group_without_measurements(tmp_path: Path) -> None:
    """An interrupted-only group stays visible with no statistical estimate."""
    experiment = _experiment(tmp_path, repeats=Repeats(key="seed", values=[0, 1]), n_trials=2)
    study = new_phase_study(experiment, "p")
    _add_trial(
        study,
        x=1,
        seed=0,
        value=None,
        attrs={TRIAL_OUTCOME_ATTR: trial_outcome(1, "interrupted")},
    )

    (table,) = read_trial_tables(experiment).phases
    (group,) = table.groups

    assert group.n == 0 and group.expected == 2
    assert group.mean is group.sd is group.min is group.max is group.aggregate is None
    assert not group.eligible and not group.winner


def test_repeat_observed_means_report_their_own_contributor_counts(tmp_path: Path) -> None:
    """Each observed mean names how many completed members actually recorded it."""
    experiment = _experiment(tmp_path, repeats=Repeats(key="seed", values=[0, 1, 2]), n_trials=3)
    study = new_phase_study(experiment, "p")
    _add_trial(study, x=1, seed=0, value=0.4, attrs={"observed:heldout": 0.6})
    _add_trial(study, x=1, seed=1, value=0.5, attrs={"observed:heldout": 0.8, "observed:cost": 2.0})
    _add_trial(study, x=1, seed=2, value=0.6)

    tables = read_trial_tables(experiment)
    (table,) = tables.phases
    (group,) = table.groups

    assert group.n == 3
    assert group.observed == {"cost": 2.0, "heldout": 0.7}
    assert group.observed_n == {"cost": 1, "heldout": 2}
    rendered = _render_trial_tables(tables, persisted=True)
    assert "observed:heldout" in rendered and "observed_n:heldout" in rendered
    assert "observed:cost" in rendered and "observed_n:cost" in rendered
