"""Correcting a metric or constraint bound re-judges a study's recorded trials without retraining."""

from __future__ import annotations

from pathlib import Path

import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Constraint, Experiment, Phase
from phasesweep.engine.errors import ResumeBoundsChangedError
from phasesweep.engine.paths import _experiment_dir, _generations_dir
from phasesweep.engine.read import read_trial_tables
from phasesweep.engine.selection import PhaseAbortRefusedError, PhaseTrialsInfeasibleError
from phasesweep.evidence.models import LogRegexExtractor
from tests.conftest import bounded_metric, make_experiment, phase_trials, write_trainer

# Trial 0 measures x=1.0, trial 1 x=0.3 and trial 2 x=1.1; each also prints its memory use.
_TRAINER = """
import os
from pathlib import Path
trial = os.environ["PHASESWEEP_TRIAL_ID"]
with Path({launches!r}).open("a") as log:
    log.write(trial + "\\n")
print("x=" + {{"0": "1.0", "1": "0.3", "2": "1.1"}}[trial])
print("mem=" + {{"0": "10", "1": "30", "2": "20"}}[trial])
"""


def _experiment(
    tmp_path: Path, *, memory_max: float | None = None, **metric_bounds: float
) -> Experiment:
    """Build the same persistent three-trial study under the given bounds.

    :param Path tmp_path: Directory holding the trainer, its launch log, the ledger and runs.
    :param float | None memory_max: Bound of a ``memory`` constraint, defaults to no constraint.
    :param float metric_bounds: ``metric.min`` and ``metric.max``, each unbounded by default.
    :return Experiment: The experiment.
    """
    trainer = write_trainer(
        tmp_path / "trainer.py", _TRAINER.format(launches=str(tmp_path / "launches"))
    )
    constraints = (
        []
        if memory_max is None
        else [
            Constraint(
                name="memory",
                extractor=LogRegexExtractor(type="log_regex", pattern=r"mem=(?P<value>[0-9.]+)"),
                max=memory_max,
            )
        ]
    )
    return make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        metric=bounded_metric(**metric_bounds),
        constraints=constraints,
        n_trials=3,
        max_consecutive_failures=3,
    )


def _launches(tmp_path: Path) -> int:
    """Count the trainer launches so far.

    :param Path tmp_path: Directory holding the launch log.
    :return int: Number of launches.
    """
    return len((tmp_path / "launches").read_text().splitlines())


def test_loosening_a_bound_publishes_from_the_recorded_trials_and_clears_the_streak_abort(
    tmp_path: Path,
) -> None:
    """Three trials outside metric.max trip the streak abort; a loosened bound re-judges them, clears the abort and publishes a winner with no launch, and tightening it again restores the abort."""
    tight = _experiment(tmp_path, max=0.0001)
    with pytest.raises(PhaseTrialsInfeasibleError, match="aborted"):
        run_experiment(tight)
    assert _launches(tmp_path) == 3

    loose = _experiment(tmp_path, max=10.0)
    winner = run_experiment(loose)["p"]

    assert (winner.trial_number, winner.metric) == (1, 0.3)
    assert _launches(tmp_path) == 3
    (table,) = read_trial_tables(loose).phases
    assert [row["state"] for row in table.rows] == ["COMPLETE"] * 3
    assert "reason" not in table.columns

    with pytest.raises(PhaseAbortRefusedError, match="previously aborted") as refused:
        run_experiment(tight)
    assert "re-judges the recorded trials" in str(refused.value.remediation)
    assert _launches(tmp_path) == 3
    (table,) = read_trial_tables(tight).phases
    assert [row["state"] for row in table.rows] == ["INFEASIBLE"] * 3
    assert all(
        str(row["reason"]).startswith("outside the metric bounds or constraints: metric")
        for row in table.rows
    )


@pytest.mark.parametrize("bound", ["metric", "constraint"])
def test_tightening_a_bound_replaces_a_winner_it_rules_out_without_retraining(
    tmp_path: Path, bound: str
) -> None:
    """A published winner a tightened bound makes infeasible is not replayed: selection reruns over the recorded values and publishes the best trial still inside, with no launch."""
    loose = _experiment(tmp_path, **({} if bound == "metric" else {"memory_max": 100.0}))
    assert run_experiment(loose)["p"].trial_number == 1
    assert _launches(tmp_path) == 3

    # Trial 1 measured x=0.3 and memory 30; only trials 0 and 2 stay inside these bounds.
    tightened = (
        _experiment(tmp_path, min=0.5)
        if bound == "metric"
        else _experiment(tmp_path, memory_max=25.0)
    )
    winner = run_experiment(tightened)["p"]

    assert (winner.trial_number, winner.metric) == (0, 1.0)
    assert _launches(tmp_path) == 3
    rows = read_trial_tables(tightened).phases[0].rows
    assert {row["trial"]: row["state"] for row in rows} == {
        0: "COMPLETE",
        1: "INFEASIBLE",
        2: "COMPLETE",
    }
    assert [trial.number for trial in phase_trials(tightened)] == [0, 1, 2]


@pytest.mark.parametrize("bound", ["metric", "constraint"])
@pytest.mark.parametrize("tighten", [True, False], ids=["tighten", "loosen"])
def test_from_phase_requires_reselection_when_bounds_change(
    tmp_path: Path, bound: str, tighten: bool
) -> None:
    """A skipped selection cannot be carried after either direction of a bound edit.

    :param Path tmp_path: Scratch trainer, journal and artifacts.
    :param str bound: Whether the objective or a constraint provides the bound.
    :param bool tighten: Tighten the bounds if true, otherwise loosen them.
    """
    loose = _experiment(tmp_path, **({} if bound == "metric" else {"memory_max": 100.0}))
    tight = (
        _experiment(tmp_path, min=0.5)
        if bound == "metric"
        else _experiment(tmp_path, memory_max=25.0)
    )
    child = Phase(name="q", n_trials=1, inherits=["p"], fixed_overrides={"x": 0}, gpu_policy="none")
    previous, current = (loose, tight) if tighten else (tight, loose)
    previous = previous.model_copy(update={"phases": [*previous.phases, child]})
    current = current.model_copy(update={"phases": [*current.phases, child]})
    original = run_experiment(previous)
    assert original["p"].metric == (0.3 if tighten else 1.0)
    assert _launches(tmp_path) == 4
    generations = _generations_dir(_experiment_dir(previous))
    before = set(generations.iterdir())

    for dry_run in (True, False):
        with pytest.raises(
            ResumeBoundsChangedError, match="Skipped phase 'p' was selected under different"
        ) as refused:
            run_experiment(current, from_phase="q", dry_run=dry_run)
        assert refused.value.remediation == (
            "Run with `--from-phase p` to reselect from the phase's recorded trials."
        )
    assert _launches(tmp_path) == 4
    assert set(generations.iterdir()) == before

    replay = run_experiment(current, from_phase="p")
    assert replay["p"].metric == (1.0 if tighten else 0.3)
    assert replay["p"].generation_id == original["p"].generation_id
    assert _launches(tmp_path) == 4
    assert run_experiment(current, from_phase="q")["p"] == replay["p"]
    assert _launches(tmp_path) == 4
