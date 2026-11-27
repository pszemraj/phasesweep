"""A reading that differs only in the metric's goal ranks kept results as recorded, unread."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import CategoricalParam, Sampler
from phasesweep.engine.errors import TrialEvidenceMissingError
from phasesweep.engine.state import TRIAL_DIR_ATTR
from phasesweep.evidence.models import JsonExtractor
from tests.conftest import addressed_phase_study, make_experiment, write_trainer


def _trainer(tmp_path: Path) -> tuple[Path, Path]:
    """Drop a trainer that scores ``--x`` as itself and appends one line per launch."""
    launches = tmp_path / "launches.txt"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        ap.add_argument("--x", type=float, required=True)
        args, _ = ap.parse_known_args()
        with open({str(launches)!r}, "a") as handle:
            handle.write("launch\\n")
        Path(args.out).write_text(json.dumps({{"score": args.x}}))
        """,
    )
    return trainer, launches


Goal = Literal["minimize", "maximize"]


def _config(tmp_path: Path, trainer: Path, goal: Goal) -> Experiment:
    """A three-cell grid over x in {1, 2, 3}, read with ``goal``."""
    metric = Metric(
        name="score", goal=goal, extractor=JsonExtractor(type="json", path="r.json", key="score")
    )
    phase = Phase(
        name="p",
        n_trials=3,
        gpu_policy="none",
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[1, 2, 3])},
    )
    return make_experiment(persistent=tmp_path, trainer=trainer, metric=metric, phases=[phase])


def _trial_dir(experiment: Experiment, x: int) -> Path:
    """Return the directory of the trial that trained ``x``."""
    [trial] = [t for t in addressed_phase_study(experiment).trials if t.params["x"] == x]
    return Path(trial.user_attrs[TRIAL_DIR_ATTR])


@pytest.mark.parametrize("moved", [3, 1])
def test_a_flipped_goal_ranks_recorded_scores_whose_directories_are_gone(
    tmp_path: Path, moved: int
) -> None:
    """Each kept score ranks as recorded; only a winner without its directory is refused.

    With the score-3 trial's directory gone, maximizing must neither fail that trial nor
    publish 2: the run refuses with the directory to restore and then selects it. With the
    score-1 trial's directory gone, that trial keeps its value and x=3 wins. Nothing retrains.
    """
    trainer, launches = _trainer(tmp_path)
    minimize = _config(tmp_path, trainer, "minimize")
    run_experiment(minimize)
    directory = _trial_dir(minimize, moved)
    parked = tmp_path / "parked"
    directory.rename(parked)
    maximize = _config(tmp_path, trainer, "maximize")

    if moved == 3:
        with pytest.raises(TrialEvidenceMissingError, match="would select trial") as refused:
            run_experiment(maximize)
        assert str(directory) in str(refused.value.remediation)
        parked.rename(directory)
    winner = run_experiment(maximize)["p"]

    assert winner.params == {"x": 3}
    assert len(launches.read_text().splitlines()) == 3
    rows = addressed_phase_study(maximize).trials
    assert sorted((row.params["x"], row.value) for row in rows) == [(1, 1.0), (2, 2.0), (3, 3.0)]


def test_a_flipped_goal_refuses_an_edited_score_file_as_a_replay_does(tmp_path: Path) -> None:
    """A score file edited after it was measured is refused, never published, under either goal."""
    trainer, _ = _trainer(tmp_path)
    minimize = _config(tmp_path, trainer, "minimize")
    run_experiment(minimize)
    (_trial_dir(minimize, 3) / "r.json").write_text(json.dumps({"score": 30.0}))

    goals: tuple[Goal, ...] = ("minimize", "maximize")
    for goal in goals:
        with pytest.raises(TrialEvidenceMissingError, match="recorded its objective evidence"):
            run_experiment(_config(tmp_path, trainer, goal))


@pytest.mark.parametrize("damage", ["moved", "edited"])
def test_a_flipped_goal_dry_run_refuses_as_the_run_does(tmp_path: Path, damage: str) -> None:
    """The dry run checks the carried rows' evidence and refuses with the run's own error.

    The winner under the flipped goal has lost its directory, or its score file was edited.
    """
    trainer, launches = _trainer(tmp_path)
    minimize = _config(tmp_path, trainer, "minimize")
    run_experiment(minimize)
    directory = _trial_dir(minimize, 3)
    if damage == "moved":
        directory.rename(tmp_path / "parked")
    else:
        (directory / "r.json").write_text(json.dumps({"score": 30.0}))
    maximize = _config(tmp_path, trainer, "maximize")

    with pytest.raises(TrialEvidenceMissingError) as previewed:
        run_experiment(maximize, dry_run=True)
    with pytest.raises(TrialEvidenceMissingError) as refused:
        run_experiment(maximize)

    assert str(previewed.value) == str(refused.value)
    assert len(launches.read_text().splitlines()) == 3
