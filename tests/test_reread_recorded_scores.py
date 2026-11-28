"""A score another study already read under the current extraction is carried, never read again."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from click.testing import CliRunner

from phasesweep import run_experiment
from phasesweep.cli import cli
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import CategoricalParam, Sampler
from phasesweep.engine.errors import TrialEvidenceMissingError
from phasesweep.engine.state import TRIAL_DIR_ATTR
from phasesweep.evidence.models import JsonExtractor
from tests.conftest import (
    addressed_phase_study,
    make_experiment,
    write_experiment_config,
    write_trainer,
)

Goal = Literal["minimize", "maximize"]


def _trainer(tmp_path: Path) -> tuple[Path, Path]:
    """Drop a trainer that writes ``score`` as ``--x`` and ``alt`` as ten times it, once per launch."""
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
        Path(args.out).write_text(json.dumps({{"score": args.x, "alt": 10 * args.x}}))
        """,
    )
    return trainer, launches


def _config(tmp_path: Path, trainer: Path, key: str, goal: Goal) -> Experiment:
    """A three-cell grid over x in {1, 2, 3}, scored by ``key`` with ``goal``."""
    metric = Metric(
        name="score", goal=goal, extractor=JsonExtractor(type="json", path="r.json", key=key)
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


def _reread_under_alt(tmp_path: Path) -> tuple[Path, Path]:
    """Train under ``score``, then re-read the three trials under ``alt`` as 10, 20 and 30."""
    trainer, launches = _trainer(tmp_path)
    run_experiment(_config(tmp_path, trainer, "score", "minimize"))
    reread = _config(tmp_path, trainer, "alt", "minimize")
    assert run_experiment(reread)["p"].params == {"x": 1}
    rows = addressed_phase_study(reread).trials
    assert sorted((row.params["x"], row.value) for row in rows) == [(1, 10.0), (2, 20.0), (3, 30.0)]
    return trainer, launches


def test_a_goal_flip_carries_the_scores_a_changed_extractor_recorded(tmp_path: Path) -> None:
    """A score file edited after the re-read never reaches a later study that flips only the goal.

    The ``alt`` study recorded 10, 20 and 30. Editing the score-10 file to 90 and maximizing must
    carry those recorded scores, not read the file again, so x=3 wins with 30.
    """
    trainer, launches = _reread_under_alt(tmp_path)
    minimize = _config(tmp_path, trainer, "alt", "minimize")
    (_trial_dir(minimize, 1) / "r.json").write_text(json.dumps({"score": 1.0, "alt": 90.0}))
    maximize = _config(tmp_path, trainer, "alt", "maximize")

    winner = run_experiment(maximize)["p"]

    assert winner.params == {"x": 3}
    rows = addressed_phase_study(maximize).trials
    assert sorted((row.params["x"], row.value) for row in rows) == [(1, 10.0), (2, 20.0), (3, 30.0)]
    assert len(launches.read_text().splitlines()) == 3


def test_a_goal_flip_names_the_missing_winner_directory_a_changed_extractor_scored(
    tmp_path: Path,
) -> None:
    """A carried winner whose directory is gone is refused with that directory, not passed over.

    The run names the directory to restore, as for a study that never changed its reading, and
    selects x=3 once it is back; the runner-up never wins and nothing retrains.
    """
    trainer, launches = _reread_under_alt(tmp_path)
    directory = _trial_dir(_config(tmp_path, trainer, "alt", "minimize"), 3)
    parked = tmp_path / "parked"
    directory.rename(parked)
    maximize = _config(tmp_path, trainer, "alt", "maximize")

    with pytest.raises(TrialEvidenceMissingError, match="would select trial") as refused:
        run_experiment(maximize)
    assert str(directory) in str(refused.value.remediation)
    parked.rename(directory)

    assert run_experiment(maximize)["p"].params == {"x": 3}
    assert len(launches.read_text().splitlines()) == 3


def test_status_names_the_missing_directory_of_a_carried_winner(tmp_path: Path) -> None:
    """Both status forms name the directory to restore when a flipped goal's winner lost it.

    Every score is recorded, so the next run would refuse, not re-read; status says so as
    ``run --dry-run`` does instead of telling the operator to run the configuration.
    """
    trainer, _ = _trainer(tmp_path)
    run_experiment(_config(tmp_path, trainer, "score", "minimize"))
    directory = _trial_dir(_config(tmp_path, trainer, "score", "minimize"), 3)
    directory.rename(tmp_path / "parked")
    config = write_experiment_config(
        tmp_path / "maximize.yaml", _config(tmp_path, trainer, "score", "maximize")
    )

    for flags in ([], ["--trials"]):
        status = CliRunner().invoke(cli, ["status", str(config), *flags])
        assert status.exit_code == 0, status.output
        text = " ".join(status.output.split())
        assert "would select trial" in text
        assert str(directory) in text
        assert "Run the same configuration" not in text
