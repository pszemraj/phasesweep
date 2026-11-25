"""Reuse sibling training at the target's own draws, budgets and failure boundaries."""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import CategoricalParam, FloatParam, IntParam, Sampler
from phasesweep.engine import reread as reread_ops
from phasesweep.engine.errors import PhaseTimeoutError
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.read import read_status
from phasesweep.engine.state import ATTEMPT_ID_ATTR, RERUN_OF_ATTR, RETURN_CODE_ATTR
from phasesweep.evidence.models import JsonExtractor
from tests.conftest import make_experiment, study_name, write_trainer


def _trainer(tmp_path: Path) -> tuple[Path, Path]:
    """Write two equivalent readings, and optionally fail x=2 after counting its launch."""
    launches = tmp_path / "launches"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse, json, sys
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        ap.add_argument("--x", type=float, default=1)
        ap.add_argument("--y", type=float, default=0)
        args = ap.parse_args()
        with Path({str(launches)!r}).open("a") as handle:
            print(args.x, file=handle)
        if args.x == 2 and Path({str(tmp_path / "fail")!r}).exists():
            sys.exit(7)
        loss = (args.x - 3) ** 2 + args.y
        Path(args.out).write_text(json.dumps({{"loss": loss, "acc": 100 + loss}}))
        """,
    )
    return trainer, launches


def _phase(sampler: Literal["grid", "random", "tpe"], n_trials: int) -> Phase:
    """Build a partial grid or seeded stochastic search with the same training contract."""
    space = (
        {"x": CategoricalParam(type="categorical", choices=[1, 2, 3])}
        if sampler == "grid"
        else {"x": FloatParam(type="float", low=0.1, high=3)}
    )
    settings = Sampler(type="grid") if sampler == "grid" else Sampler(type=sampler, seed=11)
    if sampler == "tpe":
        settings = settings.model_copy(update={"n_startup_trials": 1})
    return Phase(
        name="p",
        n_trials=n_trials,
        gpu_policy="none",
        allow_partial_grid=sampler == "grid",
        sampler=settings,
        search_space=space,
    )


def _config(
    tmp_path: Path,
    trainer: Path,
    phase: Phase,
    key: str = "loss",
    goal: str = "minimize",
) -> Experiment:
    """Read a trainer's selected JSON key without changing its launch command."""
    return make_experiment(
        persistent=tmp_path,
        trial_command=f"{shlex.quote(str(trainer))} --out {{trial_dir}}/r.json {{overrides}}",
        metric=Metric(
            name="score", goal=goal, extractor=JsonExtractor(type="json", path="r.json", key=key)
        ),
        phases=[phase],
    )


def _study(experiment: Experiment) -> optuna.Study:
    """Read the study addressed by these reading settings, among the phase's siblings."""
    return optuna.load_study(
        study_name=study_name(experiment), storage=_resolve_storage(experiment.resolved_storage)
    )


def _launches(path: Path) -> int:
    """Count the actual trainer executions."""
    return len(path.read_text().splitlines())


def test_returning_grid_reuses_finished_draws_without_launch_resources(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Preview and replay preserve sibling physical attempts even when the trainer is gone."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("grid", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    raised = phase.model_copy(update={"n_trials": 3})
    wider = phase.model_copy(
        update={
            "n_trials": 4,
            "search_space": {"x": CategoricalParam(type="categorical", choices=[1, 2, 3, 4])},
        }
    )
    sibling = _config(tmp_path, trainer, wider, "acc")
    run_experiment(sibling)
    donors = _study(sibling).trials
    target = _config(tmp_path, trainer, raised)
    trainer.unlink()
    caplog.set_level(logging.INFO)

    run_experiment(target, dry_run=True)
    assert "would re-read 2 finished trials from a sibling study" in caplog.text
    assert "may need to launch up to 2 trials depending on the re-read results" in caplog.text
    run_experiment(target)
    rows = _study(target).trials

    assert _launches(launches) == 4
    assert sorted(row.params["x"] for row in rows) == [1, 2, 3]
    by_value = {row.params["x"]: row for row in donors}
    assert [row.user_attrs[ATTEMPT_ID_ATTR] for row in rows] == [
        by_value[row.params["x"]].user_attrs[ATTEMPT_ID_ATTR] for row in rows
    ]
    assert all(RERUN_OF_ATTR not in row.user_attrs for row in rows)
    assert [row.value for row in rows] == [(row.params["x"] - 3) ** 2 for row in rows]
    assert read_status(target)["publication_integrity"] == "ok"
    run_experiment(target)
    assert _launches(launches) == 4


def test_repeated_random_configs_keep_independent_draw_slots(tmp_path: Path) -> None:
    """Equal parameters at distinct positions consume distinct physical attempts."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("random", 1).model_copy(
        update={"search_space": {"x": IntParam(type="int", low=1, high=1)}}
    )
    run_experiment(_config(tmp_path, trainer, phase))
    sibling = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": 3}), "acc")
    run_experiment(sibling)
    donors = _study(sibling).trials
    for target in (2, 3, 4):
        config = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": target}))
        run_experiment(config)
        rows = _study(config).trials
        assert len(rows) == target
        assert len({row.user_attrs[ATTEMPT_ID_ATTR] for row in rows}) == target
        assert _launches(launches) == max(3, target)
        assert [row.user_attrs[ATTEMPT_ID_ATTR] for row in rows[:3]] == [
            row.user_attrs[ATTEMPT_ID_ATTR] for row in donors[:target]
        ]


def test_ordinary_trainer_failure_is_spent_without_another_launch(tmp_path: Path) -> None:
    """A sibling's nonzero trainer exit spends the same grid slot in the target."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("grid", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    (tmp_path / "fail").touch()
    raised = phase.model_copy(update={"n_trials": 2})
    sibling = _config(tmp_path, trainer, raised, "acc")
    run_experiment(sibling)
    failed = _study(sibling).trials[-1]
    assert failed.state == optuna.trial.TrialState.FAIL
    trainer.unlink()
    target = _config(tmp_path, trainer, raised)

    run_experiment(target)

    adopted = _study(target).trials[-1]
    assert _launches(launches) == 2
    assert adopted.state == optuna.trial.TrialState.FAIL
    assert adopted.user_attrs[RETURN_CODE_ATTR] == 7
    assert adopted.user_attrs[ATTEMPT_ID_ATTR] == failed.user_attrs[ATTEMPT_ID_ATTR]


def test_grown_grid_reuses_a_later_donor_between_parallel_launches(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unmatched next cell neither hides a later donor nor inflates the preview's count."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("grid", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    sibling = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": 2}), "acc")
    run_experiment(sibling)
    donor = _study(sibling).trials[-1]
    grown = phase.model_copy(
        update={
            "n_trials": 4,
            "n_jobs": 2,
            "allow_no_gpu_isolation": True,
            "search_space": {"x": CategoricalParam(type="categorical", choices=[1, 2, 3, 4])},
        }
    )
    target = _config(tmp_path, trainer, grown)
    caplog.set_level(logging.INFO)

    run_experiment(target, dry_run=True)
    assert "would launch 2 trials if every retained draw supplies a result" in caplog.text
    assert "may need to launch up to 3 trials depending on the re-read results" in caplog.text
    assert _launches(launches) == 2
    run_experiment(target)

    rows = _study(target).trials
    assert _launches(launches) == 4
    assert sorted(row.params["x"] for row in rows) == [1, 2, 3, 4]
    assert donor.user_attrs[ATTEMPT_ID_ATTR] in [row.user_attrs[ATTEMPT_ID_ATTR] for row in rows]
    assert read_status(target)["publication_integrity"] == "ok"


def test_adaptive_reuse_checks_the_next_proposal_after_each_read(tmp_path: Path) -> None:
    """Reversing the objective order does not import the sibling's different TPE proposal."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("tpe", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    raised = phase.model_copy(update={"n_trials": 3})
    sibling = _config(tmp_path, trainer, raised, "acc", "maximize")
    run_experiment(sibling)
    donors = _study(sibling).trials
    target = _config(tmp_path, trainer, raised)

    run_experiment(target)

    rows = _study(target).trials
    assert rows[1].user_attrs[ATTEMPT_ID_ATTR] == donors[1].user_attrs[ATTEMPT_ID_ATTR]
    assert rows[2].params != donors[2].params
    assert rows[2].user_attrs[ATTEMPT_ID_ATTR] != donors[2].user_attrs[ATTEMPT_ID_ATTR]
    assert _launches(launches) == 4


def test_stochastic_preview_stops_after_the_final_possible_donor(tmp_path: Path) -> None:
    """A large top-up with one early donor forecasts only the positions it could supply."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("random", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    run_experiment(_config(tmp_path, trainer, phase.model_copy(update={"n_trials": 2}), "acc"))
    target = _config(tmp_path, trainer, phase.model_copy(update={"n_trials": 100000}))
    with patch.object(reread_ops, "_suggest_all", wraps=reread_ops._suggest_all) as suggest:
        run_experiment(target, dry_run=True)
    assert suggest.call_count < 10
    assert _launches(launches) == 2


def test_mid_phase_adoption_deadline_preserves_new_training_on_restart(tmp_path: Path) -> None:
    """A stopped sibling read between batches keeps the just-finished new grid cell."""
    trainer, launches = _trainer(tmp_path)
    phase = _phase("grid", 1)
    run_experiment(_config(tmp_path, trainer, phase))
    run_experiment(_config(tmp_path, trainer, phase.model_copy(update={"n_trials": 2}), "acc"))
    grown = phase.model_copy(
        update={
            "n_trials": 4,
            "timeout_seconds_per_phase": 30,
            "search_space": {"x": CategoricalParam(type="categorical", choices=[1, 2, 3, 4])},
        }
    )
    target = _config(tmp_path, trainer, grown)
    with (
        patch.object(
            reread_ops, "_reread_trial", side_effect=reread_ops._fill_timeout(target, grown)
        ),
        pytest.raises(PhaseTimeoutError, match="phase guard after 2/4"),
    ):
        run_experiment(target)
    assert len(_study(target).trials) == 2
    run_experiment(target)
    assert sorted(row.params["x"] for row in _study(target).trials) == [1, 2, 3, 4]
    assert _launches(launches) == 4
