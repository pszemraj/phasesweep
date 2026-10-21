"""Raised persistent grid targets retry failed cells without repeating completed cells."""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest

import phasesweep.engine.phase as phase_ops
from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import IntParam, Repeats, Sampler
from phasesweep.engine.errors import TrialTargetRefusedError
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.selection import NoFeasibleTrialError
from tests.conftest import make_experiment, write_flaky_trainer


def _grid(tmp_path: Path, trainer: Path, target: int, *, repeats: bool = False) -> Experiment:
    """Build a persistent two-cell grid, optionally using the repeat axis."""
    phase = Phase(
        name="p",
        n_trials=target,
        max_consecutive_failures=2,
        sampler=Sampler(type="grid", seed=0),
        search_space={"x": IntParam(type="int", low=1, high=1 if repeats else 2)},
        repeats=Repeats(key="seed", values=[1, 2]) if repeats else None,
    )
    return make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase])


def _trials(experiment: Experiment) -> list[optuna.trial.FrozenTrial]:
    """Read the grid's recorded attempts."""
    study = optuna.load_study(
        study_name=f"{experiment.experiment}::p",
        storage=_resolve_storage(experiment.resolved_storage),
    )
    return study.get_trials(deepcopy=False)


@pytest.mark.parametrize("repeats", [False, True], ids=["grid", "repeat-grid"])
def test_fully_failed_grid_recovers_only_the_raised_target(tmp_path: Path, repeats: bool) -> None:
    """A recorded abort stays refused until a raised target buys failed-cell retries."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=2)
    original = _grid(tmp_path, trainer, 2, repeats=repeats)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(original)
    assert [trial.state.name for trial in _trials(original)] == ["FAIL", "FAIL"]
    with pytest.raises(NoFeasibleTrialError, match="previously aborted"):
        run_experiment(original)
    assert len(calls.read_text().splitlines()) == 2

    raised = _grid(tmp_path, trainer, 3, repeats=repeats)
    if repeats:
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(raised)
    else:
        assert run_experiment(raised)["p"].metric == 1.0
    trials = _trials(raised)
    assert [trial.state.name for trial in trials] == ["FAIL", "FAIL", "COMPLETE"]
    assert trials[2].params == trials[0].params
    assert len(calls.read_text().splitlines()) == 3

    recovered = _grid(tmp_path, trainer, 4, repeats=repeats)
    assert run_experiment(recovered)["p"].metric == 1.0
    trials = _trials(recovered)
    assert trials[3].params == trials[1].params
    assert len(calls.read_text().splitlines()) == 4
    assert run_experiment(recovered)["p"].metric == 1.0
    with pytest.raises(TrialTargetRefusedError, match="completed"):
        run_experiment(_grid(tmp_path, trainer, 5, repeats=repeats))
    assert len(calls.read_text().splitlines()) == 4


@pytest.mark.parametrize("repeats", [False, True], ids=["grid", "repeat-grid"])
def test_partly_failed_grid_retries_the_failed_cell_only(tmp_path: Path, repeats: bool) -> None:
    """A successful cell closes even when another cell needs an extra attempt."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=1)
    original = _grid(tmp_path, trainer, 2, repeats=repeats)
    if repeats:
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(original)
    else:
        assert run_experiment(original)["p"].metric == 1.0
    with pytest.raises(TrialTargetRefusedError):
        run_experiment(_grid(tmp_path, trainer, 4, repeats=repeats))
    assert len(calls.read_text().splitlines()) == 2

    raised = _grid(tmp_path, trainer, 3, repeats=repeats)
    assert run_experiment(raised)["p"].metric == 1.0
    trials = _trials(raised)
    assert [trial.state.name for trial in trials] == ["FAIL", "COMPLETE", "COMPLETE"]
    assert trials[2].params == trials[0].params
    assert trials[2].params != trials[1].params
    assert run_experiment(raised)["p"].metric == 1.0
    assert len(calls.read_text().splitlines()) == 3


def test_fresh_grid_refuses_retry_slots_without_failed_cells(tmp_path: Path) -> None:
    """A target beyond the matrix requires failed cells already recorded in the study."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=0)
    experiment = _grid(tmp_path, trainer, 3)
    for dry_run in (True, False):
        with pytest.raises(TrialTargetRefusedError, match="grid"):
            run_experiment(experiment, dry_run=dry_run)
    assert not calls.exists()


def test_failed_grid_retry_consumes_the_explicit_extra_attempt(tmp_path: Path) -> None:
    """Another failed retry stays spent until the operator raises the target again."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=3)
    original = _grid(tmp_path, trainer, 2)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(original)
    raised = _grid(tmp_path, trainer, 3)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(raised)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(raised)
    assert len(calls.read_text().splitlines()) == 3
    assert [trial.state.name for trial in _trials(raised)] == ["FAIL", "FAIL", "FAIL"]
    assert run_experiment(_grid(tmp_path, trainer, 4))["p"].metric == 1.0
    assert len(calls.read_text().splitlines()) == 4


def test_grid_retry_queue_recovers_after_target_was_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted raised target remains retryable if enqueueing never happened."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=2)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_grid(tmp_path, trainer, 2))
    raised = _grid(tmp_path, trainer, 3)

    def fail_before_enqueue(*args: object) -> None:
        raise RuntimeError("stopped before enqueue")

    with monkeypatch.context() as patch:
        patch.setattr(phase_ops, "_record_reruns", fail_before_enqueue)
        with pytest.raises(RuntimeError, match="stopped before enqueue"):
            run_experiment(raised)
    assert len(_trials(raised)) == 2
    assert run_experiment(raised)["p"].metric == 1.0
    assert len(calls.read_text().splitlines()) == 3


def test_grid_seed_audit_retries_only_its_failed_repeat(tmp_path: Path) -> None:
    """A grid with no search keys can recover one failed repeat member."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=1)

    def attempt(target: int) -> Experiment:
        phase = Phase(
            name="p",
            n_trials=target,
            sampler=Sampler(type="grid", seed=0),
            fixed_overrides={"x": 1},
            repeats=Repeats(key="seed", values=[1, 2]),
        )
        return make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase])

    with pytest.raises(NoFeasibleTrialError):
        run_experiment(attempt(2))
    recovered = attempt(3)
    assert run_experiment(recovered)["p"].metric == 1.0
    trials = _trials(recovered)
    assert trials[2].params == trials[0].params
    assert len(calls.read_text().splitlines()) == 3
