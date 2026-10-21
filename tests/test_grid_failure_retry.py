"""Raised persistent grid targets retry failed cells without repeating completed cells."""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest

import phasesweep.engine.phase as phase_ops
from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import IntParam, Repeats, Sampler
from phasesweep.engine.errors import TrialTargetRefusedError
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.evidence.models import LogRegexExtractor
from tests.conftest import make_experiment, write_flaky_trainer


def _grid(
    tmp_path: Path,
    trainer: Path,
    target: int,
    *,
    repeats: bool = False,
    cells: int = 2,
    metric: Metric | None = None,
) -> Experiment:
    """Build a persistent grid of ``cells`` cells, or two with the repeat axis."""
    phase = Phase(
        name="p",
        n_trials=target,
        max_consecutive_failures=2,
        sampler=Sampler(type="grid", seed=0),
        search_space={"x": IntParam(type="int", low=1, high=1 if repeats else cells)},
        repeats=Repeats(key="seed", values=[1, 2]) if repeats else None,
    )
    return make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase], metric=metric)


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


def _advice(from_target: int, to_target: int, action: str) -> str:
    """Return the recovery advice a grid's abort and its refusals carry."""
    return (
        f"Fix the cause, then raise phases[p].n_trials from {from_target} to {to_target} "
        f"and rerun to {action}; completed cells are not repeated."
    )


@pytest.mark.parametrize("repeats", [False, True], ids=["grid", "repeat-grid"])
def test_failed_grid_names_the_target_that_retries_every_failed_cell(
    tmp_path: Path, repeats: bool
) -> None:
    """The abort, the identical rerun's refusal and the dry run name 4, which retries both cells."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=2)
    original = _grid(tmp_path, trainer, 2, repeats=repeats)
    advice = _advice(2, 4, "retry its 2 failed cells")
    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(original)
    assert str(aborted.value).endswith(f"for stderr logs. {advice}")
    for dry_run in (False, True):
        with pytest.raises(NoFeasibleTrialError, match="previously aborted") as refused:
            run_experiment(original, dry_run=dry_run)
        assert str(refused.value).endswith(f"as a successful phase. {advice}")

    # The advertised target is exactly the one that retries both failed cells.
    recovered = _grid(tmp_path, trainer, 4, repeats=repeats)
    assert run_experiment(recovered)["p"].metric == 1.0
    assert [trial.state.name for trial in _trials(recovered)] == [
        "FAIL",
        "FAIL",
        "COMPLETE",
        "COMPLETE",
    ]
    assert len(calls.read_text().splitlines()) == 4


def test_aborted_grid_names_the_cells_it_never_reached(tmp_path: Path) -> None:
    """An abort that left cells untried names the target that also runs them."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=2)
    original = _grid(tmp_path, trainer, 4, cells=4)
    advice = _advice(4, 6, "retry its 2 failed cells and run its 2 remaining cells")
    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(original)
    assert [trial.state.name for trial in _trials(original)] == ["FAIL", "FAIL"]
    assert str(aborted.value).endswith(f"for stderr logs. {advice}")
    with pytest.raises(NoFeasibleTrialError, match="previously aborted") as refused:
        run_experiment(original)
    assert str(refused.value).endswith(f"as a successful phase. {advice}")

    recovered = _grid(tmp_path, trainer, 6, cells=4)
    assert run_experiment(recovered)["p"].metric == 1.0
    assert [trial.state.name for trial in _trials(recovered)] == ["FAIL", "FAIL"] + ["COMPLETE"] * 4
    assert len(calls.read_text().splitlines()) == 6


def test_aborted_grid_without_a_failed_cell_says_raising_n_trials_schedules_nothing(
    tmp_path: Path,
) -> None:
    """Completed cells are closed, so a grid whose cells all measured infeasible has nothing to retry."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=0, value=5.0)
    bounded = Metric(
        extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        max=1.0,
    )
    original = _grid(tmp_path, trainer, 2, metric=bounded)
    advice = (
        "No grid cell has a failed attempt to retry, so raising n_trials would schedule nothing."
    )
    with pytest.raises(NoFeasibleTrialError, match="aborted after 2") as aborted:
        run_experiment(original)
    assert str(aborted.value).endswith(f"for stderr logs. {advice}")
    with pytest.raises(NoFeasibleTrialError, match="previously aborted") as refused:
        run_experiment(original)
    assert str(refused.value).endswith(f"as a successful phase. {advice}")
    assert len(calls.read_text().splitlines()) == 2


def test_failure_streak_refusal_names_the_grid_retry_target(tmp_path: Path) -> None:
    """A resumed streak at the limit quotes the same target, beside the limit alternative."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=2)
    original = _grid(tmp_path, trainer, 2)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(original)

    with pytest.raises(NoFeasibleTrialError) as refused:
        phase_ops._raise_streak_at_limit(original.phases[0], 2, 2, _trials(original))
    assert str(refused.value).endswith(
        f"at its accepted n_trials=2. {_advice(2, 4, 'retry its 2 failed cells')} "
        "Or raise max_consecutive_failures."
    )


def test_sampling_phase_keeps_the_generic_recovery_advice(tmp_path: Path) -> None:
    """Random draws have no per-cell retry rule, so the refusal still asks for a higher target."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=2)
    phase = Phase(
        name="p",
        n_trials=2,
        max_consecutive_failures=2,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": IntParam(type="int", low=1, high=9)},
    )
    experiment = make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase])
    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(experiment)
    assert str(aborted.value).endswith("for stderr logs.")
    with pytest.raises(NoFeasibleTrialError, match="previously aborted") as refused:
        run_experiment(experiment)
    assert str(refused.value).endswith(
        "as a successful phase. Increase n_trials above 2 to explicitly schedule new "
        "recovery attempts."
    )
