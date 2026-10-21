"""Raised persistent grid targets retry failed cells without repeating completed cells.

Recovery messages name the target that retries every failed cell, and a winner chosen while
some cell has no successful trial says so.
"""

from __future__ import annotations

import logging
from pathlib import Path

import optuna
import pytest

import phasesweep.engine.phase as phase_ops
from phasesweep import run_experiment
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import IntParam, Repeats, Sampler
from phasesweep.engine.errors import TrialTargetRefusedError
from phasesweep.engine.ledger import _resolve_storage
from phasesweep.engine.selection import NoFeasibleTrialError, _warn_unsuccessful_grid_cells
from phasesweep.evidence.models import LogRegexExtractor
from tests.conftest import (
    invoke_cli_boundary,
    make_experiment,
    write_flaky_trainer,
    write_trainer,
)
from tests.experiments import write_experiment


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


_NOTHING_TO_RETRY = (
    "No grid cell has a failed attempt to retry, so raising n_trials would schedule nothing."
)
_REOPEN = "this study's recorded abort cannot be reopened."


def test_fully_measured_infeasible_grid_points_to_a_fresh_namespace(tmp_path: Path) -> None:
    """Completed cells are closed and no cell is left to run, so only a fresh study can recover."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=0, value=5.0)
    bounded = Metric(
        extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        max=1.0,
    )
    original = _grid(tmp_path, trainer, 2, metric=bounded)
    # The test config names an explicit journal, which a new workdir alone does not redirect.
    advice = (
        f"{_NOTHING_TO_RETRY} Fix the cause and rerun into a fresh namespace, for example "
        f"`phasesweep run CONFIG --workdir NEW_DIR --set storage=auto`; {_REOPEN}"
    )
    with pytest.raises(NoFeasibleTrialError, match="aborted after 2") as aborted:
        run_experiment(original)
    assert str(aborted.value).endswith(f"for stderr logs. {advice}")
    with pytest.raises(NoFeasibleTrialError, match="previously aborted") as refused:
        run_experiment(original)
    assert str(refused.value).endswith(f"as a successful phase. {advice}")
    assert len(calls.read_text().splitlines()) == 2


def _untried_cells_config(tmp_path: Path, trainer: Path, *, limit: int, workdir: Path) -> Path:
    """Write a four-cell grid whose first two cells measure infeasible, with auto storage."""
    return write_experiment(
        tmp_path / "exp.yaml",
        experiment="t",
        trial_command=f"python {trainer} {{overrides}}",
        override_format="argparse",
        metric={
            "extractor": {"type": "log_regex", "pattern": r"loss=(?P<value>[0-9.eE+-]+)"},
            "max": 1.0,
        },
        storage="auto",
        workdir=str(workdir),
        provenance={"revision": "test"},
        phases=[
            {
                "name": "p",
                "n_trials": 4,
                "max_consecutive_failures": limit,
                "sampler": {"type": "grid", "seed": 0},
                "search_space": {"x": {"type": "int", "low": 1, "high": 4}},
            }
        ],
    )


def test_infeasible_streak_with_untried_cells_points_to_a_fresh_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A recorded streak abort cannot be reopened, so the way out is a higher limit in a new workdir."""
    trainer = write_trainer(
        tmp_path / "trainer.py",
        """
        import sys

        x = next(int(arg.removeprefix("--x=")) for arg in sys.argv if arg.startswith("--x="))
        print(f"loss={5.0 if x >= 3 else 0.5}")
        """,
    )
    original = _untried_cells_config(tmp_path, trainer, limit=2, workdir=tmp_path / "runs")
    advice = (
        f"{_NOTHING_TO_RETRY} Raise phases[p].max_consecutive_failures above 2 and rerun into a "
        f"fresh namespace, for example `phasesweep run CONFIG --workdir NEW_DIR`; {_REOPEN}"
    )

    assert invoke_cli_boundary(["run", str(original)], monkeypatch) != 0
    aborted = capsys.readouterr().err
    assert "aborted after 2 consecutive failures" in aborted
    assert advice in aborted
    for argv in (["run", str(original)], ["run", str(original), "--dry-run"]):
        assert invoke_cli_boundary(argv, monkeypatch) != 0
        refused = capsys.readouterr().err
        assert "previously aborted" in refused
        assert advice in refused

    # The raised limit alone changes nothing: the recorded abort still stands in the old namespace.
    raised = _untried_cells_config(tmp_path, trainer, limit=3, workdir=tmp_path / "runs")
    assert invoke_cli_boundary(["run", str(raised)], monkeypatch) != 0
    assert "previously aborted" in capsys.readouterr().err

    fresh = tmp_path / "fresh"
    assert invoke_cli_boundary(["run", str(raised), "--workdir", str(fresh)], monkeypatch) == 0
    assert list(fresh.rglob("winner.yaml"))


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


def _coverage_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the advisories saying some grid cell has no successful trial."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "phasesweep.engine.selection" and "grid cells succeeded" in record.message
    ]


def test_winner_from_a_partly_failed_grid_names_the_cells_that_never_succeeded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A raised target that retries one of two failed cells publishes, and the log says so."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=2)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_grid(tmp_path, trainer, 2))
    raised = _grid(tmp_path, trainer, 3)

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        assert run_experiment(raised)["p"].metric == 1.0
    trials = _trials(raised)
    assert [trial.state.name for trial in trials] == ["FAIL", "FAIL", "COMPLETE"]
    expected = (
        "[p] only 1 of 2 grid cells succeeded (advisory); the winner was chosen without a "
        f"successful trial for: {trials[1].params}."
    )
    assert _coverage_warnings(caplog) == [expected]

    # A replay selects from the recorded trials again and repeats the advisory.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        assert run_experiment(raised)["p"].metric == 1.0
    assert _coverage_warnings(caplog) == [expected]


def test_grid_winner_warns_about_cells_the_abort_never_reached(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A cell with no trial at all is as unsuccessful as one that failed."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=2)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_grid(tmp_path, trainer, 4, cells=4))
    raised = _grid(tmp_path, trainer, 5, cells=4)

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        assert run_experiment(raised)["p"].metric == 1.0
    succeeded = {trial.params["x"] for trial in _trials(raised) if trial.state.name == "COMPLETE"}
    (missing,) = {1, 2, 3, 4} - succeeded
    assert _coverage_warnings(caplog) == [
        "[p] only 3 of 4 grid cells succeeded (advisory); the winner was chosen without a "
        f"successful trial for: {{'x': {missing}}}."
    ]


def test_repeat_grid_winner_warns_about_the_failed_member(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A repeat group that lost a member cannot rank, so the cell names its repeat value."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=1)
    phase = Phase(
        name="p",
        n_trials=4,
        sampler=Sampler(type="grid", seed=0),
        search_space={"x": IntParam(type="int", low=1, high=2)},
        repeats=Repeats(key="seed", values=[1, 2]),
    )
    experiment = make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase])

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        assert run_experiment(experiment)["p"].metric == 1.0
    (failed,) = (trial for trial in _trials(experiment) if trial.state.name == "FAIL")
    (warning,) = _coverage_warnings(caplog)
    assert warning.startswith("[p] only 3 of 4 grid cells succeeded (advisory); ")
    assert f"'x': {failed.params['x']}" in warning and f"'seed': {failed.params['seed']}" in warning


@pytest.mark.parametrize("target", [2, 4], ids=["fresh-success", "every-failure-retried"])
def test_grid_winner_with_a_successful_trial_in_every_cell_stays_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, target: int
) -> None:
    """Failed attempts do not warn once their cells have a successful retry."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=0 if target == 2 else 2)
    if target == 4:
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(_grid(tmp_path, trainer, 2))
    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        assert run_experiment(_grid(tmp_path, trainer, target))["p"].metric == 1.0
    assert _coverage_warnings(caplog) == []


def test_sampling_winner_after_a_failure_has_no_cell_advisory(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Only a grid has cells to cover."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=1)
    phase = Phase(
        name="p",
        n_trials=3,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": IntParam(type="int", low=1, high=9)},
    )
    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        run_experiment(make_experiment(persistent=tmp_path, trainer=trainer, phases=[phase]))
    assert _coverage_warnings(caplog) == []


def test_cell_advisory_spells_out_ten_cells_and_counts_the_rest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A wide grid with one successful cell keeps the advisory to one readable line."""
    phase = Phase(
        name="p",
        n_trials=15,
        sampler=Sampler(type="grid", seed=0),
        search_space={"x": IntParam(type="int", low=0, high=14)},
    )
    trial = optuna.trial.create_trial(
        value=0.0,
        params={"x": 0},
        distributions={"x": optuna.distributions.IntDistribution(0, 14)},
    )

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        _warn_unsuccessful_grid_cells(phase, [trial])

    (warning,) = _coverage_warnings(caplog)
    assert warning.startswith("[p] only 1 of 15 grid cells succeeded (advisory); ")
    assert "{'x': 10}" in warning and "{'x': 11}" not in warning
    assert warning.endswith("; and 4 more.")


def test_in_memory_grid_abort_gets_no_recovery_advice(tmp_path: Path) -> None:
    """Without storage there is no study to recover, and a raised grid target is refused at load."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=2)
    phase = Phase(
        name="p",
        n_trials=2,
        max_consecutive_failures=2,
        sampler=Sampler(type="grid", seed=0),
        search_space={"x": IntParam(type="int", low=1, high=2)},
    )
    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(make_experiment(trainer=trainer, phases=[phase]))
    assert str(aborted.value).endswith("for stderr logs.")
