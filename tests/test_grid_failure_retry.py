"""Raised persistent grid targets retry failed cells without repeating completed cells.

Recovery messages name the target that retries every failed cell, a winner chosen while
some cell has no successful trial says so, and the winner's survivorship check counts only the
failures that no successful retry replaced.
"""

from __future__ import annotations

import logging
from pathlib import Path

import optuna
import pytest

import phasesweep.engine.phase as phase_ops
from phasesweep import load_experiment, run_experiment
from phasesweep.cli import _operational_error_text
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import IntParam, Repeats, Sampler
from phasesweep.engine.errors import TrialTargetRefusedError
from phasesweep.engine.preview import validate_grid_targets
from phasesweep.engine.selection import (
    NoFeasibleTrialError,
    PhaseAbortRefusedError,
    _warn_unsuccessful_grid_cells,
)
from phasesweep.engine.state import PHASE_RECOVERY_ATTR
from phasesweep.engine.study_policy import _load_phase_policy_state
from phasesweep.evidence.models import LogRegexExtractor
from tests.conftest import (
    assert_advises_target,
    invoke_cli_boundary,
    make_experiment,
    phase_study,
    phase_trials,
    write_experiment_config,
    write_flaky_trainer,
    write_trainer,
)
from tests.experiments import write_experiment
from tests.ledger_fixtures import published_generation_id, winner_payload


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


@pytest.mark.parametrize("repeats", [False, True], ids=["grid", "repeat-grid"])
def test_fully_failed_grid_recovers_only_the_raised_target(tmp_path: Path, repeats: bool) -> None:
    """A recorded abort stays refused until a raised target buys failed-cell retries."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=2)
    original = _grid(tmp_path, trainer, 2, repeats=repeats)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(original)
    assert [trial.state.name for trial in phase_trials(original)] == ["FAIL", "FAIL"]
    with pytest.raises(PhaseAbortRefusedError, match="previously aborted"):
        run_experiment(original)
    assert len(calls.read_text().splitlines()) == 2

    raised = _grid(tmp_path, trainer, 3, repeats=repeats)
    validate_grid_targets(raised)
    if repeats:
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(raised)
    else:
        assert run_experiment(raised)["p"].metric == 1.0
    trials = phase_trials(raised)
    assert [trial.state.name for trial in trials] == ["FAIL", "FAIL", "COMPLETE"]
    assert trials[2].params == trials[0].params
    assert len(calls.read_text().splitlines()) == 3

    recovered = _grid(tmp_path, trainer, 4, repeats=repeats)
    assert run_experiment(recovered)["p"].metric == 1.0
    trials = phase_trials(recovered)
    assert trials[3].params == trials[1].params
    assert len(calls.read_text().splitlines()) == 4
    assert run_experiment(recovered)["p"].metric == 1.0
    with pytest.raises(
        TrialTargetRefusedError, match="COMPLETE and INFEASIBLE cells are never repeated"
    ):
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
    trials = phase_trials(raised)
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
    assert [trial.state.name for trial in phase_trials(raised)] == ["FAIL", "FAIL", "FAIL"]
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
    assert len(phase_trials(raised)) == 2
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
    trials = phase_trials(recovered)
    assert trials[2].params == trials[0].params
    assert len(calls.read_text().splitlines()) == 3


def _assert_advises_limit(message: str, above: int, left: int, target: int) -> None:
    """Assert the message tells the operator to raise phase ``p``'s failure limit in the config."""
    assert f"phases[p].max_consecutive_failures above {above} in the config" in message
    assert f"run the {left} trials still left at n_trials={target}" in message
    assert "the failure count starts again from zero" in message


def _assert_no_fresh_namespace(message: str) -> None:
    """Assert the message does not send the operator to a new workdir."""
    assert "namespace" not in message
    assert "workdir" not in message


def _assert_no_recovery(message: str) -> None:
    """Assert the message offers no setting of phase ``p`` that continues the study."""
    assert "phases[p]" not in message
    _assert_no_fresh_namespace(message)


@pytest.mark.parametrize("repeats", [False, True], ids=["grid", "repeat-grid"])
def test_failed_grid_names_the_target_that_retries_every_failed_cell(
    tmp_path: Path, repeats: bool
) -> None:
    """The abort, the identical rerun's refusal and the dry run name 4, which retries both cells."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=2)
    original = _grid(tmp_path, trainer, 2, repeats=repeats)
    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(original)
    messages = [_operational_error_text(aborted.value)]
    for dry_run in (False, True):
        with pytest.raises(PhaseAbortRefusedError, match="previously aborted") as refused:
            run_experiment(original, dry_run=dry_run)
        messages.append(_operational_error_text(refused.value))
    for message in messages:
        assert_advises_target(message, 2, 4, "retry its 2 failed cells")
        # No cell is unrun, so a higher failure limit would schedule nothing.
        assert "max_consecutive_failures above" not in message

    # The advertised target is exactly the one that retries both failed cells.
    recovered = _grid(tmp_path, trainer, 4, repeats=repeats)
    assert run_experiment(recovered)["p"].metric == 1.0
    assert [trial.state.name for trial in phase_trials(recovered)] == [
        "FAIL",
        "FAIL",
        "COMPLETE",
        "COMPLETE",
    ]
    assert len(calls.read_text().splitlines()) == 4
    # Every cell succeeded after its retry, so survivorship lost nothing.
    assert _published_survivorship(recovered) == {
        "verdict": "ok",
        "detail": {"ranked": 2, "infeasible": 0, "incomplete_group": 0, "failed": 0},
    }


@pytest.mark.parametrize(
    ("target", "launch"),
    [
        pytest.param(3, "1 trial, 1 of them a rerun", id="one-retry"),
        pytest.param(4, "2 trials, 2 of them reruns", id="both-retried"),
    ],
)
def test_phase_header_counts_the_failed_cells_a_raised_target_retries(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, target: int, launch: str
) -> None:
    """The header's reruns agrees with the RERUN lines, in the dry run as in the run."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=2)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_grid(tmp_path, trainer, 2))
    raised = _grid(tmp_path, trainer, target)
    retried = target - 2
    header = f"phase=p study=t::p finished=2 remaining={retried} reruns={retried} n_jobs=1"
    caplog.set_level(logging.INFO)

    run_experiment(raised, dry_run=True)
    assert header in [record.getMessage() for record in caplog.records]
    assert f"DRY RUN phase=p would launch {launch}" in [r.getMessage() for r in caplog.records]

    caplog.clear()
    run_experiment(raised)
    lines = [record.getMessage() for record in caplog.records]
    assert header in lines
    rerun_lines = [line.split(" reason=")[0] for line in lines if " RERUN " in line]
    assert rerun_lines == ["phase=p RERUN trial=2 of=0", "phase=p RERUN trial=3 of=1"][:retried]


def test_aborted_grid_names_the_cells_it_never_reached(tmp_path: Path) -> None:
    """An abort that left cells untried names both the target that also runs them and the limit."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=2)
    original = _grid(tmp_path, trainer, 4, cells=4)
    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(original)
    assert [trial.state.name for trial in phase_trials(original)] == ["FAIL", "FAIL"]
    with pytest.raises(PhaseAbortRefusedError, match="previously aborted") as refused:
        run_experiment(original)
    for message in (_operational_error_text(aborted.value), _operational_error_text(refused.value)):
        assert_advises_target(
            message, 4, 6, "retry its 2 failed cells and run its 2 remaining cells"
        )
        _assert_advises_limit(message, 2, 2, 4)

    recovered = _grid(tmp_path, trainer, 6, cells=4)
    assert run_experiment(recovered)["p"].metric == 1.0
    states = [trial.state.name for trial in phase_trials(recovered)]
    assert states == ["FAIL"] * 2 + ["COMPLETE"] * 4
    assert len(calls.read_text().splitlines()) == 6


def test_fully_measured_infeasible_grid_says_nothing_can_continue(tmp_path: Path) -> None:
    """Every cell completed out of bounds, so no setting continues the study and none is offered."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=0, value=5.0)
    bounded = Metric(
        extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        max=1.0,
    )
    original = _grid(tmp_path, trainer, 2, metric=bounded)
    with pytest.raises(NoFeasibleTrialError, match="2 consecutive failed/infeasible") as aborted:
        run_experiment(original)
    messages = [_operational_error_text(aborted.value)]
    for dry_run in (False, True):
        with pytest.raises(PhaseAbortRefusedError, match="previously aborted") as refused:
            run_experiment(original, dry_run=dry_run)
        messages.append(_operational_error_text(refused.value))
    for message in messages:
        assert "Every cell was measured and none is feasible" in message
        assert "change the search space or bounds" in message
        _assert_no_recovery(message)
    assert len(calls.read_text().splitlines()) == 2

    # Neither a larger limit nor a larger target reaches a study with nothing left to run.
    roomier = original.model_copy(
        update={"phases": [original.phases[0].model_copy(update={"max_consecutive_failures": 3})]}
    )
    with pytest.raises(PhaseAbortRefusedError, match="previously aborted"):
        run_experiment(roomier)
    with pytest.raises(TrialTargetRefusedError):
        run_experiment(_grid(tmp_path, trainer, 3, metric=bounded))
    assert len(calls.read_text().splitlines()) == 2


def _write_cell_trainer(tmp_path: Path) -> tuple[Path, Path]:
    """Write a trainer that logs its cell and measures out of bounds for x of 3 or more."""
    calls = tmp_path / "cells.log"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import sys
        from pathlib import Path

        x = next(int(arg.removeprefix("--x=")) for arg in sys.argv if arg.startswith("--x="))
        with Path({str(calls)!r}).open("a") as log:
            log.write(f"{{x}}\\n")
        print(f"loss={{5.0 if x >= 3 else 0.5}}")
        """,
    )
    return trainer, calls


def _untried_cells_config(tmp_path: Path, trainer: Path, *, limit: int) -> Path:
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
        workdir=str(tmp_path / "runs"),
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


def test_raised_failure_limit_runs_only_the_cells_an_infeasible_streak_left_unrun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A limit above the recorded streak retires the abort, and the dry run predicts the same."""
    caplog.set_level(logging.INFO)
    trainer, calls = _write_cell_trainer(tmp_path)

    def config(limit: int) -> Path:
        return _untried_cells_config(tmp_path, trainer, limit=limit)

    def ran() -> list[str]:
        return sorted(calls.read_text().splitlines())

    assert invoke_cli_boundary(["run", str(config(2))], monkeypatch) != 0
    aborted = capsys.readouterr().err
    assert "2 consecutive failed/infeasible trials reached max_consecutive_failures=2" in aborted
    _assert_advises_limit(aborted, 2, 2, 4)
    # The target retries nothing: both measured cells are closed.
    assert "n_trials from" not in aborted
    _assert_no_fresh_namespace(aborted)
    measured = ran()
    assert sorted(measured) == ["3", "4"]

    # A limit that is no higher than the recorded streak leaves the abort in place.
    for limit in (1, 2):
        for argv in (["run"], ["run", "--dry-run"]):
            assert invoke_cli_boundary([*argv, str(config(limit))], monkeypatch) != 0
            refused = capsys.readouterr().err
            assert "previously aborted" in refused
            _assert_advises_limit(refused, 2, 2, 4)
    assert ran() == measured

    # The dry run predicts the recovery: it launches the two unrun cells, and writes nothing.
    caplog.clear()
    assert invoke_cli_boundary(["run", str(config(3)), "--dry-run"], monkeypatch) == 0
    lines = [record.getMessage() for record in caplog.records]
    assert "phase=p study=t::p finished=2 remaining=2 reruns=0 n_jobs=1" in lines
    assert "DRY RUN phase=p would launch 2 trials" in lines
    assert ran() == measured

    recovered = config(3)
    assert invoke_cli_boundary(["run", str(recovered)], monkeypatch) == 0
    assert list((tmp_path / "runs").rglob("winner.yaml"))
    # Each cell ran exactly once: the completed ones were not repeated.
    assert ran() == ["1", "2", "3", "4"]
    experiment = load_experiment(recovered)
    trials = phase_trials(experiment)
    assert [trial.state.name for trial in trials] == ["COMPLETE"] * 4
    assert sorted(trial.params["x"] for trial in trials[2:]) == [1, 2]
    study = phase_study(experiment)
    assert _load_phase_policy_state(study).abort is None
    assert study.user_attrs[PHASE_RECOVERY_ATTR]["trial_target"] == 4

    # The recovery spent no further work, and a later lower limit never reinterprets its outcomes.
    for limit in (3, 2):
        assert invoke_cli_boundary(["run", str(config(limit))], monkeypatch) == 0
    assert ran() == ["1", "2", "3", "4"]
    capsys.readouterr()
    assert invoke_cli_boundary(["show-winners", str(recovered)], monkeypatch) == 0
    assert "=== p ===" in capsys.readouterr().out


def test_failure_streak_refusal_names_the_grid_retry_target(tmp_path: Path) -> None:
    """A resumed streak at the limit quotes the same target, beside the limit alternative."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=2)
    original = _grid(tmp_path, trainer, 4, cells=4)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(original)
    phase = original.phases[0]
    streak = phase_ops._failure_policy_abort_record(
        phase, consecutive_failures=2, completion_sequence=2, trial_target=4
    )

    with pytest.raises(PhaseAbortRefusedError) as refused:
        phase_ops._raise_streak_at_limit(phase, streak, phase_study(original))
    message = _operational_error_text(refused.value)
    assert "has 2 consecutive failed/infeasible trials" in message
    assert "meets max_consecutive_failures=2" in message
    assert "at its accepted n_trials=4" in message
    assert_advises_target(message, 4, 6, "retry its 2 failed cells and run its 2 remaining cells")
    _assert_advises_limit(message, 2, 2, 4)


_FRESH_START = (
    "The recorded trial results stay in the study; if a defect you have since fixed "
    "produced them, set a new `experiment` name in the config to start fresh. "
    "Changing metric.min, metric.max, constraints, or the search space changes the study "
    "identity and requires a new `experiment` name."
)


def _out_of_bounds_metric() -> Metric:
    """Return a ``x=<value>`` metric whose upper bound a logged 5.0 exceeds."""
    return Metric(
        extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        max=1.0,
    )


def test_infeasible_streak_advice_says_recorded_results_stay_unless_the_experiment_is_renamed(
    tmp_path: Path,
) -> None:
    """Fixing the trainer does not re-measure cells it mis-measured, so every refusal says so."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=0, value=5.0)
    original = _grid(tmp_path, trainer, 4, cells=4, metric=_out_of_bounds_metric())
    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(original)
    assert [trial.state.name for trial in phase_trials(original)] == ["COMPLETE", "COMPLETE"]
    with pytest.raises(PhaseAbortRefusedError, match="previously aborted") as refused:
        run_experiment(original)
    phase = original.phases[0]
    streak = phase_ops._failure_policy_abort_record(
        phase, consecutive_failures=2, completion_sequence=2, trial_target=4
    )
    with pytest.raises(PhaseAbortRefusedError) as at_limit:
        phase_ops._raise_streak_at_limit(phase, streak, phase_study(original))

    for message in map(_operational_error_text, (aborted.value, refused.value, at_limit.value)):
        _assert_advises_limit(message, 2, 2, 4)
        assert message.endswith(_FRESH_START)
        _assert_no_fresh_namespace(message)
    headline = str(aborted.value)
    assert "stderr" not in headline
    assert (
        "The tripping trials were INFEASIBLE: outside the metric bounds or constraints." in headline
    )


def test_failed_streak_abort_points_at_the_logs_even_when_an_infeasible_trial_is_in_it(
    tmp_path: Path,
) -> None:
    """One trainer failure among the tripping trials leaves a stderr log worth reading."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=1, value=5.0)
    original = _grid(tmp_path, trainer, 4, cells=4, metric=_out_of_bounds_metric())
    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(original)
    assert [trial.state.name for trial in phase_trials(original)] == ["FAIL", "COMPLETE"]
    message = _operational_error_text(aborted.value)
    assert " for stderr logs." in message
    assert "completed outside the metric bounds" not in message
    assert message.endswith(_FRESH_START)


def test_sampling_phase_asks_for_a_higher_target_without_a_retry_rule(tmp_path: Path) -> None:
    """Random draws have no per-cell retry rule, so any target above the aborted one recovers."""
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
    with pytest.raises(PhaseAbortRefusedError, match="previously aborted") as refused:
        run_experiment(experiment)
    for message in (_operational_error_text(aborted.value), _operational_error_text(refused.value)):
        assert "phases[p].n_trials above 2 in the config" in message
        assert "the failure count starts again from zero" in message
        # Both trials are spent, so a higher limit would schedule nothing.
        assert "max_consecutive_failures above" not in message


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
    trials = phase_trials(raised)
    assert [trial.state.name for trial in trials] == ["FAIL", "FAIL", "COMPLETE"]
    expected = (
        "[p] only 1 of 2 grid cells succeeded (advisory); the winner was chosen without a "
        f"successful trial for: {trials[1].params}."
    )
    assert _coverage_warnings(caplog) == [expected]

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        assert run_experiment(raised)["p"].metric == 1.0
    assert _coverage_warnings(caplog) == []


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
    succeeded = {
        trial.params["x"] for trial in phase_trials(raised) if trial.state.name == "COMPLETE"
    }
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
    (failed,) = (trial for trial in phase_trials(experiment) if trial.state.name == "FAIL")
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


def _published_survivorship(experiment: Experiment) -> dict[str, object]:
    """Read the survivorship check that phase ``p``'s published winner.yaml records."""
    generation_id = published_generation_id(experiment)
    assert generation_id is not None
    confound = winner_payload(experiment, generation_id)["confound"]
    assert isinstance(confound, dict)
    checks = confound["checks"]
    assert isinstance(checks, dict)
    survivorship = checks["survivorship"]
    assert isinstance(survivorship, dict)
    return survivorship


@pytest.mark.parametrize(
    ("failures", "target", "failed"),
    [
        pytest.param(2, 3, 1, id="cell-never-retried"),
        pytest.param(3, 4, 2, id="cell-retried-and-failed-again"),
    ],
)
def test_survivorship_still_counts_the_failures_of_a_cell_that_never_succeeded(
    tmp_path: Path, failures: int, target: int, failed: int
) -> None:
    """Replacing a failure discounts it only when a later attempt of its cell succeeded."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=failures)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_grid(tmp_path, trainer, 2))
    if failures == 3:
        with pytest.raises(NoFeasibleTrialError):
            run_experiment(_grid(tmp_path, trainer, 3))
    raised = _grid(tmp_path, trainer, target)
    assert run_experiment(raised)["p"].metric == 1.0
    succeeded = {
        trial.params["x"] for trial in phase_trials(raised) if trial.state.name == "COMPLETE"
    }
    assert len(succeeded) == 1

    assert _published_survivorship(raised) == {
        "verdict": "heterogeneous",
        "detail": {"ranked": 1, "infeasible": 0, "incomplete_group": 0, "failed": failed},
    }


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


def test_validate_refuses_an_oversized_fresh_persistent_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validation applies the launch ceiling before a fresh grid can spend any trials."""
    experiment = _grid(tmp_path, tmp_path / "trainer.py", 3)
    path = tmp_path / "grid.yaml"
    write_experiment_config(path, experiment)
    assert invoke_cli_boundary(["validate", str(path)], monkeypatch) == 1
    error = capsys.readouterr().err
    assert "fix:" in error
    assert "Set phases[p].n_trials to at most 2" in error
    assert "correct the experiment configuration" not in error
