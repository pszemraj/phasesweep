"""Raised persistent grid targets retry failed cells without repeating completed cells.

Recovery messages name the target that retries every failed cell, and a winner chosen while
some cell has no result, failed or never tried, says so through its survivorship check.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import phasesweep.engine.phase as phase_ops
from phasesweep import load_experiment, run_experiment
from phasesweep.cli import _operational_error_text
from phasesweep.config.models import Experiment, Metric, Phase
from phasesweep.config.search import IntParam, Repeats, Sampler
from phasesweep.engine.errors import TrialTargetRefusedError
from phasesweep.engine.preview import validate_trial_targets
from phasesweep.engine.selection import (
    NoFeasibleTrialError,
    PhaseAbortRefusedError,
    _streak_at_limit_error,
)
from phasesweep.engine.state import PHASE_RECOVERY_ATTR
from phasesweep.engine.study_policy import _failure_policy_abort_record, _load_phase_policy_state
from phasesweep.errors import REJUDGE_BOUNDS_STEP
from phasesweep.evidence.models import LogRegexExtractor
from tests.conftest import (
    assert_advises_target,
    invoke_cli_boundary,
    make_experiment,
    phase_study,
    phase_trials,
    study_name,
    write_experiment_config,
    write_flaky_trainer,
    write_trainer,
)
from tests.experiments import write_experiment
from tests.ledger_fixtures import published_survivorship


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
    validate_trial_targets(raised)
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
    with pytest.raises(TrialTargetRefusedError, match="allow at most 4") as refused:
        run_experiment(_grid(tmp_path, trainer, 5, repeats=repeats))
    # 4 is both the accepted target and the ceiling, so the step names it exactly.
    assert refused.value.remediation == "Set phases[p].n_trials to 4."
    assert len(calls.read_text().splitlines()) == 4


@pytest.mark.parametrize("repeats", [False, True], ids=["grid", "repeat-grid"])
def test_partly_failed_grid_retries_the_failed_cell_only(tmp_path: Path, repeats: bool) -> None:
    """A successful cell closes even when another cell needs an extra attempt."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=1)
    original = _grid(tmp_path, trainer, 2, repeats=repeats)
    if repeats:
        with pytest.raises(NoFeasibleTrialError) as refused:
            run_experiment(original)
        assert "raise phases[p].n_trials from 2 to 3" in str(refused.value.remediation)
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


def _assert_advises_limit(message: str, above: int, left: int) -> None:
    """Assert the message tells the operator to raise phase ``p``'s failure limit."""
    assert f"phases[p].max_consecutive_failures above {above}" in message
    assert f"run the {left} trials still left" in message


def _assert_rejudge_remedy(message: str) -> None:
    """Assert the message says a corrected bound re-judges the recorded trials, and never sends an infeasible streak to a fresh study."""
    assert REJUDGE_BOUNDS_STEP in message
    assert "new `experiment` name" not in message


def _assert_no_recovery(message: str) -> None:
    """Assert the message offers no setting of phase ``p`` that continues the study."""
    assert "phases[p]" not in message
    _assert_rejudge_remedy(message)


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
    assert published_survivorship(recovered) == {
        "verdict": "ok",
        "detail": {
            "ranked": 1 if repeats else 2,
            "infeasible": 0,
            "failed": 0,
            "untried": 0,
            "failed_trials": [],
        },
    }


@pytest.mark.parametrize(
    ("target", "launch", "started"),
    [
        pytest.param(
            3,
            "1 trial, 1 of them a rerun",
            "1 trial to run, 1 of them a rerun, 1 at a time, 2 already finished",
            id="one-retry",
        ),
        pytest.param(
            4,
            "2 trials, 2 of them reruns",
            "2 trials to run, 2 of them reruns, 1 at a time, 2 already finished",
            id="both-retried",
        ),
    ],
)
def test_phase_header_counts_the_failed_cells_a_raised_target_retries(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, target: int, launch: str, started: str
) -> None:
    """The header's reruns agrees with the RERUN lines, in the dry run as in the run."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=2)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_grid(tmp_path, trainer, 2))
    raised = _grid(tmp_path, trainer, target)
    retried = target - 2
    header = (
        f"phase=p study={study_name(raised)} finished=2 remaining={retried} reruns={retried} "
        "n_jobs=1"
    )
    caplog.set_level(logging.INFO)

    run_experiment(raised, dry_run=True)
    assert header in [record.getMessage() for record in caplog.records]
    assert f"DRY RUN phase=p would launch {launch}" in [r.getMessage() for r in caplog.records]

    caplog.clear()
    run_experiment(raised)
    lines = [record.getMessage() for record in caplog.records]
    assert f"p  {started}" in lines
    rerun_lines = [line.split("  reason: ")[0] for line in lines if "  RERUN of " in line]
    assert rerun_lines == ["p  trial 2  RERUN of trial 0", "p  trial 3  RERUN of trial 1"][:retried]


def test_aborted_grid_names_the_cells_it_never_reached(tmp_path: Path) -> None:
    """An abort that left cells untried names the target that retries the failed ones and runs the rest."""
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
        # The raised target is the one step; the limit is offered only when no target can help.
        assert "max_consecutive_failures above" not in message

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
    _assert_advises_limit(aborted, 2, 2)
    # The target retries nothing: both measured cells are closed.
    assert "n_trials from" not in aborted
    _assert_rejudge_remedy(aborted)
    measured = ran()
    assert sorted(measured) == ["3", "4"]

    # A limit that is no higher than the recorded streak leaves the abort in place.
    for limit in (1, 2):
        for argv in (["run"], ["run", "--dry-run"]):
            assert invoke_cli_boundary([*argv, str(config(limit))], monkeypatch) != 0
            refused = capsys.readouterr().err
            assert "previously aborted" in refused
            _assert_advises_limit(refused, 2, 2)
    assert ran() == measured

    # The dry run predicts the recovery: it launches the two unrun cells, and writes nothing.
    caplog.clear()
    assert invoke_cli_boundary(["run", str(config(3)), "--dry-run"], monkeypatch) == 0
    lines = [record.getMessage() for record in caplog.records]
    study = study_name(load_experiment(config(3)))
    assert f"phase=p study={study} finished=2 remaining=2 reruns=0 n_jobs=1" in lines
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
    assert _load_phase_policy_state(study, experiment).abort is None
    assert study.user_attrs[PHASE_RECOVERY_ATTR]["trial_target"] == 4

    # The recovery spent no further work, and a later lower limit never reinterprets its outcomes.
    for limit in (3, 2):
        assert invoke_cli_boundary(["run", str(config(limit))], monkeypatch) == 0
    assert ran() == ["1", "2", "3", "4"]
    capsys.readouterr()
    assert invoke_cli_boundary(["show-winners", str(recovered)], monkeypatch) == 0
    assert "=== p ===" in capsys.readouterr().out


def test_failure_streak_refusal_names_the_grid_retry_target(tmp_path: Path) -> None:
    """A resumed streak at the limit quotes the same target as the abort."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=2)
    original = _grid(tmp_path, trainer, 4, cells=4)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(original)
    phase = original.phases[0]
    streak = _failure_policy_abort_record(
        phase, consecutive_failures=2, completion_sequence=2, trial_target=4
    )

    refused = _streak_at_limit_error(original, phase, streak, phase_study(original))
    message = _operational_error_text(refused)
    assert "has 2 consecutive failed/infeasible trials" in message
    assert "meets max_consecutive_failures=2" in message
    assert_advises_target(message, 4, 6, "retry its 2 failed cells and run its 2 remaining cells")


def _out_of_bounds_metric() -> Metric:
    """Return a ``x=<value>`` metric whose upper bound a logged 5.0 exceeds."""
    return Metric(
        extractor=LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        max=1.0,
    )


def test_infeasible_streak_advice_says_a_bounds_change_rejudges_the_trials(
    tmp_path: Path,
) -> None:
    """Every refusal of an infeasible streak says a corrected bound re-judges the recorded trials, then gives the step that keeps the bounds."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=0, value=5.0)
    original = _grid(tmp_path, trainer, 4, cells=4, metric=_out_of_bounds_metric())
    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(original)
    assert [trial.state.name for trial in phase_trials(original)] == ["COMPLETE", "COMPLETE"]
    with pytest.raises(PhaseAbortRefusedError, match="previously aborted") as refused:
        run_experiment(original)
    phase = original.phases[0]
    streak = _failure_policy_abort_record(
        phase, consecutive_failures=2, completion_sequence=2, trial_target=4
    )
    at_limit = _streak_at_limit_error(original, phase, streak, phase_study(original))

    for message in map(_operational_error_text, (aborted.value, refused.value, at_limit)):
        _assert_advises_limit(message, 2, 2)
        assert message.index("outside the metric bounds or constraints.") < message.index(
            REJUDGE_BOUNDS_STEP
        )
        assert "Fix the cause" not in message
        assert message.index(REJUDGE_BOUNDS_STEP) < message.index(
            "Otherwise, raise phases[p].max_consecutive_failures"
        )
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
    assert "Fix the cause, then raise phases[p].n_trials from 4 to 5" in message


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
        assert "phases[p].n_trials above 2" in message
        # Both trials are spent, so a higher limit would schedule nothing.
        assert "max_consecutive_failures above" not in message


def test_infeasible_sampling_streak_offers_the_raised_target_under_unchanged_limits(
    tmp_path: Path,
) -> None:
    """An infeasible streak leads with the re-judge step for a bounds change, then the target raise that keeps the bounds."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=0, value=5.0)
    phase = Phase(
        name="p",
        n_trials=2,
        max_consecutive_failures=2,
        sampler=Sampler(type="random", seed=0),
        search_space={"x": IntParam(type="int", low=1, high=9)},
    )
    experiment = make_experiment(
        persistent=tmp_path, trainer=trainer, phases=[phase], metric=_out_of_bounds_metric()
    )
    with pytest.raises(NoFeasibleTrialError) as aborted:
        run_experiment(experiment)
    message = _operational_error_text(aborted.value)
    assert "Otherwise, raise phases[p].n_trials above 2" in message
    assert "Fix the cause" not in message
    assert message.index(REJUDGE_BOUNDS_STEP) < message.index("Otherwise, raise")


def _survivorship_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the selection warnings that carry a flagged survivorship check."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "phasesweep.engine.selection" and "survivorship:" in record.message
    ]


def test_winner_from_a_partly_failed_grid_names_the_cell_that_never_succeeded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A raised target that retries one of two failed cells publishes, and the log names the other."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=2)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_grid(tmp_path, trainer, 2))
    raised = _grid(tmp_path, trainer, 3)

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        assert run_experiment(raised)["p"].metric == 1.0
    trials = phase_trials(raised)
    assert [trial.state.name for trial in trials] == ["FAIL", "FAIL", "COMPLETE"]
    (warning,) = _survivorship_warnings(caplog)
    assert (
        f"survivorship: 1 of 2 configurations has no result (trial 1 x={trials[1].params['x']} "
        "failed)"
    ) in warning

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        assert run_experiment(raised)["p"].metric == 1.0
    assert _survivorship_warnings(caplog) == []


def test_grid_winner_counts_the_cell_a_raised_target_never_reached(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Failed-cell retries run before unvisited cells, so a finished target can skip a cell."""
    trainer, _calls = write_flaky_trainer(tmp_path, failures=2)
    with pytest.raises(NoFeasibleTrialError):
        run_experiment(_grid(tmp_path, trainer, 4, cells=4))
    raised = _grid(tmp_path, trainer, 5, cells=4)

    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        assert run_experiment(raised)["p"].metric == 1.0
    trials = phase_trials(raised)
    assert [trial.state.name for trial in trials] == ["FAIL", "FAIL", *["COMPLETE"] * 3]
    (missing,) = {1, 2, 3, 4} - {trial.params["x"] for trial in trials}
    assert published_survivorship(raised) == {
        "verdict": "heterogeneous",
        "detail": {"ranked": 3, "infeasible": 0, "failed": 0, "untried": 1, "failed_trials": []},
    }
    (warning,) = _survivorship_warnings(caplog)
    assert f"survivorship: 1 of 4 configurations has no result (x={missing} was never tried)" in (
        warning
    )

    # The documented retry runs the cell that was never tried, and nothing flags afterwards.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="phasesweep.engine.selection"):
        assert run_experiment(_grid(tmp_path, trainer, 6, cells=4))["p"].metric == 1.0
    assert {trial.params["x"] for trial in phase_trials(raised)} == {1, 2, 3, 4}
    assert _survivorship_warnings(caplog) == []


def test_repeat_grid_winner_names_the_group_its_failed_member_left_without_a_result(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A repeat group that lost a member cannot rank; the warning names the member's params."""
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
    (warning,) = _survivorship_warnings(caplog)
    assert (
        f"survivorship: 1 of 2 configurations has no result (trial {failed.number} "
        f"x={failed.params['x']} seed={failed.params['seed']} failed)"
    ) in warning


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
    assert _survivorship_warnings(caplog) == []


@pytest.mark.parametrize(
    ("failures", "target"),
    [
        pytest.param(2, 3, id="cell-never-retried"),
        pytest.param(3, 4, id="cell-retried-and-failed-again"),
    ],
)
def test_survivorship_still_counts_the_failures_of_a_cell_that_never_succeeded(
    tmp_path: Path, failures: int, target: int
) -> None:
    """A cell counts once: by a successful retry, else by its latest failed attempt."""
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
    attempts = [
        trial.number for trial in phase_trials(raised) if trial.params["x"] not in succeeded
    ]
    assert len(attempts) == failures - 1
    failed_trials = [max(attempts)]

    assert published_survivorship(raised) == {
        "verdict": "heterogeneous",
        "detail": {
            "ranked": 1,
            "infeasible": 0,
            "failed": 1,
            "untried": 0,
            "failed_trials": failed_trials,
        },
    }


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
    assert str(aborted.value).endswith("/p/attempts for stderr logs.")


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
    diagnosis = error.splitlines()[0].removeprefix("phasesweep: ")
    remedy = next(
        line.removeprefix("fix: ") for line in error.splitlines() if line.startswith("fix: ")
    )
    assert diagnosis == "Phase 'p': n_trials=3 exceeds the grid cardinality 2."
    assert diagnosis not in remedy
    assert "available attempts" not in error
    assert "correct the experiment configuration" not in error
    assert "new `experiment` name" not in error
    assert "old study" not in error


@pytest.mark.integration
def test_oversized_grid_target_over_a_recorded_study_names_the_target_it_takes(
    tmp_path: Path,
) -> None:
    """A target past a measured grid names the one target it takes and launches nothing."""
    trainer, calls = write_flaky_trainer(tmp_path, failures=0)
    assert run_experiment(_grid(tmp_path, trainer, 2))["p"].metric == 1.0
    with pytest.raises(TrialTargetRefusedError) as refused:
        run_experiment(_grid(tmp_path, trainer, 4))
    assert refused.value.remediation == "Set phases[p].n_trials to 2."
    assert len(calls.read_text().splitlines()) == 2
