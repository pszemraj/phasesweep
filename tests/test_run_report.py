"""The live run's report: exact text of each block at 80 and 60 columns, and its one-line form."""

from __future__ import annotations

import pytest

from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import IntParam
from phasesweep.engine.state import Winner
from phasesweep.run_report import (
    PhaseStart,
    phase_start,
    rerun_line,
    skipped_line,
    trial_line,
    winner_line,
)
from phasesweep.runtime.terminal import ReportBlock
from tests.conftest import FAKE_TRAIN_METRIC, make_experiment

STAMP = "12:00:00  "
OOM = "non-zero exit code 1; stderr: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"


def console(block: ReportBlock, width: int = 80) -> list[str]:
    """Draw ``block`` as the console prints it: the time before the first line, blanks after."""
    lines = block.render(width - len(STAMP), color=False)
    return [(STAMP if index == 0 else " " * len(STAMP)) + line for index, line in enumerate(lines)]


@pytest.fixture
def experiment() -> Experiment:
    """Two phases whose names are 5 and 13 characters wide, objective ``eval_loss``."""
    space = {"x": IntParam(type="int", low=0, high=10)}
    return make_experiment(
        metric=FAKE_TRAIN_METRIC,
        phases=[
            Phase(name="depth", n_trials=4, search_space=space),
            Phase(name="learning_rate", n_trials=4, search_space=space),
        ],
    )


def test_phase_start_says_what_will_run(experiment: Experiment) -> None:
    """A fresh phase names its trials and parallelism; finished and rerun counts appear when nonzero."""
    depth, lr = experiment.phases
    fresh = phase_start(experiment, depth, finished=0, remaining=4, reruns=0)
    assert console(fresh) == ["12:00:00  depth          4 trials to run, 1 at a time"]
    assert str(fresh) == "depth  4 trials to run, 1 at a time"

    resumed = phase_start(experiment, lr, finished=2, remaining=2, reruns=2)
    assert console(resumed) == [
        "12:00:00  learning_rate  2 trials to run, 2 of them reruns, 1 at a time,",
        "                         2 already finished",
    ]
    assert str(resumed) == (
        "learning_rate  2 trials to run, 2 of them reruns, 1 at a time, 2 already finished"
    )
    assert console(resumed, 60) == [
        "12:00:00  learning_rate  2 trials to run, 2 of them reruns,",
        "                         1 at a time, 2 already finished",
    ]

    rerun = phase_start(experiment, depth, finished=3, remaining=1, reruns=1)
    assert str(rerun) == "depth  1 trial to run, 1 of them a rerun, 1 at a time, 3 already finished"

    replay = phase_start(experiment, depth, finished=4, remaining=0, reruns=0)
    assert console(replay) == [
        "12:00:00  depth          nothing to run, 4 trials already finished,",
        "                         replaying the recorded result",
    ]
    assert (
        str(replay)
        == "depth  nothing to run, 4 trials already finished, replaying the recorded result"
    )


def test_trial_params_share_the_line_only_when_it_fits(experiment: Experiment) -> None:
    """The configuration sits beside the trial when the whole line fits, else under its text."""
    depth, lr = experiment.phases
    trial = trial_line(
        experiment,
        depth,
        number=0,
        state="COMPLETE",
        value=0.3,
        seconds=0.5,
        finished=1,
        params={"model.n_layers": 8},
        repeat=None,
        rerun_of=None,
        reason=None,
    )
    assert console(trial, 120) == [
        "12:00:00  depth          trial 0  COMPLETE  eval_loss 0.3  0.5s  1/4  model.n_layers=8"
    ]
    assert console(trial) == [
        "12:00:00  depth          trial 0  COMPLETE  eval_loss 0.3  0.5s  1/4",
        "                         model.n_layers=8",
    ]
    assert str(trial) == "depth  trial 0  COMPLETE  eval_loss 0.3  0.5s  1/4  model.n_layers=8"

    wide = trial_line(
        experiment,
        lr,
        number=1,
        state="COMPLETE",
        value=123456789.0,
        seconds=61.04,
        finished=2,
        params={"optimizer.lr": 0.00031234, "model.n_layers": 8},
        repeat=("seed", 1),
        rerun_of=0,
        reason=None,
    )
    assert console(wide) == [
        "12:00:00  learning_rate  trial 1  COMPLETE  eval_loss 123456789  61.0s  2/4",
        "                         optimizer.lr=0.0003123  model.n_layers=8  repeat seed=1",
        "                         rerun of 0",
    ]
    assert str(wide) == (
        "learning_rate  trial 1  COMPLETE  eval_loss 123456789  61.0s  2/4  optimizer.lr=0.0003123"
        "  model.n_layers=8  repeat seed=1  rerun of 0"
    )


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        pytest.param(
            80,
            [
                "12:00:00  depth          trial 3  FAIL  0.5s  3/4  model.n_layers=8",
                "                         reason: non-zero exit code 1; stderr:",
                "                         torch.OutOfMemoryError: CUDA out of memory. Tried to",
                "                         allocate 2.00 GiB",
            ],
            id="80",
        ),
        pytest.param(
            60,
            [
                "12:00:00  depth          trial 3  FAIL  0.5s  3/4",
                "                         model.n_layers=8",
                "                         reason: non-zero exit code 1;",
                "                         stderr: torch.OutOfMemoryError:",
                "                         CUDA out of memory. Tried to",
                "                         allocate 2.00 GiB",
            ],
            id="60",
        ),
    ],
)
def test_failed_trial_wraps_its_reason_to_the_terminal(
    experiment: Experiment, width: int, expected: list[str]
) -> None:
    """A failure has no objective, and its cause wraps under the text without overrunning."""
    failed = trial_line(
        experiment,
        experiment.phases[0],
        number=3,
        state="FAIL",
        value=None,
        seconds=0.5,
        finished=3,
        params={"model.n_layers": 8},
        repeat=None,
        rerun_of=None,
        reason=OOM,
    )
    assert console(failed, width) == expected
    assert all(len(line) <= width for line in console(failed, width))
    assert str(failed) == f"depth  trial 3  FAIL  0.5s  3/4  model.n_layers=8  reason: {OOM}"


def test_infeasible_and_interrupted_trials_keep_what_was_recorded(experiment: Experiment) -> None:
    """An infeasible trial shows its measured objective; an interrupted one has none and no time."""
    depth = experiment.phases[0]
    infeasible = trial_line(
        experiment,
        depth,
        number=2,
        state="INFEASIBLE",
        value=52.5,
        seconds=0.5,
        finished=3,
        params={"model.n_layers": 8},
        repeat=None,
        rerun_of=None,
        reason="objective 52.5 > max=50",
    )
    assert console(infeasible) == [
        "12:00:00  depth          trial 2  INFEASIBLE  eval_loss 52.5  0.5s  3/4",
        "                         model.n_layers=8",
        "                         reason: objective 52.5 > max=50",
    ]
    interrupted = trial_line(
        experiment,
        depth,
        number=4,
        state="INTERRUPTED",
        value=None,
        seconds=None,
        finished=3,
        params={"model.n_layers": 8},
        repeat=None,
        rerun_of=None,
        reason="orchestrator interrupted by signal 2",
    )
    assert str(interrupted) == (
        "depth  trial 4  INTERRUPTED  3/4  model.n_layers=8  "
        "reason: orchestrator interrupted by signal 2"
    )


def test_rerun_and_skipped_lines(experiment: Experiment) -> None:
    """A rerun names its source and that trial's cause; a skipped phase says where its winner is."""
    depth = experiment.phases[0]
    rerun = rerun_line(
        experiment, depth, number=4, source=2, reason="orchestrator interrupted by signal 2"
    )
    assert console(rerun) == [
        "12:00:00  depth          trial 4  RERUN of trial 2",
        "                         reason: orchestrator interrupted by signal 2",
    ]
    assert (
        str(rerun)
        == "depth  trial 4  RERUN of trial 2  reason: orchestrator interrupted by signal 2"
    )
    assert console(rerun_line(experiment, depth, number=5, source=3, reason=None)) == [
        "12:00:00  depth          trial 5  RERUN of trial 3"
    ]

    skipped = skipped_line(experiment, depth, "no published or required winner")
    assert console(skipped) == ["12:00:00  depth          skipped: no published or required winner"]
    assert console(skipped, 60) == [
        "12:00:00  depth          skipped: no published or required",
        "                         winner",
    ]
    assert str(skipped) == "depth  skipped: no published or required winner"


def test_winner_line_forms(experiment: Experiment) -> None:
    """The winner shows its trial, objective and configuration; repeats and a partial result add theirs."""
    depth, lr = experiment.phases
    plain = winner_line(
        experiment,
        depth,
        Winner(trial_number=0, params={"model.n_layers": 8}, effective_overrides={}, metric=0.3),
    )
    assert console(plain) == [
        "12:00:00  depth          winner: trial 0  eval_loss 0.3  model.n_layers=8"
    ]
    assert str(plain) == "depth  winner: trial 0  eval_loss 0.3  model.n_layers=8"

    repeats = {
        "key": "seed",
        "reduce": "mean",
        "spread": {"n": 3, "mean": 0.31, "sd": 0.012345678, "min": 0.3, "max": 0.32},
        "members": [{"trial_number": 5}, {"trial_number": 6}, {"trial_number": 7}],
    }
    partial = {
        "incomplete": True,
        "finished_trials": 3,
        "requested_trials": 4,
        "timeout_scope": "phase",
    }
    winner = Winner(
        trial_number=7,
        params={"model.n_layers": 8, "optimizer.lr": 0.00031234},
        effective_overrides={},
        metric=0.31,
        completion=partial,
        repeats=repeats,
    )
    block = winner_line(experiment, lr, winner)
    assert console(block) == [
        "12:00:00  learning_rate  winner: trial 7  eval_loss 0.31",
        "                         INCOMPLETE 3/4 finished  timeout scope phase",
        "                         model.n_layers=8  optimizer.lr=0.0003123",
        "                         mean of 3 repeats of seed  sd 0.01235  range 0.3..0.32",
        "                         members 5, 6, 7",
    ]
    narrow = console(block, 60)
    assert narrow == [
        "12:00:00  learning_rate  winner: trial 7  eval_loss 0.31",
        "                         INCOMPLETE 3/4 finished",
        "                         timeout scope phase",
        "                         model.n_layers=8",
        "                         optimizer.lr=0.0003123",
        "                         mean of 3 repeats of seed",
        "                         sd 0.01235  range 0.3..0.32",
        "                         members 5, 6, 7",
    ]
    assert all(len(line) <= 60 for line in narrow)
    assert str(block) == (
        "learning_rate  winner: trial 7  eval_loss 0.31  INCOMPLETE 3/4 finished  "
        "timeout scope phase  model.n_layers=8  optimizer.lr=0.0003123  mean of 3 repeats of seed  "
        "sd 0.01235  range 0.3..0.32  members 5, 6, 7"
    )


def test_every_block_is_a_one_line_log_message(experiment: Experiment) -> None:
    """``str()`` never breaks a line, so a ``run.log`` record stays one line."""
    blocks: list[ReportBlock] = [
        PhaseStart("p", 1, 2, 0, 0, 1),
        phase_start(experiment, experiment.phases[0], finished=0, remaining=4, reruns=0),
        rerun_line(experiment, experiment.phases[0], number=1, source=0, reason="cause"),
    ]
    assert all("\n" not in str(block) for block in blocks)
