"""The live run's report: exact text of each block at 80 and 60 columns, and its one-line form."""

from __future__ import annotations

import io
import logging
import re
import shlex
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.cli import _console_logging, _rerun_command, _resume_hint
from phasesweep.config.models import ExecutionContext, Experiment, Metric, Phase
from phasesweep.config.search import CategoricalParam, IntParam, Repeats, Sampler
from phasesweep.engine.ledger import validate_ledger
from phasesweep.engine.paths import _phase_dir
from phasesweep.engine.read import read_trial_tables
from phasesweep.engine.run import _phase_counts
from phasesweep.engine.selection import PhaseTrialsFailedError
from phasesweep.engine.state import GENERATION_ID_ATTR, TRIAL_OUTCOME_ATTR, Winner
from phasesweep.errors import UnsafeProcessCleanupError, operator_step
from phasesweep.evidence.models import RequiredFileGate, WandbExtractor
from phasesweep.run_report import (
    LogLine,
    banner,
    phase_start,
    rerun_line,
    skipped_line,
    trial_line,
    winner_line,
)
from phasesweep.runtime.console import ConsoleHandler
from phasesweep.runtime.gpu import _HOST_LOCK_WAIT
from phasesweep.runtime.shutdown import PhaseSweepShutdown, ShutdownCleanupReport
from phasesweep.runtime.terminal import REPORT_LOGGER, ReportBlock, ReportText
from phasesweep.tracking.models import MlflowSettings
from tests.conftest import (
    FAKE_TRAIN_METRIC,
    grid_experiment,
    invoke_cli_boundary,
    make_experiment,
    phase_study,
    run_log_text,
    stub_run_command,
    trial_outcome,
    write_constant_trainer,
    write_experiment_config,
    write_param_echo_trainer,
)

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

    many = phase_start(experiment, depth, finished=0, remaining=1500, reruns=1200)
    assert str(many) == "depth  1,500 trials to run, 1,200 of them reruns, 1 at a time"

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
                "                                 torch.OutOfMemoryError: CUDA out of memory.",
                "                                 Tried to allocate 2.00 GiB",
            ],
            id="80",
        ),
        pytest.param(
            60,
            [
                "12:00:00  depth          trial 3  FAIL  0.5s  3/4",
                "                         model.n_layers=8",
                "                         reason: non-zero exit code 1;",
                "                                 stderr:",
                "                                 torch.OutOfMemoryError:",
                "                                 CUDA out of memory. Tried",
                "                                 to allocate 2.00 GiB",
            ],
            id="60",
        ),
    ],
)
def test_failed_trial_wraps_its_reason_to_the_terminal(
    experiment: Experiment, width: int, expected: list[str]
) -> None:
    """A failure has no objective, and its cause hangs under the text after ``reason:``."""
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
        reason="orchestrator interrupted by SIGINT",
    )
    assert str(interrupted) == (
        "depth  trial 4  INTERRUPTED  3/4  model.n_layers=8  "
        "reason: orchestrator interrupted by SIGINT"
    )


def test_rerun_and_skipped_lines(experiment: Experiment) -> None:
    """A rerun names its source and that trial's cause; a skipped phase says where its winner is."""
    depth = experiment.phases[0]
    rerun = rerun_line(
        experiment, depth, number=4, source=2, reason="orchestrator interrupted by SIGINT"
    )
    assert console(rerun) == [
        "12:00:00  depth          trial 4  RERUN of trial 2",
        "                         reason: orchestrator interrupted by SIGINT",
    ]
    assert (
        str(rerun) == "depth  trial 4  RERUN of trial 2  reason: orchestrator interrupted by SIGINT"
    )
    assert console(rerun_line(experiment, depth, number=5, source=3, reason=None)) == [
        "12:00:00  depth          trial 5  RERUN of trial 3"
    ]
    assert console(rerun_line(experiment, depth, number=4, source=2, reason=OOM), 60) == [
        "12:00:00  depth          trial 4  RERUN of trial 2",
        "                         reason: non-zero exit code 1;",
        "                                 stderr:",
        "                                 torch.OutOfMemoryError:",
        "                                 CUDA out of memory. Tried",
        "                                 to allocate 2.00 GiB",
    ]

    skipped = skipped_line(experiment, depth, "no published or required winner")
    assert console(skipped) == ["12:00:00  depth          skipped: no published or required winner"]
    assert console(skipped, 60) == [
        "12:00:00  depth          skipped: no published or required",
        "                                  winner",
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


class FakeStderr(io.StringIO):
    """A stderr whose terminal-ness the test chooses."""

    def __init__(self, *, tty: bool = False) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


NOON = time.mktime((2026, 1, 1, 12, 0, 0, 0, 0, -1))


def record(
    name: str,
    level: int,
    message: object,
    *args: object,
    exc_info: logging._SysExcInfoType | None = None,
) -> logging.LogRecord:
    """Build a record created at noon, as a logger would hand the handler."""
    made = logging.LogRecord(name, level, __file__, 0, message, args, exc_info)
    made.created = NOON
    return made


@pytest.fixture
def redirect_stderr(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeStderr]:
    """Give a test a function that swaps ``sys.stderr`` for a stream 80 columns wide.

    The swap happens in the test body because pytest's capture reinstalls its own stream at the
    start of each test phase, after fixtures have run.
    """

    def redirect(*, tty: bool = False, no_color: str | None = None) -> FakeStderr:
        fake = FakeStderr(tty=tty)
        monkeypatch.setattr(sys, "stderr", fake)
        monkeypatch.setenv("COLUMNS", "80")
        monkeypatch.delenv("NO_COLOR", raising=False)
        if no_color is not None:
            monkeypatch.setenv("NO_COLOR", no_color)
        return fake

    return redirect


def test_a_report_record_prints_bare_with_the_time(
    redirect_stderr: Callable[..., FakeStderr], experiment: Experiment
) -> None:
    """A report block prints as the time and its text, with no level or logger name."""
    stderr = redirect_stderr()
    handler = ConsoleHandler()
    start = phase_start(experiment, experiment.phases[1], finished=0, remaining=4, reruns=0)
    handler.handle(record(REPORT_LOGGER, logging.INFO, start))
    handler.handle(record(REPORT_LOGGER, logging.INFO, "DRY RUN example command:\n  run --x 1"))
    assert stderr.getvalue() == (
        "12:00:00  learning_rate  4 trials to run, 1 at a time\n"
        "12:00:00  DRY RUN example command:\n"
        "            run --x 1\n"
    )


def test_a_report_record_at_warning_level_is_still_a_report_line(
    redirect_stderr: Callable[..., FakeStderr],
) -> None:
    """Dry-run warnings keep their own wording, so they print bare, not under a ``warning:`` label."""
    stderr = redirect_stderr()
    ConsoleHandler().handle(record(REPORT_LOGGER, logging.WARNING, "DRY RUN warning: %s", "x"))
    assert stderr.getvalue() == "12:00:00  DRY RUN warning: x\n"


def test_info_records_from_other_loggers_print_nothing(
    redirect_stderr: Callable[..., FakeStderr],
) -> None:
    """Trainer commands, GPU notes and the like belong to run.log alone."""
    stderr = redirect_stderr()
    handler = ConsoleHandler()
    handler.handle(record("phasesweep.engine.trial", logging.INFO, "[p/trial_0] train --x 1"))
    handler.handle(record("phasesweep.runtime.gpu", logging.DEBUG, "GPU assigned"))
    assert stderr.getvalue() == ""


def test_a_dry_run_prints_info_records_of_the_package_as_report_lines(
    redirect_stderr: Callable[..., FakeStderr],
) -> None:
    """In a dry run an INFO record of any ``phasesweep`` logger prints as the time and its text."""
    stderr = redirect_stderr()
    handler = ConsoleHandler(dry_run=True)
    handler.handle(record("phasesweep.runtime.gpu", logging.INFO, "GPU pool: %d device(s).", 2))
    handler.handle(record("phasesweep.engine.preview", logging.INFO, "phase=p\n  second line"))
    handler.handle(record("phasesweep.runtime.gpu", logging.DEBUG, "GPU assigned"))
    handler.handle(record("urllib3.connectionpool", logging.INFO, "Starting new connection"))
    handler.handle(record("phasesweepish.other", logging.INFO, "not the package"))
    assert stderr.getvalue() == (
        "12:00:00  GPU pool: 2 device(s).\n12:00:00  phase=p\n            second line\n"
    )


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        pytest.param(
            80,
            [
                "12:00:00  Waiting for a GPU: another process holds its lock, such as another",
                "          sweep or a trial that is still shutting down.",
                "12:00:00  MLflow: delivered 12 record(s) left pending by earlier runs.",
            ],
            id="80",
        ),
        pytest.param(
            60,
            [
                "12:00:00  Waiting for a GPU: another process holds its lock,",
                "          such as another sweep or a trial that is still",
                "          shutting down.",
                "12:00:00  MLflow: delivered 12 record(s) left pending by",
                "          earlier runs.",
            ],
            id="60",
        ),
    ],
)
def test_prose_report_lines_wrap_to_the_terminal(
    redirect_stderr: Callable[..., FakeStderr],
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    expected: list[str],
) -> None:
    """A prose report line wraps with its continuation under the text, not under the time."""
    stderr = redirect_stderr()
    monkeypatch.setenv("COLUMNS", str(width))
    handler = ConsoleHandler()
    handler.handle(record(REPORT_LOGGER, logging.INFO, _HOST_LOCK_WAIT))
    handler.handle(
        record(
            REPORT_LOGGER,
            logging.INFO,
            ReportText("MLflow: delivered 12 record(s) left pending by earlier runs."),
        )
    )
    assert stderr.getvalue().splitlines() == expected
    assert all(len(line) <= width for line in expected)
    assert str(_HOST_LOCK_WAIT).startswith("Waiting for a GPU: another process holds its lock")


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        pytest.param(
            80,
            [
                "12:00:00  warning: [learning_rate] only 1 of 2 grid cells succeeded; the winner",
                "                   was chosen without a successful trial for: {'optimizer.lr':",
                "                   0.0003}.",
            ],
            id="80",
        ),
        pytest.param(
            60,
            [
                "12:00:00  warning: [learning_rate] only 1 of 2 grid cells",
                "                   succeeded; the winner was chosen without",
                "                   a successful trial for: {'optimizer.lr':",
                "                   0.0003}.",
            ],
            id="60",
        ),
    ],
)
def test_a_warning_wraps_under_the_text_after_its_label(
    redirect_stderr: Callable[..., FakeStderr],
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    expected: list[str],
) -> None:
    """A long warning wraps to the terminal, every later line hanging under the text, not the label."""
    stderr = redirect_stderr()
    monkeypatch.setenv("COLUMNS", str(width))
    message = (
        "[learning_rate] only 1 of 2 grid cells succeeded; the winner was chosen without a "
        "successful trial for: {'optimizer.lr': 0.0003}."
    )
    handler = ConsoleHandler()
    handler.handle(record("phasesweep.engine.selection", logging.WARNING, message))
    assert stderr.getvalue().splitlines() == expected
    assert all(len(line) <= width for line in expected)
    stderr.truncate(0)
    stderr.seek(0)
    handler.handle(
        record("phasesweep.engine.phase", logging.ERROR, "phase=%s ABORTED: %s", "p", "x")
    )
    assert stderr.getvalue() == "12:00:00  error: phase=p ABORTED: x\n"
    stderr.truncate(0)
    stderr.seek(0)
    handler.handle(record("phasesweep.engine.phase", logging.ERROR, "first line\nsecond line"))
    assert stderr.getvalue() == "12:00:00  error: first line\n                 second line\n"


def test_a_failure_record_prints_no_traceback(redirect_stderr: Callable[..., FakeStderr]) -> None:
    """The exception behind a logged failure stays in run.log."""
    stderr = redirect_stderr()
    try:
        raise ValueError("secret detail")
    except ValueError:
        failed = record(
            "phasesweep.engine.run",
            logging.WARNING,
            "MLflow delivery did not complete",
            exc_info=sys.exc_info(),
        )
    ConsoleHandler().handle(failed)
    assert stderr.getvalue() == "12:00:00  warning: MLflow delivery did not complete\n"


@pytest.mark.parametrize("tty", [False, True])
@pytest.mark.parametrize("no_color", [None, "1", ""], ids=["unset", "set", "empty"])
def test_styling_needs_a_terminal_and_no_no_color(
    redirect_stderr: Callable[..., FakeStderr],
    experiment: Experiment,
    tty: bool,
    no_color: str | None,
) -> None:
    """Escapes appear only on a terminal with ``NO_COLOR`` absent; an empty value disables them too."""
    fake = redirect_stderr(tty=tty, no_color=no_color)
    failed = trial_line(
        experiment,
        experiment.phases[0],
        number=3,
        state="FAIL",
        value=None,
        seconds=0.5,
        finished=3,
        params={},
        repeat=None,
        rerun_of=None,
        reason=None,
    )
    handler = ConsoleHandler()
    handler.handle(record(REPORT_LOGGER, logging.INFO, failed))
    handler.handle(record("phasesweep.engine.phase", logging.WARNING, "slow"))
    handler.handle(record("phasesweep.engine.phase", logging.ERROR, "broken"))
    styled = tty and no_color is None
    assert ("\x1b[31mFAIL\x1b[0m" in fake.getvalue()) is styled
    assert ("\x1b[33mwarning:\x1b[0m" in fake.getvalue()) is styled
    assert ("\x1b[31merror:\x1b[0m" in fake.getvalue()) is styled
    plain = re.sub(r"\x1b\[[0-9;]*m", "", fake.getvalue())
    assert plain.splitlines() == [
        "12:00:00  depth          trial 3  FAIL  0.5s  3/4",
        "12:00:00  warning: slow",
        "12:00:00  error: broken",
    ]


class ThreeLines(ReportBlock):
    """A block that always draws three lines, to expose any interleaving."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def render(self, width: int, *, color: bool) -> list[str]:
        return [f"{self.tag} first", f"{self.tag} second", f"{self.tag} third"]

    def __str__(self) -> str:
        return self.tag


def test_records_from_parallel_trials_stay_contiguous(
    redirect_stderr: Callable[..., FakeStderr],
) -> None:
    """Workers log from threads; each record's lines reach the stream as one write."""
    stderr = redirect_stderr()
    handler = ConsoleHandler()

    def worker(tag: str) -> None:
        for _ in range(50):
            handler.handle(record(REPORT_LOGGER, logging.INFO, ThreeLines(tag)))

    threads = [threading.Thread(target=worker, args=(f"t{n}",)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lines = stderr.getvalue().splitlines()
    assert len(lines) == 8 * 50 * 3
    for start in range(0, len(lines), 3):
        tag = lines[start].split()[1]
        assert lines[start] == f"12:00:00  {tag} first"
        assert lines[start + 1] == f"          {tag} second"
        assert lines[start + 2] == f"          {tag} third"


@pytest.mark.parametrize("verbose", [False, True])
def test_console_logging_is_scoped_and_keeps_optuna_at_error(verbose: bool) -> None:
    """``-v`` changes only what run.log records; Optuna stays at ERROR, and nothing outlives the run."""
    root, package = logging.getLogger(), logging.getLogger("phasesweep")
    handlers, level = root.handlers[:], package.level
    optuna.logging.set_verbosity(optuna.logging.INFO)
    try:
        with _console_logging(verbose):
            assert optuna.logging.get_verbosity() == optuna.logging.ERROR
            assert package.level == (logging.DEBUG if verbose else logging.INFO)
            assert any(isinstance(handler, ConsoleHandler) for handler in root.handlers)
        assert optuna.logging.get_verbosity() == optuna.logging.INFO
        assert package.level == level
        assert root.handlers == handlers
    finally:
        optuna.logging.set_verbosity(optuna.logging.INFO)


def _run_config(tmp_path: Path, name: str) -> Path:
    """Write a one-phase, two-trial config whose trials succeed, with its own workdir."""
    experiment = make_experiment(
        persistent=tmp_path / name,
        trainer=write_constant_trainer(tmp_path),
        n_trials=2,
        gpu_policy="none",
        sampler=Sampler(type="grid"),
        search_space={"x": CategoricalParam(type="categorical", choices=[1, 2])},
    )
    return write_experiment_config(tmp_path / f"{name}.yaml", experiment)


def _console_text(err: str, tmp_path: Path, name: str) -> str:
    """Normalize what differs between two runs of the same sweep: times, durations, ids, paths."""
    text = re.sub(r"(?m)^\d\d:\d\d:\d\d", "HH:MM:SS", err.replace(str(tmp_path), "TMP"))
    text = re.sub(r"\d+\.\ds", "Ns", text)
    return re.sub(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])", "ID", text).replace(name, "NAME")


@pytest.mark.integration
def test_verbose_leaves_the_console_unchanged_and_deepens_run_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same sweep prints the same console text with and without ``-v``; only run.log differs."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # Wide enough that no path wraps, so the two runs' differing path lengths break nowhere.
    monkeypatch.setenv("COLUMNS", "500")
    transcripts = {}
    logs = {}
    for name, flags in (("plain", []), ("verbose", ["-v"])):
        config = _run_config(tmp_path, name)
        assert invoke_cli_boundary(["run", str(config), *flags], monkeypatch) == 0
        captured = capsys.readouterr()
        assert captured.out == ""
        transcripts[name] = _console_text(captured.err, tmp_path, name)
        logs[name] = (tmp_path / name / "runs" / "t" / "run.log").read_text(encoding="utf-8")
    assert transcripts["plain"] == transcripts["verbose"]
    assert "trial 1  COMPLETE" in transcripts["plain"]
    # The summary names the winning trial's directory, relative to the working directory or whole.
    assert re.search(
        r"(?m)^ +\S*/NAME/runs/t/p/trial_00000__generation_ID__attempt_ID$", transcripts["plain"]
    )
    assert not re.search(r"^\S+ \S+ D ", logs["plain"], re.MULTILINE)
    assert re.search(r"^\S+ \S+ D phasesweep\.", logs["verbose"], re.MULTILINE)


def test_a_dry_run_prints_its_lines_on_stderr_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A dry run keeps its message text, prefixed only by the time, and prints nothing on stdout."""
    config = _run_config(tmp_path, "dry")
    assert invoke_cli_boundary(["run", str(config), "--dry-run"], monkeypatch) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = [re.sub(r"^\d\d:\d\d:\d\d  ", "", line) for line in captured.err.splitlines()]
    assert "phase=p study=t::p finished=0 remaining=2 reruns=0 n_jobs=1" in lines
    assert "DRY RUN phase=p would launch 2 trials" in lines
    assert lines[-1] == "DRY RUN complete. No trials launched, no summary written."
    assert not any(line.startswith(("phasesweep run", "Finished in", "log:")) for line in lines)


def test_a_dry_run_prints_every_info_line_of_the_starter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The dry run is the launch contract, so its console lines, GPU notes included, are pinned."""
    monkeypatch.setenv("COLUMNS", "500")
    config = tmp_path / "starter.yaml"
    assert invoke_cli_boundary(["init", "-o", str(config)], monkeypatch) == 0
    capsys.readouterr()

    assert invoke_cli_boundary(["run", str(config), "--dry-run"], monkeypatch) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    root = str(tmp_path.resolve())
    # A later line of a record sits under the text, where the first line has the time.
    lines = [re.sub(r"^(\d\d:\d\d:\d\d| {8})  ", "", line) for line in captured.err.splitlines()]
    assert [line.replace(root, "TMP") for line in lines] == [
        "DRY RUN metric='eval_loss' goal=minimize extractor={'type': 'json_envelope', "
        "'path': 'result.json', 'objective_name': 'eval_loss', 'split': 'validation', "
        "'policy': 'synthetic', 'checkpoint': 'toy_formula', 'expected_step': 100, "
        "'evaluation_axis': 'step'}",
        "DRY RUN evaluation_point check: supported (axis 'step' from the envelope; "
        "any other axis from evaluation.progress)",
        "DRY RUN environment: inherit_env=none (inherited when set, and digested: PATH, HOME, "
        "LANG, LC_ALL, USER, LOGNAME, TZ); env keys: none; passthrough_env: TMPDIR (built in; "
        "inherited when set, excluded from the digest); a persistent study refuses to resume "
        "when any digested value changes",
        "DRY RUN trainer cwd: TMP (execution.cwd; part of the study identity)",
        "DRY RUN artifact root: TMP/runs/phasesweep_starter",
        "DRY RUN phase 'depth': sampler=grid (resumable, reproducible)",
        "DRY RUN grid order: configurations are shuffled by sampler.seed, not listed choice order.",
        "DRY RUN phase 'learning_rate': sampler=grid (resumable, reproducible)",
        "DRY RUN grid order: configurations are shuffled by sampler.seed, not listed choice order.",
        "phase=depth study=phasesweep_starter::depth finished=0 remaining=2 reruns=0 n_jobs=1",
        "GPU isolation disabled by gpu_policy='none'.",
        "DRY RUN phase=depth would launch 2 trials",
        "DRY RUN example command:",
        "  phasesweep-fake-train --config "
        "TMP/runs/phasesweep_starter/depth/trial_dryrun/trainer_config.yaml",
        "phase=learning_rate study=phasesweep_starter::learning_rate finished=0 remaining=2 "
        "reruns=0 n_jobs=1",
        "GPU isolation disabled by gpu_policy='none'.",
        "DRY RUN phase=learning_rate would launch 2 trials",
        "DRY RUN example command (placeholders until 'depth' publishes a winner: model.n_layers):",
        "  phasesweep-fake-train --config "
        "TMP/runs/phasesweep_starter/learning_rate/trial_dryrun/trainer_config.yaml",
        "DRY RUN total: would launch 4 trials across 2 phases",
        "DRY RUN complete. No trials launched, no summary written.",
    ]


def test_banner_for_the_starter() -> None:
    """The banner names the config, each phase's plan, storage, tracker and where output goes."""
    block = banner(
        grid_experiment(),
        config_path="starter.yaml",
        workdir="runs/phasesweep_starter",
        log_path="/home/me/runs/phasesweep_starter/run.log",
        from_phase=None,
    )
    assert block.render(80, color=False) == [
        "phasesweep run  phasesweep_starter",
        "",
        "  config    starter.yaml",
        "  phases    depth          2 trials  grid  1 at a time  gpu none",
        "            learning_rate  2 trials  grid  1 at a time  gpu none",
        "  storage   auto",
        "  tracker   none",
        "  workdir   runs/phasesweep_starter",
        "  log       /home/me/runs/phasesweep_starter/run.log",
        "",
    ]


def _tracked_experiment() -> Experiment:
    """MLflow and W&B targets, explicit GPU ids and device tokens, a fixed second phase."""
    space = {"x": IntParam(type="int", low=0, high=10)}
    return make_experiment(
        experiment="tracked",
        metric=Metric(
            name="eval_loss",
            extractor=WandbExtractor(
                type="wandb", entity="lab", project="sweeps", metric_key="eval/loss"
            ),
        ),
        storage="journal:///data/sweeps/tracked.journal",
        phases=[
            Phase(
                name="depth",
                n_trials=4,
                n_jobs=2,
                gpu_ids=[0, 1],
                sampler=Sampler(type="random", seed=0),
                search_space=space,
            ),
            Phase(
                name="learning_rate", n_trials=8, gpu_policy="whole_node", gpu_devices=["GPU-aa"]
            ),
        ],
    ).model_copy(
        update={
            "mlflow": MlflowSettings(tracking_uri="http://localhost:5000", experiment_name="sweeps")
        }
    )


def test_banner_names_trackers_devices_and_the_starting_phase() -> None:
    """Targets and device lists are shown, a fixed configuration says so, ``from`` names the start."""
    block = banner(
        _tracked_experiment(),
        config_path="configs/tracked.yaml",
        workdir="runs/tracked",
        log_path="/home/me/projects/sweeps/runs/tracked/run.log",
        from_phase="learning_rate",
    )
    assert block.render(100, color=False) == [
        "phasesweep run  tracked",
        "",
        "  config    configs/tracked.yaml",
        "  phases    depth          4 trials  random  2 at a time  gpu single_per_trial ids 0,1",
        "            learning_rate  8 trials  fixed  1 at a time  gpu whole_node devices GPU-aa",
        "  from      learning_rate",
        "  storage   journal:///data/sweeps/tracked.journal",
        "  tracker   mlflow http://localhost:5000, wandb lab/sweeps",
        "  workdir   runs/tracked",
        "  log       /home/me/projects/sweeps/runs/tracked/run.log",
        "",
    ]
    assert block.render(60, color=False)[2:9] == [
        "  config    configs/tracked.yaml",
        "  phases    depth          4 trials",
        "                           random  2 at a time",
        "                           gpu single_per_trial ids 0,1",
        "            learning_rate  8 trials",
        "                           fixed  1 at a time",
        "                           gpu whole_node devices GPU-aa",
    ]


def test_banner_breaks_a_long_path_at_slashes_and_never_shows_the_environment() -> None:
    """A path wider than the terminal breaks after a slash; env values and passthrough stay out."""
    experiment = grid_experiment().model_copy(
        update={
            "env": {"API_TOKEN": "hunter2"},
            "execution": ExecutionContext(inherit_env="none", passthrough_env=["SECRET_NAME"]),
        }
    )
    block = banner(
        experiment,
        config_path="starter.yaml",
        workdir="runs/phasesweep_starter",
        log_path="/home/someone/projects/a-long-project-name/runs/phasesweep_starter/run.log",
        from_phase=None,
    )
    assert block.render(50, color=False)[-5:] == [
        "  workdir   runs/phasesweep_starter",
        "  log       /home/someone/projects/",
        "            a-long-project-name/runs/",
        "            phasesweep_starter/run.log",
        "",
    ]
    shown = "".join(block.render(80, color=False))
    assert "hunter2" not in shown
    assert "API_TOKEN" not in shown
    assert "SECRET_NAME" not in shown


@pytest.mark.parametrize("columns", ["500", "80", "60"])
def test_the_log_path_comes_last_after_an_operational_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    columns: str,
) -> None:
    """The error and its fix line come first, then the absolute path of ``run.log`` fitted to the width."""
    monkeypatch.setenv("COLUMNS", columns)
    config = make_experiment(persistent=tmp_path)
    error = PhaseTrialsFailedError("no feasible trial in 'p'")
    log = stub_run_command(monkeypatch, error, config=config, writes_log=True)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")

    assert invoke_cli_boundary(["run", str(config_path)], monkeypatch) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    expected = [
        "phasesweep: no feasible trial in 'p'",
        f"fix: {operator_step(error)}",
        *LogLine(str(log.resolve())).render(int(columns), color=False),
    ]
    assert captured.err.splitlines()[-len(expected) :] == expected


_RESUME_HINT_AT = {
    500: [
        "phasesweep: stopped. Rerun the same command to resume: finished trials are kept and "
        "interrupted ones rerun first."
    ],
    80: [
        "phasesweep: stopped. Rerun the same command to resume: finished trials are kept",
        "and interrupted ones rerun first.",
    ],
    60: [
        "phasesweep: stopped. Rerun the same command to resume:",
        "finished trials are kept and interrupted ones rerun first.",
    ],
}


@pytest.mark.parametrize("columns", [500, 80, 60])
def test_the_log_path_comes_last_after_ctrl_c(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    columns: int,
) -> None:
    """An interrupted run prints its resume hint, wrapped to the width, then names the log, and exits 130."""
    monkeypatch.setenv("COLUMNS", str(columns))
    config = make_experiment(persistent=tmp_path)
    shutdown = PhaseSweepShutdown(
        signal.SIGINT,
        ShutdownCleanupReport(signum=signal.SIGINT, cleanup_confirmed=True, child_pgids=()),
    )
    log = stub_run_command(monkeypatch, shutdown, config=config, writes_log=True)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")

    assert invoke_cli_boundary(["run", str(config_path)], monkeypatch) == 128 + signal.SIGINT

    expected = [
        *_RESUME_HINT_AT[columns],
        f"phasesweep run {config_path}",
        *LogLine(str(log.resolve())).render(columns, color=False),
    ]
    assert capsys.readouterr().err.splitlines()[-len(expected) :] == expected


def test_the_stop_hint_repeats_the_command_quoted_with_its_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A path with spaces, a quoted ``--set`` value and ``--from-phase`` all come back pasteable."""
    monkeypatch.setenv("COLUMNS", "500")
    config = make_experiment(persistent=tmp_path)
    shutdown = PhaseSweepShutdown(
        signal.SIGTERM,
        ShutdownCleanupReport(signum=signal.SIGTERM, cleanup_confirmed=True, child_pgids=()),
    )
    stub_run_command(monkeypatch, shutdown, config=config, writes_log=True)
    monkeypatch.setattr("phasesweep.cli._command_config", lambda _path, settings: config)
    config_path = tmp_path / "my sweeps" / "experiment.yaml"
    config_path.parent.mkdir()
    config_path.write_text("placeholder: true\n")

    assert (
        invoke_cli_boundary(
            [
                "run",
                str(config_path),
                "--set",
                "phases.p.n_trials=12",
                "--set",
                "provenance.note=two words",
                "--from-phase",
                "p",
            ],
            monkeypatch,
        )
        == 128 + signal.SIGTERM
    )

    command = capsys.readouterr().err.splitlines()[-2]
    assert command == (
        f"phasesweep run '{config_path}' --set phases.p.n_trials=12 "
        "--set 'provenance.note=two words' --from-phase p"
    )
    assert shlex.split(command)[2:] == [
        str(config_path),
        "--set",
        "phases.p.n_trials=12",
        "--set",
        "provenance.note=two words",
        "--from-phase",
        "p",
    ]


@pytest.mark.parametrize(
    ("persistent", "width", "expected"),
    [
        pytest.param(
            True,
            80,
            [
                "phasesweep: stopped. Rerun the same command to resume: finished trials are kept",
                "and interrupted ones rerun first.",
                "phasesweep run sweep.yaml --set optimizer.lr=0.01",
            ],
            id="persistent-80",
        ),
        pytest.param(
            True,
            60,
            [
                "phasesweep: stopped. Rerun the same command to resume:",
                "finished trials are kept and interrupted ones rerun first.",
                "phasesweep run sweep.yaml --set optimizer.lr=0.01",
            ],
            id="persistent-60",
        ),
        pytest.param(
            False,
            80,
            [
                "phasesweep: stopped. Rerun the same command to start over: storage: null keeps",
                "no completed trials between runs.",
                "phasesweep run sweep.yaml --set optimizer.lr=0.01",
            ],
            id="in-memory-80",
        ),
        pytest.param(
            False,
            60,
            [
                "phasesweep: stopped. Rerun the same command to start over:",
                "storage: null keeps no completed trials between runs.",
                "phasesweep run sweep.yaml --set optimizer.lr=0.01",
            ],
            id="in-memory-60",
        ),
    ],
)
def test_the_stop_hint_wraps_its_sentence_and_keeps_the_command_on_one_line(
    tmp_path: Path, persistent: bool, width: int, expected: list[str]
) -> None:
    """Both forms wrap the prose to the terminal, and the command line below it is never broken."""
    config = make_experiment(persistent=tmp_path if persistent else None)
    command = _rerun_command(Path("sweep.yaml"), ("optimizer.lr=0.01",), None)

    assert _resume_hint(config, command=command, width=width).splitlines() == expected


def test_the_stop_hint_never_breaks_a_command_wider_than_the_terminal(tmp_path: Path) -> None:
    """A long config path stays whole on its line at 60 columns, so the line can be pasted."""
    long_path = Path("experiments/a-long-project-name/phasesweep_starter/sweeps/sweep.yaml")
    command = _rerun_command(long_path, (), "depth")

    lines = _resume_hint(
        make_experiment(persistent=tmp_path), command=command, width=60
    ).splitlines()

    assert lines[-1] == f"phasesweep run {long_path} --from-phase depth"
    assert len(lines[-1]) > 60
    assert all(len(line) <= 60 for line in lines[:-1])


LONG_LOG = (
    "/home/someone/projects/a-long-project-name/experiments/phasesweep_starter/runs/"
    "phasesweep_starter/run.log"
)


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        pytest.param(
            120,
            ["log: " + LONG_LOG],
            id="120-fits-on-one-line",
        ),
        pytest.param(
            80,
            [
                "log: /home/someone/projects/a-long-project-name/experiments/phasesweep_starter/",
                "     runs/phasesweep_starter/run.log",
            ],
            id="80",
        ),
        pytest.param(
            60,
            [
                "log: /home/someone/projects/a-long-project-name/experiments/",
                "     phasesweep_starter/runs/phasesweep_starter/run.log",
            ],
            id="60",
        ),
    ],
)
def test_the_log_line_breaks_a_long_path_after_slashes(width: int, expected: list[str]) -> None:
    """A path wider than the terminal breaks after a slash, every component whole and indented under the path."""
    lines = LogLine(LONG_LOG).render(width, color=False)
    assert lines == expected
    assert all(len(line) <= width for line in lines)
    assert "".join(line.removeprefix("log: ").strip() for line in lines) == LONG_LOG


@pytest.mark.parametrize("tty", [False, True])
@pytest.mark.parametrize("no_color", [None, "1", ""], ids=["unset", "set", "empty"])
def test_the_closing_log_line_fits_the_terminal_and_is_styled_only_on_a_terminal(
    redirect_stderr: Callable[..., FakeStderr],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tty: bool,
    no_color: str | None,
) -> None:
    """The last line breaks a long path at 80 columns; escapes need a terminal and no ``NO_COLOR``."""
    workdir = tmp_path / "a-long-project-name" / "experiments" / "phasesweep_starter"
    log = stub_run_command(
        monkeypatch,
        PhaseTrialsFailedError("no feasible trial in 'p'"),
        config=make_experiment(persistent=workdir),
        writes_log=True,
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")
    stderr = redirect_stderr(tty=tty, no_color=no_color)

    assert invoke_cli_boundary(["run", str(config_path)], monkeypatch) == 1

    styled = tty and no_color is None
    text = stderr.getvalue()
    assert ("\x1b[1mlog:\x1b[0m" in text) is styled
    assert ("\x1b[2m" in text) is styled
    plain = re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines()
    start = plain.index(next(line for line in plain if line.startswith("log: ")))
    drawn = plain[start:]
    assert len(drawn) > 1
    assert all(len(line) <= 80 for line in drawn)
    assert all(line.startswith("     ") for line in drawn[1:])
    assert "".join(line.removeprefix("log: ").strip() for line in drawn) == str(log.resolve())


def test_no_log_line_when_the_engine_never_wrote_a_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that failed before creating ``run.log`` names no file."""
    config = make_experiment(persistent=tmp_path)
    stub_run_command(monkeypatch, PhaseTrialsFailedError("no feasible trial in 'p'"), config=config)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder: true\n")

    assert invoke_cli_boundary(["run", str(config_path)], monkeypatch) == 1

    assert "log:" not in capsys.readouterr().err


def test_a_phase_abort_prints_once_and_stays_in_run_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The raised abort carries the fact to the console; the ABORTED line is INFO in run.log."""
    monkeypatch.setenv("COLUMNS", "500")
    experiment = make_experiment(
        persistent=tmp_path,
        trial_command="false {overrides}",
        n_trials=4,
        max_consecutive_failures=2,
    )
    config = write_experiment_config(tmp_path / "abort.yaml", experiment)

    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 1

    err = capsys.readouterr().err
    assert err.count("2 consecutive failed/infeasible trials") == 1
    assert "ABORTED" not in err
    assert "error:" not in err
    cause = "2 consecutive failed/infeasible trials reached max_consecutive_failures=2."
    assert f" I phasesweep.engine.phase phase=p ABORTED: {cause}" in run_log_text(experiment)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("objective_key", "gates", "non_finite", "reason", "logged"),
    [
        pytest.param(
            "not_the_objective",
            [],
            False,
            "reason: metric extractor:",
            " I phasesweep.engine.trial [p/trial_0] metric extraction failed: ",
            id="missing-objective",
        ),
        pytest.param(
            "x",
            [],
            True,
            "reason: metric extractor returned non-finite value: inf",
            " I phasesweep.engine.trial [p/trial_0] metric extractor returned non-finite value: inf",
            id="non-finite-objective",
        ),
        pytest.param(
            "x",
            [RequiredFileGate(type="required_file", path="never_written.txt")],
            False,
            "reason: evidence gates failed: never_written.txt is missing",
            " I phasesweep.engine.trial [p/trial_0] evidence gate(s) failed: "
            "never_written.txt is missing",
            id="failing-gate",
        ),
    ],
)
def test_a_trial_failure_the_reason_states_prints_once_and_stays_in_run_log_at_info(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    objective_key: str,
    gates: list[RequiredFileGate],
    non_finite: bool,
    reason: str,
    logged: str,
) -> None:
    """The failed trial's ``reason:`` is the console's account; the same cause is INFO in ``run.log``."""
    monkeypatch.setenv("COLUMNS", "500")
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_constant_trainer(tmp_path, key=objective_key),
        n_trials=2,
        max_consecutive_failures=5,
        gates=gates,
    )
    if non_finite:
        monkeypatch.setattr(
            "phasesweep.engine.trial.run_extractor", lambda *_args, **_kwargs: float("inf")
        )
    config = write_experiment_config(tmp_path / "failing.yaml", experiment)

    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 1

    err = capsys.readouterr().err
    assert err.count(reason) == 2
    assert "warning:" not in err
    assert logged in run_log_text(experiment)


def test_a_fatal_objective_error_prints_once_and_stays_in_run_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The error that ends the run is the console's account; its FATAL line is INFO in run.log."""
    monkeypatch.setenv("COLUMNS", "500")
    experiment = make_experiment(persistent=tmp_path, n_trials=2)
    config = write_experiment_config(tmp_path / "fatal.yaml", experiment)

    def unsafe(*_args: object, **_kwargs: object) -> float:
        raise UnsafeProcessCleanupError("trainer group may still be alive")

    monkeypatch.setattr("phasesweep.engine.phase._execute_objective", unsafe)

    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 1

    err = capsys.readouterr().err
    assert err.count("trainer group may still be alive") == 1
    assert "FATAL OBJECTIVE ERROR" not in err
    assert (
        " I phasesweep.engine.phase phase=p FATAL OBJECTIVE ERROR (UnsafeProcessCleanupError): "
        "trainer group may still be alive"
    ) in run_log_text(experiment)


@pytest.mark.integration
def test_a_repeats_run_summary_lists_each_winning_member_under_one_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A real repeats phase names every member of its winning group, in trial order, under one parent."""
    monkeypatch.setenv("COLUMNS", "500")
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=write_param_echo_trainer(tmp_path),
        phases=[
            Phase(
                name="p",
                n_trials=4,
                sampler=Sampler(type="random", seed=0),
                search_space={"x": IntParam(type="int", low=3, high=9)},
                repeats=Repeats(key="seed", values=[0, 1]),
            )
        ],
    )
    config = write_experiment_config(tmp_path / "repeats.yaml", experiment)

    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 0

    lines = capsys.readouterr().err.splitlines()
    members = sorted(
        row["trial"]
        for row in read_trial_tables(experiment, latest=True).phases[0].rows
        if row["winner"] == "*"
    )
    assert len(members) == 2
    written = next(i for i, line in enumerate(lines) if line.startswith("Written under"))
    summary_lines = lines[lines.index("  p  4 complete") : written - 1]
    names = [line for line in summary_lines if "__generation_" in line]
    assert [int(name.strip().split("__")[0].removeprefix("trial_")) for name in names] == members
    assert all(re.fullmatch(r"\s+trial_\d{5}__generation_\w{32}__attempt_\w{32}", n) for n in names)
    parent = str(_phase_dir(experiment, "p")) + "/"
    assert summary_lines.count(f"     {parent}") == 1
    assert not any(parent + "trial_" in line for line in summary_lines)


def test_a_live_run_opens_with_the_banner_and_closes_with_summary_then_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A real two-trial sweep prints the banner, its progress, the summary and the log path."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("COLUMNS", "500")
    config = _run_config(tmp_path, "first")
    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    shape = [re.sub(r"^\d\d:\d\d:\d\d", "##:##:##", line) for line in lines]
    assert shape[0] == "phasesweep run  t"
    assert "##:##:##  p  2 trials to run, 1 at a time" in shape
    headline = next(line for line in lines if line.startswith("Finished in "))
    assert re.fullmatch(r"Finished in \d+\.\ds: 1 phase, 2 trials, 0 from a previous run", headline)
    after = lines[next(i for i, line in enumerate(lines) if line.startswith("Written under")) + 1 :]
    written = after[: after.index("")]
    assert [line.split("/")[-1].strip() for line in written] == ["summary.yaml", "trials.csv"]
    assert lines[-2] == f"Next: phasesweep status {config} --trials"
    assert lines[-1] == f"log: {tmp_path / 'first' / 'runs' / 't' / 'run.log'}"
    # The banner and the summary are written to stderr only; run.log holds the progress lines.
    run_log = (tmp_path / "first" / "runs" / "t" / "run.log").read_text(encoding="utf-8")
    assert "phasesweep.report p  trial 0  COMPLETE" in run_log
    assert "phasesweep run" not in run_log
    assert "Finished in" not in run_log
    assert "Written under" not in run_log


def test_a_replay_says_nothing_ran_and_lists_no_written_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The second run of a finished sweep launches nothing, so its closing is the one sentence."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("COLUMNS", "500")
    config = _run_config(tmp_path, "again")
    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 0
    capsys.readouterr()

    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 0

    err = capsys.readouterr().err
    assert "p  nothing to run, 2 trials already finished, replaying the recorded result" in err
    assert re.search(
        r"Finished in \d+\.\ds: nothing to run, 2 trials already finished, and generation "
        r"[0-9a-f]{32} already publishes these winners\.",
        err,
    )
    assert "Written under" not in err
    assert err.splitlines()[-1].startswith("log: ")


def test_a_resumed_run_counts_the_trials_a_previous_run_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Raising ``n_trials`` runs the new trials and reports the old ones as from a previous run."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("COLUMNS", "500")
    experiment = make_experiment(
        persistent=tmp_path / "grown",
        trainer=write_constant_trainer(tmp_path),
        n_trials=2,
        gpu_policy="none",
        sampler=Sampler(type="random", seed=0),
    )
    config = write_experiment_config(tmp_path / "grown.yaml", experiment)
    assert invoke_cli_boundary(["run", str(config)], monkeypatch) == 0
    capsys.readouterr()

    assert (
        invoke_cli_boundary(["run", str(config), "--set", "phases[p].n_trials=4"], monkeypatch) == 0
    )

    err = capsys.readouterr().err
    assert re.search(r"p  2 trials to run, 1 at a time, 2 already finished", err)
    assert re.search(r"Finished in \d+\.\ds: 1 phase, 4 trials, 2 from a previous run", err)
    assert f"Next: phasesweep status {config} --set 'phases[p].n_trials=4' --trials" in err


def test_phase_counts_keep_an_interrupted_trial_out_of_the_counted_and_earlier_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline's counts follow the engine's rule: an interrupted trial never counts, nor is it earlier."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    experiment = make_experiment(
        persistent=tmp_path, trainer=write_constant_trainer(tmp_path), n_trials=2, gpu_policy="none"
    )
    winner = run_experiment(experiment)["p"]
    phase_study(experiment).add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.FAIL,
            user_attrs={
                GENERATION_ID_ATTR: "an-earlier-generation",
                TRIAL_OUTCOME_ATTR: trial_outcome(9, "interrupted", cause="interrupted by SIGINT"),
            },
        )
    )

    counts = _phase_counts(validate_ledger(experiment), experiment.phases[0], "this-run", winner)

    assert counts is not None
    assert (counts.complete, counts.interrupted) == (2, 1)
    assert (counts.counted, counts.earlier) == (2, 2)
