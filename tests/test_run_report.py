"""The live run's report: exact text of each block at 80 and 60 columns, and its one-line form."""

from __future__ import annotations

import io
import logging
import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import optuna
import pytest

from phasesweep.cli import _console_logging
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import CategoricalParam, IntParam, Sampler
from phasesweep.engine.state import Winner
from phasesweep.run_report import (
    PhaseStart,
    phase_start,
    rerun_line,
    skipped_line,
    trial_line,
    winner_line,
)
from phasesweep.runtime.console import ConsoleHandler
from phasesweep.runtime.terminal import REPORT_LOGGER, ReportBlock
from tests.conftest import (
    FAKE_TRAIN_METRIC,
    invoke_cli_boundary,
    make_experiment,
    write_constant_trainer,
    write_experiment_config,
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


def test_a_warning_wraps_under_its_label(redirect_stderr: Callable[..., FakeStderr]) -> None:
    """A long warning wraps to the terminal with a hanging indent under ``warning:``."""
    stderr = redirect_stderr()
    message = (
        "[learning_rate] only 1 of 2 grid cells succeeded; the winner was chosen without a "
        "successful trial for: {'optimizer.lr': 0.0003}."
    )
    handler = ConsoleHandler()
    handler.handle(record("phasesweep.engine.selection", logging.WARNING, message))
    assert stderr.getvalue() == (
        "12:00:00  warning: [learning_rate] only 1 of 2 grid cells succeeded; the winner\n"
        "          was chosen without a successful trial for: {'optimizer.lr': 0.0003}.\n"
    )
    stderr.truncate(0)
    stderr.seek(0)
    handler.handle(
        record("phasesweep.engine.phase", logging.ERROR, "phase=%s ABORTED: %s", "p", "x")
    )
    assert stderr.getvalue() == "12:00:00  error: phase=p ABORTED: x\n"


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


def _console_text(err: str) -> str:
    """Normalize what varies between two runs of the same sweep: the time and trial durations."""
    return re.sub(r"\d+\.\ds", "Ns", re.sub(r"(?m)^\d\d:\d\d:\d\d", "HH:MM:SS", err))


def test_verbose_leaves_the_console_unchanged_and_deepens_run_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same sweep prints the same console text with and without ``-v``; only run.log differs."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    transcripts = {}
    logs = {}
    for name, flags in (("plain", []), ("verbose", ["-v"])):
        config = _run_config(tmp_path, name)
        assert invoke_cli_boundary(["run", str(config), *flags], monkeypatch) == 0
        captured = capsys.readouterr()
        assert captured.out == ""
        transcripts[name] = _console_text(captured.err)
        logs[name] = (tmp_path / name / "runs" / "t" / "run.log").read_text(encoding="utf-8")
    assert transcripts["plain"] == transcripts["verbose"]
    assert "trial 1  COMPLETE" in transcripts["plain"]
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
