"""A shutdown that lands while Optuna records a trial stops the run as that shutdown, and the next run completes."""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

import optuna.study._optimize as optuna_optimize
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import Experiment
from phasesweep.runtime.shutdown import (
    PhaseSweepShutdown,
    ShutdownCleanupReport,
    _shutdown_handler,
    shutdown_raised_over,
)
from tests.conftest import (
    console_script,
    make_experiment,
    phase_trials,
    write_experiment_config,
    write_trainer,
)

# Runs the CLI with Optuna's tell wrapped so that the first call sends this process SIGINT
# before or after the real tell. os.kill services a signal sent to its own process before
# it returns, so the real handler raises inside the wrapper, where Optuna's finally block
# then raises UnboundLocalError over it.
_RUN_WITH_SIGINT_IN_TELL = r"""
import os, signal, sys
import optuna.study._optimize as optimize
from phasesweep.cli import main

when, config = sys.argv[1], sys.argv[2]
real_tell = optimize._tell_with_warning
interrupted = False

def tell(**kwargs):
    global interrupted
    if interrupted:
        return real_tell(**kwargs)
    interrupted = True
    if when == "before":
        os.kill(os.getpid(), signal.SIGINT)
    told = real_tell(**kwargs)
    os.kill(os.getpid(), signal.SIGINT)
    return told

optimize._tell_with_warning = tell
sys.argv = ["phasesweep", "run", config]
main()
"""


def _counting_experiment(tmp_path: Path) -> tuple[Experiment, Path]:
    """Build a persistent two-trial experiment whose trainer records each start.

    :param Path tmp_path: Directory holding the ledger, the workdir and the start log.
    :return tuple[Experiment, Path]: The experiment and the file each trainer start appends
        its trial directory name to.
    """
    starts = tmp_path / "starts.txt"
    trainer = write_trainer(
        tmp_path / "trainer.py",
        f"""
        import argparse
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument("--out")
        parser.add_argument("--x", type=int, default=0)
        args, _ = parser.parse_known_args()
        with open({str(starts)!r}, "a") as handle:
            handle.write(Path(args.out).parent.name + "\\n")
        print(f"x={{args.x}}")
        """,
    )
    return make_experiment(persistent=tmp_path, trainer=trainer, n_trials=2), starts


def _assert_completed_with_one_start_each(experiment: Experiment, starts: Path) -> None:
    """Require every trial COMPLETE with the objective of its one and only training.

    :param Experiment experiment: The experiment whose ledger is read.
    :param Path starts: The trainer's start log.
    """
    trials = phase_trials(experiment)
    assert [t.state.name for t in trials] == ["COMPLETE", "COMPLETE"]
    # The trainer reports its x, so the value is the one its single training produced.
    assert [t.value for t in trials] == [t.params["x"] for t in trials]
    names = starts.read_text().splitlines()
    assert sorted(name.split("__")[0] for name in names) == ["trial_00000", "trial_00001"]


def _raised(chain: str) -> BaseException:
    """Raise and catch one exception chain, as Python links it.

    :param str chain: ``cause`` raises a RuntimeError from a shutdown, ``context`` raises one
        while handling a shutdown, and ``none`` chains a RuntimeError to a ValueError.
    :return BaseException: The outermost exception, with its links set.
    """
    shutdown = PhaseSweepShutdown(signal.SIGINT, ShutdownCleanupReport(signal.SIGINT, True, ()))
    inner: BaseException = ValueError("inner") if chain == "none" else shutdown
    try:
        try:
            raise inner
        except BaseException as handled:
            if chain == "cause":
                raise RuntimeError("outer") from handled
            raise RuntimeError("outer")  # noqa: B904
    except RuntimeError as outer:
        return outer
    raise AssertionError("the chain was not raised")


@pytest.mark.parametrize("chain", ["cause", "context", "none"])
def test_a_shutdown_is_found_through_either_link_of_a_chain(chain: str) -> None:
    """A shutdown linked as the cause or as the context of the exception leaving ``study.optimize`` is found; a chain without one gives nothing."""
    outer = _raised(chain)
    if chain == "cause":
        # Python sets the context too; the cause alone must be enough.
        outer.__context__ = None

    found = shutdown_raised_over(outer)

    if chain == "none":
        assert found is None
    else:
        assert isinstance(found, PhaseSweepShutdown) and found.signum == signal.SIGINT


@pytest.mark.integration
@pytest.mark.parametrize("when", ["before", "after"], ids=["before-tell", "after-tell"])
def test_run_experiment_raises_the_shutdown_optuna_raised_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    """A shutdown the handler raises inside Optuna's tell leaves the run as that shutdown, not Optuna's UnboundLocalError."""
    experiment, starts = _counting_experiment(tmp_path)
    real_tell = optuna_optimize._tell_with_warning
    calls = 0

    def tell(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1 and when == "before":
            _shutdown_handler(signal.SIGINT, None)
        told = real_tell(**kwargs)
        if calls == 1:
            _shutdown_handler(signal.SIGINT, None)
        return told

    monkeypatch.setattr(optuna_optimize, "_tell_with_warning", tell)
    with pytest.raises(PhaseSweepShutdown) as stopped:
        run_experiment(experiment)
    assert stopped.value.signum == signal.SIGINT

    monkeypatch.setattr(optuna_optimize, "_tell_with_warning", real_tell)
    run_experiment(experiment)
    _assert_completed_with_one_start_each(experiment, starts)


@pytest.mark.integration
@pytest.mark.parametrize("when", ["before", "after"], ids=["before-tell", "after-tell"])
def test_sigint_during_optuna_tell_exits_130_and_the_rerun_completes(
    tmp_path: Path, when: str
) -> None:
    """SIGINT while Optuna records trial 0 exits 130 with the stop line and no traceback; rerunning finishes every trial without retraining."""
    experiment, starts = _counting_experiment(tmp_path)
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)

    stopped = subprocess.run(
        [sys.executable, "-c", _RUN_WITH_SIGINT_IN_TELL, when, str(config)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = stopped.stdout + stopped.stderr
    assert stopped.returncode == 128 + signal.SIGINT, output
    assert "Interrupted (SIGINT)" in output
    assert "phasesweep: stopped. Rerun the same command to resume" in output
    assert "Traceback" not in output
    assert "internal error" not in output

    resumed = subprocess.run(
        [str(console_script("phasesweep")), "run", str(config)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    _assert_completed_with_one_start_each(experiment, starts)
