"""How an attempt ended, as its lifecycle record says, and the verdict each ending earns."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
from optuna.trial import FrozenTrial

from phasesweep import run_experiment
from phasesweep.config.models import Experiment
from phasesweep.engine import phase as phase_module
from phasesweep.engine.cleanup import _reap_stale_trials
from phasesweep.engine.optuna import _trial_count_state
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    TRIAL_OUTCOME_ATTR,
    _parsed_trial_outcome,
    _trial_attempt_dir,
)
from phasesweep.engine.study_policy import _load_phase_policy_state, _record_reruns
from phasesweep.errors import PhaseSweepError
from phasesweep.runtime.process import (
    AttemptEnd,
    read_attempt_lifecycle,
    run_supervised,
    write_attempt_lifecycle,
)
from phasesweep.runtime.reaper import read_proc_starttime
from tests.conftest import (
    make_experiment,
    phase_study,
    phase_trials,
    write_experiment_config,
    write_trainer,
)
from tests.recovery_helpers import fabricate_stale_trial, runner_once, write_trial_identity


def _outcome(trial: FrozenTrial) -> tuple[str, str | None]:
    """Return a finished trial's count state and recorded cause.

    :param FrozenTrial trial: The trial.
    :return tuple[str, str | None]: ``(state, cause)``.
    """
    parsed = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    return _trial_count_state(trial), None if parsed is None else parsed.cause


def _exec_trainer(tmp_path: Path, body: str, **phase: object) -> Experiment:
    """Build a one-trial persistent experiment whose trainer replaces its shell.

    :param Path tmp_path: Scratch directory.
    :param str body: Trainer script body.
    :param object phase: Default-phase overrides.
    :return Experiment: The experiment.
    """
    experiment = make_experiment(
        persistent=tmp_path, trainer=write_trainer(tmp_path / "trainer.py", body), **phase
    )
    return experiment.model_copy(update={"trial_command": "exec " + experiment.trial_command})


@pytest.mark.parametrize(
    ("command", "env_extra", "cwd", "timeout", "cap", "ending", "return_code"),
    [
        pytest.param("exit 3", {}, None, None, False, "trainer", 3, id="nonzero-exit"),
        pytest.param("exit 75", {}, None, None, False, "trainer", 75, id="trainer-exits-75"),
        pytest.param("kill -9 $$", {}, None, None, False, "trainer", -9, id="own-sigkill"),
        pytest.param("true", {}, "/nonexistent", None, False, "not_started", None, id="bad-cwd"),
        # One environment string past MAX_ARG_STRLEN makes ``execve`` fail with E2BIG.
        pytest.param(
            "true", {"BIG": "x" * 200_000}, None, None, False, "not_started", None, id="execve"
        ),
        pytest.param("exec sleep 30", {}, None, 0.5, False, "timed_out", None, id="timeout"),
        pytest.param("exec sleep 30", {}, None, 0.5, True, "cancelled", None, id="deadline-cap"),
        pytest.param(
            "kill -9 $PPID; exec sleep 30", {}, None, None, False, "guardian_lost", None, id="lost"
        ),
    ],
)
def test_each_ending_leaves_its_own_record(
    tmp_path: Path,
    command: str,
    env_extra: dict[str, str],
    cwd: str | None,
    timeout: float | None,
    cap: bool,
    ending: AttemptEnd,
    return_code: int | None,
) -> None:
    """The record names how the attempt ended, and only a trainer's own ending carries a code."""
    with (tmp_path / "stdout.log").open("w") as out, (tmp_path / "stderr.log").open("w") as err:
        run_supervised(
            command,
            env={**os.environ, **env_extra},
            stdout=out,
            stderr=err,
            timeout=None if cap else timeout,
            wallclock_deadline=time.monotonic() + timeout if cap and timeout else None,
            attempt_dir=tmp_path,
            attempt_id="attempt",
            cwd=cwd,
        )
    lifecycle = read_attempt_lifecycle(tmp_path, expected_attempt_id="attempt")
    assert lifecycle is not None
    assert (lifecycle.state, lifecycle.cleanup_confirmed) == ("exited", True)
    assert (lifecycle.ended_by, lifecycle.return_code) == (ending, return_code)


@pytest.mark.parametrize(
    ("body", "state", "cause"),
    [
        pytest.param(
            """
            import os, signal, sys, time
            open(sys.argv[-1], "w").close()
            os.kill(os.getppid(), signal.SIGTERM)
            time.sleep(30)
            """,
            "INTERRUPTED",
            "orchestrator interrupted by SIGINT",
            id="cancelled-by-its-guardian",
        ),
        pytest.param(
            """
            import os, signal, sys
            open(sys.argv[-1], "w").close()
            os.kill(os.getpid(), signal.SIGKILL)
            """,
            "FAIL",
            "killed by signal 9 (SIGKILL); the kernel's out-of-memory killer is a common sender",
            id="killed-by-another-sender",
        ),
    ],
)
def test_a_shutdown_interrupts_only_a_trainer_phasesweep_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str, state: str, cause: str
) -> None:
    """While a shutdown is serviced, a guardian cancellation is an interruption and a kill nobody in PhaseSweep sent is a failure."""
    started = tmp_path / "started"
    experiment = _exec_trainer(tmp_path, body, n_trials=1, max_consecutive_failures=5)
    experiment = experiment.model_copy(
        update={"trial_command": f"{experiment.trial_command} {started}"}
    )
    monkeypatch.setattr(
        phase_module,
        "servicing_shutdown_signal",
        lambda: signal.SIGINT if started.exists() else None,
    )
    with pytest.raises((RuntimeError, NoFeasibleTrialError)):
        run_experiment(experiment)
    (trial,) = phase_trials(experiment)
    assert _outcome(trial) == (state, cause)


@pytest.mark.integration
def test_a_crash_keeps_its_failure_when_a_ctrl_c_lands_in_the_helper_settle(
    tmp_path: Path,
) -> None:
    """A trainer that exited 1 keeps its failure and stderr reason when Ctrl-C reaches the run during the guardian's helper settle."""
    exited = tmp_path / "exited"
    experiment = _exec_trainer(
        tmp_path,
        f"""
        import subprocess, sys
        from pathlib import Path
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
        print("loss went NaN at step 7", file=sys.stderr, flush=True)
        Path({str(exited)!r}).touch()
        sys.exit(1)
        """,
        n_trials=1,
        max_consecutive_failures=5,
    )
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    with runner_once(config, exited, timeout=15) as runner:
        time.sleep(0.5)  # the root has exited; its helper keeps the guardian settling
        runner.send_signal(signal.SIGINT)
        assert runner.wait(timeout=30) == 128 + signal.SIGINT
    (trial,) = phase_trials(experiment)
    assert _outcome(trial) == ("FAIL", "non-zero exit code 1; stderr: loss went NaN at step 7")
    assert _load_phase_policy_state(phase_study(experiment)).consecutive_failures == 1
    assert _record_reruns(phase_study(experiment)) == []


@pytest.mark.integration
def test_a_crash_whose_runner_was_killed_is_recovered_as_a_failure_that_trips_the_streak(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A trainer that exited 1 while its runner was SIGKILLed is recovered as that failure, and ``max_consecutive_failures: 1`` aborts instead of rerunning it."""
    exited = tmp_path / "exited"
    experiment = _exec_trainer(
        tmp_path,
        f"""
        import subprocess, sys
        from pathlib import Path
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
        print("assertion failed in data loader", file=sys.stderr, flush=True)
        Path({str(exited)!r}).touch()
        sys.exit(1)
        """,
        n_trials=3,
        max_consecutive_failures=1,
    )
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    with runner_once(config, exited, timeout=15):
        pass  # killed: the guardian, in a session of its own, still records the exit
    (stale,) = phase_trials(experiment)
    attempt = stale.user_attrs[ATTEMPT_ID_ATTR]
    attempt_dir = _trial_attempt_dir(stale)
    assert attempt_dir is not None
    deadline = time.monotonic() + 15
    while (
        lifecycle := read_attempt_lifecycle(attempt_dir, expected_attempt_id=attempt)
    ) is None or lifecycle.state != "exited":
        assert time.monotonic() < deadline, "the guardian never recorded the exit"
        time.sleep(0.05)
    assert stale.state.name == "RUNNING" and TRIAL_OUTCOME_ATTR not in stale.user_attrs

    with (
        caplog.at_level(logging.INFO, logger="phasesweep"),
        pytest.raises(PhaseSweepError, match="max_consecutive_failures"),
    ):
        run_experiment(experiment)

    (trial,) = phase_trials(experiment)
    assert _outcome(trial) == (
        "FAIL",
        "non-zero exit code 1; stderr: assertion failed in data loader",
    )
    # Recovery reports the crash as the trial's own line, noted recovered, not as a warning.
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("p  trial 0")]
    assert line.startswith("p  trial 0  FAIL  ")
    assert line.endswith(
        "  recovered  reason: non-zero exit code 1; stderr: assertion failed in data loader"
    )
    assert all(r.levelno < logging.WARNING for r in caplog.records if "trial 0" in r.getMessage())
    parsed = _parsed_trial_outcome(trial.user_attrs[TRIAL_OUTCOME_ATTR])
    assert parsed is not None and parsed.abort is not None
    assert _record_reruns(phase_study(experiment)) == []


def test_a_reboot_leaves_its_launching_attempt_interrupted(tmp_path: Path) -> None:
    """An attempt a reboot cut off, still ``launching`` with an identity from another boot, reruns as interrupted, and the live process that now holds its pid and start time is left alone."""
    experiment = make_experiment(persistent=tmp_path, n_trials=1)
    stale = fabricate_stale_trial(experiment, "p", attempt_id="rebooted")
    write_attempt_lifecycle(stale.attempt_dir, attempt_id="rebooted", state="launching")
    # Only the boot id tells this process from the trainer the identity names.
    holder = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        write_trial_identity(
            stale.attempt_dir,
            attempt_id="rebooted",
            pid=holder.pid,
            starttime=read_proc_starttime(holder.pid),
            boot_id="previous-boot",
        )

        assert _reap_stale_trials(stale.study, experiment, "p", confirm=True) == 1

        assert holder.poll() is None
    finally:
        holder.kill()
        holder.wait(timeout=5)
    trial = stale.study.get_trials(deepcopy=False)[stale.number]
    assert _outcome(trial)[0] == "INTERRUPTED"
    lifecycle = read_attempt_lifecycle(stale.attempt_dir, expected_attempt_id="rebooted")
    assert lifecycle is not None and lifecycle.ended_by == "cancelled"
