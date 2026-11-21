"""The lease guardian gives a trainer's helper processes a bounded settle after the trainer root exits.

A helper that exits by itself within the settle, as CPython's resource tracker or an orphaned
DataLoader worker does, costs nothing: the trial keeps its root's exit code and its objective.
A member still running when the settle ends is terminated, named in the failure reason, and
fails the trial even after a zero exit. Once the root has exited, a cancellation request no
longer cuts the settle short, and every wait that covers the guardian grows by the settle.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.engine.state import FAILURE_REASON_ATTR
from phasesweep.errors import PhaseSweepError
from phasesweep.runtime import process, reaper, supervisor
from phasesweep.runtime.process import ProcessResult, read_attempt_lifecycle, run_supervised
from phasesweep.runtime.reaper import PROCESS_IDENTITY_FILE
from phasesweep.runtime.shutdown import PhaseSweepShutdown
from tests.conftest import (
    assert_descendant_dies,
    make_experiment,
    phase_trials,
    use_guardian_timing,
    write_trainer,
)

pytestmark = pytest.mark.integration

#: Forks a helper that waits for the root to exit, lingers ``sys.argv[2]`` seconds and exits;
#: the root exits with ``sys.argv[3]``. The helper records a SIGTERM instead of dying of it.
_HELPER_TRAINER = """
import os, signal, sys, time
from pathlib import Path
marks = Path(sys.argv[1])
linger = float(sys.argv[2])
parent = os.getpid()
if os.fork() == 0:
    signal.signal(signal.SIGTERM, lambda *_: (marks / "helper_sigterm").touch())
    while os.getppid() == parent:
        time.sleep(0.005)
    (marks / "root_gone").touch()
    time.sleep(linger)
    (marks / "helper_done").touch()
    os._exit(0)
os._exit(int(sys.argv[3]))
"""

#: Records its PID, command line and the time each SIGTERM arrives, which it survives.
_PERSISTENT_CHILD = """
import pathlib, signal, sys, time, os
marks = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGTERM, lambda *_: (marks / "child_sigterm").write_text(repr(time.monotonic())))
(marks / "child_cmdline").write_bytes(pathlib.Path("/proc/self/cmdline").read_bytes() if sys.platform == "linux" else b"")
(marks / "child_pid").write_text(str(os.getpid()))
time.sleep(60)
"""

#: Starts ``_PERSISTENT_CHILD`` in its own group, records when it exits, and exits 0.
_PERSISTENT_TRAINER = f"""
import subprocess, sys, time
from pathlib import Path
marks = Path(sys.argv[1])
subprocess.Popen([sys.executable, "-c", {_PERSISTENT_CHILD!r}, str(marks)])
while not (marks / "child_pid").exists():
    time.sleep(0.005)
(marks / "root_exit").write_text(repr(time.monotonic()))
"""


def _trainer(tmp_path: Path, body: str) -> tuple[Path, Path]:
    """Write a trainer script and the directory it leaves its marks in.

    :param Path tmp_path: Test directory.
    :param str body: Trainer source.
    :return tuple[Path, Path]: The trainer script and its marks directory.
    """
    marks = tmp_path / "marks"
    marks.mkdir()
    return write_trainer(tmp_path / "trainer.py", body), marks


def _supervise(trial_dir: Path, argv: list[str], *, timeout: float) -> ProcessResult:
    """Run ``argv`` as a supervised trial in ``trial_dir``.

    :param Path trial_dir: Attempt directory, created here.
    :param list[str] argv: Trainer command, run through ``exec`` so it is the group root.
    :param float timeout: Trial wallclock budget.
    :return ProcessResult: The supervision result.
    """
    trial_dir.mkdir()
    with (trial_dir / "out.log").open("w") as out, (trial_dir / "err.log").open("w") as err:
        return run_supervised(
            "exec " + shlex.join(argv),
            env=os.environ.copy(),
            stdout=out,
            stderr=err,
            timeout=timeout,
            trial_dir=trial_dir,
            attempt_id=trial_dir.name,
        )


def _assert_exited(trial_dir: Path, return_code: int) -> None:
    """Assert the guardian recorded a confirmed exit with ``return_code``.

    :param Path trial_dir: Attempt directory whose name is its attempt ID.
    :param int return_code: Expected recorded return code.
    """
    lifecycle = read_attempt_lifecycle(trial_dir, expected_attempt_id=trial_dir.name)
    assert lifecycle is not None
    assert (lifecycle.state, lifecycle.return_code, lifecycle.cleanup_confirmed) == (
        "exited",
        return_code,
        True,
    )


@pytest.mark.parametrize("code", [0, 3])
def test_a_helper_that_exits_within_the_settle_keeps_the_roots_exit(
    tmp_path: Path, code: int
) -> None:
    """A helper outliving its root by 0.2 s is waited for, not signalled, at any exit code."""
    trainer, marks = _trainer(tmp_path, _HELPER_TRAINER)

    started = time.monotonic()
    result = _supervise(
        tmp_path / f"helper-{code}",
        [sys.executable, str(trainer), str(marks), "0.2", str(code)],
        timeout=30.0,
    )
    elapsed = time.monotonic() - started

    assert (result.return_code, result.failure_reason, result.cleanup_confirmed) == (
        code,
        None,
        True,
    )
    _assert_exited(tmp_path / f"helper-{code}", code)
    assert (marks / "helper_done").exists()
    assert not (marks / "helper_sigterm").exists()
    assert result.duration_seconds >= 0.2
    assert elapsed < supervisor._HELPER_SETTLE_SECONDS / 2


def test_a_persistent_child_is_terminated_after_the_settle_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SIGTERM-ignoring child is signalled only once the settle ends, killed, and named."""
    use_guardian_timing(monkeypatch, tmp_path, settle_seconds=0.5, term_grace_seconds=0.5)
    trainer, marks = _trainer(tmp_path, _PERSISTENT_TRAINER)
    trial_dir = tmp_path / "persistent"

    result = _supervise(trial_dir, [sys.executable, str(trainer), str(marks)], timeout=30.0)

    child_pid = int((marks / "child_pid").read_text())
    assert_descendant_dies(child_pid, on_timeout_msg=f"child {child_pid} survived the guardian")
    assert result.return_code == 0
    assert result.cleanup_confirmed
    assert result.failure_reason is not None
    assert result.failure_reason.startswith(
        f"trainer root exited with code 0, but process group {result.pid} still had live "
        "descendants 0.5s later"
    )
    assert result.failure_reason.endswith(
        ". They were terminated. Make the trainer wait for its children before exiting"
    )
    if sys.platform == "linux":
        command = (marks / "child_cmdline").read_bytes().replace(b"\0", b" ").decode()
        assert f": {child_pid} ({command[:40]}" in result.failure_reason
    _assert_exited(trial_dir, 1)
    assert (trial_dir / PROCESS_IDENTITY_FILE).exists()
    root_exit = float((marks / "root_exit").read_text())
    assert float((marks / "child_sigterm").read_text()) - root_exit >= 0.5


def test_the_settle_is_outside_the_trial_budget(tmp_path: Path) -> None:
    """A helper that outlives a 1 s budget by itself neither times out nor fails the trial."""
    trainer, marks = _trainer(tmp_path, _HELPER_TRAINER)
    budget = 1.0

    result = _supervise(
        tmp_path / "budget", [sys.executable, str(trainer), str(marks), "1.5", "0"], timeout=budget
    )

    assert (result.timed_out, result.failure_reason, result.return_code) == (False, None, 0)
    assert result.duration_seconds > budget
    assert not (marks / "helper_sigterm").exists()
    _assert_exited(tmp_path / "budget", 0)


def test_descendant_termination_is_outside_the_trial_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root that exits in budget stays a descendant failure while the guardian's grace runs past it."""
    use_guardian_timing(monkeypatch, tmp_path, settle_seconds=0.3, term_grace_seconds=3.0)
    ready = tmp_path / "descendant_ready"
    root_pid = tmp_path / "root_pid"
    budget = 1.0
    child = (
        "import os, pathlib, signal, time; "
        f"signal.signal(signal.SIGTERM, lambda *_: (time.sleep({budget}), os._exit(0))); "
        f"pathlib.Path({str(ready)!r}).touch(); "
        "time.sleep(60)"
    )
    root = (
        "import os, pathlib, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        f"while not pathlib.Path({str(ready)!r}).exists(): time.sleep(0.005)\n"
        f"pathlib.Path({str(root_pid)!r}).write_text(str(os.getpid()))\n"
    )
    real_wait = process._wait_for_guardian_exit

    def wait_for_cleanup(proc: subprocess.Popen[bytes]) -> bool:
        # The root-exit status has already arrived; the guardian still owns cleanup.
        assert not reaper.is_pid_alive(int(root_pid.read_text()))
        assert proc.poll() is None
        assert time.monotonic() - started < budget, "root exit did not precede the deadline"
        return real_wait(proc)

    monkeypatch.setattr(process, "_wait_for_guardian_exit", wait_for_cleanup)
    started = time.monotonic()
    result = _supervise(tmp_path / "grace", [sys.executable, "-c", root], timeout=budget)

    assert (result.timed_out, result.return_code, result.cleanup_confirmed) == (False, 0, True)
    assert result.failure_reason is not None
    assert "still had live descendants 0.3s later" in result.failure_reason
    assert result.duration_seconds > budget, "test did not exercise cleanup beyond the budget"


def _spawn_guardian(tmp_path: Path, argv: list[str]) -> tuple[subprocess.Popen[bytes], int, int]:
    """Start a guardian for ``argv`` and wait until it reports the trainer root's exit.

    :param Path tmp_path: Attempt directory, whose lifecycle the guardian writes.
    :param list[str] argv: Trainer command.
    :return tuple[subprocess.Popen[bytes], int, int]: Guardian, trainer group and the
        status pipe, already past its ``X`` byte.
    """
    process.write_attempt_lifecycle(tmp_path, attempt_id="settle", state="launching")
    with (tmp_path / "guardian.log").open("w") as output:
        guardian, pgid, ack_write, status_read = process._spawn_blocked_supervisor(
            stdout=output,
            stderr=output,
            lifecycle_args=(str(tmp_path), "settle", str(time.monotonic())),
        )
    try:
        process._write_all(
            ack_write,
            process._encode_launch_payload("exec " + shlex.join(argv), os.environ.copy(), None),
        )
    finally:
        os.close(ack_write)
    assert os.read(status_read, 1) == supervisor._TRAINER_ROOT_EXITED
    return guardian, pgid, status_read


def _finish_guardian(guardian: subprocess.Popen[bytes], pgid: int, status_read: int) -> bytes:
    """Wait for the guardian and return the rest of its status frame.

    :param subprocess.Popen[bytes] guardian: Guardian to wait for.
    :param int pgid: Its trainer group, killed if the guardian left it behind.
    :param int status_read: Status pipe, closed here.
    :return bytes: Everything the guardian wrote after ``X``.
    """
    try:
        guardian.wait(timeout=10)
        return os.read(status_read, 4096)
    finally:
        os.close(status_read)
        if reaper._process_group_alive(pgid):
            reaper._terminate_process_group(pgid, grace_seconds=0.1)
        process._unregister(pgid)


@pytest.mark.parametrize("child", ["helper", "persistent"])
def test_cancellation_after_root_exit_does_not_cut_the_settle_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, child: str
) -> None:
    """A SIGTERM to the guardian after ``X`` changes neither the helper's exit nor a leak's verdict."""
    trainer, marks = _trainer(
        tmp_path, _HELPER_TRAINER if child == "helper" else _PERSISTENT_TRAINER
    )
    if child == "helper":
        use_guardian_timing(monkeypatch, tmp_path, settle_seconds=5.0, term_grace_seconds=0.5)
        argv = [sys.executable, str(trainer), str(marks), "0.5", "0"]
    else:
        use_guardian_timing(monkeypatch, tmp_path, settle_seconds=0.3, term_grace_seconds=0.3)
        argv = [sys.executable, str(trainer), str(marks)]

    guardian, pgid, status_read = _spawn_guardian(tmp_path, argv)
    guardian.send_signal(signal.SIGTERM)
    rest = _finish_guardian(guardian, pgid, status_read)

    assert guardian.returncode == 0
    lifecycle = read_attempt_lifecycle(tmp_path, expected_attempt_id="settle")
    assert lifecycle is not None and lifecycle.state == "exited" and lifecycle.cleanup_confirmed
    if child == "helper":
        assert rest == b""
        assert lifecycle.return_code == 0
        assert (marks / "helper_done").exists() and not (marks / "helper_sigterm").exists()
    else:
        assert rest.startswith(supervisor._DESCENDANTS_REAPED)
        assert lifecycle.return_code == 1
        child_pid = int((marks / "child_pid").read_text())
        assert_descendant_dies(child_pid, on_timeout_msg=f"child {child_pid} survived the guardian")


def test_the_guarded_shutdown_window_covers_the_settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shutdown confirms a guarded group whose helper outlives the plain kill window."""
    use_guardian_timing(monkeypatch, tmp_path, settle_seconds=4.0, term_grace_seconds=0.5)
    trainer, marks = _trainer(tmp_path, _HELPER_TRAINER)
    grace = 0.1
    linger = 2.6  # beyond grace plus the 2 s kill window, inside it plus the settle

    guardian, pgid, status_read = _spawn_guardian(
        tmp_path, [sys.executable, str(trainer), str(marks), str(linger), "0"]
    )
    try:
        verdict = reaper._terminate_process_groups(
            (pgid,), grace_seconds=grace, guardians={pgid: guardian}
        )
    finally:
        rest = _finish_guardian(guardian, pgid, status_read)

    assert verdict == {pgid: True}
    assert rest == b""
    assert (marks / "helper_done").exists() and not (marks / "helper_sigterm").exists()
    lifecycle = read_attempt_lifecycle(tmp_path, expected_attempt_id="settle")
    assert lifecycle is not None
    assert (lifecycle.state, lifecycle.return_code, lifecycle.cleanup_confirmed) == (
        "exited",
        0,
        True,
    )


#: Logs one line per launch, prints the objective, and leaves a helper that outlives
#: the root by 0.2 s, or by ``LINGER`` seconds when set; exits with ``EXIT_CODE``.
_ENGINE_TRAINER = """
import os, signal, sys, time
from pathlib import Path
marks = Path({marks!r})
with (marks / "calls").open("a") as calls:
    calls.write("run\\n")
print("x=0.5", flush=True)
parent = os.getpid()
if os.fork() == 0:
    signal.signal(signal.SIGTERM, lambda *_: (marks / "helper_sigterm").touch())
    while os.getppid() == parent:
        time.sleep(0.005)
    (marks / "root_gone").touch()
    time.sleep({linger!r})
    (marks / "helper_done").touch()
    os._exit(0)
os._exit({code!r})
"""


@pytest.mark.parametrize("code", [0, 3])
def test_an_engine_trial_with_a_short_lived_helper_keeps_its_own_outcome(
    tmp_path: Path, code: int
) -> None:
    """The trial completes with its objective at exit 0 and keeps its exit-code reason at exit 3."""
    marks = tmp_path / "marks"
    marks.mkdir()
    trainer = write_trainer(
        tmp_path / "trainer.py", _ENGINE_TRAINER.format(marks=str(marks), linger=0.2, code=code)
    )
    experiment = make_experiment(
        persistent=tmp_path, trainer=trainer, n_trials=1, gpu_policy="none"
    )

    if code == 0:
        winners = run_experiment(experiment)
        assert winners["p"].metric == 0.5
    else:
        with pytest.raises(PhaseSweepError):
            run_experiment(experiment)

    (trial,) = phase_trials(experiment)
    if code == 0:
        assert (trial.state, trial.value) == (optuna.trial.TrialState.COMPLETE, 0.5)
        assert FAILURE_REASON_ATTR not in trial.user_attrs
    else:
        assert trial.state == optuna.trial.TrialState.FAIL
        assert str(trial.user_attrs[FAILURE_REASON_ATTR]).startswith("non-zero exit code 3")
    assert not (marks / "helper_sigterm").exists()


@pytest.mark.signals_own_pid
def test_a_shutdown_during_the_settle_keeps_the_finished_result_for_the_next_run(
    tmp_path: Path,
) -> None:
    """A SIGTERM while a helper outlives its root waits for the helper; the next run extracts without retraining."""
    marks = tmp_path / "marks"
    marks.mkdir()
    trainer = write_trainer(
        tmp_path / "trainer.py", _ENGINE_TRAINER.format(marks=str(marks), linger=1.0, code=0)
    )
    experiment = make_experiment(
        persistent=tmp_path, trainer=trainer, n_trials=1, gpu_policy="none"
    )

    def interrupt_once_the_root_exits() -> None:
        deadline = time.monotonic() + 30.0
        while not (marks / "root_gone").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGTERM)

    sender = threading.Thread(target=interrupt_once_the_root_exits, daemon=True)
    sender.start()
    with pytest.raises(PhaseSweepShutdown):
        run_experiment(experiment)
    sender.join(timeout=5.0)

    assert (marks / "helper_done").exists(), "shutdown returned before the helper exited"
    assert not (marks / "helper_sigterm").exists()
    (trial,) = phase_trials(experiment)
    assert trial.state == optuna.trial.TrialState.RUNNING

    winners = run_experiment(experiment)

    (trial,) = phase_trials(experiment)
    assert (trial.state, trial.value) == (optuna.trial.TrialState.COMPLETE, 0.5)
    assert winners["p"].metric == 0.5
    assert (marks / "calls").read_text().splitlines() == ["run"]


@pytest.mark.skipif(sys.platform != "linux", reason="reads Linux procfs command lines")
def test_survivor_descriptions_fit_one_pipe_write(tmp_path: Path) -> None:
    """Four entries of at most 100 characters and a count of the rest stay under 512 bytes."""
    argument = "x" * 300 + "é"
    procs = [
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", argument])
        for _ in range(6)
    ]
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and any(
            argument not in Path(f"/proc/{proc.pid}/cmdline").read_text(errors="replace")
            for proc in procs
        ):
            time.sleep(0.01)
        described = supervisor._describe_members([proc.pid for proc in procs])
    finally:
        for proc in procs:
            proc.kill()
            proc.wait(timeout=5)

    entries = described.split(", ")
    assert entries[-1] == "+2 more"
    assert [entry.split(" ", 1)[0] for entry in entries[:4]] == [str(p.pid) for p in procs[:4]]
    assert all(len(entry) <= 100 and entry.endswith("...)") for entry in entries[:4])
    assert described.isascii() and len(supervisor._DESCENDANTS_REAPED) + len(described) < 512
