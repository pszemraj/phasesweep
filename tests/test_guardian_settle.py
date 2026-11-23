"""The lease guardian gives a trainer's helper processes a bounded settle after the trainer root exits.

A helper that exits by itself within the settle, as CPython's resource tracker or an orphaned
DataLoader worker does, costs nothing: the trial keeps its root's exit code and its objective.
A member still running when the settle ends is terminated, named in the failure reason, and
fails the trial even after a zero exit. Once the root has exited, a cancellation request no
longer cuts the settle short, and every wait that covers the guardian grows by the settle.
On Linux the guardian adopts descendants that left the root's group, such as ranks torchrun
starts in sessions of their own, reaps each one as it exits, and the same rules hold for them.
"""

from __future__ import annotations

import errno
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.engine.paths import _run_log_path
from phasesweep.engine.state import ATTEMPT_ID_ATTR, FAILURE_REASON_ATTR, TRIAL_DIR_ATTR
from phasesweep.errors import PhaseSweepError
from phasesweep.mcp.runs import RUNNER_STOP_GRACE_SECONDS
from phasesweep.runtime import process, reaper, supervisor
from phasesweep.runtime.process import ProcessResult, read_attempt_lifecycle, run_supervised
from phasesweep.runtime.reaper import PROCESS_IDENTITY_FILE, StaleProcessIdentity
from phasesweep.runtime.shutdown import PhaseSweepShutdown
from tests.conftest import (
    assert_descendant_dies,
    console_script,
    make_experiment,
    phase_trials,
    use_guardian_timing,
    write_experiment_config,
    write_trainer,
)

pytestmark = pytest.mark.integration

#: Forks a helper that waits for the root to exit, lingers ``sys.argv[2]`` seconds and exits;
#: the root exits with ``sys.argv[3]``. The helper records a SIGTERM instead of dying of it,
#: and leaves the root's session first when ``sys.argv[4]`` is ``session``.
_HELPER_TRAINER = """
import os, signal, sys, time
from pathlib import Path
marks = Path(sys.argv[1])
linger = float(sys.argv[2])
parent = os.getpid()
if os.fork() == 0:
    if sys.argv[4:] == ["session"]:
        os.setsid()
    signal.signal(signal.SIGTERM, lambda *_: (marks / "helper_sigterm").touch())
    while os.getppid() == parent:
        time.sleep(0.005)
    (marks / "root_gone").touch()
    time.sleep(linger)
    (marks / "helper_done").touch()
    os._exit(0)
os._exit(int(sys.argv[3]))
"""

#: Records its PID and the time each SIGTERM arrives, which it survives.
_PERSISTENT_CHILD = """
import pathlib, signal, sys, time, os
marks = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGTERM, lambda *_: (marks / "child_sigterm").write_text(repr(time.monotonic())))
(marks / "child_pid.tmp").write_text(str(os.getpid()))
(marks / "child_pid.tmp").replace(marks / "child_pid")
time.sleep(60)
"""

#: Starts ``_PERSISTENT_CHILD`` in its own group, records when it exits, and exits with
#: ``sys.argv[2]``, 0 when omitted.
_PERSISTENT_TRAINER = f"""
import subprocess, sys, time
from pathlib import Path
marks = Path(sys.argv[1])
subprocess.Popen([sys.executable, "-c", {_PERSISTENT_CHILD!r}, str(marks)])
while not (marks / "child_pid").exists():
    time.sleep(0.005)
(marks / "root_exit").write_text(repr(time.monotonic()))
sys.exit(int(sys.argv[2]) if sys.argv[2:] else 0)
"""

#: Leaves five orphans that re-parent to the guardian and exit a second later, waits up to
#: five seconds for none of them to remain as a live or zombie child of the guardian, and
#: writes how many it saw adopted and how many are left as zombies.
_ORPHANING_TRAINER = """
import os, subprocess, sys, time
from pathlib import Path
marks = Path(sys.argv[1])
guardian = os.getppid()
def adopted():
    children = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or entry.name == str(os.getpid()):
            continue
        try:
            stat = (entry / "stat").read_bytes()
        except OSError:
            continue
        fields = stat[stat.rfind(b")") + 2 :].split()
        if int(fields[1]) == guardian:
            children[int(entry.name)] = fields[0]
    return children
for _ in range(5):
    subprocess.run(["/bin/sh", "-c", "sleep 1 &"], check=True)
seen = set()
deadline = time.monotonic() + 5.0
while True:
    children = adopted()
    seen.update(children)
    if not children or time.monotonic() >= deadline:
        break
    time.sleep(0.02)
zombies = sum(state == b"Z" for state in children.values())
(marks / "orphans").write_text(f"{len(seen)} {zombies}")
"""

#: Starts ``_PERSISTENT_CHILD`` in a session of its own, as torchrun starts a rank, records
#: when it dies, and SIGKILLs itself, as a killed launcher ends.
_KILLED_LAUNCHER = f"""
import os, signal, subprocess, sys, time
from pathlib import Path
marks = Path(sys.argv[1])
subprocess.Popen([sys.executable, "-c", {_PERSISTENT_CHILD!r}, str(marks)], start_new_session=True)
while not (marks / "child_pid").exists():
    time.sleep(0.005)
(marks / "root_exit").write_text(repr(time.monotonic()))
os.kill(os.getpid(), signal.SIGKILL)
"""

#: Ignores SIGTERM, starts ``_PERSISTENT_CHILD`` in a session of its own, records its own PID
#: once the child is up, and runs until killed.
_STUBBORN_LAUNCHER = f"""
import os, signal, subprocess, sys, time
from pathlib import Path
marks = Path(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
subprocess.Popen([sys.executable, "-c", {_PERSISTENT_CHILD!r}, str(marks)], start_new_session=True)
while not (marks / "child_pid").exists():
    time.sleep(0.005)
(marks / "root_pid.tmp").write_text(str(os.getpid()))
(marks / "root_pid.tmp").replace(marks / "root_pid")
time.sleep(60)
"""

#: ``use_guardian_timing`` patch under which the kernel refuses ``PR_SET_CHILD_SUBREAPER``.
_REFUSED_SUBREAPER_PATCH = """
import ctypes
class _RefusingLibc:
    def prctl(self, *_args):
        return -1
ctypes.CDLL = lambda *_args, **_kwargs: _RefusingLibc()
"""

_LINUX_ADOPTION = pytest.mark.skipif(
    sys.platform != "linux", reason="adoption needs a Linux child subreaper"
)


def _trainer(tmp_path: Path, body: str) -> tuple[Path, Path]:
    """Write a trainer script and the directory it leaves its marks in.

    :param Path tmp_path: Test directory.
    :param str body: Trainer source.
    :return tuple[Path, Path]: The trainer script and its marks directory.
    """
    marks = tmp_path / "marks"
    marks.mkdir()
    return write_trainer(tmp_path / "trainer.py", body), marks


def _supervise(
    trial_dir: Path, argv: list[str], *, timeout: float, gpu_lease_fds: tuple[int, ...] = ()
) -> ProcessResult:
    """Run ``argv`` as a supervised trial in ``trial_dir``.

    :param Path trial_dir: Attempt directory, created here.
    :param list[str] argv: Trainer command, run through ``exec`` so it is the group root.
    :param float timeout: Trial wallclock budget.
    :param tuple[int, ...] gpu_lease_fds: Descriptors the guardian inherits as its leases.
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
            gpu_lease_fds=gpu_lease_fds,
        )


def _parent_of(pid: int) -> int:
    """Return the parent PID that ``/proc/<pid>/stat`` records.

    :param int pid: Live process.
    :return int: Its parent's PID.
    """
    stat = Path(f"/proc/{pid}/stat").read_bytes()
    return int(stat[stat.rfind(b")") + 2 :].split()[1])


def _wait_for_mark(mark: Path) -> None:
    """Wait up to ten seconds for a trainer to create ``mark``.

    :param Path mark: File the trainer creates.
    """
    deadline = time.monotonic() + 10.0
    while not mark.exists():
        assert time.monotonic() < deadline, f"{mark.name} never appeared"
        time.sleep(0.01)


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
    assert elapsed >= result.duration_seconds + 0.2, "the trial's time includes the helper"
    assert elapsed < supervisor._HELPER_SETTLE_SECONDS / 2


def test_a_trials_time_ends_when_its_root_exits(tmp_path: Path) -> None:
    """A root that exits at once and leaves a helper alive for three seconds reports its own short time, in the result and in the lifecycle record."""
    trainer, marks = _trainer(tmp_path, _HELPER_TRAINER)
    trial_dir = tmp_path / "time"

    started = time.monotonic()
    result = _supervise(
        trial_dir, [sys.executable, str(trainer), str(marks), "3.0", "0"], timeout=30.0
    )
    elapsed = time.monotonic() - started

    assert (result.return_code, result.failure_reason) == (0, None)
    assert (marks / "helper_done").exists(), "the guardian did not wait for the helper"
    lifecycle = read_attempt_lifecycle(trial_dir, expected_attempt_id=trial_dir.name)
    assert lifecycle is not None and lifecycle.duration_seconds is not None
    assert elapsed - result.duration_seconds > 2.0
    assert elapsed - lifecycle.duration_seconds > 2.0


def test_a_persistent_child_is_terminated_after_the_settle_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A SIGTERM-ignoring child is signalled only once the settle ends, killed, and named; the trial's reason names it, so run.log alone keeps the INFO line."""
    caplog.set_level(logging.INFO, logger="phasesweep.runtime.process")
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
        "trainer root exited with code 0, but the trainer still had live descendants 0.5s later"
    )
    assert result.failure_reason.endswith(
        ". They were terminated. Make the trainer wait for its children before exiting."
    )
    if sys.platform == "linux":
        executable = Path(sys.executable).resolve().name
        assert f": {child_pid} ({executable})." in result.failure_reason
    named = [r.levelno for r in caplog.records if "still had live descendants" in r.getMessage()]
    assert named == [logging.INFO]
    _assert_exited(trial_dir, 1)
    assert (trial_dir / PROCESS_IDENTITY_FILE).exists()
    root_exit = float((marks / "root_exit").read_text())
    assert float((marks / "child_sigterm").read_text()) - root_exit >= 0.5


def test_the_settle_is_outside_the_trial_budget(tmp_path: Path) -> None:
    """A helper that outlives a 1 s budget by itself neither times out nor fails the trial."""
    trainer, marks = _trainer(tmp_path, _HELPER_TRAINER)
    budget = 1.0

    started = time.monotonic()
    result = _supervise(
        tmp_path / "budget", [sys.executable, str(trainer), str(marks), "1.5", "0"], timeout=budget
    )

    assert (result.timed_out, result.failure_reason, result.return_code) == (False, None, 0)
    assert time.monotonic() - started > budget, "the helper did not outlive the budget"
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
    assert time.monotonic() - started > budget, "test did not exercise cleanup beyond the budget"


@_LINUX_ADOPTION
def test_a_killed_launchers_session_escaped_child_holds_the_lease_until_terminated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The guardian adopts a child its SIGKILLed root started in a new session, holds the lease while it runs, then terminates it and names it in the log, leaving the kill as the trial's reason."""
    caplog.set_level(logging.INFO, logger="phasesweep.runtime.process")
    use_guardian_timing(monkeypatch, tmp_path, settle_seconds=0.5, term_grace_seconds=0.5)
    trainer, marks = _trainer(tmp_path, _KILLED_LAUNCHER)
    lease_read, lease_write = os.pipe()
    real_wait = process._wait_for_guardian_exit

    def wait_for_cleanup(proc: subprocess.Popen[bytes]) -> bool:
        # The root is dead and its child still runs, now as the guardian's own child.
        child_pid = int((marks / "child_pid").read_text())
        assert reaper.is_pid_alive(child_pid)
        assert _parent_of(child_pid) == proc.pid
        assert proc.poll() is None
        lease = os.readlink(f"/proc/self/fd/{lease_write}")
        assert os.readlink(f"/proc/{proc.pid}/fd/{lease_write}") == lease
        return real_wait(proc)

    monkeypatch.setattr(process, "_wait_for_guardian_exit", wait_for_cleanup)
    trial_dir = tmp_path / "launcher"
    try:
        result = _supervise(
            trial_dir,
            [sys.executable, str(trainer), str(marks)],
            timeout=30.0,
            gpu_lease_fds=(lease_write,),
        )
    finally:
        os.close(lease_write)
        os.close(lease_read)

    child_pid = int((marks / "child_pid").read_text())
    assert_descendant_dies(child_pid, on_timeout_msg=f"child {child_pid} survived the guardian")
    assert (result.return_code, result.failure_reason, result.cleanup_confirmed) == (
        -signal.SIGKILL,
        None,
        True,
    )
    executable = Path(sys.executable).resolve().name
    named = f"still had live descendants 0.5s later ({child_pid} ({executable}))"
    assert [r.levelno for r in caplog.records if named in r.getMessage()] == [logging.INFO]
    _assert_exited(trial_dir, -signal.SIGKILL)
    root_exit = float((marks / "root_exit").read_text())
    assert float((marks / "child_sigterm").read_text()) - root_exit >= 0.5


@_LINUX_ADOPTION
@pytest.mark.parametrize("code", [0, 3])
def test_a_session_escaped_helper_that_exits_within_the_settle_keeps_the_roots_exit(
    tmp_path: Path, code: int
) -> None:
    """A helper in a session of its own is waited for, not signalled, and the root's exit stands."""
    trainer, marks = _trainer(tmp_path, _HELPER_TRAINER)
    trial_dir = tmp_path / f"escaped-{code}"

    result = _supervise(
        trial_dir,
        [sys.executable, str(trainer), str(marks), "0.3", str(code), "session"],
        timeout=30.0,
    )

    assert (result.return_code, result.failure_reason, result.cleanup_confirmed) == (
        code,
        None,
        True,
    )
    assert (marks / "helper_done").exists(), "the guardian did not wait for the adopted helper"
    assert not (marks / "helper_sigterm").exists()
    _assert_exited(trial_dir, code)


@_LINUX_ADOPTION
def test_a_refused_subreaper_leaves_only_the_roots_group_watched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the kernel refuses ``PR_SET_CHILD_SUBREAPER``, a helper in its own session is not waited for."""
    use_guardian_timing(
        monkeypatch,
        tmp_path,
        settle_seconds=5.0,
        term_grace_seconds=0.5,
        patch=_REFUSED_SUBREAPER_PATCH,
    )
    trainer, marks = _trainer(tmp_path, _HELPER_TRAINER)
    trial_dir = tmp_path / "refused"

    result = _supervise(
        trial_dir,
        [sys.executable, str(trainer), str(marks), "1.0", "0", "session"],
        timeout=30.0,
    )

    assert (result.return_code, result.failure_reason, result.cleanup_confirmed) == (0, None, True)
    assert not (marks / "helper_done").exists(), "the guardian waited for a helper it cannot adopt"
    _assert_exited(trial_dir, 0)
    _wait_for_mark(marks / "helper_done")


@_LINUX_ADOPTION
def test_orphans_that_exit_while_the_root_runs_are_reaped_at_once(tmp_path: Path) -> None:
    """An adopted process that exits while the root still runs leaves no zombie under the guardian."""
    trainer, marks = _trainer(tmp_path, _ORPHANING_TRAINER)

    result = _supervise(
        tmp_path / "orphans", [sys.executable, str(trainer), str(marks)], timeout=30
    )

    assert (result.return_code, result.failure_reason, result.cleanup_confirmed) == (0, None, True)
    assert (marks / "orphans").read_text() == "5 0"


#: Leaves a helper in its own group for ``sys.argv[2]`` seconds and, through a parent that
#: exits at once, an orphan in a session of its own that exits half a second after the root.
_ORPHAN_DURING_SETTLE = """
import os, sys, time
from pathlib import Path
marks = Path(sys.argv[1])
if os.fork() == 0:
    time.sleep(float(sys.argv[2]))
    (marks / "helper_done").touch()
    os._exit(0)
if os.fork() == 0:
    if os.fork() == 0:
        os.setsid()
        while not (marks / "root_gone").exists():
            time.sleep(0.005)
        time.sleep(0.5)
        (marks / "orphan_done").touch()
        os._exit(0)
    os._exit(0)
(marks / "root_gone").touch()
os._exit(0)
"""


def _zombie_children(parent: int) -> list[int]:
    """List the children of ``parent`` that procfs shows as zombies.

    :param int parent: Process whose children are listed.
    :return list[int]: PIDs in state ``Z`` whose parent is ``parent``.
    """
    zombies = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_bytes()
        except OSError:
            continue
        fields = stat[stat.rfind(b")") + 2 :].split()
        if fields[0] == b"Z" and int(fields[1]) == parent:
            zombies.append(int(entry.name))
    return zombies


@_LINUX_ADOPTION
def test_an_orphan_that_exits_while_a_helper_keeps_the_group_alive_is_reaped_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the root exits, an adopted child that exits during the helper wait leaves no zombie under the guardian, though a helper keeps the root's group alive."""
    trainer, marks = _trainer(tmp_path, _ORPHAN_DURING_SETTLE)
    real_wait = process._wait_for_guardian_exit
    zombies: list[int] = []

    def wait_for_cleanup(proc: subprocess.Popen[bytes]) -> bool:
        _wait_for_mark(marks / "orphan_done")
        # Past the guardian's longest poll interval, so it has looked at least once.
        time.sleep(2 * supervisor._GROUP_MAX_POLL_SECONDS)
        assert not (marks / "helper_done").exists(), "the helper ended before the check"
        zombies.extend(_zombie_children(proc.pid))
        return real_wait(proc)

    monkeypatch.setattr(process, "_wait_for_guardian_exit", wait_for_cleanup)
    result = _supervise(
        tmp_path / "settle-orphan", [sys.executable, str(trainer), str(marks), "3.0"], timeout=30
    )

    assert zombies == []
    assert (result.return_code, result.failure_reason, result.cleanup_confirmed) == (0, None, True)
    assert (marks / "helper_done").exists()


def _spawn_guardian(
    tmp_path: Path, argv: list[str], *, until_root_exit: bool = True
) -> tuple[subprocess.Popen[bytes], int, int]:
    """Start a guardian for ``argv`` and, by default, wait until it reports the root's exit.

    :param Path tmp_path: Attempt directory, whose lifecycle the guardian writes.
    :param list[str] argv: Trainer command.
    :param bool until_root_exit: Whether to read the ``X`` byte before returning.
    :return tuple[subprocess.Popen[bytes], int, int]: Guardian, trainer group and the
        status pipe, past its ``X`` byte when ``until_root_exit`` is set.
    """
    process.write_attempt_lifecycle(tmp_path, attempt_id="settle", state="launching")
    with (tmp_path / "guardian.log").open("w") as output:
        guardian, pgid, ack_write, status_read = process._spawn_blocked_supervisor(
            stdout=output,
            stderr=output,
            lifecycle_args=(str(tmp_path), "settle", str(time.monotonic())),
            label="trial",
        )
    try:
        process._write_all(
            ack_write,
            process._encode_launch_payload("exec " + shlex.join(argv), os.environ.copy(), None),
        )
    finally:
        os.close(ack_write)
    if until_root_exit:
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


#: Makes the guardian SIGTERM itself while its root is a zombie, before or after its own
#: blocking ``waitpid`` reaps it; ``os.kill`` runs the handler before it returns. The root
#: is the guardian's first child to exit, because the trainer's helpers outlive it.
_RACE_PATCH = """
import os, signal
real_waitpid = os.waitpid
fired = []
def racing_waitpid(pid, options):
    if options != 0 or fired:
        return real_waitpid(pid, options)
    fired.append(True)
    if {before_reap!r}:
        os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOWAIT)
        os.kill(os.getpid(), signal.SIGTERM)
        return real_waitpid(pid, options)
    result = real_waitpid(pid, options)
    os.kill(os.getpid(), signal.SIGTERM)
    return result
os.waitpid = racing_waitpid
"""


@pytest.mark.skipif(sys.platform != "linux", reason="holds the zombie root with os.waitid")
@pytest.mark.parametrize("before_reap", [True, False], ids=["before-reap", "after-reap"])
def test_cancellation_between_root_exit_and_reap_keeps_the_finished_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, before_reap: bool
) -> None:
    """A SIGTERM landing after the root exited, before the guardian ignores SIGTERM, is not a cancellation."""
    use_guardian_timing(
        monkeypatch,
        tmp_path,
        settle_seconds=5.0,
        term_grace_seconds=0.5,
        patch=_RACE_PATCH.format(before_reap=before_reap),
    )
    trainer, marks = _trainer(tmp_path, _HELPER_TRAINER)

    guardian, pgid, status_read = _spawn_guardian(
        tmp_path, [sys.executable, str(trainer), str(marks), "0.3", "0"]
    )
    rest = _finish_guardian(guardian, pgid, status_read)

    assert (guardian.returncode, rest) == (0, b"")
    lifecycle = read_attempt_lifecycle(tmp_path, expected_attempt_id="settle")
    assert lifecycle is not None
    assert (lifecycle.state, lifecycle.return_code, lifecycle.cleanup_confirmed) == (
        "exited",
        0,
        True,
    )
    assert (marks / "helper_done").exists() and not (marks / "helper_sigterm").exists()


@_LINUX_ADOPTION
def test_a_cancellation_after_root_exit_ends_the_wait_while_an_adopted_child_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handler that reaps an exited root ends the guardian's wait though an adopted child runs on, and the root's exit 3 is recorded."""
    use_guardian_timing(
        monkeypatch,
        tmp_path,
        settle_seconds=0.3,
        term_grace_seconds=0.3,
        patch=_RACE_PATCH.format(before_reap=True),
    )
    trainer, marks = _trainer(tmp_path, _PERSISTENT_TRAINER)

    guardian, pgid, status_read = _spawn_guardian(
        tmp_path, [sys.executable, str(trainer), str(marks), "3"]
    )
    rest = _finish_guardian(guardian, pgid, status_read)

    child_pid = int((marks / "child_pid").read_text())
    assert_descendant_dies(child_pid, on_timeout_msg=f"child {child_pid} survived the guardian")
    assert guardian.returncode == 3
    assert rest.startswith(supervisor._DESCENDANTS_REAPED + f"{child_pid} (".encode())
    lifecycle = read_attempt_lifecycle(tmp_path, expected_attempt_id="settle")
    assert lifecycle is not None
    assert (lifecycle.state, lifecycle.return_code, lifecycle.cleanup_confirmed) == (
        "exited",
        3,
        True,
    )


def test_the_guarded_shutdown_window_covers_the_settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Shutdown confirms a guarded group whose helper outlives the plain kill window, without a SIGKILL warning when the guardian exits inside the window."""
    caplog.set_level(logging.WARNING, logger="phasesweep.runtime.process")
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
    assert not [r.getMessage() for r in caplog.records if "SIGKILL" in r.getMessage()]
    lifecycle = read_attempt_lifecycle(tmp_path, expected_attempt_id="settle")
    assert lifecycle is not None
    assert (lifecycle.state, lifecycle.return_code, lifecycle.cleanup_confirmed) == (
        "exited",
        0,
        True,
    )


def test_the_guarded_shutdown_window_allows_what_the_guardian_exit_wait_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guardian that never exits is waited for as long as a trial's own wait for its guardian allows, and an MCP runner's stop grace outlasts that."""
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(reaper, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(reaper, "_process_group_alive", lambda pgid: True)
    monkeypatch.setattr(reaper, "_process_group_alive_with_members", lambda pgid, members: True)
    monkeypatch.setattr(reaper, "_group_member_pids", lambda pgid: SimpleNamespace(pids=()))
    signalled: list[int] = []
    guardian = SimpleNamespace(pid=4321, poll=lambda: None, send_signal=signalled.append)
    waited: list[float] = []

    def wait(timeout: float) -> None:
        waited.append(timeout)

    verdict = reaper._terminate_process_groups(
        (1234,), grace_seconds=reaper._KILL_GRACE_SECONDS, guardians={1234: guardian}
    )
    assert process._wait_for_guardian_exit(SimpleNamespace(pid=4321, wait=wait))

    allowance = reaper.guardian_exit_timeout()
    assert (verdict, signalled, waited) == ({1234: False}, [signal.SIGTERM], [allowance])
    assert allowance <= clock[0] < allowance + 0.5
    assert allowance < RUNNER_STOP_GRACE_SECONDS


@_LINUX_ADOPTION
def test_cancellation_kills_a_session_escaped_child_once_its_launcher_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled launcher that ignores SIGTERM is killed after the grace; its child in a new session is then adopted and killed at once, before cleanup is confirmed."""
    use_guardian_timing(monkeypatch, tmp_path, settle_seconds=5.0, term_grace_seconds=0.5)
    trainer, marks = _trainer(tmp_path, _STUBBORN_LAUNCHER)

    guardian, pgid, status_read = _spawn_guardian(
        tmp_path, [sys.executable, str(trainer), str(marks)], until_root_exit=False
    )
    sent = time.monotonic()
    try:
        _wait_for_mark(marks / "root_pid")
        sent = time.monotonic()
    finally:
        guardian.send_signal(signal.SIGTERM)
        rest = _finish_guardian(guardian, pgid, status_read)
    elapsed = time.monotonic() - sent

    root_pid = int((marks / "root_pid").read_text())
    child_pid = int((marks / "child_pid").read_text())
    assert not reaper.is_pid_alive(root_pid)
    assert not reaper.is_pid_alive(child_pid), "cleanup was confirmed while the child ran"
    assert not (marks / "child_sigterm").exists(), "the adopted child got a grace of its own"
    assert (guardian.returncode, rest) == (-signal.SIGTERM, supervisor._TRAINER_ROOT_EXITED)
    lifecycle = read_attempt_lifecycle(tmp_path, expected_attempt_id="settle")
    assert lifecycle is not None
    assert (
        lifecycle.state,
        lifecycle.ended_by,
        lifecycle.return_code,
        lifecycle.cleanup_confirmed,
    ) == ("exited", "cancelled", None, True)
    assert 0.5 <= elapsed < 5.0


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


#: Prints the objective, starts ``_PERSISTENT_CHILD``, which outlives any settle, prints an
#: error line on stderr and exits 3, as a crash that leaves a helper running ends.
_ENGINE_CRASH = """
import subprocess, sys, time
from pathlib import Path
marks = Path({marks!r})
print("x=0.5", flush=True)
subprocess.Popen([sys.executable, "-c", {child!r}, str(marks)])
while not (marks / "child_pid").exists():
    time.sleep(0.005)
print("torch.OutOfMemoryError: CUDA out of memory", file=sys.stderr, flush=True)
sys.exit(3)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="names survivors from procfs")
def test_a_crash_that_leaves_a_child_running_keeps_its_own_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exit 3 whose child outlives the settle fails with the exit code and the stderr line; run.log names the child."""
    use_guardian_timing(monkeypatch, tmp_path, settle_seconds=0.3, term_grace_seconds=0.3)
    marks = tmp_path / "marks"
    marks.mkdir()
    trainer = write_trainer(
        tmp_path / "trainer.py",
        _ENGINE_CRASH.format(marks=str(marks), child=_PERSISTENT_CHILD),
    )
    experiment = make_experiment(
        persistent=tmp_path, trainer=trainer, n_trials=1, gpu_policy="none"
    )

    with pytest.raises(PhaseSweepError):
        run_experiment(experiment)

    child_pid = int((marks / "child_pid").read_text())
    assert_descendant_dies(child_pid, on_timeout_msg=f"child {child_pid} survived the guardian")
    (trial,) = phase_trials(experiment)
    assert trial.state == optuna.trial.TrialState.FAIL
    assert trial.user_attrs[FAILURE_REASON_ATTR] == (
        "non-zero exit code 3; stderr: torch.OutOfMemoryError: CUDA out of memory"
    )
    run_log = _run_log_path(experiment).read_text(encoding="utf-8")
    assert (
        f"exited with code 3 but the trainer still had live descendants 0.3s later ({child_pid} ("
        in run_log
    )


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


@_LINUX_ADOPTION
@pytest.mark.parametrize("older_kernel", [False, True], ids=["native", "no-pidfd-syscall"])
def test_restart_during_the_settle_keeps_the_killed_controllers_finished_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, older_kernel: bool
) -> None:
    """Recovery waits for an escaped helper's guardian and retains the original objective."""
    trainer, marks = _trainer(
        tmp_path,
        r"""
        import os, signal, time
        from pathlib import Path
        marks = Path(__file__).parent / "marks"
        with (marks / "calls").open("a") as calls:
            calls.write("run\n")
        print("x=0.5", flush=True)
        parent = os.getpid()
        if os.fork() == 0:
            os.setsid()
            signal.signal(signal.SIGTERM, lambda *_: (marks / "helper_sigterm").touch())
            while os.getppid() == parent:
                time.sleep(0.005)
            (marks / "helper_pid").write_text(str(os.getpid()))
            (marks / "root_gone").touch()
            deadline = time.monotonic() + 30
            while not (marks / "release").exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            (marks / "helper_done").touch()
            os._exit(0)
        os._exit(0)
        """,
    )
    experiment = make_experiment(
        persistent=tmp_path, trainer=trainer, n_trials=1, gpu_policy="none"
    )
    experiment = experiment.model_copy(update={"trial_command": "exec " + experiment.trial_command})
    config = write_experiment_config(tmp_path / "experiment.yaml", experiment)
    with (tmp_path / "controller.log").open("w") as output:
        controller = subprocess.Popen(
            [str(console_script("phasesweep")), "run", str(config)],
            stdout=output,
            stderr=output,
        )
    try:
        _wait_for_mark(marks / "root_gone")
        helper_pid = int((marks / "helper_pid").read_text())
        (trial,) = phase_trials(experiment)
        assert trial.state == optuna.trial.TrialState.RUNNING
        attempt = trial.user_attrs[ATTEMPT_ID_ATTR]
        trial_dir = Path(trial.user_attrs[TRIAL_DIR_ATTR])
        identity = reaper.read_stale_process_identity(trial_dir, expected_attempt_id=attempt)
        assert identity.guardian_starttime is not None
        assert _parent_of(helper_pid) == identity.guardian_pid
        deadline = time.monotonic() + 5
        while reaper._process_group_alive(identity.pgid):
            assert time.monotonic() < deadline, "trainer group still lives"
            time.sleep(0.01)
        controller.kill()
        assert controller.wait(timeout=5) == -signal.SIGKILL

        if older_kernel:

            def no_pidfd(pid: int) -> int:
                raise OSError(errno.ENOSYS, "pidfd_open unavailable")

            monkeypatch.setattr(os, "pidfd_open", no_pidfd, raising=False)

        settle = reaper._stop_stale_guardian

        def release_helper(stale: StaleProcessIdentity) -> bool:
            """Release the helper only once recovery has found its guardian.

            :param StaleProcessIdentity stale: Verified identity selected for guardian cleanup.
            :return bool: Whether the guardian has exited.
            """
            assert stale == identity
            assert reaper.is_pid_alive(helper_pid)
            (marks / "release").touch()
            return settle(stale)

        monkeypatch.setattr(reaper, "_stop_stale_guardian", release_helper)
        winners = run_experiment(experiment)
        assert (marks / "helper_done").exists()
        assert not (marks / "helper_sigterm").exists()
        assert not reaper.is_pid_alive(helper_pid)
        (recovered,) = phase_trials(experiment)
        assert (recovered.state, recovered.value) == (optuna.trial.TrialState.COMPLETE, 0.5)
        assert winners["p"].trial_number == trial.number
        assert (marks / "calls").read_text().splitlines() == ["run"]
        lifecycle = read_attempt_lifecycle(trial_dir, expected_attempt_id=attempt)
        assert lifecycle is not None
        assert (lifecycle.state, lifecycle.return_code, lifecycle.cleanup_confirmed) == (
            "exited",
            0,
            True,
        )
    finally:
        (marks / "release").touch()
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=5)


#: A rank its launcher starts in a session of its own: records its PID, then on SIGTERM
#: records the time and exits.
_ENGINE_RANK = """
import os, pathlib, signal, sys, time
marks = pathlib.Path(sys.argv[1])
def stop(*_):
    (marks / "rank_stopped").write_text(repr(time.monotonic()))
    os._exit(0)
signal.signal(signal.SIGTERM, stop)
(marks / "rank_pid.tmp").write_text(str(os.getpid()))
(marks / "rank_pid.tmp").replace(marks / "rank_pid")
time.sleep(60)
"""

#: Logs each launch's start time and prints the objective; the first launch starts
#: ``_ENGINE_RANK`` in a new session and SIGKILLs itself, as a killed launcher ends.
_ENGINE_LAUNCHER = """
import os, signal, subprocess, sys, time
from pathlib import Path
marks = Path({marks!r})
with (marks / "calls").open("a") as calls:
    calls.write(repr(time.monotonic()) + "\\n")
print("x=0.5", flush=True)
if not (marks / "rank_pid").exists():
    subprocess.Popen([sys.executable, "-c", {rank!r}, str(marks)], start_new_session=True)
    while not (marks / "rank_pid").exists():
        time.sleep(0.005)
    os.kill(os.getpid(), signal.SIGKILL)
"""


@_LINUX_ADOPTION
def test_the_next_trial_waits_for_the_rank_a_killed_launcher_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trial whose launcher is killed fails with the kill as its reason and its rank named in run.log, and the next trial starts only once that rank is gone."""
    use_guardian_timing(monkeypatch, tmp_path, settle_seconds=0.3, term_grace_seconds=0.3)
    marks = tmp_path / "marks"
    marks.mkdir()
    trainer = write_trainer(
        tmp_path / "trainer.py", _ENGINE_LAUNCHER.format(marks=str(marks), rank=_ENGINE_RANK)
    )
    experiment = make_experiment(
        persistent=tmp_path,
        trainer=trainer,
        n_trials=2,
        gpu_policy="none",
        max_consecutive_failures=5,
    )

    winners = run_experiment(experiment)

    first, second = sorted(phase_trials(experiment), key=lambda trial: trial.number)
    rank_pid = int((marks / "rank_pid").read_text())
    assert first.state == optuna.trial.TrialState.FAIL
    # The killed launcher's exit is the reason; run.log names the rank that outlived it.
    assert str(first.user_attrs[FAILURE_REASON_ATTR]).startswith("non-zero exit code ")
    run_log = _run_log_path(experiment).read_text(encoding="utf-8")
    assert f"still had live descendants 0.3s later ({rank_pid} (" in run_log
    assert (second.state, second.value) == (optuna.trial.TrialState.COMPLETE, 0.5)
    assert winners["p"].metric == 0.5
    starts = [float(line) for line in (marks / "calls").read_text().splitlines()]
    assert len(starts) == 2
    assert starts[1] > float((marks / "rank_stopped").read_text())
    assert not reaper.is_pid_alive(rank_pid)


@pytest.mark.skipif(sys.platform != "linux", reason="reads the Linux procfs executable link")
def test_survivor_description_omits_command_arguments() -> None:
    """A helper is identified by PID and executable without leaking a credential in its arguments."""
    argument = "--token=sentinel-argument-only"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", argument])
    try:
        deadline = time.monotonic() + 5.0
        cmdline = Path(f"/proc/{proc.pid}/cmdline")
        while time.monotonic() < deadline and argument.encode() not in cmdline.read_bytes():
            time.sleep(0.01)
        assert argument.encode() in cmdline.read_bytes()
        described = supervisor._describe_members([proc.pid])
    finally:
        proc.kill()
        proc.wait(timeout=5)

    assert argument not in described
    assert described == f"{proc.pid} ({Path(sys.executable).resolve().name})"


def test_survivor_descriptions_fit_one_pipe_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """Four entries with long executable names and a count of the rest stay under 512 bytes."""
    executable = "xé" * 100
    pids = [101, 102, 103, 104, 105, 106]
    with monkeypatch.context() as patch:
        patch.setattr(supervisor.os, "readlink", lambda _path: f"/bin/{executable}")
        described = supervisor._describe_members(pids)

    entries = described.split(", ")
    assert entries[-1] == "+2 more"
    assert [entry.split(" ", 1)[0] for entry in entries[:4]] == [str(pid) for pid in pids[:4]]
    assert all(len(entry) <= 100 and entry.endswith("...)") for entry in entries[:4])
    assert described.isascii() and len(supervisor._DESCENDANTS_REAPED) + len(described) < 512
