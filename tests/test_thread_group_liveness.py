"""A process whose main thread ended while its other threads run is live to every group probe.

A thread-group leader that calls ``pthread_exit``, as a C or C++ trainer can, reads ``Z``
in ``/proc/<pid>/stat`` while its other threads keep running. The controller's reaper and
the lease guardian's stdlib copy both count it live, signal it and only then report it
gone, while a dead, unreaped zombie still counts as gone.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NoReturn

import pytest

from phasesweep.runtime import reaper, supervisor
from phasesweep.runtime.process import read_attempt_lifecycle, run_supervised
from tests.conftest import is_pid_zombie, use_guardian_timing

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="reads Linux procfs thread listings"
)

#: Writes its PID to ``sys.argv[1]``, keeps one thread running, and ends its main thread.
_LEADER_EXIT_SCRIPT = (
    "import ctypes, os, pathlib, sys, threading, time\n"
    "threading.Thread(target=time.sleep, args=(60,)).start()\n"
    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
    "ctypes.CDLL(None).pthread_exit(None)\n"
)


def _wait_for(condition: Callable[[], bool], what: str) -> None:
    """Poll ``condition`` for up to five seconds.

    :param Callable[[], bool] condition: Check to repeat until it holds.
    :param str what: Description used in the failure message.
    """
    deadline = time.monotonic() + 5.0
    while not condition():
        if time.monotonic() >= deadline:
            pytest.fail(f"timed out waiting for {what}")
        time.sleep(0.02)


def _leader_exited(pid: int) -> bool:
    """Return whether ``pid`` reads ``Z`` while ``/proc/<pid>/task`` lists another thread.

    :param int pid: Process to inspect.
    :return bool: Whether its main thread has ended and another thread still runs.
    """
    stat = reaper._read_proc_stat(Path("/proc") / str(pid))
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except FileNotFoundError:
        return False
    return stat is not None and stat.state == "Z" and tasks != [str(pid)]


@pytest.fixture
def leader_exited(tmp_path: Path) -> Iterator[subprocess.Popen[bytes]]:
    """Start a process, in its own group, whose main thread has ended while a thread runs."""
    marker = tmp_path / "pid"
    proc = subprocess.Popen(
        [sys.executable, "-c", _LEADER_EXIT_SCRIPT, str(marker)], start_new_session=True
    )
    try:
        _wait_for(lambda: _leader_exited(proc.pid), "the main thread to end")
        yield proc
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


@pytest.mark.integration
def test_leader_exited_process_is_live_to_reaper_and_guardian(
    leader_exited: subprocess.Popen[bytes],
) -> None:
    """Every group and identity probe counts a ``Z`` leader with a running thread as live."""
    pid = leader_exited.pid

    assert reaper._process_group_alive(pid)
    assert not reaper._group_member_pids(pid).all_members_terminal
    assert reaper._member_pids_alive(pid, {pid})
    assert reaper.is_same_live_process(pid, reaper.read_proc_starttime(pid))
    assert supervisor._trainer_group_alive(pid)


@pytest.mark.integration
def test_leader_exited_group_is_signalled_then_reported_gone(
    leader_exited: subprocess.Popen[bytes],
) -> None:
    """Stale cleanup signals the group, the signal reaches the running thread, then it is gone."""
    pid = leader_exited.pid
    starttime = reaper.read_proc_starttime(pid)
    assert starttime is not None

    decision = reaper._stale_group_decision(pid, starttime, pgid=pid)
    assert (decision.action, decision.pgid) == ("terminate", pid)

    assert reaper.kill_stale_group(pid, starttime, pgid=pid, grace_seconds=5.0)
    # The kill is confirmed while the process is still an unreaped zombie of this test.
    assert is_pid_zombie(pid) and os.listdir(f"/proc/{pid}/task") == [str(pid)]
    assert leader_exited.wait(timeout=5) == -signal.SIGTERM


@pytest.mark.integration
def test_guardian_terminates_a_leader_exited_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover descendant whose main thread ended fails the trial and is terminated."""
    use_guardian_timing(monkeypatch, tmp_path, settle_seconds=0.2, term_grace_seconds=5.0)
    marker = tmp_path / "descendant_pid"
    root = (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"subprocess.Popen([sys.executable, '-c', {_LEADER_EXIT_SCRIPT!r}, {str(marker)!r}])\n"
        f"marker = Path({str(marker)!r})\n"
        "deadline = time.monotonic() + 5\n"
        "while time.monotonic() < deadline:\n"
        "    if marker.exists() and marker.read_text():\n"
        "        stat = Path('/proc', marker.read_text(), 'stat').read_bytes()\n"
        "        if stat[stat.rfind(b')') + 2 :].split()[0] == b'Z':\n"
        "            break\n"
        "    time.sleep(0.02)\n"
    )
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    try:
        with (
            (attempt_dir / "out.log").open("w") as out,
            (attempt_dir / "err.log").open("w") as err,
        ):
            result = run_supervised(
                f"exec {shlex.quote(sys.executable)} -c {shlex.quote(root)}",
                env=os.environ.copy(),
                stdout=out,
                stderr=err,
                timeout=30.0,
                attempt_dir=attempt_dir,
                attempt_id="leader-exited-descendant",
            )

        assert result.failure_reason is not None
        assert "still had live descendants 0.2s later" in result.failure_reason
        assert result.cleanup_confirmed
        lifecycle = read_attempt_lifecycle(
            attempt_dir, expected_attempt_id="leader-exited-descendant"
        )
        assert lifecycle is not None and lifecycle.cleanup_confirmed
        descendant = int(marker.read_text())
        _wait_for(lambda: not reaper.is_pid_alive(descendant), "the descendant to be reaped")
    finally:
        if marker.exists() and marker.read_text():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(marker.read_text()), signal.SIGKILL)


@pytest.mark.integration
def test_dead_unreaped_zombie_still_counts_as_gone() -> None:
    """A child that exited and is not yet reaped lists only itself and is gone to every probe."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    try:
        starttime = reaper.read_proc_starttime(proc.pid)
        _wait_for(lambda: is_pid_zombie(proc.pid), "the child to exit")
        assert os.listdir(f"/proc/{proc.pid}/task") == [str(proc.pid)]

        assert not reaper._process_group_alive(proc.pid)
        assert reaper._group_member_pids(proc.pid).all_members_terminal
        assert not reaper.is_same_live_process(proc.pid, starttime)
        assert reaper._stale_group_decision(proc.pid, starttime, pgid=proc.pid).action == "gone"
        assert not supervisor._trainer_group_alive(proc.pid)
    finally:
        proc.wait(timeout=5)


@pytest.mark.parametrize(
    ("error", "exited"),
    [
        pytest.param(PermissionError, False, id="unreadable-listing-keeps-it-live"),
        pytest.param(FileNotFoundError, True, id="vanished-process-has-exited"),
        pytest.param(ProcessLookupError, True, id="released-process-has-exited"),
    ],
)
def test_zombie_without_a_thread_listing_is_live_only_while_it_may_still_run(
    monkeypatch: pytest.MonkeyPatch, error: type[OSError], exited: bool
) -> None:
    """Both copies of the rule fail closed on an unreadable ``task`` listing."""
    real_iterdir = Path.iterdir

    def fail_task_listing(path: Path) -> Iterator[Path]:
        if path.name == "task":
            raise error(str(path))
        return real_iterdir(path)

    def fail_scandir(path: str) -> NoReturn:
        raise error(path)

    monkeypatch.setattr(Path, "iterdir", fail_task_listing)
    monkeypatch.setattr("phasesweep.runtime.supervisor.os.scandir", fail_scandir)

    assert reaper._has_exited(4242, "Z") is exited
    assert supervisor._has_exited(4242, b"Z") is exited
    assert reaper._has_exited(4242, "X") and supervisor._has_exited(4242, b"X")
    assert not reaper._has_exited(4242, "S") and not supervisor._has_exited(4242, b"S")
