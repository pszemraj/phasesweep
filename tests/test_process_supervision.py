"""Supervised subprocess lifecycle: Popen registration, signal handler installation and forwarding, descendant cleanup, cleanup_confirmed propagation, and the launch-window deadlock guard."""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import select
import shlex
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.config.models import ExecutionContext
from phasesweep.engine.cleanup import _reap_stale_trials
from phasesweep.engine.state import ATTEMPT_ID_ATTR, TRIAL_DIR_ATTR
from phasesweep.errors import UnsafeProcessCleanupError
from phasesweep.runtime import supervisor
from phasesweep.runtime.process import (
    _kill_group,
    _spawn_blocked_supervisor,
    _wait_for_guardian_exit,
    read_attempt_lifecycle,
    run_supervised,
)
from phasesweep.runtime.reaper import (
    PROCESS_IDENTITY_FILE,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    StaleProcessIdentity,
    _group_member_pids,
    _GroupMemberScan,
    _process_group_alive,
    _process_group_alive_with_members,
    _ProcStat,
    _terminate_process_group,
    _terminate_process_groups,
    cleanup_stale_trial_process,
    is_pid_alive,
    read_stale_process_identity,
    reap_child,
)
from phasesweep.runtime.shutdown import (
    PhaseSweepShutdown,
    _shutdown_handler,
)
from tests.conftest import is_pid_zombie, make_experiment, raise_after_first_successful_call

pytestmark = pytest.mark.integration


def _report_uncertain_after_real_terminate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch group termination to clean up the group but report uncertainty."""
    import phasesweep.runtime.reaper as _reaper

    real_terminate = _reaper._terminate_process_groups

    def terminate_then_report_uncertain(
        pgids: tuple[int, ...],
        *,
        grace_seconds: float,
        guardians: Mapping[int, subprocess.Popen[bytes]] | None = None,
    ) -> dict[int, bool]:
        real_terminate(pgids, grace_seconds=0.05, guardians=guardians)
        return dict.fromkeys(pgids, False)

    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_groups",
        terminate_then_report_uncertain,
    )


def _run_supervised(
    trial_dir: Path,
    cmd: str,
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    attempt_id: str,
    cwd: str | None = None,
):
    """Create ``trial_dir``, open its logs, and delegate to ``run_supervised``."""
    trial_dir.mkdir(exist_ok=True)
    with (trial_dir / "out.log").open("w") as fout, (trial_dir / "err.log").open("w") as ferr:
        return run_supervised(
            cmd,
            env=env if env is not None else os.environ.copy(),
            stdout=fout,
            stderr=ferr,
            timeout=timeout,
            trial_dir=trial_dir,
            attempt_id=attempt_id,
            cwd=cwd,
        )


def _patch_process_identity_write_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make process-identity persistence fail while preserving other atomic writes."""
    import phasesweep.runtime.process as process

    real_atomic_write_text = process.atomic_write_text

    def fail_identity_write(path: Path, text: str) -> None:
        if path.name == PROCESS_IDENTITY_FILE:
            raise OSError("identity disk full")
        real_atomic_write_text(path, text)

    monkeypatch.setattr(process, "atomic_write_text", fail_identity_write)


@pytest.mark.parametrize(
    ("command", "return_code", "attempt_id"),
    [
        pytest.param("kill -9 $$", -signal.SIGKILL, "failure-attempt", id="killed-trainer"),
        pytest.param("true", 0, "success-attempt", id="clean-exit"),
    ],
)
def test_run_supervised_preserves_identity_and_records_terminal_lifecycle(
    tmp_path: Path, command: str, return_code: int, attempt_id: str
) -> None:
    """Every completed attempt retains its identity and records confirmed terminal cleanup.

    Retained identity proves cleanup after an orchestrator death between process
    exit and Optuna's terminal commit, including clean trainer exits.
    """
    trial_dir = tmp_path / "trial"
    result = _run_supervised(
        trial_dir,
        command,
        timeout=None,
        attempt_id=attempt_id,
    )
    assert result.return_code == return_code
    identity = read_stale_process_identity(
        trial_dir,
        expected_attempt_id=attempt_id,
    )
    assert identity.pid == result.pid
    assert identity.pgid == result.pid
    assert identity.proc_starttime is not None
    assert identity.boot_id is not None
    assert result.duration_seconds >= 0.0
    assert (trial_dir / PROCESS_IDENTITY_FILE).exists()
    lifecycle = read_attempt_lifecycle(trial_dir, expected_attempt_id=attempt_id)
    assert lifecycle is not None
    assert lifecycle.state == "exited"
    assert lifecycle.return_code == return_code
    assert lifecycle.cleanup_confirmed is True


def test_run_supervised_terminates_child_when_identity_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A metadata write failure after Popen must not leave the child running."""
    _patch_process_identity_write_failure(monkeypatch)

    trial_dir = tmp_path / "trial"
    trainer_started = tmp_path / "trainer-started"
    result = _run_supervised(
        trial_dir,
        f'{sys.executable} -c "from pathlib import Path; '
        f"Path({str(trainer_started)!r}).write_text('started')\"",
        timeout=None,
        attempt_id="identity-write-failure",
    )

    assert result.cleanup_confirmed is True
    assert "failed to persist process identity" in (result.failure_reason or "")
    assert not trainer_started.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(result.pid, 0)


@pytest.mark.parametrize("shutdown", [False, True], ids=["spawn-error", "shutdown"])
def test_pre_spawn_failure_records_confirmed_terminal_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shutdown: bool,
) -> None:
    """A failed or interrupted spawn cannot strand a childless attempt."""
    import phasesweep.runtime.process as process
    import phasesweep.runtime.shutdown as shutdown_mod

    def fail_before_spawn(**_kwargs: object) -> None:
        if shutdown:
            raise PhaseSweepShutdown(
                signal.SIGTERM,
                shutdown_mod.ShutdownCleanupReport(signal.SIGTERM, True, ()),
            )
        raise OSError("injected pre-spawn failure")

    monkeypatch.setattr(process, "_spawn_blocked_supervisor", fail_before_spawn)
    trial_dir = tmp_path / "trial"

    expected = PhaseSweepShutdown if shutdown else OSError
    with pytest.raises(expected, match=None if shutdown else "injected pre-spawn failure"):
        _run_supervised(
            trial_dir,
            "true",
            timeout=None,
            attempt_id="pre-spawn-failure",
        )

    lifecycle = read_attempt_lifecycle(
        trial_dir,
        expected_attempt_id="pre-spawn-failure",
    )
    assert lifecycle is not None
    assert lifecycle.state == "exited"
    assert lifecycle.return_code is None
    assert lifecycle.cleanup_confirmed is True
    assert not (trial_dir / PROCESS_IDENTITY_FILE).exists()


def test_pre_spawn_failure_requires_terminal_lifecycle_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A childless attempt cannot hide loss of its only terminal evidence."""
    import phasesweep.runtime.process as process

    real_write = process.write_attempt_lifecycle

    def fail_terminal_write(
        trial_dir: Path,
        *,
        attempt_id: str,
        state: str,
        return_code: int | None = None,
        cleanup_confirmed: bool | None = None,
    ) -> None:
        if state == "exited":
            raise OSError("injected terminal lifecycle write failure")
        real_write(
            trial_dir,
            attempt_id=attempt_id,
            state=state,
            return_code=return_code,
            cleanup_confirmed=cleanup_confirmed,
        )

    spawn_error = RuntimeError("injected pre-spawn failure")

    def fail_before_spawn(**_kwargs: object) -> None:
        raise spawn_error

    monkeypatch.setattr(process, "write_attempt_lifecycle", fail_terminal_write)
    monkeypatch.setattr(process, "_spawn_blocked_supervisor", fail_before_spawn)
    trial_dir = tmp_path / "trial"

    with pytest.raises(OSError, match="terminal lifecycle write failure") as exc_info:
        _run_supervised(
            trial_dir,
            "true",
            timeout=None,
            attempt_id="pre-spawn-terminal-write-failure",
        )

    assert exc_info.value.__cause__ is spawn_error

    lifecycle = read_attempt_lifecycle(
        trial_dir,
        expected_attempt_id="pre-spawn-terminal-write-failure",
    )
    assert lifecycle is not None
    assert lifecycle.state == "launching"
    assert not (trial_dir / PROCESS_IDENTITY_FILE).exists()


@pytest.mark.parametrize("cleanup_confirmed", [True, False])
def test_spawn_interrupt_after_popen_preserves_cleanup_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_confirmed: bool,
) -> None:
    """A control-flow interrupt cannot turn a spawned guardian into no child."""
    import phasesweep.runtime.process as process

    real_popen = subprocess.Popen
    spawned: list[subprocess.Popen] = []

    def capture_popen(argv: list[str], **kwargs: object) -> subprocess.Popen:
        proc = real_popen(argv, **kwargs)  # type: ignore[arg-type]
        spawned.append(proc)
        return proc

    def interrupt_read(_fd: int, _size: int, *, deadline: float) -> bytes:
        del deadline
        raise KeyboardInterrupt

    real_abort = process._abort_launch

    def abort_with_verdict(proc: subprocess.Popen, pgid: int | None) -> bool:
        real_abort(proc, pgid)
        return cleanup_confirmed

    monkeypatch.setattr(process.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(process, "_read_pipe_frame", interrupt_read)
    monkeypatch.setattr(process, "_abort_launch", abort_with_verdict)
    trial_dir = tmp_path / "trial"

    expected = KeyboardInterrupt if cleanup_confirmed else UnsafeProcessCleanupError
    with pytest.raises(expected):
        _run_supervised(
            trial_dir,
            "true",
            timeout=None,
            attempt_id="interrupted-supervisor-spawn",
        )

    assert len(spawned) == 1
    assert spawned[0].poll() is not None
    lifecycle = read_attempt_lifecycle(
        trial_dir,
        expected_attempt_id="interrupted-supervisor-spawn",
    )
    assert lifecycle is not None
    if cleanup_confirmed:
        assert lifecycle.state == "exited"
        assert lifecycle.cleanup_confirmed is True
    else:
        # The guardian may finish independently of the caller's uncertain verdict.
        assert lifecycle.state in {"launching", "terminating", "exited"}
        if lifecycle.state == "exited":
            assert lifecycle.cleanup_confirmed is True
            assert lifecycle.return_code != 0
            assert lifecycle.duration_seconds is not None
        elif lifecycle.state == "terminating":
            assert lifecycle.return_code == -signal.SIGTERM
            assert lifecycle.cleanup_confirmed is False
    assert not (trial_dir / PROCESS_IDENTITY_FILE).exists()


@pytest.mark.parametrize("cleanup_confirmed", [True, False])
def test_payload_delivery_interrupt_cleans_registered_trainer_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_confirmed: bool,
) -> None:
    """An interrupt after payload delivery cannot leave training alive."""
    import phasesweep.runtime.process as process
    import phasesweep.runtime.reaper as reaper
    import phasesweep.runtime.shutdown as shutdown

    real_write_all = process._write_all

    def deliver_then_interrupt(
        fd: int,
        data: bytes,
        *,
        deadline: float | None = None,
        pid: int = -1,
    ) -> None:
        real_write_all(fd, data, deadline=deadline, pid=pid)
        raise KeyboardInterrupt

    real_abort = process._abort_launch

    def abort_with_verdict(proc: subprocess.Popen, pgid: int | None) -> bool:
        real_abort(proc, pgid)
        return cleanup_confirmed

    monkeypatch.setattr(process, "_write_all", deliver_then_interrupt)
    monkeypatch.setattr(process, "_abort_launch", abort_with_verdict)
    trial_dir = tmp_path / "trial"

    expected = KeyboardInterrupt if cleanup_confirmed else UnsafeProcessCleanupError
    with pytest.raises(expected):
        _run_supervised(
            trial_dir,
            "exec sleep 30",
            timeout=None,
            attempt_id="interrupted-payload-delivery",
        )

    identity = read_stale_process_identity(
        trial_dir,
        expected_attempt_id="interrupted-payload-delivery",
    )
    assert not reaper._process_group_alive(identity.pgid)
    with shutdown._lock:
        assert identity.pgid not in shutdown._active_children
    lifecycle = read_attempt_lifecycle(
        trial_dir,
        expected_attempt_id="interrupted-payload-delivery",
    )
    assert lifecycle is not None
    assert lifecycle.state == "exited"
    assert lifecycle.cleanup_confirmed is True
    assert lifecycle.return_code is not None and lifecycle.return_code != 0
    assert lifecycle.duration_seconds is not None and lifecycle.duration_seconds >= 0


def test_identity_write_failure_marks_launch_before_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guardian exit evidence survives a parent cleanup probe reporting uncertainty."""
    import phasesweep.runtime.process as process

    real_abort_launch = process._abort_launch

    def abort_but_report_uncertain(proc: object, pgid: int | None) -> bool:
        real_abort_launch(proc, pgid)  # type: ignore[arg-type]
        return False

    _patch_process_identity_write_failure(monkeypatch)
    monkeypatch.setattr(process, "_abort_launch", abort_but_report_uncertain)

    trial_dir = tmp_path / "trial"
    result = _run_supervised(
        trial_dir,
        "true",
        timeout=None,
        attempt_id="identity-uncertain-attempt",
    )

    assert result.cleanup_confirmed is False
    lifecycle = read_attempt_lifecycle(
        trial_dir,
        expected_attempt_id="identity-uncertain-attempt",
    )
    assert lifecycle is not None
    assert lifecycle.state == "exited"
    assert lifecycle.cleanup_confirmed is True
    assert lifecycle.return_code is not None and lifecycle.return_code != 0
    assert lifecycle.duration_seconds is not None and lifecycle.duration_seconds >= 0
    assert not _process_group_alive(result.pid)
    assert not (trial_dir / PROCESS_IDENTITY_FILE).exists()


@pytest.mark.parametrize("cleanup_return_code", [None, 0, -signal.SIGTERM])
def test_cleanup_retains_cancellation_after_guardian_dies_before_final_publication(
    tmp_path: Path, cleanup_return_code: int | None
) -> None:
    """Cleanup after guardian death retains cancellation through repeated durable confirmations."""
    from phasesweep.runtime.process import _record_attempt_exited, write_attempt_lifecycle

    write_attempt_lifecycle(
        tmp_path,
        attempt_id="cancelled",
        state="terminating",
        return_code=-signal.SIGTERM,
        cleanup_confirmed=False,
    )
    for _ in range(2):
        _record_attempt_exited(tmp_path, attempt_id="cancelled", return_code=cleanup_return_code)
        lifecycle = read_attempt_lifecycle(tmp_path, expected_attempt_id="cancelled")
        assert lifecycle is not None and lifecycle.state == "exited"
        assert lifecycle.return_code == -signal.SIGTERM
        assert lifecycle.cleanup_confirmed and lifecycle.duration_seconds is None


def test_kill_group_reports_unconfirmed_when_direct_child_reap_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct child that never reaps within the bounded wait must report cleanup
    as unconfirmed, not silently ``True``."""
    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_groups",
        lambda pgids, *, grace_seconds, guardians: dict.fromkeys(pgids, True),
    )

    proc = Mock(pid=424243, returncode=None)
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd="fake", timeout=0.0)

    assert _kill_group(999999, proc) is False


def test_hard_parent_death_before_identity_commit_never_starts_trainer(tmp_path: Path) -> None:
    """The supervisor exits on acknowledgement-pipe EOF without execing the trainer."""
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    trainer_started = tmp_path / "trainer-started"
    command = (
        f'{sys.executable} -c "from pathlib import Path; '
        f"Path({str(trainer_started)!r}).write_text('started')\""
    )
    parent_code = f"""
import os
import time
from pathlib import Path
import phasesweep.runtime.process as process
import phasesweep.runtime.reaper as reaper

real_atomic_write_text = process.atomic_write_text
def stall_identity_write(path: Path, text: str) -> None:
    if path.name != reaper.PROCESS_IDENTITY_FILE:
        real_atomic_write_text(path, text)
        return
    print(text, flush=True)
    while True:
        time.sleep(1)

process.atomic_write_text = stall_identity_write
env = os.environ.copy()
env["PHASESWEEP_TRIAL_DIR"] = {str(trial_dir)!r}
with open(os.devnull, "w") as output:
    process.run_supervised(
        {command!r},
        env=env,
        stdout=output,
        stderr=output,
        timeout=None,
        trial_dir=Path({str(trial_dir)!r}),
        attempt_id="pre-commit-attempt",
    )
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    supervisor_pid: int | None = None
    try:
        assert parent.stdout is not None
        readable, _, _ = select.select([parent.stdout], [], [], 10.0)
        assert readable, f"identity write was not reached; parent returncode={parent.poll()}"
        supervisor_pid = json.loads(parent.stdout.readline())["pid"]

        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)
        deadline = time.time() + 5
        while (
            is_pid_alive(supervisor_pid)
            and not is_pid_zombie(supervisor_pid)
            and time.time() < deadline
        ):
            time.sleep(0.05)

        assert not trainer_started.exists()
        assert not (trial_dir / PROCESS_IDENTITY_FILE).exists()
        assert not is_pid_alive(supervisor_pid) or is_pid_zombie(supervisor_pid)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if supervisor_pid is not None and is_pid_alive(supervisor_pid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(supervisor_pid, signal.SIGKILL)


def test_hard_parent_death_after_ack_leaves_recoverable_identity(tmp_path: Path) -> None:
    """Once the trainer starts, a complete identity exists and can reap its group."""
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    trainer_started = tmp_path / "trainer-started"
    command = (
        f'{sys.executable} -c "from pathlib import Path; import time; '
        f"Path({str(trainer_started)!r}).write_text('started'); time.sleep(60)\""
    )
    parent_code = f"""
import os
from pathlib import Path
from phasesweep.runtime.process import run_supervised

env = os.environ.copy()
env["PHASESWEEP_TRIAL_DIR"] = {str(trial_dir)!r}
with open(os.devnull, "w") as output:
    run_supervised(
        {command!r},
        env=env,
        stdout=output,
        stderr=output,
        timeout=None,
        trial_dir=Path({str(trial_dir)!r}),
        attempt_id="post-ack-attempt",
    )
"""
    parent = subprocess.Popen([sys.executable, "-c", parent_code])
    identity = None
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            if trainer_started.exists() and (trial_dir / PROCESS_IDENTITY_FILE).exists():
                identity = read_stale_process_identity(
                    trial_dir,
                    expected_attempt_id="post-ack-attempt",
                )
                break
            if parent.poll() is not None:
                pytest.fail(f"launch parent exited early with code {parent.returncode}")
            time.sleep(0.05)
        assert identity is not None

        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)
        assert cleanup_stale_trial_process(identity, grace_seconds=0.2) is True
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if identity is not None and is_pid_alive(identity.pid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(identity.pgid, signal.SIGKILL)


def _spy_write_process_identity(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Wrap ``_write_process_identity`` so a test can prove it ran, without disturbing it.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to wrap the writer.
    :return list[bool]: Appended to (a single ``True``) once the real writer
        has completed.
    """
    import phasesweep.runtime.process as process

    real_write_identity = process._write_process_identity
    written: list[bool] = []

    def spy(path: Path, identity: StaleProcessIdentity) -> None:
        real_write_identity(path, identity)
        written.append(True)

    monkeypatch.setattr(process, "_write_process_identity", spy)
    return written


@pytest.mark.parametrize(
    ("poison_file", "marker_name", "attempt_id"),
    [
        # The supervisor runs as a stdlib script under a PATH-only env: ``-m`` under the trainer env would import a shadowed ``phasesweep`` before ``process_identity.json`` exists.
        pytest.param(
            "phasesweep/__init__.py",
            "poison-marker",
            "poison-pythonpath",
            id="phasesweep-package",
        ),
        # Even without an ``-m phasesweep...`` import, a bare ``python`` runs ``sitecustomize``/``usercustomize`` during site initialization when their directory is importable at process start; ``-S`` disables that.
        pytest.param(
            "sitecustomize.py",
            "sitecustomize-marker",
            "poison-sitecustomize",
            id="sitecustomize",
        ),
    ],
)
def test_supervisor_never_imports_poisoned_modules_from_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    poison_file: str,
    marker_name: str,
    attempt_id: str,
) -> None:
    """A poisoned PYTHONPATH on the trainer env must never reach the supervisor pre-ACK."""
    poison_root = tmp_path / "poison"
    poison_path = poison_root / poison_file
    poison_path.parent.mkdir(parents=True)
    marker = tmp_path / marker_name
    poison_path.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('poisoned')\n"
    )

    written = _spy_write_process_identity(monkeypatch)

    trial_dir = tmp_path / "trial"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(poison_root)
    result = _run_supervised(trial_dir, "true", env=env, timeout=None, attempt_id=attempt_id)

    assert result.return_code == 0
    assert not marker.exists(), f"supervisor loaded the poisoned {poison_file}"
    assert written == [True], "process_identity.json was never written before exec"


def test_run_supervised_delivers_trainer_env_via_payload(tmp_path: Path) -> None:
    """The framed ack payload must still deliver the full trainer environment
    to the exec'd command."""
    trial_dir = tmp_path / "trial"
    out_path = tmp_path / "env-var.txt"
    env = os.environ.copy()
    env["PHASESWEEP_TEST_MARKER"] = "trial-env-value"
    cmd = (
        f'{sys.executable} -c "import os, pathlib; '
        f"pathlib.Path({str(out_path)!r}).write_text(os.environ['PHASESWEEP_TEST_MARKER'])\""
    )
    result = _run_supervised(trial_dir, cmd, env=env, timeout=None, attempt_id="env-flow-attempt")

    assert result.return_code == 0
    assert out_path.read_text() == "trial-env-value"


def test_run_supervised_enters_payload_cwd_before_exec(tmp_path: Path) -> None:
    """The supervisor chdirs into the payload's working directory before exec,
    so relative-path trainer commands resolve against the execution contract's
    cwd."""
    trial_dir = tmp_path / "trial"
    trainer_home = tmp_path / "trainer_home"
    trainer_home.mkdir()

    result = _run_supervised(
        trial_dir, "pwd", timeout=None, attempt_id="cwd-attempt", cwd=str(trainer_home)
    )

    assert result.return_code == 0
    assert (trial_dir / "out.log").read_text().strip() == str(trainer_home)


def test_run_supervised_fails_with_chdir_exit_code_when_cwd_vanishes(tmp_path: Path) -> None:
    """A payload cwd that cannot be entered must fail loudly (exit 76) instead
    of silently running the trainer from the wrong directory. launch_trial
    validates existence up front; this covers the race where the directory
    disappears between validation and exec."""
    trial_dir = tmp_path / "trial"

    result = _run_supervised(
        trial_dir,
        "pwd",
        timeout=None,
        attempt_id="cwd-race-attempt",
        cwd=str(tmp_path / "vanished"),
    )

    assert result.return_code == 76


def test_spawn_blocked_supervisor_launch_argv_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_spawn_blocked_supervisor`` must launch the stdlib script directly under
    ``-I -S`` with a minimal sanitized environment; the trainer command must
    never appear in argv."""
    import phasesweep.runtime.process as process
    import phasesweep.runtime.shutdown as shutdown

    captured: dict[str, object] = {}
    real_popen = subprocess.Popen

    def spy_popen(argv: list[str], **kwargs: object) -> subprocess.Popen:
        captured["argv"] = list(argv)
        captured["env"] = dict(kwargs.get("env") or {})  # type: ignore[arg-type]
        return real_popen(argv, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(process.subprocess, "Popen", spy_popen)

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    with (trial_dir / "out.log").open("w") as fout, (trial_dir / "err.log").open("w") as ferr:
        proc, pgid, ack_write, status_read = _spawn_blocked_supervisor(
            stdout=fout,
            stderr=ferr,
        )
    try:
        argv = captured["argv"]
        assert isinstance(argv, list)
        assert len(argv) == 6
        assert argv[:4] == [
            sys.executable,
            "-I",
            "-S",
            str(Path(supervisor.__file__).resolve()),
        ]
        # Last two elements are the dynamic ready/ack pipe fd numbers; the
        # trainer command must never appear anywhere in argv.
        assert all(fd.isdigit() for fd in argv[4:])
        assert set(captured["env"]) == {"PATH"}
    finally:
        os.close(ack_write)
        os.close(status_read)
        _kill_group(pgid, proc)
        shutdown._unregister(pgid)


def test_spawn_blocked_supervisor_invalidates_pipe_fd_before_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A close interruption must propagate without a second close masking it."""
    import phasesweep.runtime.process as process

    class InjectedClose(RuntimeError):
        """Interrupt the first parent-side pipe close after the descriptor closes."""

    fake_proc = object()
    close_then_fail, close_calls = raise_after_first_successful_call(
        os.close,
        InjectedClose(),
    )
    monkeypatch.setattr(process.subprocess, "Popen", lambda *args, **kwargs: fake_proc)
    monkeypatch.setattr(process, "_abort_launch", lambda proc, pgid: True)
    monkeypatch.setattr(process.os, "close", close_then_fail)

    with (
        (tmp_path / "out.log").open("w") as fout,
        (tmp_path / "err.log").open("w") as ferr,
        pytest.raises(InjectedClose),
    ):
        process._spawn_blocked_supervisor(stdout=fout, stderr=ferr)

    assert close_calls


def _frame(body: bytes) -> bytes:
    """Build one length-prefixed supervisor payload frame for test fixtures."""
    return f"{len(body):0{supervisor._HEADER_LEN}d}".encode("ascii") + body


@pytest.mark.parametrize(
    "raw_payload",
    [
        pytest.param(None, id="eof"),
        pytest.param(b"X" * supervisor._HEADER_LEN, id="bad_header"),
        pytest.param(
            f"{10:0{supervisor._HEADER_LEN}d}".encode("ascii") + b"{}",
            id="truncated_body",
        ),
        pytest.param(_frame(b"[1, 2, 3]"), id="non_dict_json"),
        pytest.param(
            _frame(json.dumps({"cmd": "true", "env": {"X": 1}}).encode("utf-8")),
            id="non_str_env_value",
        ),
    ],
)
def test_supervisor_main_rejects_malformed_payload(
    monkeypatch: pytest.MonkeyPatch, raw_payload: bytes | None
) -> None:
    """EOF or any payload-shape violation must return 75 without execing."""
    exec_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(supervisor.os, "execve", lambda *a: exec_calls.append(a))

    ready_read, ready_write = os.pipe()
    ack_read, ack_write = os.pipe()
    if raw_payload is not None:
        os.write(ack_write, raw_payload)
    os.close(ack_write)

    exit_code = supervisor.main([str(ready_write), str(ack_read)])

    # 75 = supervisor's malformed/EOF-payload exit -- see supervisor.main() docstring.
    assert exit_code == 75
    assert exec_calls == []
    os.close(ready_read)


@pytest.mark.parametrize(
    ("waitpid_result", "reaped"),
    [
        pytest.param((0, 0), False, id="still-running"),
        pytest.param((12345, 0), True, id="reaped"),
    ],
)
def test_reap_child_is_strictly_nonblocking_and_reports_whether_it_reaped(
    monkeypatch: pytest.MonkeyPatch,
    waitpid_result: tuple[int, int],
    reaped: bool,
) -> None:
    calls: list[tuple[int, int]] = []

    def fake_waitpid(pid: int, flags: int) -> tuple[int, int]:
        calls.append((pid, flags))
        return waitpid_result

    monkeypatch.setattr("phasesweep.runtime.reaper.os.waitpid", fake_waitpid)

    assert reap_child(12345) is reaped

    assert calls == [(12345, os.WNOHANG)]


_SIGTERM_IGNORING_CHILD_SCRIPT = (
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
)


def _assert_descendant_dies(
    pid: int, *, deadline_seconds: float = 5.0, on_timeout_msg: str
) -> None:
    """Poll until ``pid`` is gone; if it outlives the deadline, SIGKILL it and fail.

    :param int pid: Descendant PID expected to be reaped by supervisor cleanup.
    :param float deadline_seconds: How long to poll before giving up.
    :param str on_timeout_msg: ``pytest.fail`` message used if still alive at the deadline.
    """
    deadline = time.time() + deadline_seconds
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return  # success
        time.sleep(0.05)

    # Cleanup before failing so we don't leak a python process across the test run.
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)
    pytest.fail(on_timeout_msg)


def test_timeout_kills_descendant_when_root_exits_after_sigterm(tmp_path: Path) -> None:
    """Timeout cleanup kills a SIGTERM-ignoring child after its root shell exits."""
    trial_dir = tmp_path / "trial"
    marker = tmp_path / "child_pid.txt"

    # Inline Python child + parent so the test is self-contained.
    parent_script = (
        f"import os, subprocess, sys, time; "
        f"p = subprocess.Popen([sys.executable, '-c', {_SIGTERM_IGNORING_CHILD_SCRIPT!r}]); "
        f"open({str(marker)!r}, 'w').write(str(p.pid)); "
        f"sys.stdout.flush(); "
        # Parent exits cleanly (and immediately) on SIGTERM, abandoning child.
        f"import signal as _s; "
        f"_s.signal(_s.SIGTERM, lambda *_: (sys.stdout.flush(), os._exit(0))); "
        f"time.sleep(60)"
    )
    cmd = f"python -c {parent_script!r}"

    result = _run_supervised(trial_dir, cmd, timeout=2.0, attempt_id="timeout-attempt")

    assert result.timed_out, "trial should have hit the configured timeout"
    assert marker.exists(), "child PID marker was never written; test setup is wrong"

    child_pid = int(marker.read_text().strip())

    # The child must be dead within a reasonable window after run_supervised returns.
    # If _kill_group only waited for the root, the child would still be alive here.
    _assert_descendant_dies(
        child_pid, on_timeout_msg=f"timeout left descendant process {child_pid} alive"
    )


@pytest.mark.parametrize(
    ("timeout", "cleanup_confirmed"),
    [
        pytest.param(None, True, id="unbounded-cleanup-confirmed"),
        pytest.param(30.0, True, id="deadline-cleanup-confirmed"),
        pytest.param(30.0, False, id="deadline-cleanup-uncertain"),
    ],
)
def test_unexpected_status_read_failure_preserves_cleanup_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout: float | None,
    cleanup_confirmed: bool,
) -> None:
    """A post-launch wait failure cannot leave an untracked trainer running."""
    import phasesweep.runtime.process as process
    import phasesweep.runtime.reaper as reaper
    import phasesweep.runtime.shutdown as shutdown

    trial_dir = tmp_path / "trial"
    real_read_frame = process._read_pipe_frame
    frame_reads = 0

    def fail_execution_status_read(_fd: int, _size: int) -> bytes:
        raise OSError("injected execution status read failure")

    def read_ready_then_arm_failure(fd: int, size: int, *, deadline: float) -> bytes | None:
        nonlocal frame_reads
        frame_reads += 1
        if frame_reads == 2:
            raise OSError("injected execution status read failure")
        frame = real_read_frame(fd, size, deadline=deadline)
        monkeypatch.setattr(process.os, "read", fail_execution_status_read)
        return frame

    monkeypatch.setattr(process, "_read_pipe_frame", read_ready_then_arm_failure)
    if not cleanup_confirmed:
        _report_uncertain_after_real_terminate(monkeypatch)

    error_type = OSError if cleanup_confirmed else UnsafeProcessCleanupError
    message = (
        "injected execution status read failure"
        if cleanup_confirmed
        else "cleanup could not be confirmed"
    )
    with pytest.raises(error_type, match=message) as excinfo:
        _run_supervised(
            trial_dir,
            "exec sleep 30",
            timeout=timeout,
            attempt_id="status-read-failure",
        )

    if not cleanup_confirmed:
        assert isinstance(excinfo.value.__cause__, OSError)
    identity = read_stale_process_identity(
        trial_dir,
        expected_attempt_id="status-read-failure",
    )
    assert not reaper._process_group_alive(identity.pgid)
    with shutdown._lock:
        assert identity.pgid not in shutdown._active_children
    lifecycle = read_attempt_lifecycle(
        trial_dir,
        expected_attempt_id="status-read-failure",
    )
    assert lifecycle is not None
    assert lifecycle.state == "exited"
    assert lifecycle.cleanup_confirmed is True
    assert lifecycle.return_code is not None and lifecycle.return_code != 0
    assert lifecycle.duration_seconds is not None and lifecycle.duration_seconds >= 0


@pytest.mark.skipif(
    sys.platform != "linux", reason="reads Linux procfs SigBlk for the exec'd trainer"
)
def test_trainer_starts_with_unblocked_shutdown_signals(tmp_path: Path) -> None:
    """The exec'd trainer receives unblocked shutdown signals despite launch-time deferral."""
    trial_dir = tmp_path / "trial"
    status_out = tmp_path / "status.txt"
    cmd = f"cp /proc/self/status {status_out}"

    result = _run_supervised(trial_dir, cmd, timeout=30.0, attempt_id="mask-attempt")

    assert result.return_code == 0
    blocked = int(status_out.read_text().split("SigBlk:")[1].split()[0], 16)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        assert not blocked & (1 << (sig - 1)), f"signal {sig} is blocked in the trainer"


@pytest.mark.parametrize(
    ("identity_delay", "timeout", "command", "attempt_id", "wait_after", "pre_payload"),
    [
        pytest.param(
            0.5,
            0.2,
            "touch {marker}",
            "pre-payload-deadline-attempt",
            0.2,
            True,
            id="deadline-expires-before-payload",
        ),
        pytest.param(
            0.6,
            1.0,
            "sleep 0.6 && touch {marker}",
            "remaining-budget-attempt",
            0.0,
            False,
            id="wait-uses-remaining-budget",
        ),
    ],
)
def test_supervised_deadline_uses_remaining_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_delay: float,
    timeout: float,
    command: str,
    attempt_id: str,
    wait_after: float,
    pre_payload: bool,
) -> None:
    """Launch bookkeeping consumes the same budget used by trainer waiting."""
    trial_dir = tmp_path / "trial"
    marker = tmp_path / "trainer_ran.txt"

    import phasesweep.runtime.process as _process

    real_write_identity = _process._write_process_identity

    def slow_write_identity(path, identity):
        real_write_identity(path, identity)
        time.sleep(identity_delay)

    monkeypatch.setattr(_process, "_write_process_identity", slow_write_identity)

    result = _run_supervised(
        trial_dir,
        command.format(marker=marker),
        timeout=timeout,
        attempt_id=attempt_id,
    )

    assert result.timed_out
    if pre_payload:
        assert result.failure_reason is not None
        assert "before trainer launch" in result.failure_reason
        assert result.cleanup_confirmed
    if wait_after:
        time.sleep(wait_after)
    assert not marker.exists()


def test_payload_delivery_consumes_total_trial_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow payload write cannot turn post-deadline execution into success."""
    import phasesweep.runtime.process as process

    real_write_all = process._write_all

    def slow_payload_write(
        fd: int,
        data: bytes,
        *,
        deadline: float | None = None,
        pid: int = -1,
    ) -> None:
        real_write_all(fd, data, deadline=deadline, pid=pid)
        time.sleep(0.2)

    monkeypatch.setattr(process, "_write_all", slow_payload_write)
    trial_dir = tmp_path / "trial"

    result = _run_supervised(
        trial_dir,
        "exec sleep 30",
        timeout=0.05,
        attempt_id="payload-deadline-attempt",
    )

    assert result.timed_out
    assert result.cleanup_confirmed
    assert result.failure_reason in {
        "timeout after 0.05s",
        "timeout after 0.05s before trainer launch",
    }


def test_terminal_status_read_cannot_turn_expired_trial_into_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A status byte observed after the deadline still means the trial timed out."""
    import phasesweep.runtime.process as process

    real_fd_ready = process.fd_ready
    read_waits = 0

    def delay_terminal_status_read(fd: int, *, timeout: float, write: bool = False) -> bool:
        nonlocal read_waits
        if not write:
            read_waits += 1
            if read_waits == 2:
                time.sleep(0.8)
        return real_fd_ready(fd, timeout=timeout, write=write)

    monkeypatch.setattr(process, "fd_ready", delay_terminal_status_read)
    trial_dir = tmp_path / "trial"

    result = _run_supervised(
        trial_dir,
        "exec sleep 0.6",
        timeout=0.5,
        attempt_id="late-status-read",
    )

    assert read_waits >= 2
    assert result.timed_out
    assert result.cleanup_confirmed
    assert result.failure_reason == "timeout after 0.5s"


def test_payload_write_timeout_covers_a_nonreading_pipe() -> None:
    """A full ACK pipe expires the payload deadline without depending on supervisor startup."""
    import threading

    import phasesweep.runtime.process as process

    read_fd, write_fd = os.pipe()
    with (
        os.fdopen(read_fd, "rb", buffering=0) as reader,
        os.fdopen(write_fd, "wb", buffering=0) as writer,
    ):
        os.set_blocking(writer.fileno(), False)
        while True:
            try:
                os.write(writer.fileno(), b"x" * 4096)
            except BlockingIOError:
                break
        # A broken deadline must fail the test rather than leave a blocked writer behind.
        release = threading.Timer(1.0, reader.close)
        release.start()
        try:
            with pytest.raises(process._LaunchDeadlineExpired, match="while delivering") as raised:
                process._write_all(
                    writer.fileno(), b"payload", deadline=time.monotonic() + 0.05, pid=123
                )
        finally:
            release.cancel()
            release.join()
        assert raised.value.pid == 123


def test_supervisor_ready_wait_is_capped_by_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delayed real trainer child cannot outlive the trial's own budget.

    The readiness wait is capped by the remaining budget, not a fixed
    ``_SUPERVISOR_READY_TIMEOUT_SECONDS``; with a 10s allowance a 0.15s trial
    budget would otherwise stall far past its deadline and then still launch.
    """
    trial_dir = tmp_path / "trial"
    marker = tmp_path / "trainer_ran.txt"
    child_pid_path = tmp_path / "trainer_child.pid"

    slow_supervisor = tmp_path / "slow_supervisor.py"
    source = Path(supervisor.__file__).read_text()
    line = '    ready_frame = _READY_TAG + f"{os.getpid():0{_READY_PID_WIDTH}d}".encode("ascii")\n'
    assert line in source
    slow_supervisor.write_text(
        source.replace(
            line,
            (
                f"    with open({str(child_pid_path)!r}, 'w', encoding='utf-8') as child_file:\n"
                "        child_file.write(str(os.getpid()))\n"
                "    time.sleep(1.0)\n" + line
            ),
            1,
        )
    )
    monkeypatch.setattr("phasesweep.runtime.process._SUPERVISOR_SCRIPT_PATH", str(slow_supervisor))

    started = time.monotonic()
    result = _run_supervised(
        trial_dir,
        f"touch {marker}",
        timeout=0.15,
        attempt_id="ready-wait-deadline-attempt",
    )
    elapsed = time.monotonic() - started

    assert result.timed_out
    assert result.failure_reason is not None
    assert "before trainer launch" in result.failure_reason
    # Bounded by the budget plus kill/reap grace — nowhere near the slow
    # supervisor's 1s startup + fixed 10s readiness allowance.
    assert elapsed < 5.0
    assert not marker.exists()
    assert child_pid_path.exists(), "real supervisor never forked the trainer child"
    child_pid = int(child_pid_path.read_text())

    child_deadline = time.monotonic() + 5.0
    while time.monotonic() < child_deadline:
        if not is_pid_alive(child_pid) or is_pid_zombie(child_pid):
            break
        time.sleep(0.05)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)
        pytest.fail("deadline abort left the pre-ACK trainer child running")
    assert not marker.exists()


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(
            "phasesweep.runtime.process._write_process_identity",
            id="identity-persisted-before-payload",
        ),
        pytest.param(
            "phasesweep.engine.phase.prepare_trainer_input",
            id="input-prepared-before-lease",
        ),
    ],
)
def test_phase_deadline_expiring_before_payload_prevents_trainer_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """A phase deadline consumed at either launch boundary prevents trainer execution."""
    marker = tmp_path / "trainer_ran.txt"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f"touch {marker} && echo x=1.0 {{overrides}}",
        n_trials=1,
        timeout_seconds_per_phase=0.2,
    )
    module_name, name = target.rsplit(".", 1)
    module = import_module(module_name)
    original = getattr(module, name)

    def delayed_call(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        time.sleep(0.5)
        return result

    monkeypatch.setattr(module, name, delayed_call)

    with pytest.raises(TimeoutError, match="timed out via (phase|run) guard after 0/1"):
        run_experiment(experiment)
    assert not marker.exists(), "trainer started after the phase deadline expired"


@pytest.mark.parametrize("scope", ["phase", "run"])
@pytest.mark.parametrize("trial_limit", [30.0, 0.5])
def test_deadline_expiring_during_launch_preparation_prevents_trainer_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    trial_limit: float,
) -> None:
    """Launch preparation consumes the original phase/run deadline."""
    import phasesweep.engine.trial as trial_module

    marker = tmp_path / "trainer_ran"
    budget = 1.0
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f"touch {shlex.quote(str(marker))} && echo x=1.0 {{overrides}}",
        execution=ExecutionContext(record_env=True),
        n_trials=1,
        timeout_seconds_per_phase=budget if scope == "phase" else None,
        timeout_seconds_per_trial=trial_limit,
        gpu_policy="none",
    )
    if scope == "run":
        experiment = experiment.model_copy(update={"timeout_seconds_per_run": budget})

    real_record = trial_module._record_trainer_environment

    def delayed_record(*args: object, **kwargs: object) -> None:
        real_record(*args, **kwargs)
        time.sleep(budget + 0.2)

    monkeypatch.setattr(trial_module, "_record_trainer_environment", delayed_record)

    with pytest.raises(TimeoutError, match="timed out via (phase|run) guard after 0/1"):
        run_experiment(experiment)

    assert not marker.exists()


def test_normal_root_exit_kills_background_descendant(tmp_path: Path) -> None:
    """Root exits 0 while a child ignores SIGTERM."""
    trial_dir = tmp_path / "trial"
    marker = tmp_path / "child_pid.txt"

    parent_script = (
        "import os, subprocess, sys; "
        f"p = subprocess.Popen([sys.executable, '-c', {_SIGTERM_IGNORING_CHILD_SCRIPT!r}]); "
        f"open({str(marker)!r}, 'w').write(str(p.pid)); "
        "sys.stdout.flush(); "
        "os._exit(0)"
    )

    cmd = f"python -c {parent_script!r}"

    result = _run_supervised(trial_dir, cmd, timeout=10.0, attempt_id="descendant-attempt")

    # Must be flagged as a lifecycle failure, not a clean exit.
    assert result.failure_reason is not None
    assert "still had live descendants" in result.failure_reason

    # The atomic identity must be preserved for forensics.
    assert (trial_dir / PROCESS_IDENTITY_FILE).exists()

    # Descendant must be dead.
    assert marker.exists(), "child PID marker was never written; test setup broken"
    child_pid = int(marker.read_text().strip())
    _assert_descendant_dies(
        child_pid, on_timeout_msg=f"background descendant {child_pid} survived root exit"
    )


def test_descendant_reaping_grace_is_outside_trial_wallclock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-budget root exit remains a lifecycle failure, not a timeout."""
    import phasesweep.runtime.process as process

    trial_dir = tmp_path / "trial"
    ready = tmp_path / "descendant_ready"
    root_pid = tmp_path / "root_pid"
    budget = 5.0
    child_script = (
        "import os, pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, "
        f"lambda *_: (time.sleep({budget + 0.5}), os._exit(0))); "
        f"pathlib.Path({str(ready)!r}).touch(); "
        "time.sleep(60)"
    )
    parent_script = (
        "import os, pathlib, subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child_script!r}]); "
        f"ready = pathlib.Path({str(ready)!r}); "
        "\nwhile not ready.exists(): time.sleep(0.005)\n"
        f"pathlib.Path({str(root_pid)!r}).write_text(str(os.getpid())); "
        "os._exit(0)"
    )

    real_wait = process._wait_for_guardian_exit

    def wait_for_cleanup(proc: subprocess.Popen[bytes]) -> bool:
        # The root-exit status has already arrived; the guardian still owns cleanup.
        assert root_pid.exists()
        assert not is_pid_alive(int(root_pid.read_text()))
        assert proc.poll() is None
        assert time.monotonic() - started < budget, "root exit did not precede the deadline"
        return real_wait(proc)

    monkeypatch.setattr(process, "_wait_for_guardian_exit", wait_for_cleanup)
    started = time.monotonic()
    result = _run_supervised(
        trial_dir,
        shlex.join([sys.executable, "-c", parent_script]),
        timeout=budget,
        attempt_id="post-root-grace-attempt",
    )

    assert result.timed_out is False
    assert result.return_code == 0
    assert result.cleanup_confirmed is True
    assert result.failure_reason is not None
    assert "still had live descendants" in result.failure_reason
    assert result.duration_seconds > budget, "test did not exercise cleanup beyond the budget"


def test_in_band_guardian_wait_reports_uncertainty_instead_of_hanging() -> None:
    """A fail-closed detached guardian cannot park the orchestrator forever."""
    guardian = Mock(pid=4242)
    guardian.wait.side_effect = subprocess.TimeoutExpired("guardian", 0.0)

    assert _wait_for_guardian_exit(guardian) is False
    guardian.wait.assert_called_once()
    timeout = guardian.wait.call_args.kwargs["timeout"]
    assert timeout is not None
    assert timeout > supervisor._GROUP_TERM_GRACE_SECONDS


@pytest.mark.parametrize(
    ("unreadable_pgid", "expected"),
    [
        pytest.param(9999, ((11,), True, True), id="unrelated-entry-is-irrelevant"),
        pytest.param(1234, ((11, 22), False, False), id="possible-member-is-uncertain"),
    ],
)
def test_proc_scan_scopes_unreadable_entries_to_the_target_group(
    monkeypatch: pytest.MonkeyPatch,
    unreadable_pgid: int,
    expected: tuple[tuple[int, ...], bool, bool],
) -> None:
    """Host-global hidepid failures do not poison an unrelated trainer group."""
    monkeypatch.setattr("phasesweep.runtime.reaper.sys", SimpleNamespace(platform="linux"))
    real_iterdir = Path.iterdir

    def fake_iterdir(path: Path):
        if path == Path("/proc"):
            return iter((Path("/proc/11"), Path("/proc/22")))
        return real_iterdir(path)

    def fake_stat(path: Path) -> tuple[_ProcStat | None, bool]:
        if path.name == "11":
            return _ProcStat(state="Z", pgrp=1234, starttime=1), True
        return None, False

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    monkeypatch.setattr("phasesweep.runtime.reaper._read_proc_stat_result", fake_stat)
    monkeypatch.setattr("phasesweep.runtime.reaper.os.getpgid", lambda _pid: unreadable_pgid)

    scan = _group_member_pids(1234)

    assert (scan.pids, scan.complete, scan.all_members_terminal) == expected


@pytest.mark.parametrize(
    ("unreadable_pgid", "expected_alive"),
    [
        pytest.param(9999, False, id="unrelated-entry-is-irrelevant"),
        pytest.param(1234, True, id="possible-member-keeps-lease"),
    ],
)
def test_guardian_proc_scan_ignores_only_proven_unrelated_entries(
    monkeypatch: pytest.MonkeyPatch,
    unreadable_pgid: int,
    expected_alive: bool,
) -> None:
    """The stdlib guardian applies the same target-scoped completeness rule."""
    monkeypatch.setattr(supervisor, "sys", SimpleNamespace(platform="linux"))

    real_open = builtins.open

    def fake_open(path: str, mode: str = "r", *args: object, **kwargs: object):
        if path == "/proc/11/stat":
            return io.BytesIO(b"11 (trainer) Z 1 1234 0")
        if path == "/proc/22/stat":
            raise PermissionError(path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("phasesweep.runtime.supervisor.os.killpg", lambda *_args: None)
    entries = MagicMock()
    entries.__iter__.return_value = (SimpleNamespace(name="11"), SimpleNamespace(name="22"))
    monkeypatch.setattr("phasesweep.runtime.supervisor.os.scandir", Mock(return_value=entries))
    monkeypatch.setattr("phasesweep.runtime.supervisor.os.getpgid", lambda _pid: unreadable_pgid)
    monkeypatch.setattr(builtins, "open", fake_open)

    assert supervisor._trainer_group_alive(1234) is expected_alive


def test_terminate_process_groups_shares_grace_across_groups(tmp_path: Path) -> None:
    """Two SIGTERM-ignoring groups burn ONE shared grace window, not one each.

    The shutdown handler kills every active trial group of an ``n_jobs > 1``
    phase through this path; serial escalation would multiply worst-case
    shutdown latency by the trial parallelism.
    """
    grace = 3.0
    procs: list[subprocess.Popen] = []
    try:
        for idx in range(2):
            ready = tmp_path / f"ready_{idx}"
            script = (
                "import pathlib, signal, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"pathlib.Path({str(ready)!r}).touch(); "
                "time.sleep(60)"
            )
            procs.append(subprocess.Popen([sys.executable, "-c", script], start_new_session=True))
        deadline = time.time() + 10.0
        while not all((tmp_path / f"ready_{i}").exists() for i in range(2)):
            if time.time() > deadline:
                pytest.fail("children never installed their SIGTERM ignore handlers")
            time.sleep(0.02)

        start = time.monotonic()
        verdicts = _terminate_process_groups(tuple(proc.pid for proc in procs), grace_seconds=grace)
        elapsed = time.monotonic() - start
    finally:
        for proc in procs:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)

    assert verdicts == {proc.pid: True for proc in procs}
    # Serial escalation pays a full grace per group (>= 2 * grace); sharing leaves a grace of slack.
    assert elapsed < 2 * grace, f"escalation took {elapsed:.2f}s; grace not shared"


@pytest.mark.parametrize("guardian_state", [None, "dead", "dying"])
def test_terminate_process_groups_reports_per_group_verdicts(
    guardian_state: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live group gets cleaned after its guardian exits before or during cancellation."""
    dead = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    dead.wait()
    live = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    guardian = (
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
        )
        if guardian_state == "dying"
        else dead
    )
    real_kill = os.kill

    def kill(pid: int, signum: int) -> None:
        assert pid != dead.pid or signum == 0
        real_kill(pid, signum)

    monkeypatch.setattr(os, "kill", kill)
    try:
        verdicts = _terminate_process_groups(
            (dead.pid, live.pid),
            grace_seconds=5.0,
            guardians=None if guardian_state is None else {live.pid: guardian},
        )
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(live.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            live.wait(timeout=5)
        if guardian.poll() is None:
            guardian.kill()
        guardian.wait(timeout=5)

    assert verdicts == {dead.pid: True, live.pid: True}


def test_shutdown_handler_uses_initial_pgid_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown cleanup must target groups seen in the initial snapshot.

    A worker thread can unregister a PGID while cleanup is underway. Using the
    initial snapshot prevents that race from hiding a group from the shutdown
    report.
    """
    terminated: list[int] = []
    active = {1234: SimpleNamespace(pid=4321)}

    monkeypatch.setattr("phasesweep.runtime.shutdown._active_children", active)

    def fake_terminate(
        pgids: tuple[int, ...],
        *,
        grace_seconds: float,
        guardians: Mapping[int, subprocess.Popen[bytes]],
    ) -> dict[int, bool]:
        assert guardians == active
        terminated.extend(pgids)
        active.clear()
        return dict.fromkeys(pgids, True)

    monkeypatch.setattr("phasesweep.runtime.shutdown._terminate_process_groups", fake_terminate)

    with pytest.raises(PhaseSweepShutdown) as excinfo:
        _shutdown_handler(signal.SIGTERM, None)

    assert terminated == [1234]
    assert excinfo.value.report.cleanup_confirmed is True
    assert excinfo.value.report.child_pgids == (1234,)


def test_shutdown_handler_reports_uncertain_when_group_termination_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "phasesweep.runtime.shutdown._active_children", {1234: SimpleNamespace(pid=4321)}
    )
    monkeypatch.setattr(
        "phasesweep.runtime.shutdown._terminate_process_groups",
        lambda pgids, *, grace_seconds, guardians: dict.fromkeys(pgids, False),
    )

    with pytest.raises(PhaseSweepShutdown) as excinfo:
        _shutdown_handler(signal.SIGTERM, None)

    assert int(excinfo.value.code) == 143
    assert excinfo.value.report.cleanup_confirmed is False
    assert excinfo.value.report.child_pgids == (1234,)


def test_shutdown_handler_ignores_reentrant_signal_during_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first shutdown signal owns cleanup evidence until its pass completes."""
    active = {1234: SimpleNamespace(pid=4321), 5678: SimpleNamespace(pid=8765)}
    terminated: list[int] = []
    reentered = False

    monkeypatch.setattr("phasesweep.runtime.shutdown._active_children", active)

    def fake_terminate(
        pgids: tuple[int, ...],
        *,
        grace_seconds: float,
        guardians: Mapping[int, subprocess.Popen[bytes]],
    ) -> dict[int, bool]:
        assert guardians == active
        nonlocal reentered
        terminated.extend(pgids)
        if not reentered:
            reentered = True
            assert _shutdown_handler(signal.SIGINT, None) is None
        return dict.fromkeys(pgids, True)

    monkeypatch.setattr("phasesweep.runtime.shutdown._terminate_process_groups", fake_terminate)

    with pytest.raises(PhaseSweepShutdown) as excinfo:
        _shutdown_handler(signal.SIGTERM, None)

    assert terminated == [1234, 5678]
    assert excinfo.value.signum == signal.SIGTERM
    assert excinfo.value.report.signum == signal.SIGTERM
    assert excinfo.value.report.cleanup_confirmed is True
    assert excinfo.value.report.child_pgids == (1234, 5678)

    active.clear()
    with pytest.raises(PhaseSweepShutdown) as subsequent:
        _shutdown_handler(signal.SIGINT, None)
    assert subsequent.value.signum == signal.SIGINT


def test_process_group_alive_uses_cached_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux procfs probes reuse live cached members without rescanning."""
    monkeypatch.setattr("phasesweep.runtime.reaper.sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr("phasesweep.runtime.reaper._PROC", tmp_path)
    monkeypatch.setattr("phasesweep.runtime.reaper._process_group_exists", lambda pgid: True)
    monkeypatch.setattr(
        "phasesweep.runtime.reaper._group_member_pids",
        lambda pgid: (_ for _ in ()).throw(AssertionError("must not rescan /proc")),
    )
    monkeypatch.setattr("phasesweep.runtime.reaper._member_pids_alive", lambda pgid, pids: True)

    assert _process_group_alive_with_members(1234, {11}) is True


def test_process_group_alive_refreshes_when_cached_members_are_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux procfs probes replace exhausted cached members with the current scan."""
    monkeypatch.setattr("phasesweep.runtime.reaper.sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr("phasesweep.runtime.reaper._PROC", tmp_path)
    member_sets: list[set[int]] = []
    scans: list[int] = []

    monkeypatch.setattr("phasesweep.runtime.reaper._process_group_exists", lambda pgid: True)

    def fake_group_member_pids(pgid: int) -> _GroupMemberScan:
        scans.append(pgid)
        return _GroupMemberScan(pids=(22,), complete=True, all_members_terminal=False)

    def fake_member_pids_alive(pgid: int, pids: set[int] | list[int]) -> bool:
        member_sets.append(set(pids))
        return 22 in pids

    monkeypatch.setattr("phasesweep.runtime.reaper._group_member_pids", fake_group_member_pids)
    monkeypatch.setattr("phasesweep.runtime.reaper._member_pids_alive", fake_member_pids_alive)

    member_pids = {11}

    assert _process_group_alive_with_members(1234, member_pids) is True
    assert member_pids == {22}
    assert scans == [1234]
    assert member_sets == [{11}, {22}]


@pytest.mark.parametrize(
    ("scan", "expected_alive"),
    [
        pytest.param(
            _GroupMemberScan(pids=(), complete=True, all_members_terminal=False),
            True,
            id="empty-kernel-visible-group-is-uncertain",
        ),
        pytest.param(
            _GroupMemberScan(pids=(22,), complete=False, all_members_terminal=True),
            True,
            id="incomplete-zombie-scan-is-uncertain",
        ),
        pytest.param(
            _GroupMemberScan(pids=(22,), complete=True, all_members_terminal=True),
            False,
            id="complete-zombie-only-group-is-dead",
        ),
    ],
)
def test_process_group_absence_requires_complete_terminal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scan: _GroupMemberScan,
    expected_alive: bool,
) -> None:
    """Linux procfs marks a visible group dead only with complete terminal evidence."""
    monkeypatch.setattr("phasesweep.runtime.reaper.sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr("phasesweep.runtime.reaper._PROC", tmp_path)
    monkeypatch.setattr("phasesweep.runtime.reaper._process_group_exists", lambda pgid: True)
    monkeypatch.setattr("phasesweep.runtime.reaper._group_member_pids", lambda pgid: scan)
    monkeypatch.setattr("phasesweep.runtime.reaper._member_pids_alive", lambda pgid, pids: False)

    assert _process_group_alive_with_members(1234, None) is expected_alive


@pytest.mark.parametrize(
    ("kill_error", "group_survives", "expected", "required_signals"),
    [
        pytest.param(None, True, False, {signal.SIGTERM, signal.SIGKILL}, id="survives-sigkill"),
        pytest.param(ProcessLookupError, False, True, {signal.SIGTERM}, id="already-gone"),
        pytest.param(PermissionError, False, False, {signal.SIGTERM}, id="permission-denied"),
    ],
)
def test_terminate_process_group_reports_cleanup_status(
    monkeypatch: pytest.MonkeyPatch,
    kill_error: type[OSError] | None,
    group_survives: bool,
    expected: bool,
    required_signals: set[signal.Signals],
) -> None:
    """Group termination reports confirmed cleanup only when it can prove it."""
    calls: list[tuple[int, int]] = []

    def fake_killpg(pgid: int, sig: int) -> None:
        calls.append((pgid, sig))
        if kill_error is not None:
            raise kill_error

    monkeypatch.setattr("phasesweep.runtime.reaper.os.killpg", fake_killpg)
    monkeypatch.setattr(
        "phasesweep.runtime.reaper._process_group_alive_with_members",
        lambda pgid, member_pids: group_survives,
    )

    assert _terminate_process_group(1234, grace_seconds=0.0) is expected
    assert required_signals.issubset({sig for _, sig in calls})


def test_reaper_raises_when_cleanup_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uncertain cleanup prevents the reaper from advancing the trial."""

    def fake_identity(*_args: object, **_kwargs: object) -> StaleProcessIdentity:
        return StaleProcessIdentity(
            schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
            attempt_id="uncertain-attempt",
            pid=99999,
            pgid=99999,
            proc_starttime=12345,
            boot_id="test-boot",
            pid_namespace="pid:[4026539001]@100",
        )

    monkeypatch.setattr("phasesweep.engine.attempts.read_stale_process_identity", fake_identity)
    monkeypatch.setattr(
        "phasesweep.engine.attempts.cleanup_stale_trial_process",
        lambda _identity, **_kwargs: False,
    )

    exp = make_experiment(workdir=tmp_path / "runs")
    study = optuna.create_study(direction="maximize")

    # Inject one RUNNING trial so the reaper has something to chew on.
    trial = study.ask()
    trial.set_user_attr(ATTEMPT_ID_ATTR, "uncertain-attempt")
    trial.set_user_attr(TRIAL_DIR_ATTR, str(tmp_path / "runs" / "t" / "p" / "trial_00000"))
    assert study.trials[trial.number].state == optuna.trial.TrialState.RUNNING

    with pytest.raises(RuntimeError, match="process cleanup could not be confirmed"):
        _reap_stale_trials(study, exp, exp.phases[0].name, confirm=True)


def test_run_supervised_reports_uncertain_cleanup_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When group termination reports uncertain cleanup,
    ``run_supervised`` must surface that in ``ProcessResult.cleanup_confirmed``
    so the orchestrator can abort instead of launching more trials."""
    # Pattern: actually kill the group with the real implementation, then
    # report uncertain. Exercises the orchestrator's "cleanup uncertainty"
    # branch without leaving real ``time.sleep(60)`` zombies behind for the
    # rest of the test run.
    _report_uncertain_after_real_terminate(monkeypatch)

    result = _run_supervised(
        tmp_path,
        f"{sys.executable} -c 'import time; time.sleep(60)'",
        timeout=0.1,
        attempt_id="uncertain-attempt",
    )

    assert result.timed_out is True
    assert result.cleanup_confirmed is False
    # The atomic identity must be preserved for forensics.
    assert (tmp_path / PROCESS_IDENTITY_FILE).exists()


@pytest.mark.parametrize(
    ("n_trials", "n_jobs", "gpu_ids", "failure_limit"),
    [
        pytest.param(1, 1, None, 5, id="serial-optimize-raises"),
        pytest.param(2, 2, [0], 100, id="parallel-fatal-slot-stops-queued-gpu-worker"),
    ],
)
def test_uncertain_cleanup_aborts_without_reusing_gpu_and_preserves_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_trials: int,
    n_jobs: int,
    gpu_ids: list[int] | None,
    failure_limit: int,
) -> None:
    """Unsafe cleanup stops launches and preserves forensic CSV on both optimize exit paths.

    Serial optimize propagates the exception directly. Parallel optimize can
    absorb worker exceptions, so its fatal slot must re-raise after draining.
    With one GPU and two workers the abort is recorded before releasing the
    lease, preventing the queued worker from launching onto an unresolved group.
    """
    _report_uncertain_after_real_terminate(monkeypatch)
    if gpu_ids is not None:
        monkeypatch.setattr(
            "phasesweep.runtime.gpu._detect_gpu_inventory",
            lambda: ([], {"0": "GPU-test-0"}),
        )

    exp = make_experiment(
        workdir=tmp_path / "runs",
        n_trials=n_trials,
        n_jobs=n_jobs,
        gpu_ids=gpu_ids,
        timeout_seconds_per_trial=0.2,
        max_consecutive_failures=failure_limit,
        trial_command=f"{sys.executable} -c 'import time; time.sleep(60)' {{overrides}}",
    )
    phase_dir = tmp_path / "runs" / exp.experiment / exp.phases[0].name

    try:
        with pytest.raises(UnsafeProcessCleanupError):
            run_experiment(exp)

        launched_identities = sorted(phase_dir.glob(f"trial_*/{PROCESS_IDENTITY_FILE}"))
        assert len(launched_identities) == 1, (
            f"A queued trial launched after unsafe cleanup. "
            f"Found identities: {[p.parent.name for p in launched_identities]}. "
            "Unsafe cleanup must hard-abort BEFORE the GPU lease is released "
            "to a queued worker."
        )
        csv_path = phase_dir / "trials.csv"
        assert csv_path.exists(), (
            f"trials.csv missing after hard abort. Forensic data must survive "
            f"every exit path. Looked at {csv_path}."
        )
        content = csv_path.read_text()
        assert "number" in content and "state" in content, (
            f"CSV header missing expected columns: {content[:200]!r}"
        )
        assert content.count("\n") >= 2, f"CSV has no trial rows, only header: {content!r}"
        assert "FAIL" in content, f"CSV missing FAIL row for hard-aborted trial: {content!r}"
    finally:
        for identity_file in phase_dir.glob(f"trial_*/{PROCESS_IDENTITY_FILE}"):
            with contextlib.suppress(Exception):
                os.killpg(json.loads(identity_file.read_text())["pgid"], signal.SIGKILL)


def test_uncertain_cleanup_cancels_queued_host_gpu_lease_wait(tmp_path: Path) -> None:
    """A guardian-held GPU flock cannot strand a queued parallel objective."""
    script = textwrap.dedent(
        f"""
        import os
        import time
        from pathlib import Path
        from types import SimpleNamespace

        import phasesweep.engine.phase as phase_module
        from phasesweep import run_experiment
        from phasesweep.errors import UnsafeProcessCleanupError
        from phasesweep.runtime.gpu import GpuDevice, GpuPool
        from phasesweep.runtime.process import ProcessResult
        from tests.conftest import make_experiment

        root = Path({str(tmp_path)!r})
        lock_dir = root / "locks"
        lock_dir.mkdir(mode=0o700)
        lock_dir.chmod(0o700)
        os.environ["PHASESWEEP_LOCK_DIR"] = str(lock_dir)
        held_fds = []
        phase_module.GpuPool = SimpleNamespace(
            for_phase=lambda phase, **kwargs: GpuPool([GpuDevice("0", "GPU-test")])
        )

        def fake_launch_trial(**kwargs):
            held_fds.append(os.dup(kwargs["gpu_lease_fds"][0]))
            time.sleep(0.5)
            return SimpleNamespace(
                process=ProcessResult(
                    return_code=-9,
                    timed_out=True,
                    pid=12345,
                    duration_seconds=0.5,
                    failure_reason="injected cleanup uncertainty",
                    cleanup_confirmed=False,
                )
            )

        phase_module.launch_trial = fake_launch_trial
        experiment = make_experiment(
            workdir=root / "runs",
            n_trials=2,
            n_jobs=2,
            gpu_ids=[0],
            timeout_seconds_per_trial=1.0,
            max_consecutive_failures=100,
            trial_command="echo {{overrides}}",
        )
        try:
            try:
                run_experiment(experiment)
            except UnsafeProcessCleanupError:
                print(f"unsafe:{{len(held_fds)}}", flush=True)
            else:
                raise AssertionError("unsafe cleanup unexpectedly succeeded")
        finally:
            for fd in held_fds:
                os.close(fd)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],  # The script imports the tests package.
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "unsafe:1"
