"""Supervised subprocess lifecycle: Popen registration, signal handler installation and forwarding, descendant cleanup, cleanup_confirmed propagation, and the launch-window deadlock guard. Some tests run as real subprocesses because the behavior is POSIX signal delivery, not Python state we can mock."""

from __future__ import annotations

import contextlib
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.engine.guards import _reap_stale_trials
from phasesweep.engine.state import ATTEMPT_ID_ATTR, TRIAL_DIR_ATTR
from phasesweep.engine.trial import UnsafeProcessCleanupError
from phasesweep.runtime import supervisor
from phasesweep.runtime.process import (
    ATTEMPT_LIFECYCLE_FILE,
    PROCESS_IDENTITY_FILE,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    PhaseSweepShutdown,
    StaleProcessIdentity,
    _kill_group,
    _process_group_alive_with_members,
    _shutdown_handler,
    _spawn_blocked_supervisor,
    _terminate_process_group,
    _terminate_process_groups,
    cleanup_stale_trial_process,
    defer_shutdown_signals,
    is_pid_alive,
    read_attempt_lifecycle,
    read_stale_process_identity,
    reap_child,
    run_supervised,
)
from tests.conftest import is_pid_zombie, make_experiment


def _report_uncertain_after_real_terminate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch group termination to clean up the group but report uncertainty."""
    import phasesweep.runtime.process as _process

    real_terminate = _process._terminate_process_group

    def terminate_then_report_uncertain(pgid: int, *, grace_seconds: float) -> bool:
        real_terminate(pgid, grace_seconds=0.05)
        return False

    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_group",
        terminate_then_report_uncertain,
    )


def _install_signal_probe(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """Patch ``signal_handler_scope`` so tests can observe whether it was entered.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to replace the scope.
    :return dict[str, bool]: Mutable probe; ``"called"`` flips to ``True`` on entry.
    """
    installed = {"called": False}

    @contextlib.contextmanager
    def fake_scope():
        installed["called"] = True
        yield

    monkeypatch.setattr("phasesweep.engine.run.signal_handler_scope", fake_scope)
    return installed


def _run_supervised(
    trial_dir: Path,
    cmd: str,
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    attempt_id: str,
    cwd: str | None = None,
):
    """Open ``trial_dir``'s out/err logs and delegate to ``run_supervised``.

    Collapses the repeated "open two log files, call run_supervised"
    boilerplate shared by most call sites in this module.
    """
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


def test_run_supervised_persists_pgid_on_failure(tmp_path: Path) -> None:
    """Failing trials leave one complete atomic identity for forensic recovery."""
    if not Path("/proc/self/stat").exists():
        pytest.skip("Linux-only test")

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    result = _run_supervised(trial_dir, "false", timeout=None, attempt_id="failure-attempt")
    assert result.return_code != 0
    identity = read_stale_process_identity(
        trial_dir,
        expected_attempt_id="failure-attempt",
    )
    assert identity.pid == result.pid
    assert identity.pgid == result.pid
    assert identity.proc_starttime is not None
    assert identity.boot_id is not None


def test_run_supervised_retains_identity_and_records_exit_on_success(tmp_path: Path) -> None:
    """Clean exit retains the identity and durably records the 'exited' state.

    The identity used to be unlinked on clean exit, which made an orchestrator
    death between process exit and the Optuna terminal commit (evidence
    extraction, gates) indistinguishable from a missing identity — recovery
    then failed closed forever (review v0.5.17 / blocker 2 gap B).
    """
    if not Path("/proc/self/stat").exists():
        pytest.skip("Linux-only test")

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    result = _run_supervised(trial_dir, "true", timeout=None, attempt_id="success-attempt")
    assert result.return_code == 0
    assert result.duration_seconds >= 0.0
    assert (trial_dir / PROCESS_IDENTITY_FILE).exists()
    lifecycle = read_attempt_lifecycle(trial_dir, expected_attempt_id="success-attempt")
    assert lifecycle is not None
    assert lifecycle.state == "exited"
    assert lifecycle.return_code == 0
    assert lifecycle.cleanup_confirmed is True


def test_read_attempt_lifecycle_rejects_unreadable_record(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    (trial_dir / ATTEMPT_LIFECYCLE_FILE).mkdir(parents=True)

    with pytest.raises(ValueError, match="is unreadable"):
        read_attempt_lifecycle(trial_dir, expected_attempt_id="expected")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{", "Malformed attempt lifecycle record"),
        ("[]", "must be a JSON object"),
        (
            '{"schema_version": 2, "attempt_id": "expected", "state": "allocated"}',
            "Unsupported attempt lifecycle schema",
        ),
        (
            '{"schema_version": 1, "attempt_id": "other", "state": "allocated"}',
            "belongs to another attempt",
        ),
        (
            '{"schema_version": 1, "attempt_id": "expected", "state": "launching"}',
            "Unknown attempt lifecycle state",
        ),
        (
            '{"schema_version": 1, "attempt_id": "expected", "state": "exited", '
            '"return_code": true}',
            "field 'return_code' is invalid",
        ),
        (
            '{"schema_version": 1, "attempt_id": "expected", "state": "exited", '
            '"return_code": 0, "cleanup_confirmed": 1}',
            "field 'cleanup_confirmed' is invalid",
        ),
    ],
)
def test_read_attempt_lifecycle_rejects_every_malformed_shape(
    tmp_path: Path, payload: str, message: str
) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    (trial_dir / ATTEMPT_LIFECYCLE_FILE).write_text(payload)

    with pytest.raises(ValueError, match=message):
        read_attempt_lifecycle(trial_dir, expected_attempt_id="expected")


def test_run_supervised_terminates_child_when_identity_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A metadata write failure after Popen must not leave the child running.

    Looped ~30 times (review v0.5.15 / blocker 4): pre-fix, ``_kill_group``
    only reaped the direct child when a single nonblocking ``poll()`` already
    observed it exited, so a child that died right after ``poll()`` returned
    ``None`` could remain an unreaped zombie despite ``cleanup_confirmed``
    being reported ``True``. The bounded, unconditional ``wait()`` fix reaps
    deterministically; the loop guards against a single lucky run masking a
    reintroduced race.
    """
    import phasesweep.runtime.process as process

    real_atomic_write_text = process.atomic_write_text

    def fail_pid_write(path: Path, text: str) -> None:
        if path.name == PROCESS_IDENTITY_FILE:
            raise OSError("identity disk full")
        real_atomic_write_text(path, text)

    monkeypatch.setattr("phasesweep.runtime.process.atomic_write_text", fail_pid_write)

    for i in range(30):
        trial_dir = tmp_path / f"trial_{i}"
        trial_dir.mkdir()
        trainer_started = tmp_path / f"trainer-started-{i}"
        result = _run_supervised(
            trial_dir,
            f'{sys.executable} -c "from pathlib import Path; '
            f"Path({str(trainer_started)!r}).write_text('started')\"",
            timeout=None,
            attempt_id=f"identity-write-failure-{i}",
        )

        assert result.cleanup_confirmed is True, i
        assert "failed to persist process identity" in (result.failure_reason or ""), i
        assert not trainer_started.exists(), i
        with pytest.raises(ProcessLookupError):
            os.kill(result.pid, 0)


def test_kill_group_reports_unconfirmed_when_direct_child_reap_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct child that never reaps within the bounded wait must report cleanup
    as unconfirmed, not silently ``True`` (review v0.5.15 / blocker 4)."""
    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_group",
        lambda pgid, *, grace_seconds: True,
    )

    class _NeverReapsProc:
        pid = 424243
        returncode = None

        def wait(self, timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0.0)

    assert _kill_group(999999, _NeverReapsProc()) is False  # type: ignore[arg-type]


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

def stall_identity_write(path: Path, text: str) -> None:
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
            with __import__("contextlib").suppress(ProcessLookupError):
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
            with __import__("contextlib").suppress(ProcessLookupError):
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


def test_supervisor_never_imports_poisoned_phasesweep_from_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A poisoned PYTHONPATH on the trainer env must never reach the supervisor.

    Pre-fix (review v0.5.15 / blocker 1), the supervisor ran ``-m
    phasesweep.runtime.supervisor`` under the FULL trainer environment,
    including any ``PYTHONPATH`` composed from ``experiment.env``. A
    shadowed/poisoned ``phasesweep`` package on that path would execute at
    import time, before ``process_identity.json`` exists. The fix launches a
    stdlib-only script directly under a sanitized ``{"PATH": ...}``
    environment, so the poisoned package must never even be importable.
    """
    poison_root = tmp_path / "poison"
    package_dir = poison_root / "phasesweep"
    package_dir.mkdir(parents=True)
    marker = tmp_path / "poison-marker"
    (package_dir / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('poisoned')\n"
    )

    written = _spy_write_process_identity(monkeypatch)

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(poison_root)
    result = _run_supervised(
        trial_dir, "true", env=env, timeout=None, attempt_id="poison-pythonpath"
    )

    assert result.return_code == 0
    assert not marker.exists(), "supervisor imported the poisoned phasesweep package"
    assert written == [True], "process_identity.json was never written before exec"


def test_supervisor_never_imports_poisoned_sitecustomize_from_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``-S`` must keep site initialization (and thus ``sitecustomize``) off pre-ACK.

    Even without an ``-m phasesweep...`` import, a bare ``python`` invocation
    still runs ``sitecustomize``/``usercustomize`` during site initialization
    when their directory is importable at process start — e.g. via a
    ``PYTHONPATH`` composed into the trainer environment. ``-S`` disables
    that (review v0.5.15 / blocker 1).
    """
    poison_root = tmp_path / "site-poison"
    poison_root.mkdir()
    marker = tmp_path / "sitecustomize-marker"
    (poison_root / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('poisoned')\n"
    )

    written = _spy_write_process_identity(monkeypatch)

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(poison_root)
    result = _run_supervised(
        trial_dir, "true", env=env, timeout=None, attempt_id="poison-sitecustomize"
    )

    assert result.return_code == 0
    assert not marker.exists(), "supervisor ran a poisoned sitecustomize.py"
    assert written == [True], "process_identity.json was never written before exec"


def test_run_supervised_delivers_trainer_env_via_payload(tmp_path: Path) -> None:
    """The framed ack payload must still deliver the full trainer environment
    to the exec'd command (review v0.5.15 / blocker 1)."""
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
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
    cwd (review v0.5.17 / blocker 4)."""
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
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
    trial_dir.mkdir()

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
    never appear in argv (review v0.5.15 / blocker 1)."""
    import phasesweep.runtime.process as process

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
        proc, pgid, ack_write = _spawn_blocked_supervisor(stdout=fout, stderr=ferr)
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
        _kill_group(pgid, proc)
        process._unregister(pgid)


def test_spawn_blocked_supervisor_invalidates_pipe_fd_before_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A close interruption must propagate without a second close masking it."""
    import phasesweep.runtime.process as process

    class InjectedClose(RuntimeError):
        """Interrupt the first parent-side pipe close after the descriptor closes."""

    fake_proc = object()
    real_close = os.close
    fired = False

    def close_then_fail(fd: int) -> None:
        nonlocal fired
        real_close(fd)
        if not fired:
            fired = True
            raise InjectedClose

    monkeypatch.setattr(process.subprocess, "Popen", lambda *args, **kwargs: fake_proc)
    monkeypatch.setattr(process, "_abort_launch", lambda proc, pgid: True)
    monkeypatch.setattr(process.os, "close", close_then_fail)

    with (
        (tmp_path / "out.log").open("w") as fout,
        (tmp_path / "err.log").open("w") as ferr,
        pytest.raises(InjectedClose),
    ):
        process._spawn_blocked_supervisor(stdout=fout, stderr=ferr)

    assert fired


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


def test_reap_child_is_strictly_nonblocking(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_waitpid(pid: int, flags: int) -> tuple[int, int]:
        calls.append((pid, flags))
        return (0, 0)

    monkeypatch.setattr("phasesweep.runtime.process.os.waitpid", fake_waitpid)

    assert reap_child(12345) is False

    assert calls == [(12345, os.WNOHANG)]


def test_reap_child_reports_when_it_reaped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "phasesweep.runtime.process.os.waitpid",
        lambda pid, flags: (pid, 0),
    )

    assert reap_child(12345) is True


# Both duplicate-victim tests below spawn a background child that ignores
# SIGTERM, so ``run_supervised`` must escalate to SIGKILL against the whole
# process group rather than stopping once the root process is gone.
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
    """Root shell exits cleanly on SIGTERM, but child ignores it.

    The previous code path (proc.wait() only) returned as soon as the shell
    died, leaving the child running with a GPU lease released. This must now
    poll the whole process group and escalate to SIGKILL.
    """
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
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

    # Wait for child PID file before timeout fires.
    # Use a thread to write the marker check; simpler: just timeout=2.0
    # and confirm the marker was written (child started).
    result = _run_supervised(trial_dir, cmd, timeout=2.0, attempt_id="timeout-attempt")

    assert result.timed_out, "trial should have hit the configured timeout"
    assert marker.exists(), "child PID marker was never written; test setup is wrong"

    child_pid = int(marker.read_text().strip())

    # The child must be dead within a reasonable window after run_supervised returns.
    # If _kill_group only waited for the root, the child would still be alive here.
    _assert_descendant_dies(
        child_pid, on_timeout_msg=f"timeout left descendant process {child_pid} alive"
    )


def test_trainer_starts_with_unblocked_shutdown_signals(tmp_path: Path) -> None:
    """The exec'd trainer must not inherit the launcher's blocked signal mask.

    ``run_supervised`` spawns the supervisor from inside its shutdown-signal
    deferral window, and a blocked mask survives both fork and exec. Without
    the supervisor's explicit mask reset, every trainer ran with
    SIGTERM/SIGINT/SIGHUP blocked: graceful trainer shutdown was impossible
    and the SIGTERM -> grace -> SIGKILL escalation always burned the full
    grace period.
    """
    if not Path("/proc/self/status").exists():
        pytest.skip("Linux-only test")

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    status_out = tmp_path / "status.txt"
    cmd = f"cp /proc/self/status {status_out}"

    result = _run_supervised(trial_dir, cmd, timeout=30.0, attempt_id="mask-attempt")

    assert result.return_code == 0
    blocked = int(status_out.read_text().split("SigBlk:")[1].split()[0], 16)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        assert not blocked & (1 << (sig - 1)), f"signal {sig} is blocked in the trainer"


def test_deadline_expired_before_payload_never_starts_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget that expires during launch bookkeeping must not start the trainer.

    Pre-fix (review v0.5.16 / blocker 6), the timeout was applied only at
    ``proc.wait``: supervisor spawn, readiness, identity persistence, and
    payload delivery all ran on an already-expired budget and the trainer
    still started. The deadline is now re-checked before the payload crosses
    the ack pipe.
    """
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    marker = tmp_path / "trainer_ran.txt"

    import phasesweep.runtime.process as _process

    real_write_identity = _process._write_process_identity

    def slow_write_identity(path, identity):  # noqa: ANN001, ANN202
        real_write_identity(path, identity)
        time.sleep(0.5)

    monkeypatch.setattr(_process, "_write_process_identity", slow_write_identity)

    result = _run_supervised(
        trial_dir,
        f"touch {marker}",
        timeout=0.2,
        attempt_id="pre-payload-deadline-attempt",
    )

    assert result.timed_out
    assert result.failure_reason is not None
    assert "before trainer launch" in result.failure_reason
    assert result.cleanup_confirmed
    # The trainer command itself never ran: the payload was never delivered.
    time.sleep(0.2)
    assert not marker.exists()


def test_supervisor_ready_wait_is_capped_by_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow supervisor startup cannot outlive the trial's own budget.

    The readiness wait used to be a fixed ``_SUPERVISOR_READY_TIMEOUT_SECONDS``
    regardless of the remaining budget; with a 10s allowance a 0.15s trial
    budget could stall far past its deadline and then still launch (review
    v0.5.16 / blocker 6).
    """
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    marker = tmp_path / "trainer_ran.txt"

    slow_supervisor = tmp_path / "slow_supervisor.py"
    slow_supervisor.write_text(
        "import os, signal, sys, time\n"
        # Mirror the real supervisor's mask reset: the parent's launch window
        # blocks shutdown signals and the mask survives exec.
        "signal.pthread_sigmask(signal.SIG_SETMASK, set())\n"
        "time.sleep(1.0)\n"
        "os.write(int(sys.argv[1]), b'R')\n"
        "time.sleep(30)\n"
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


def test_proc_wait_uses_remaining_budget_not_original_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch overhead must consume the budget, not silently extend it.

    Pre-fix, ``proc.wait(timeout=<original duration>)`` restarted the clock
    after launch bookkeeping, so a trainer could run for (overhead + budget)
    wallclock (review v0.5.16 / blocker 6).
    """
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    marker = tmp_path / "trainer_finished.txt"

    import phasesweep.runtime.process as _process

    real_write_identity = _process._write_process_identity

    def slow_write_identity(path, identity):  # noqa: ANN001, ANN202
        real_write_identity(path, identity)
        time.sleep(0.6)

    monkeypatch.setattr(_process, "_write_process_identity", slow_write_identity)

    # The trainer needs 0.6s; the original 1.0s duration would fit it, but
    # after 0.6s of injected launch overhead only ~0.4s of budget remains.
    result = _run_supervised(
        trial_dir,
        f"sleep 0.6 && touch {marker}",
        timeout=1.0,
        attempt_id="remaining-budget-attempt",
    )

    assert result.timed_out
    assert not marker.exists()


def test_phase_deadline_expiring_during_launch_fails_as_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-trial phase whose budget dies mid-launch must not publish success.

    Review v0.5.16 / blocker 6 reproduction: with startup delay injected into
    the launch path and a tiny phase budget, the trainer used to start after
    the deadline had expired and the phase published success with
    ``incomplete: false``. Now the launch aborts pre-payload, the trial
    fails, and the phase surfaces a TimeoutError — no winner, no publication.
    """
    trial_dir_marker = tmp_path / "trainer_ran.txt"
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f"touch {trial_dir_marker} && echo x=1.0 {{overrides}}",
        n_trials=1,
        timeout_seconds_per_phase=0.2,
    )

    import phasesweep.runtime.process as _process

    real_write_identity = _process._write_process_identity

    def slow_write_identity(path, identity):  # noqa: ANN001, ANN202
        real_write_identity(path, identity)
        time.sleep(0.5)

    monkeypatch.setattr(_process, "_write_process_identity", slow_write_identity)

    with pytest.raises(TimeoutError, match="deadline"):
        run_experiment(experiment)

    assert not trial_dir_marker.exists(), "trainer started after the phase deadline expired"


def test_normal_root_exit_kills_background_descendant(tmp_path: Path) -> None:
    """Root exits 0 while a child ignores SIGTERM.

    Previous code treated root-exit-zero as clean and deleted identity files,
    leaking the child with no forensic trail for the reaper. Now this is a
    lifecycle failure: descendants are killed, identity files are preserved,
    and failure_reason is set.
    """
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
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


def test_terminate_process_groups_shares_grace_across_groups(tmp_path: Path) -> None:
    """Two SIGTERM-ignoring groups burn ONE shared grace window, not one each.

    The shutdown handler kills every active trial group of an ``n_jobs > 1``
    phase through this path; serial escalation would multiply worst-case
    shutdown latency by the trial parallelism (review v0.5.17 gap hunt).
    """
    grace = 1.5
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
    # Serial escalation would burn at least one full grace per group.
    assert elapsed < 2 * grace - 0.2, f"escalation took {elapsed:.2f}s; grace not shared"


def test_terminate_process_groups_reports_per_group_verdicts() -> None:
    """An already-gone group confirms instantly; a live group still gets killed."""
    dead = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    dead.wait()
    live = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        verdicts = _terminate_process_groups((dead.pid, live.pid), grace_seconds=5.0)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(live.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            live.wait(timeout=5)

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
    active: dict[int, object] = {1234: object()}

    monkeypatch.setattr("phasesweep.runtime.process._active_children", active)

    def fake_terminate(pgids: tuple[int, ...], *, grace_seconds: float) -> dict[int, bool]:
        terminated.extend(pgids)
        active.clear()
        return dict.fromkeys(pgids, True)

    monkeypatch.setattr("phasesweep.runtime.process._terminate_process_groups", fake_terminate)

    with pytest.raises(PhaseSweepShutdown) as excinfo:
        _shutdown_handler(signal.SIGTERM, None)

    assert terminated == [1234]
    assert excinfo.value.report.cleanup_confirmed is True
    assert excinfo.value.report.child_pgids == (1234,)


def test_shutdown_handler_reports_uncertain_when_group_termination_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("phasesweep.runtime.process._active_children", {1234: object()})
    monkeypatch.setattr(
        "phasesweep.runtime.process._terminate_process_groups",
        lambda pgids, *, grace_seconds: dict.fromkeys(pgids, False),
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
    active: dict[int, object] = {1234: object(), 5678: object()}
    terminated: list[int] = []
    reentered = False

    monkeypatch.setattr("phasesweep.runtime.process._active_children", active)

    def fake_terminate(pgids: tuple[int, ...], *, grace_seconds: float) -> dict[int, bool]:
        nonlocal reentered
        terminated.extend(pgids)
        if not reentered:
            reentered = True
            assert _shutdown_handler(signal.SIGINT, None) is None
        return dict.fromkeys(pgids, True)

    monkeypatch.setattr("phasesweep.runtime.process._terminate_process_groups", fake_terminate)

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("phasesweep.runtime.process._process_group_exists", lambda pgid: True)
    monkeypatch.setattr(
        "phasesweep.runtime.process._group_member_pids",
        lambda pgid: (_ for _ in ()).throw(AssertionError("must not rescan /proc")),
    )
    monkeypatch.setattr("phasesweep.runtime.process._member_pids_alive", lambda pgid, pids: True)

    assert _process_group_alive_with_members(1234, {11}) is True


def test_process_group_alive_refreshes_when_cached_members_are_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member_sets: list[set[int]] = []
    scans: list[int] = []

    monkeypatch.setattr("phasesweep.runtime.process._process_group_exists", lambda pgid: True)

    def fake_group_member_pids(pgid: int) -> list[int]:
        scans.append(pgid)
        return [22]

    def fake_member_pids_alive(pgid: int, pids: set[int] | list[int]) -> bool:
        member_sets.append(set(pids))
        return 22 in pids

    monkeypatch.setattr("phasesweep.runtime.process._group_member_pids", fake_group_member_pids)
    monkeypatch.setattr("phasesweep.runtime.process._member_pids_alive", fake_member_pids_alive)

    member_pids = {11}

    assert _process_group_alive_with_members(1234, member_pids) is True
    assert member_pids == {22}
    assert scans == [1234]
    assert member_sets == [{11}, {22}]


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

    monkeypatch.setattr("phasesweep.runtime.process.os.killpg", fake_killpg)
    monkeypatch.setattr(
        "phasesweep.runtime.process._process_group_alive_with_members",
        lambda pgid, member_pids: group_survives,
    )

    assert _terminate_process_group(1234, grace_seconds=0.0) is expected
    assert required_signals.issubset({sig for _, sig in calls})


def test_reaper_raises_when_cleanup_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When kill_stale_group returns False the reaper must refuse to advance.

    The previous behavior logged the survivor and still called ``study.tell``,
    which let new trials launch onto a potentially-leaked GPU.
    """

    monkeypatch.setattr(
        "phasesweep.engine.guards._read_trial_process_identity",
        lambda *_args, **_kwargs: StaleProcessIdentity(
            schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
            attempt_id="uncertain-attempt",
            pid=99999,
            pgid=99999,
            proc_starttime=12345,
            boot_id="test-boot",
        ),
    )
    monkeypatch.setattr(
        "phasesweep.engine.guards.cleanup_stale_trial_process",
        lambda _identity: False,
    )

    exp = make_experiment(workdir=tmp_path / "runs")
    study = optuna.create_study(direction="maximize")

    # Inject one RUNNING trial so the reaper has something to chew on.
    trial = study.ask()
    trial.set_user_attr(ATTEMPT_ID_ATTR, "uncertain-attempt")
    trial.set_user_attr(TRIAL_DIR_ATTR, str(tmp_path / "runs" / "t" / "p" / "trial_00000"))
    assert study.trials[trial.number].state == optuna.trial.TrialState.RUNNING

    with pytest.raises(RuntimeError, match="cleanup could not prove"):
        _reap_stale_trials(study, exp, exp.phases[0].name)


def test_public_run_experiment_enters_signal_handler_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Library callers using ``run_experiment`` directly get the same cleanup
    contract as CLI callers."""
    installed = _install_signal_probe(monkeypatch)

    # Minimal trial_command that emits the metric captured by the log extractor.
    script = "print('x=1')"
    exp = make_experiment(
        workdir=tmp_path / "runs",
        trial_command=f'{sys.executable} -c "{script}" trial_dir={{trial_dir}} {{overrides}}',
    )
    run_experiment(exp)
    assert installed["called"] is True


def test_dry_run_does_not_enter_signal_handler_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry-run launches no children, so it must not perturb the signal mask."""
    installed = _install_signal_probe(monkeypatch)

    exp = make_experiment(workdir=tmp_path / "runs")
    run_experiment(exp, dry_run=True)
    assert installed["called"] is False


def testdefer_shutdown_signals_blocks_and_restores() -> None:
    """The context manager must add SIGTERM/SIGINT to the thread mask on entry
    and restore the original mask on exit."""
    if not hasattr(signal, "pthread_sigmask"):
        pytest.skip("pthread_sigmask not available")

    before = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    try:
        with defer_shutdown_signals():
            inside = signal.pthread_sigmask(signal.SIG_BLOCK, set())
            assert signal.SIGTERM in inside
            assert signal.SIGINT in inside
        after = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        assert after == before
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, before)


def test_install_signal_handlers_unblocks_inherited_shutdown_mask() -> None:
    """Startup should recover if the orchestrator inherited blocked SIGTERM."""
    if not hasattr(signal, "pthread_sigmask"):
        pytest.skip("pthread_sigmask not available")

    code = r"""
import os, signal
from phasesweep.runtime.process import install_signal_handlers, defer_shutdown_signals
signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGTERM,))
install_signal_handlers()
with defer_shutdown_signals():
    print("queued", flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
print("post-context", flush=True)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=5.0,
        check=False,
    )
    assert "queued" in proc.stdout
    assert proc.returncode == 128 + signal.SIGTERM


def test_pending_sigterm_inside_signal_deferred_sections_does_not_deadlock() -> None:
    """Queued SIGTERM must not deadlock while launch or registry locks are held."""
    cases = [
        (
            "launch_lock",
            "queued",
            r"""
import os, signal
from phasesweep.runtime.process import install_signal_handlers, defer_shutdown_signals, _launch_lock
install_signal_handlers()
with defer_shutdown_signals(), _launch_lock:
    print("queued", flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
print("post-context", flush=True)
""",
        ),
        (
            "registry_lock",
            "locked",
            r"""
import os, signal
from phasesweep.runtime.process import install_signal_handlers, defer_shutdown_signals, _lock
install_signal_handlers()
with defer_shutdown_signals():
    with _lock:
        print("locked", flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
print("post-context", flush=True)
""",
        ),
    ]

    for case, marker, code in cases:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            timeout=5.0,
            check=False,
        )
        assert marker in proc.stdout, case
        assert proc.returncode == 128 + signal.SIGTERM, case


def test_sigterm_via_worker_thread_mid_launch_window_defers_instead_of_deadlocking() -> None:
    """A signal tripped by a non-main thread mid-window must defer, not deadlock.

    Kernel masking in ``defer_shutdown_signals`` only covers the main thread.
    Library pools (e.g. BLAS workers pulled in via numpy/optuna) keep SIGTERM
    unblocked, so a process-directed SIGTERM sent during the masked launch
    window is delivered to one of them — and CPython then runs the Python
    handler in the main thread anyway, mid-critical-section. Pre-fix the
    handler re-acquired ``_launch_lock`` held by that same thread and hung
    until the MCP server's 30s grace SIGKILLed the runner with no status.json
    written (the flaky-cancel e2e failures). The handler must record the signal
    and let the window exit service it.
    """
    code = r"""
import os, signal, threading, time
import phasesweep.runtime.process as proc_mod
from phasesweep.runtime.process import install_signal_handlers, defer_shutdown_signals, _launch_lock

install_signal_handlers()

# Stand-in for a BLAS pool worker: SIGTERM stays unblocked here, so the kernel
# delivers the process-directed signal to this thread while the main thread is
# masked inside the launch window.
ready = threading.Event()
def helper():
    ready.set()
    threading.Event().wait(30)
threading.Thread(target=helper, daemon=True).start()
assert ready.wait(5)

with defer_shutdown_signals(), _launch_lock:
    os.kill(os.getpid(), signal.SIGTERM)
    deadline = time.time() + 5
    while proc_mod._deferred_shutdown_signum is None and time.time() < deadline:
        time.sleep(0.005)
    print("recorded-mid-window" if proc_mod._deferred_shutdown_signum is not None
          else "never-recorded", flush=True)
print("post-context", flush=True)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=15.0,
        check=False,
    )
    assert "recorded-mid-window" in proc.stdout, proc.stdout + proc.stderr
    assert "post-context" not in proc.stdout  # window exit must raise the shutdown
    assert proc.returncode == 128 + signal.SIGTERM


def test_deferred_shutdown_services_at_outermost_window_exit() -> None:
    """A shutdown recorded mid-window fires only when the outermost window exits."""
    inner_exited = False
    with pytest.raises(PhaseSweepShutdown) as excinfo, defer_shutdown_signals():
        with defer_shutdown_signals():
            # Emulates CPython invoking the handler in the main thread
            # after a worker-thread delivery: it must record and return.
            _shutdown_handler(signal.SIGTERM, None)
        inner_exited = True
    assert inner_exited, "inner window exit must not service the deferred shutdown"
    assert excinfo.value.code == 128 + signal.SIGTERM


def test_run_supervised_reports_uncertain_cleanup_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``_terminate_process_group`` returns ``False`` (cleanup uncertain),
    ``run_supervised`` must surface that in ``ProcessResult.cleanup_confirmed``
    so the orchestrator can abort instead of launching more trials."""
    # Pattern: actually kill the group with the real implementation, then
    # report uncertain. Exercises the orchestrator's "cleanup uncertainty"
    # branch without leaving real ``time.sleep(60)`` zombies behind for the
    # rest of the test run (review v0.5.11).
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


def test_uncertain_cleanup_aborts_optimization_not_just_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``UnsafeProcessCleanupError`` must propagate out of ``study.optimize()``
    in the n_jobs=1 path. Pre-v0.5.10, ``_kill_group`` ignored the boolean and
    the orchestrator swallowed the leaked-process condition via
    ``TrialExecutionError``. v0.5.11 review: hard_abort state is the only
    mechanism that surfaces this for n_jobs>1; n_jobs=1 still works through
    direct exception propagation, but we re-raise from hard_abort regardless
    so both paths share a common contract."""
    _report_uncertain_after_real_terminate(monkeypatch)

    exp = make_experiment(
        workdir=tmp_path / "runs",
        n_trials=2,
        timeout_seconds_per_trial=0.1,
        trial_command=f"{sys.executable} -c 'import time; time.sleep(60)' {{overrides}}",
    )

    with pytest.raises(UnsafeProcessCleanupError):
        run_experiment(exp)


def test_uncertain_cleanup_aborts_parallel_phase_before_reusing_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsafe cleanup must be a hard phase abort under ``n_jobs > 1``.

    v0.5.11 had two coupled defects in the parallel path:
      1. The GPU lease was released before ``cleanup_confirmed`` was
         observed, so a queued worker could acquire the just-freed lease
         and launch a second trial onto the leaked process group.
      2. Optuna's threaded n_jobs>1 ``study.optimize`` does NOT propagate
         non-caught objective exceptions: it logs them and marks the
         trial FAIL. The public exception surfaced as
         ``NoFeasibleTrialError`` instead of ``UnsafeProcessCleanupError``.

    The fix is an orchestrator-owned hard_abort flag, set inside the GPU
    lease before cleanup_confirmed is checked, and re-raised after
    ``study.optimize`` returns.
    """
    import contextlib as _contextlib

    _report_uncertain_after_real_terminate(monkeypatch)

    exp = make_experiment(
        workdir=tmp_path / "runs",
        n_trials=2,
        n_jobs=2,
        gpu_ids=[0],  # one GPU, two workers — forces queueing
        timeout_seconds_per_trial=0.2,
        max_consecutive_failures=100,  # large, so we don't abort via that path
        trial_command=f"{sys.executable} -c 'import time; time.sleep(60)' {{overrides}}",
    )

    phase_dir = tmp_path / "runs" / exp.experiment / exp.phases[0].name

    try:
        with pytest.raises(UnsafeProcessCleanupError):
            run_experiment(exp)

        # The queued second worker must have pruned before launch. If the
        # GPU was reused or hard_abort was checked too late, a second
        # ``trial_*/process_identity.json`` record would exist.
        launched_identities = sorted(phase_dir.glob(f"trial_*/{PROCESS_IDENTITY_FILE}"))
        assert len(launched_identities) == 1, (
            f"A queued parallel trial launched after unsafe cleanup. "
            f"Found identities: {[p.parent.name for p in launched_identities]}. "
            "Unsafe cleanup must hard-abort BEFORE the GPU lease is released "
            "to a queued worker."
        )
    finally:
        # Defense in depth: if the fake cleanup is ever changed to actually
        # leak, kill the surviving group here so we don't pollute the host.
        for identity_file in phase_dir.glob(f"trial_*/{PROCESS_IDENTITY_FILE}"):
            with _contextlib.suppress(Exception):
                os.killpg(json.loads(identity_file.read_text())["pgid"], signal.SIGKILL)


def test_trials_csv_written_even_when_hard_abort_propagates_through_optimize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """trials.csv must be written on every exit path, including the n_jobs=1
    hard-abort path where ``UnsafeProcessCleanupError`` propagates directly
    out of ``study.optimize`` rather than being re-raised after the call
    returns. Without this, the forensic CSV is missing precisely when the
    user needs it most — after a safety-critical abort. Review v0.5.12."""
    import contextlib as _contextlib

    _report_uncertain_after_real_terminate(monkeypatch)

    exp = make_experiment(
        workdir=tmp_path / "runs",
        n_trials=1,
        n_jobs=1,  # serial path: UnsafeProcessCleanupError exits via study.optimize raise
        timeout_seconds_per_trial=0.2,
        trial_command=f"{sys.executable} -c 'import time; time.sleep(60)' {{overrides}}",
    )

    phase_dir = tmp_path / "runs" / exp.experiment / exp.phases[0].name

    try:
        with pytest.raises(UnsafeProcessCleanupError):
            run_experiment(exp)

        csv_path = phase_dir / "trials.csv"
        assert csv_path.exists(), (
            f"trials.csv missing after hard abort. Forensic data must survive "
            f"every exit path. Looked at {csv_path}."
        )
        # Must contain the failed trial's row, not just a header.
        content = csv_path.read_text()
        assert "number" in content and "state" in content, (
            f"CSV header missing expected columns: {content[:200]!r}"
        )
        assert content.count("\n") >= 2, f"CSV has no trial rows, only header: {content!r}"
        # The failed trial should be recorded as FAIL, not lost.
        assert "FAIL" in content, f"CSV missing FAIL row for hard-aborted trial: {content!r}"
    finally:
        for identity_file in phase_dir.glob(f"trial_*/{PROCESS_IDENTITY_FILE}"):
            with _contextlib.suppress(Exception):
                os.killpg(json.loads(identity_file.read_text())["pgid"], signal.SIGKILL)
