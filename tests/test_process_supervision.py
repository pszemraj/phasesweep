"""Supervised subprocess lifecycle: Popen registration, signal handler installation and forwarding, descendant cleanup, cleanup_confirmed propagation, and the launch-window deadlock guard. Some tests run as real subprocesses because the behavior is POSIX signal delivery, not Python state we can mock."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.engine.guards import _reap_stale_trials
from phasesweep.engine.state import ATTEMPT_ID_ATTR, TRIAL_DIR_ATTR
from phasesweep.engine.trial import UnsafeProcessCleanupError
from phasesweep.runtime.process import (
    PROCESS_IDENTITY_FILE,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    PhaseSweepShutdown,
    StaleProcessIdentity,
    _defer_shutdown_signals,
    _process_group_alive_with_members,
    _shutdown_handler,
    _terminate_process_group,
    cleanup_stale_trial_process,
    is_pid_alive,
    is_pid_zombie,
    read_stale_process_identity,
    reap_child,
    run_supervised,
)
from tests.conftest import make_experiment


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
    installed = {"called": False}

    def fake_install() -> None:
        installed["called"] = True

    monkeypatch.setattr("phasesweep.engine.run.install_signal_handlers", fake_install)
    return installed


def test_run_supervised_persists_pgid_on_failure(tmp_path: Path) -> None:
    """Failing trials leave one complete atomic identity for forensic recovery."""
    if not Path("/proc/self/stat").exists():
        pytest.skip("Linux-only test")

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    with (trial_dir / "out.log").open("w") as fout, (trial_dir / "err.log").open("w") as ferr:
        result = run_supervised(
            "false",
            env=os.environ.copy(),
            stdout=fout,
            stderr=ferr,
            timeout=None,
            trial_dir=trial_dir,
            attempt_id="failure-attempt",
        )
    assert result.return_code != 0
    identity = read_stale_process_identity(
        trial_dir,
        expected_attempt_id="failure-attempt",
    )
    assert identity.pid == result.pid
    assert identity.pgid == result.pid
    assert identity.proc_starttime is not None
    assert identity.boot_id is not None


def test_run_supervised_cleans_identity_files_on_success(tmp_path: Path) -> None:
    """Clean exit removes the durable process identity."""
    if not Path("/proc/self/stat").exists():
        pytest.skip("Linux-only test")

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    with (trial_dir / "out.log").open("w") as fout, (trial_dir / "err.log").open("w") as ferr:
        result = run_supervised(
            "true",
            env=os.environ.copy(),
            stdout=fout,
            stderr=ferr,
            timeout=None,
            trial_dir=trial_dir,
            attempt_id="success-attempt",
        )
    assert result.return_code == 0
    assert result.duration_seconds >= 0.0
    assert not (trial_dir / PROCESS_IDENTITY_FILE).exists()


def test_run_supervised_terminates_child_when_identity_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A metadata write failure after Popen must not leave the child running."""
    import phasesweep.runtime.process as process

    real_atomic_write_text = process.atomic_write_text

    def fail_pid_write(path: Path, text: str) -> None:
        if path.name == PROCESS_IDENTITY_FILE:
            raise OSError("identity disk full")
        real_atomic_write_text(path, text)

    monkeypatch.setattr("phasesweep.runtime.process.atomic_write_text", fail_pid_write)

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    trainer_started = tmp_path / "trainer-started"
    with (trial_dir / "out.log").open("w") as fout, (trial_dir / "err.log").open("w") as ferr:
        result = run_supervised(
            f'{sys.executable} -c "from pathlib import Path; '
            f"Path({str(trainer_started)!r}).write_text('started')\"",
            env=os.environ.copy(),
            stdout=fout,
            stderr=ferr,
            timeout=None,
            trial_dir=trial_dir,
            attempt_id="identity-write-failure",
        )

    assert result.cleanup_confirmed is True
    assert "failed to persist process identity" in (result.failure_reason or "")
    assert not trainer_started.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(result.pid, 0)


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


def test_reap_child_is_strictly_nonblocking(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_waitpid(pid: int, flags: int) -> tuple[int, int]:
        calls.append((pid, flags))
        return (0, 0)

    def fail_if_proc_polled(pid: int) -> bool:
        raise AssertionError(f"reap_child should not poll /proc for pid {pid}")

    monkeypatch.setattr("phasesweep.runtime.process.os.waitpid", fake_waitpid)
    monkeypatch.setattr("phasesweep.runtime.process.is_pid_zombie", fail_if_proc_polled)

    assert reap_child(12345) is False

    assert calls == [(12345, os.WNOHANG)]


def test_reap_child_reports_when_it_reaped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "phasesweep.runtime.process.os.waitpid",
        lambda pid, flags: (pid, 0),
    )

    assert reap_child(12345) is True


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
    child_script = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    )
    parent_script = (
        f"import os, subprocess, sys, time; "
        f"p = subprocess.Popen([sys.executable, '-c', {child_script!r}]); "
        f"open({str(marker)!r}, 'w').write(str(p.pid)); "
        f"sys.stdout.flush(); "
        # Parent exits cleanly (and immediately) on SIGTERM, abandoning child.
        f"import signal as _s; "
        f"_s.signal(_s.SIGTERM, lambda *_: (sys.stdout.flush(), os._exit(0))); "
        f"time.sleep(60)"
    )
    cmd = f"python -c {parent_script!r}"

    with (trial_dir / "stdout.log").open("w") as out, (trial_dir / "stderr.log").open("w") as err:
        # Wait for child PID file before timeout fires.
        # Use a thread to write the marker check; simpler: just timeout=2.0
        # and confirm the marker was written (child started).
        result = run_supervised(
            cmd,
            env=os.environ.copy(),
            stdout=out,
            stderr=err,
            timeout=2.0,
            trial_dir=trial_dir,
            attempt_id="timeout-attempt",
        )

    assert result.timed_out, "trial should have hit the configured timeout"
    assert marker.exists(), "child PID marker was never written; test setup is wrong"

    child_pid = int(marker.read_text().strip())

    # The child must be dead within a reasonable window after run_supervised returns.
    # If _kill_group only waited for the root, the child would still be alive here.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            return  # success
        time.sleep(0.05)

    # Cleanup before failing so we don't leak a python process across the test run.
    with __import__("contextlib").suppress(ProcessLookupError):
        os.kill(child_pid, signal.SIGKILL)
    pytest.fail(f"timeout left descendant process {child_pid} alive")


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

    child_script = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    )
    parent_script = (
        "import os, subprocess, sys; "
        f"p = subprocess.Popen([sys.executable, '-c', {child_script!r}]); "
        f"open({str(marker)!r}, 'w').write(str(p.pid)); "
        "sys.stdout.flush(); "
        "os._exit(0)"
    )

    cmd = f"python -c {parent_script!r}"

    with (trial_dir / "stdout.log").open("w") as out, (trial_dir / "stderr.log").open("w") as err:
        result = run_supervised(
            cmd,
            env=os.environ.copy(),
            stdout=out,
            stderr=err,
            timeout=10.0,
            trial_dir=trial_dir,
            attempt_id="descendant-attempt",
        )

    # Must be flagged as a lifecycle failure, not a clean exit.
    assert result.failure_reason is not None
    assert "still had live descendants" in result.failure_reason

    # The atomic identity must be preserved for forensics.
    assert (trial_dir / PROCESS_IDENTITY_FILE).exists()

    # Descendant must be dead.
    assert marker.exists(), "child PID marker was never written; test setup broken"
    child_pid = int(marker.read_text().strip())
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            return  # success
        time.sleep(0.05)

    with __import__("contextlib").suppress(ProcessLookupError):
        os.kill(child_pid, signal.SIGKILL)
    pytest.fail(f"background descendant {child_pid} survived root exit")


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

    def fake_terminate(pgid: int, *, grace_seconds: float) -> bool:
        terminated.append(pgid)
        active.clear()
        return True

    monkeypatch.setattr("phasesweep.runtime.process._terminate_process_group", fake_terminate)

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
        "phasesweep.runtime.process._terminate_process_group",
        lambda pgid, *, grace_seconds: False,
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

    def fake_terminate(pgid: int, *, grace_seconds: float) -> bool:
        nonlocal reentered
        terminated.append(pgid)
        if not reentered:
            reentered = True
            assert _shutdown_handler(signal.SIGINT, None) is None
        return True

    monkeypatch.setattr("phasesweep.runtime.process._terminate_process_group", fake_terminate)

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


def test_terminate_process_group_reports_cleanup_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group termination reports confirmed cleanup only when it can prove it."""

    def group_survives(mp: pytest.MonkeyPatch, calls: list[tuple[int, int]]) -> None:
        def fake_killpg(pgid: int, sig: int) -> None:
            calls.append((pgid, sig))

        mp.setattr("phasesweep.runtime.process.os.killpg", fake_killpg)
        mp.setattr(
            "phasesweep.runtime.process._process_group_alive_with_members",
            lambda pgid, member_pids: True,
        )

    def already_gone(mp: pytest.MonkeyPatch, calls: list[tuple[int, int]]) -> None:
        def raise_lookup(pgid: int, sig: int) -> None:
            calls.append((pgid, sig))
            raise ProcessLookupError

        mp.setattr("phasesweep.runtime.process.os.killpg", raise_lookup)

    def permission_denied(mp: pytest.MonkeyPatch, calls: list[tuple[int, int]]) -> None:
        def raise_perm(pgid: int, sig: int) -> None:
            calls.append((pgid, sig))
            raise PermissionError

        mp.setattr("phasesweep.runtime.process.os.killpg", raise_perm)

    cases = [
        ("survives_sigkill", group_survives, False, {signal.SIGTERM, signal.SIGKILL}),
        ("already_gone", already_gone, True, {signal.SIGTERM}),
        ("permission_denied", permission_denied, False, {signal.SIGTERM}),
    ]

    for case, arrange, expected, required_signals in cases:
        calls: list[tuple[int, int]] = []
        with monkeypatch.context() as mp:
            arrange(mp, calls)
            assert _terminate_process_group(1234, grace_seconds=0.0) is expected, case

        sent_signals = {sig for _, sig in calls}
        assert required_signals.issubset(sent_signals), case


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


def test_public_run_experiment_installs_signal_handlers(
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


def test_dry_run_does_not_install_signal_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry-run launches no children, so it must not perturb the signal mask."""
    installed = _install_signal_probe(monkeypatch)

    exp = make_experiment(workdir=tmp_path / "runs")
    run_experiment(exp, dry_run=True)
    assert installed["called"] is False


def test_defer_shutdown_signals_blocks_and_restores() -> None:
    """The context manager must add SIGTERM/SIGINT to the thread mask on entry
    and restore the original mask on exit."""
    if not hasattr(signal, "pthread_sigmask"):
        pytest.skip("pthread_sigmask not available")

    before = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    try:
        with _defer_shutdown_signals():
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
from phasesweep.runtime.process import install_signal_handlers, _defer_shutdown_signals
signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGTERM,))
install_signal_handlers()
with _defer_shutdown_signals():
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


def test_launch_lock_serializes_signal_handler_against_in_flight_launch() -> None:
    """The shutdown handler must wait for ``_launch_lock`` before snapshotting.

    We simulate the n_jobs > 1 race: a worker thread is mid-launch (holding the
    lock), the main thread "receives" SIGTERM. Our test stand-in for the
    handler is a function that takes the same lock and snapshots
    ``_active_children``. It must block until the worker finishes register.
    """
    from phasesweep.runtime.process import _active_children, _launch_lock, _lock

    snapshot_done = threading.Event()
    snapshot: list[int] = []

    def worker() -> None:
        with _launch_lock:
            # Pretend Popen happened; now we're between Popen and _register.
            time.sleep(0.3)
            with _lock:
                _active_children[424242] = None  # type: ignore[assignment]

    def handler_stand_in() -> None:
        with _launch_lock, _lock:
            snapshot.extend(_active_children.keys())
        snapshot_done.set()

    try:
        t_worker = threading.Thread(target=worker)
        t_handler = threading.Thread(target=handler_stand_in)

        t_worker.start()
        # Tiny delay so worker is definitely inside the critical section.
        time.sleep(0.05)
        t_handler.start()

        t_worker.join(timeout=2.0)
        t_handler.join(timeout=2.0)

        assert snapshot_done.is_set()
        assert 424242 in snapshot
    finally:
        with _lock:
            _active_children.pop(424242, None)


def test_pending_sigterm_inside_signal_deferred_sections_does_not_deadlock() -> None:
    """Queued SIGTERM must not deadlock while launch or registry locks are held."""
    cases = [
        (
            "launch_lock",
            "queued",
            r"""
import os, signal
from phasesweep.runtime.process import install_signal_handlers, _defer_shutdown_signals, _launch_lock
install_signal_handlers()
with _defer_shutdown_signals(), _launch_lock:
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
from phasesweep.runtime.process import install_signal_handlers, _defer_shutdown_signals, _lock
install_signal_handlers()
with _defer_shutdown_signals():
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

    Kernel masking in ``_defer_shutdown_signals`` only covers the main thread.
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
from phasesweep.runtime.process import install_signal_handlers, _defer_shutdown_signals, _launch_lock

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

with _defer_shutdown_signals(), _launch_lock:
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
    with pytest.raises(PhaseSweepShutdown) as excinfo, _defer_shutdown_signals():
        with _defer_shutdown_signals():
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

    with (
        (tmp_path / "stdout.log").open("w") as stdout,
        (tmp_path / "stderr.log").open("w") as stderr,
    ):
        result = run_supervised(
            f"{sys.executable} -c 'import time; time.sleep(60)'",
            env=os.environ.copy(),
            stdout=stdout,
            stderr=stderr,
            timeout=0.1,
            trial_dir=tmp_path,
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
