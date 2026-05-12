"""Supervised subprocess lifecycle: Popen registration, signal handler installation and forwarding, descendant cleanup, cleanup_confirmed propagation, and the launch-window deadlock guard. Some tests run as real subprocesses because the behavior is POSIX signal delivery, not Python state we can mock."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import optuna
import pytest

from phasesweep import run_experiment
from phasesweep.orchestrator import (
    _reap_stale_trials,
)
from phasesweep.process import (
    _defer_shutdown_signals,
    _shutdown_handler,
    _terminate_process_group,
    run_supervised,
)
from phasesweep.runner import UnsafeProcessCleanupError
from tests.conftest import make_experiment


def _report_uncertain_after_real_terminate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch group termination to clean up the group but report uncertainty."""
    import phasesweep.process as _process

    real_terminate = _process._terminate_process_group

    def terminate_then_report_uncertain(pgid: int, *, grace_seconds: float) -> bool:
        real_terminate(pgid, grace_seconds=0.05)
        return False

    monkeypatch.setattr(
        "phasesweep.process._terminate_process_group",
        terminate_then_report_uncertain,
    )


def test_run_supervised_persists_pgid_on_failure(tmp_path: Path) -> None:
    """Failing trials leave pid + pgid + starttime files for forensic recovery."""
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
        )
    assert result.return_code != 0
    assert (trial_dir / "pid").is_file(), "pid file must persist on failure"
    assert (trial_dir / "pgid").is_file(), "pgid file must persist on failure"
    assert (trial_dir / "pid_starttime").is_file(), "starttime file must persist on failure"


def test_run_supervised_cleans_identity_files_on_success(tmp_path: Path) -> None:
    """Clean exit removes all three identity files."""
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
        )
    assert result.return_code == 0
    for name in ("pid", "pgid", "pid_starttime"):
        assert not (trial_dir / name).exists(), f"{name} should be cleaned up on success"


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
        )

    # Must be flagged as a lifecycle failure, not a clean exit.
    assert result.failure_reason is not None
    assert "still had live descendants" in result.failure_reason

    # Identity files must be preserved for forensics.
    assert (trial_dir / "pid").exists()
    assert (trial_dir / "pgid").exists()

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
    """SIGKILL phase must still target groups seen during SIGTERM phase.

    A worker thread can unregister a PGID after the root exits on SIGTERM but
    before the handler escalates to SIGKILL. Using the initial snapshot
    prevents that race.
    """
    killed: list[tuple[int, int]] = []
    active: dict[int, object] = {1234: object()}

    monkeypatch.setattr("phasesweep.process._active_children", active)
    monkeypatch.setattr("phasesweep.process._process_group_alive", lambda pgid: True)

    def fake_killpg(pgid: int, sig: int) -> None:
        killed.append((pgid, sig))
        # Simulate worker thread unregistering after SIGTERM.
        if sig == signal.SIGTERM:
            active.clear()

    monkeypatch.setattr("os.killpg", fake_killpg)

    with pytest.raises(SystemExit):
        _shutdown_handler(signal.SIGTERM, None)

    assert (1234, signal.SIGTERM) in killed
    assert (1234, signal.SIGKILL) in killed, (
        "SIGKILL must target the initial snapshot even though the registry was cleared"
    )


def test_trial_command_unbalanced_brace_rejected() -> None:
    with pytest.raises(ValueError, match="trial_command failed to render"):
        make_experiment(trial_command="echo {trial_dir", n_trials=1)


def test_terminate_process_group_returns_false_when_group_survives_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``_process_group_alive`` keeps reporting alive after SIGKILL, the
    function must return False so the reaper doesn't mark the trial FAIL."""
    calls: list[tuple[int, int]] = []

    def fake_killpg(pgid: int, sig: int) -> None:
        calls.append((pgid, sig))

    monkeypatch.setattr("phasesweep.process.os.killpg", fake_killpg)
    monkeypatch.setattr("phasesweep.process._process_group_alive", lambda pgid: True)

    assert _terminate_process_group(1234, grace_seconds=0.0) is False
    # Both SIGTERM and SIGKILL must have been attempted.
    sent_signals = [sig for _, sig in calls]
    assert signal.SIGTERM in sent_signals
    assert signal.SIGKILL in sent_signals


def test_terminate_process_group_returns_true_when_already_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ProcessLookupError from ``killpg`` is the success signal: nothing to clean up."""

    def raise_lookup(pgid: int, sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("phasesweep.process.os.killpg", raise_lookup)
    assert _terminate_process_group(1234, grace_seconds=0.0) is True


def test_terminate_process_group_returns_false_on_permission_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permission denied is uncertain; we cannot prove the group is gone."""

    def raise_perm(pgid: int, sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr("phasesweep.process.os.killpg", raise_perm)
    assert _terminate_process_group(1234, grace_seconds=0.0) is False


def test_reaper_raises_when_cleanup_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When kill_stale_group returns False the reaper must refuse to advance.

    The previous behavior logged the survivor and still called ``study.tell``,
    which let new trials launch onto a potentially-leaked GPU.
    """

    monkeypatch.setattr(
        "phasesweep.process.read_stale_process_identity",
        lambda trial_dir: __import__(
            "phasesweep.process", fromlist=["StaleProcessIdentity"]
        ).StaleProcessIdentity(pid=99999, pgid=99999, starttime=12345),
    )
    monkeypatch.setattr(
        "phasesweep.process.kill_stale_group",
        lambda pid, starttime, *, pgid: False,
    )

    exp = make_experiment(workdir=tmp_path / "runs")
    study = optuna.create_study(direction="maximize")

    # Inject one RUNNING trial so the reaper has something to chew on.
    trial = study.ask()
    assert study.trials[trial.number].state == optuna.trial.TrialState.RUNNING

    with pytest.raises(RuntimeError, match="cleanup could not prove"):
        _reap_stale_trials(study, exp, exp.phases[0].name)


def test_public_run_experiment_installs_signal_handlers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Library callers using ``run_experiment`` directly get the same cleanup
    contract as CLI callers."""
    installed = {"called": False}

    def fake_install() -> None:
        installed["called"] = True

    monkeypatch.setattr("phasesweep.process.install_signal_handlers", fake_install)

    # Minimal trial_command that writes the metric file the JsonExtractor expects.
    # Avoid {} literals in the script so the override-template parser doesn't
    # mistake them for placeholders.
    script = (
        "import json, pathlib, sys; "
        "trial_dir = sys.argv[1].split('=', 1)[1]; "
        "pathlib.Path(trial_dir, 'r.json').write_text(json.dumps(dict(x=1)))"
    )
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
    installed = {"called": False}

    def fake_install() -> None:
        installed["called"] = True

    monkeypatch.setattr("phasesweep.process.install_signal_handlers", fake_install)

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


def test_launch_lock_serializes_signal_handler_against_in_flight_launch() -> None:
    """The shutdown handler must wait for ``_launch_lock`` before snapshotting.

    We simulate the n_jobs > 1 race: a worker thread is mid-launch (holding the
    lock), the main thread "receives" SIGTERM. Our test stand-in for the
    handler is a function that takes the same lock and snapshots
    ``_active_children``. It must block until the worker finishes register.
    """
    from phasesweep.process import _active_children, _launch_lock, _lock

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


def test_pending_sigterm_during_launch_context_does_not_deadlock() -> None:
    """SIGTERM queued while the launch critical section is active must not
    deadlock. The correct ordering is ``_defer_shutdown_signals()`` (outermost)
    then ``_launch_lock`` (innermost). Pre-v0.5.10 the order was reversed,
    causing the signal handler to block on ``_launch_lock`` held by the
    interrupted thread.

    The product contract is "no deadlock; clean exit with 128+SIGTERM" — NOT
    "no Python bytecode runs after the context exits". CPython's signal
    handler scheduling after ``pthread_sigmask`` unblock is non-deterministic
    relative to the next bytecode (the signal is pending; the interpreter
    notices it on the next signal-checkpoint instruction, which may or may
    not be before a subsequent ``print``). Asserting that a post-context
    statement never runs makes this test flaky for no product reason.
    Review v0.5.12.
    """
    code = r"""
import os, signal
from phasesweep.process import install_signal_handlers, _defer_shutdown_signals, _launch_lock
install_signal_handlers()
with _defer_shutdown_signals(), _launch_lock:
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


def test_signal_while_registry_lock_held_does_not_deadlock() -> None:
    """SIGTERM while ``_lock`` is held (e.g. during ``_unregister``) must not
    deadlock. The ``_unregister`` function defers shutdown signals before
    acquiring ``_lock`` so the handler cannot interrupt mid-mutation.

    See the launch-context test above for why we do NOT assert that a
    post-context statement is unreached. Review v0.5.12.
    """
    code = r"""
import os, signal
from phasesweep.process import install_signal_handlers, _defer_shutdown_signals, _lock
install_signal_handlers()
with _defer_shutdown_signals():
    with _lock:
        print("locked", flush=True)
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
    assert "locked" in proc.stdout
    assert proc.returncode == 128 + signal.SIGTERM


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
        )

    assert result.timed_out is True
    assert result.cleanup_confirmed is False
    # Identity files must be preserved for forensics.
    assert (tmp_path / "pid").exists()
    assert (tmp_path / "pgid").exists()


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
        # ``trial_*/pid`` file would exist.
        launched_pid_files = sorted(phase_dir.glob("trial_*/pid"))
        assert len(launched_pid_files) == 1, (
            f"A queued parallel trial launched after unsafe cleanup. "
            f"Found pid files: {[p.parent.name for p in launched_pid_files]}. "
            "Unsafe cleanup must hard-abort BEFORE the GPU lease is released "
            "to a queued worker."
        )
    finally:
        # Defense in depth: if the fake cleanup is ever changed to actually
        # leak, kill the surviving group here so we don't pollute CI.
        for pgid_file in phase_dir.glob("trial_*/pgid"):
            with _contextlib.suppress(Exception):
                os.killpg(int(pgid_file.read_text().strip()), signal.SIGKILL)


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
        for pgid_file in phase_dir.glob("trial_*/pgid"):
            with _contextlib.suppress(Exception):
                os.killpg(int(pgid_file.read_text().strip()), signal.SIGKILL)
