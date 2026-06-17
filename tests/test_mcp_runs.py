"""Run-handle store: persistence round-trip and derived run-state logic."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from phasesweep.mcp.runs import RunHandle, RunStore, utc_now_iso
from phasesweep.runtime.process import is_pid_zombie, read_proc_starttime


def _make_handle(
    store: RunStore,
    *,
    run_id: str,
    experiment_id: str = "exp",
    pid: int,
    starttime: int | None,
) -> RunHandle:
    return RunHandle(
        run_id=run_id,
        experiment_id=experiment_id,
        config_sha256="0" * 64,
        pid=pid,
        pgid=pid,
        pid_starttime=starttime,
        started_at=utc_now_iso(),
        log_path=str(store.log_path(run_id)),
        status_path=str(store.status_path(run_id)),
    )


def _live_handle(store: RunStore, *, run_id: str, experiment_id: str = "exp") -> RunHandle:
    pid = os.getpid()
    return _make_handle(
        store,
        run_id=run_id,
        experiment_id=experiment_id,
        pid=pid,
        starttime=read_proc_starttime(pid),
    )


def _write_status(store: RunStore, run_id: str, **payload: object) -> None:
    store.status_path(run_id).write_text(json.dumps(payload))


def test_save_get_roundtrip(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = _live_handle(store, run_id="exp-1")
    store.save(handle)
    assert store.get("exp-1") == handle
    assert store.get("missing") is None


def test_save_replaces_existing_handle_without_temp_files(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    first = _live_handle(store, run_id="exp-1", experiment_id="old")
    second = _live_handle(store, run_id="exp-1", experiment_id="new")

    store.save(first)
    store.save(second)

    assert store.get("exp-1") == second
    assert list((tmp_path / "state" / "runs").glob("*.tmp")) == []
    assert list((tmp_path / "state" / "runs").glob(".*.tmp")) == []


@pytest.mark.parametrize(
    "unsafe",
    [
        "../../etc/passwd",
        "a/b",
        "..",
        "exp-1/../../../secret",
        "exp 1",
        "exp.1",
        "exp-1\n",
        "",
    ],
)
def test_get_rejects_unsafe_run_id(tmp_path: Path, unsafe: str) -> None:
    # An agent-supplied id must never be interpolated into a path it could use
    # to escape the runs dir; an out-of-shape id reads as a missing handle.
    store = RunStore(tmp_path / "state")
    store.save(_live_handle(store, run_id="exp-1"))
    assert store.get(unsafe) is None


def test_list_handles_skips_malformed(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    store.save(_live_handle(store, run_id="exp-1"))
    store.save(_live_handle(store, run_id="exp-2"))
    # A torn/partial handle file must not crash a read.
    (tmp_path / "state" / "runs" / "broken.json").write_text("{not valid json")
    assert {h.run_id for h in store.list_handles()} == {"exp-1", "exp-2"}


def test_state_succeeded_from_status(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = _live_handle(store, run_id="exp-1")
    _write_status(store, "exp-1", returncode=0, error_class=None)
    assert store.state(handle) == "succeeded"


@pytest.mark.parametrize("code", [143, 130])
def test_state_cancelled_from_signalled_code(tmp_path: Path, code: int) -> None:
    store = RunStore(tmp_path / "state")
    handle = _live_handle(store, run_id="exp-1")
    _write_status(store, "exp-1", returncode=code, error_class="cancelled")
    assert store.state(handle) == "cancelled"


def test_state_failed_from_nonzero_status(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = _live_handle(store, run_id="exp-1")
    _write_status(store, "exp-1", returncode=1, error_class="RuntimeError")
    assert store.state(handle) == "failed"


def test_state_running_for_live_pid_without_status(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = _live_handle(store, run_id="exp-1")
    assert store.state(handle) == "running"


def test_mark_cancelled_records_cancel_when_runner_left_no_status(tmp_path: Path) -> None:
    # SIGKILL escalation leaves no runner-written status; the canceller records
    # the cause so a later read is 'cancelled', not 'failed'.
    store = RunStore(tmp_path / "state")
    handle = _live_handle(store, run_id="exp-1")
    store.mark_cancelled_if_unrecorded(handle)
    assert store.state(handle) == "cancelled"
    assert json.loads(store.status_path("exp-1").read_text())["error_class"] == "cancelled"


def test_mark_cancelled_is_noop_when_status_exists(tmp_path: Path) -> None:
    # A graceful terminal cause (or genuine failure) is never clobbered.
    store = RunStore(tmp_path / "state")
    handle = _live_handle(store, run_id="exp-1")
    _write_status(store, "exp-1", returncode=0, error_class=None)  # succeeded
    store.mark_cancelled_if_unrecorded(handle)
    assert store.state(handle) == "succeeded"


def test_state_failed_on_pid_reuse_mismatch(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    live_starttime = read_proc_starttime(os.getpid())
    if live_starttime is None:
        pytest.skip("/proc starttime unavailable (non-Linux); no PID-reuse guard")
    # PID is alive (our own) but the saved starttime does not match, so it is a
    # different process than the one we launched -> not running -> failed.
    handle = _make_handle(store, run_id="exp-x", pid=os.getpid(), starttime=live_starttime + 99_999)
    assert store.state(handle) == "failed"


def test_live_run_for_ignores_terminal_runs(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    store.save(_live_handle(store, run_id="exp-run", experiment_id="exp"))
    store.save(_live_handle(store, run_id="exp-done", experiment_id="exp"))
    _write_status(store, "exp-done", returncode=0)  # terminal: succeeded

    live = store.live_run_for("exp")
    assert live is not None
    assert live.run_id == "exp-run"
    assert store.live_run_for("other-experiment") is None


def test_state_failed_for_zombie_runner_without_status(tmp_path: Path) -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("zombie detection relies on /proc")
    store = RunStore(tmp_path / "state")
    # Spawn a child that exits immediately. As its (unreaping) parent, the pid
    # lingers as a zombie that os.kill(pid, 0) still reports as alive - exactly
    # the SIGKILL/OOM case that must report failed, not running, or relaunch
    # would be blocked forever.
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not is_pid_zombie(proc.pid):
            time.sleep(0.02)
        if not is_pid_zombie(proc.pid):
            pytest.skip("could not observe a zombie (fast reaper?)")
        handle = _make_handle(
            store, run_id="zomb", pid=proc.pid, starttime=read_proc_starttime(proc.pid)
        )
        assert store.state(handle) == "failed"
        # state() reaped the child, so the zombie is gone, not merely filtered.
        assert not is_pid_zombie(proc.pid)
    finally:
        proc.wait()
