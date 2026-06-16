"""Run-handle store: persistence round-trip and derived run-state logic."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from phasesweep.mcp.runs import RunHandle, RunStore, utc_now_iso
from phasesweep.runtime.process import read_proc_starttime


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
