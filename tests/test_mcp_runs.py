"""Run-handle store: persistence round-trip and derived run-state logic."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from phasesweep.mcp.runs import RunStore, write_status_file
from phasesweep.runtime.process import is_pid_zombie, read_proc_starttime
from tests.mcp_helpers import make_run_handle, write_run_status


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_save_get_roundtrip(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1")
    store.save(handle)
    assert store.get("exp-1") == handle
    assert store.get("missing") is None


def test_launching_handle_roundtrip_is_failed_without_status(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", launch_state="launching")

    store.save(handle)
    loaded = store.get("exp-1")

    assert loaded == handle
    assert store.state(loaded) == "failed"
    assert store.live_runs() == []


def test_save_replaces_existing_handle_without_temp_files(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    first = make_run_handle(run_id="exp-1", experiment_id="old")
    second = make_run_handle(run_id="exp-1", experiment_id="new")

    store.save(first)
    store.save(second)

    assert store.get("exp-1") == second
    assert list((tmp_path / "state" / "runs").glob("*.tmp")) == []
    assert list((tmp_path / "state" / "runs").glob(".*.tmp")) == []


def test_write_status_file_replaces_existing_status_without_temp_files(tmp_path: Path) -> None:
    status_path = tmp_path / "state" / "logs" / "exp-1.status.json"

    write_status_file(status_path, {"run_id": "exp-1", "returncode": 1})
    write_status_file(status_path, {"run_id": "exp-1", "returncode": 0})

    assert json.loads(status_path.read_text())["returncode"] == 0
    assert list(status_path.parent.glob("*.tmp")) == []
    assert list(status_path.parent.glob(".*.tmp")) == []


def test_mcp_state_files_are_private_under_permissive_umask(tmp_path: Path) -> None:
    old_umask = os.umask(0)
    try:
        store = RunStore(tmp_path / "state")
        handle = make_run_handle(run_id="exp-1")
        store.save(handle)
        write_status_file(store.status_path("exp-1"), {"run_id": "exp-1", "returncode": 0})
        store.mark_cleanup_uncertain(handle)
    finally:
        os.umask(old_umask)

    assert _mode(tmp_path / "state") == 0o700
    assert _mode(tmp_path / "state" / "runs") == 0o700
    assert _mode(tmp_path / "state" / "logs") == 0o700
    assert _mode(tmp_path / "state" / "runs" / "exp-1.json") == 0o600
    assert _mode(tmp_path / "state" / "logs" / "exp-1.status.json") == 0o600
    assert _mode(tmp_path / "state" / "logs" / "exp-1.cleanup_uncertain.json") == 0o600


def test_open_existing_is_observational_and_requires_run_store_layout(tmp_path: Path) -> None:
    missing = tmp_path / "mistyped-state"

    with pytest.raises(ValueError, match="expected directories are missing"):
        RunStore.open_existing(missing)

    assert not missing.exists()

    state_dir = tmp_path / "state"
    RunStore(state_dir)
    before_modes = {
        path: _mode(path) for path in (state_dir, state_dir / "runs", state_dir / "logs")
    }

    RunStore.open_existing(state_dir)

    assert {
        path: _mode(path) for path in (state_dir, state_dir / "runs", state_dir / "logs")
    } == before_modes


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
    store.save(make_run_handle(run_id="exp-1"))
    assert store.get(unsafe) is None


def test_list_handles_skips_malformed(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    store.save(make_run_handle(run_id="exp-1"))
    store.save(make_run_handle(run_id="exp-2"))
    # A torn/partial handle file must not crash a read.
    (tmp_path / "state" / "runs" / "broken.json").write_text("{not valid json")
    assert {h.run_id for h in store.list_handles()} == {"exp-1", "exp-2"}


def test_get_skips_malformed_handle(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    (tmp_path / "state" / "runs" / "broken.json").write_text("{not valid json")
    assert store.get("broken") is None


def test_loaded_handle_must_match_filename(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    payload = asdict(make_run_handle(run_id="other"))
    (tmp_path / "state" / "runs" / "exp-1.json").write_text(json.dumps(payload))

    assert store.get("exp-1") is None
    assert store.list_handles() == []


def test_persisted_handle_omits_derived_store_paths(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1")
    store.save(handle)

    payload = json.loads((tmp_path / "state" / "runs" / "exp-1.json").read_text())
    assert "log_path" not in payload
    assert "status_path" not in payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("experiment_id", "../bad"),
        ("pid", 0),
        ("pid", "123"),
        ("pgid", 0),
        ("pgid", "123"),
        ("pid_starttime", 0),
        ("pid_starttime", "123"),
        ("launch_state", "bogus"),
        ("started_at", "not-a-timestamp"),
        ("started_at", "2026-07-17T12:00:00"),
    ],
)
def test_loaded_handle_shape_is_validated(tmp_path: Path, field: str, value: object) -> None:
    store = RunStore(tmp_path / "state")
    payload = asdict(make_run_handle(run_id="exp-1"))
    payload[field] = value
    (tmp_path / "state" / "runs" / "exp-1.json").write_text(json.dumps(payload))

    assert store.get("exp-1") is None
    assert store.list_handles() == []


@pytest.mark.parametrize("field,value", [("pid", os.getpid()), ("pgid", os.getpid())])
def test_launching_handle_cannot_have_process_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    store = RunStore(tmp_path / "state")
    payload = asdict(make_run_handle(run_id="exp-1", launch_state="launching"))
    payload[field] = value
    (tmp_path / "state" / "runs" / "exp-1.json").write_text(json.dumps(payload))

    assert store.get("exp-1") is None
    assert store.list_handles() == []


def test_latest_run_for_computes_newest_with_stable_tiebreaker(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    older = replace(
        make_run_handle(run_id="srv-z", experiment_id="srv"),
        started_at="2026-07-17T12:00:00+00:00",
    )
    newer_a = replace(
        make_run_handle(run_id="srv-a", experiment_id="srv"),
        started_at="2026-07-17T13:00:00+00:00",
    )
    newer_b = replace(newer_a, run_id="srv-b")
    unrelated = replace(newer_a, run_id="other-a", experiment_id="other")
    for handle in (newer_b, unrelated, older, newer_a):
        store.save(handle)

    assert store.latest_run_for("srv") == newer_b
    assert store.latest_run_for("missing") is None


@pytest.mark.parametrize(
    ("returncode", "error_class", "expected_state"),
    [
        pytest.param(0, None, "succeeded", id="success"),
        pytest.param(143, "cancelled", "cancelled", id="sigterm"),
        pytest.param(130, "cancelled", "cancelled", id="sigint"),
        pytest.param(1, "RuntimeError", "failed", id="failure"),
    ],
)
def test_terminal_status_determines_state(
    tmp_path: Path,
    returncode: int,
    error_class: str | None,
    expected_state: str,
) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1")
    write_run_status(store, "exp-1", returncode=returncode, error_class=error_class)

    assert store.state(handle) == expected_state


def test_pending_result_snapshot_keeps_run_live_until_finalized(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(
        run_id="exp-1",
        experiment_id="exp",
        pid=999999,
        starttime=111,
    )
    store.save(handle)
    write_run_status(
        store,
        "exp-1",
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot_state="pending",
    )

    assert store.state(handle) == "running"
    assert store.live_runs() == [handle]
    assert store.live_run_for("exp") == handle

    write_run_status(
        store,
        "exp-1",
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot_state="complete",
        result_snapshot={},
    )

    assert store.state(handle) == "succeeded"
    assert store.live_runs() == []
    assert store.live_run_for("exp") is None


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"run_id": "other", "returncode": 0},
        {"run_id": "exp-1", "returncode": "0"},
        {"run_id": "exp-1", "returncode": True},
        {"run_id": "exp-1", "returncode": 0, "cleanup_confirmed": "yes"},
        {"run_id": "exp-1", "returncode": 0, "error_class": 3},
        {"run_id": "exp-1", "returncode": 0, "result_snapshot_state": "unknown"},
        {"run_id": "exp-1", "returncode": 0, "result_snapshot_state": "complete"},
        {"run_id": "exp-1", "returncode": 0, "result_snapshot_error": 3},
    ],
)
def test_status_payload_shape_is_validated(tmp_path: Path, payload: object) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1")
    store.status_path("exp-1").write_text(json.dumps(payload), encoding="utf-8")

    assert store.recorded_terminal_status(handle) is None
    assert store.state(handle) == "running"


def test_status_read_failures_do_not_break_state_scans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1")
    store.save(handle)

    store.status_path("exp-1").write_bytes(b"\xff")
    assert store.recorded_terminal_status(handle) is None
    assert store.state(handle) == "running"
    assert store.live_runs() == [handle]

    store.status_path("exp-1").write_text('{"returncode": 0}', encoding="utf-8")
    real_read_text = Path.read_text

    def fail_status_read(path: Path, *args: object, **kwargs: object) -> str:
        if path == store.status_path("exp-1"):
            raise OSError("status file temporarily unreadable")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_status_read)

    assert store.recorded_terminal_status(handle) is None
    assert store.state(handle) == "running"
    assert store.live_runs() == [handle]


def test_state_failed_from_ordinary_cleanup_confirmed_failure(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", pid=999999, starttime=111)
    store.save(handle)
    write_run_status(
        store,
        "exp-1",
        returncode=1,
        error_class="RuntimeError",
        cleanup_confirmed=True,
    )

    assert store.state(handle) == "failed"
    assert store.live_runs() == []
    assert store.live_run_for("exp") is None


def test_state_running_for_live_pid_without_status(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1")
    assert store.state(handle) == "running"


def test_dead_runner_without_status_stays_live_until_recovery_evidence(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", pid=999999, starttime=111)
    store.save(handle)

    assert store.state(handle) == "running"
    assert store.cleanup_uncertain(handle)
    assert store.live_runs() == [handle]

    store.cleanup_recovery_path("exp-1").write_text(
        json.dumps(
            {
                "run_id": "exp-1",
                "config_sha256": handle.config_sha256,
                "cleanup_confirmed": True,
            }
        )
    )
    store.clear_cleanup_uncertain(handle)

    assert store.state(handle) == "failed"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"run_id": "other", "cleanup_confirmed": False},
        {"run_id": "exp-1", "config_sha256": "b" * 64, "cleanup_confirmed": False},
        {"run_id": "exp-1", "cleanup_confirmed": True},
        {"run_id": "exp-1", "cleanup_confirmed": False, "pid": "123"},
        {"run_id": "exp-1", "cleanup_confirmed": False, "pgid": 0},
        {"run_id": "exp-1", "cleanup_confirmed": False, "pid_starttime": True},
    ],
)
def test_cleanup_uncertain_marker_shape_is_validated(tmp_path: Path, payload: object) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(
        run_id="exp-1",
        config_sha256="a" * 64,
        pid=999999,
        starttime=111,
    )
    store.save(handle)
    store.cleanup_uncertain_path("exp-1").write_text(json.dumps(payload), encoding="utf-8")

    assert not store.cleanup_uncertain(handle)
    assert store.state(handle) == "running"
    assert store.cleanup_uncertain(handle)


def test_cleanup_uncertain_marker_preserves_spawned_identity_for_pending_handle(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "state")
    pending = make_run_handle(
        run_id="exp-1",
        config_sha256="a" * 64,
        launch_state="launching",
    )
    spawned = make_run_handle(
        run_id="exp-1",
        config_sha256="a" * 64,
        pid=4242,
        starttime=111,
    )
    store.save(pending)

    store.mark_cleanup_uncertain(spawned)
    store.mark_cleanup_uncertain(pending)

    marker = json.loads(store.cleanup_uncertain_path("exp-1").read_text())
    assert marker["pid"] == 4242
    assert marker["pgid"] == 4242
    assert marker["pid_starttime"] == 111
    identity = store.cleanup_identity(pending)
    assert (identity.pid, identity.pgid, identity.pid_starttime) == (4242, 4242, 111)


def test_confirmed_terminal_status_overrides_stale_cleanup_marker(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", pid=999999, starttime=111)
    store.save(handle)
    store.mark_cleanup_uncertain(handle)
    write_run_status(
        store,
        "exp-1",
        returncode=143,
        error_class="cancelled",
        cleanup_confirmed=True,
    )

    assert store.state(handle) == "cancelled"
    assert not store.cleanup_uncertain_path("exp-1").exists()

    store.mark_cleanup_uncertain(handle)

    assert store.state(handle) == "cancelled"
    assert not store.cleanup_uncertain_path("exp-1").exists()


def test_terminal_cleanup_uncertain_status_keeps_run_live_until_recovered(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(
        run_id="exp-1",
        experiment_id="exp",
        config_sha256="a" * 64,
        pid=999999,
        starttime=111,
    )
    store.save(handle)
    write_run_status(
        store,
        "exp-1",
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
    )

    assert store.state(handle) == "running"
    assert store.live_runs() == [handle]
    assert store.live_run_for("exp") == handle

    store.cleanup_recovery_path("exp-1").write_text(
        json.dumps(
            {
                "run_id": "exp-1",
                "config_sha256": "a" * 64,
                "cleanup_confirmed": True,
            }
        )
    )

    assert store.state(handle) == "failed"
    assert store.live_runs() == []
    assert store.live_run_for("exp") is None


def test_terminal_cleanup_recovery_must_match_handle_hash(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(
        run_id="exp-1",
        config_sha256="a" * 64,
        pid=999999,
        starttime=111,
    )
    store.save(handle)
    write_run_status(
        store,
        "exp-1",
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
    )
    store.cleanup_recovery_path("exp-1").write_text(
        json.dumps(
            {
                "run_id": "exp-1",
                "config_sha256": "b" * 64,
                "cleanup_confirmed": True,
            }
        )
    )

    assert store.state(handle) == "running"


def test_state_cleanup_uncertain_on_pid_reuse_mismatch(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    live_starttime = read_proc_starttime(os.getpid())
    if live_starttime is None:
        pytest.skip("/proc starttime unavailable (non-Linux); no PID-reuse guard")
    # PID is alive (our own) but the saved starttime does not match, so it is a
    # different process than the one we launched. The runner is gone, but its
    # separately-sessioned descendants are not proven gone, so fail closed.
    handle = make_run_handle(run_id="exp-x", pid=os.getpid(), starttime=live_starttime + 99_999)
    assert store.state(handle) == "running"
    assert store.cleanup_uncertain(handle)


def test_live_run_for_ignores_terminal_runs(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    store.save(make_run_handle(run_id="exp-run", experiment_id="exp"))
    store.save(make_run_handle(run_id="exp-done", experiment_id="exp"))
    write_run_status(store, "exp-done", returncode=0)  # terminal: succeeded

    live = store.live_run_for("exp")
    assert live is not None
    assert live.run_id == "exp-run"
    assert store.live_run_for("other-experiment") is None


def test_state_cleanup_uncertain_for_zombie_runner_without_status(tmp_path: Path) -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("zombie detection relies on /proc")
    store = RunStore(tmp_path / "state")
    # Kill a child that is paused until signalled. As its unreaping parent, the
    # PID lingers as a zombie that os.kill(pid, 0) still reports as alive -
    # exactly the SIGKILL/OOM case that must report failed, not running.
    proc = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"])
    try:
        starttime = read_proc_starttime(proc.pid)
        os.kill(proc.pid, signal.SIGKILL)
        deadline = time.time() + 5
        while time.time() < deadline and not is_pid_zombie(proc.pid):
            time.sleep(0.02)
        assert is_pid_zombie(proc.pid)
        handle = make_run_handle(run_id="zomb", pid=proc.pid, starttime=starttime)
        assert store.state(handle) == "running"
        assert store.cleanup_uncertain(handle)
        # state() reaped the child, so the zombie is gone, not merely filtered.
        assert not is_pid_zombie(proc.pid)
    finally:
        proc.wait()
