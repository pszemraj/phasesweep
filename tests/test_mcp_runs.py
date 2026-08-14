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

import phasesweep.mcp.runs as mcp_runs
from phasesweep.mcp.runs import RunStore, write_status_file
from phasesweep.runtime.files import private_atomic_write_text
from phasesweep.runtime.process import read_boot_id, read_proc_starttime
from tests.conftest import file_mode, is_pid_zombie
from tests.mcp_helpers import make_run_handle, write_run_status


def _earlier_boot_id() -> str:
    """Return a boot id that cannot be this host's current one.

    :return str: Boot id differing from ``read_boot_id()``.
    """
    current = read_boot_id()
    if current is None:
        pytest.skip("boot id unavailable on this platform")
    other = "0" * len(current)
    return other if other != current else "1" * len(current)


def test_create_get_roundtrip(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(
        run_id="exp-1",
        allow_cancel=True,
        visible_params_at_launch=["lr", "depth"],
    )
    store.create(handle)
    assert store.get("exp-1") == handle
    persisted = json.loads((tmp_path / "state" / "runs" / "exp-1.json").read_text())
    assert persisted["allow_cancel"] is True
    assert persisted["visible_params_at_launch"] == ["lr", "depth"]
    assert store.get("missing") is None


def test_new_run_id_uses_full_uuid_width(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")

    suffix = store.new_run_id("exp").removeprefix("exp-")

    assert len(suffix) == 32
    int(suffix, 16)


def test_launching_handle_reserves_concurrency_until_outcome_is_known(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", launch_state="launching")

    store.create(handle)
    loaded = store.get("exp-1")

    assert loaded == handle
    assert store.state(loaded) == "running"
    assert store.recovery_required(loaded)
    assert store.live_runs() == [loaded]
    assert store.live_run_for("exp") == loaded


def test_create_refuses_existing_identity_without_replacement(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    first = make_run_handle(run_id="exp-1", experiment_id="old")
    second = make_run_handle(run_id="exp-1", experiment_id="new")

    store.create(first)
    with pytest.raises(FileExistsError):
        store.create(second)

    assert store.get("exp-1") == first


def test_create_serializes_before_reserving_the_final_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serialization error cannot publish an empty or partial handle."""
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-serialization")

    def fail_serialization(*_args: object, **_kwargs: object) -> str:
        raise TypeError("injected serialization failure")

    monkeypatch.setattr(mcp_runs.json, "dumps", fail_serialization)

    with pytest.raises(TypeError, match="serialization"):
        store.create(handle)

    assert not store.handle_exists(handle.run_id)
    assert list((tmp_path / "state" / "runs").glob("*.tmp")) == []


@pytest.mark.parametrize("failure_kind", ["file", "directory"])
def test_create_fsync_failure_rolls_back_the_reserved_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """A failed file or directory fsync leaves no final run reservation."""
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id=f"exp-fsync-{failure_kind}")
    real_fsync = os.fsync
    failed = False

    def fail_selected_fsync(fd: int) -> None:
        nonlocal failed
        is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if not failed and is_directory is (failure_kind == "directory"):
            failed = True
            raise OSError(f"injected {failure_kind} fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(mcp_runs.os, "fsync", fail_selected_fsync)

    with pytest.raises(OSError, match=f"{failure_kind} fsync"):
        store.create(handle)

    assert failed
    assert not store.handle_exists(handle.run_id)
    assert list((tmp_path / "state" / "runs").glob("*.tmp")) == []
    store.create(handle)
    assert store.get(handle.run_id) == handle


@pytest.mark.parametrize("failing_create", [1, 2, 3])
def test_prepare_launch_rolls_back_each_create_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_create: int,
) -> None:
    """Lease, snapshot, and handle create failures leave no capacity reservation."""
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id=f"exp-create-{failing_create}", launch_state="launching")
    real_create = mcp_runs._strict_atomic_create_text
    calls = 0

    def fail_one_create(path: Path, text: str) -> None:
        nonlocal calls
        calls += 1
        if calls == failing_create:
            raise OSError(f"injected create {failing_create} failure")
        real_create(path, text)

    monkeypatch.setattr(mcp_runs, "_strict_atomic_create_text", fail_one_create)

    with pytest.raises(OSError, match=f"create {failing_create}"):
        store.prepare_launch(handle, b"experiment: exp\n")

    assert store.get(handle.run_id) is None
    assert not store.config_snapshot_path(handle.run_id).exists()
    assert not store.launch_lease_path(handle.run_id).exists()


def test_failed_pre_spawn_cleanup_retains_recoverable_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient rollback failure retains the proof needed for retry cleanup."""
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-cleanup", launch_state="launching")
    real_unlink = mcp_runs._strict_unlink
    real_create = store.create
    failed = False

    def fail_handle_create(_handle: object) -> None:
        raise OSError("injected handle create failure")

    def fail_snapshot_cleanup(path: Path) -> None:
        nonlocal failed
        if path == store.config_snapshot_path(handle.run_id) and not failed:
            failed = True
            raise OSError("injected pre-spawn cleanup failure")
        real_unlink(path)

    monkeypatch.setattr(store, "create", fail_handle_create)
    monkeypatch.setattr(mcp_runs, "_strict_unlink", fail_snapshot_cleanup)
    with pytest.raises(OSError, match="handle create"):
        store.prepare_launch(handle, b"experiment: exp\n")

    assert failed
    assert store.launch_lease_path(handle.run_id).is_file()
    monkeypatch.setattr(store, "create", real_create)
    monkeypatch.setattr(mcp_runs, "_strict_unlink", real_unlink)
    assert store.is_pre_spawn_orphan(handle.run_id)
    store.clear_pre_spawn_orphan(handle.run_id)
    assert not store.run_evidence_exists(handle.run_id)


def test_launch_lease_distinguishes_live_child_from_abandoned_preparation(
    tmp_path: Path,
) -> None:
    """The inherited lease closes the Popen-before-receipt recovery race."""
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-lease", launch_state="launching")
    preparation = store.prepare_launch(handle, b"experiment: exp\n")

    assert not store.is_pre_spawn_orphan(handle.run_id)
    preparation.close()
    assert store.is_pre_spawn_orphan(handle.run_id)
    store.clear_pre_spawn_orphan(handle.run_id)
    assert not store.run_evidence_exists(handle.run_id)


def test_update_allows_only_spawn_transition_and_idempotent_retry(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    pending = make_run_handle(run_id="exp-1", launch_state="launching")
    spawned = replace(
        pending,
        pid=os.getpid(),
        pgid=os.getpid(),
        pid_starttime=read_proc_starttime(os.getpid()),
        launch_state="spawned",
    )
    store.create(pending)

    store.update(spawned)
    store.update(spawned)

    assert store.get("exp-1") == spawned
    with pytest.raises(ValueError, match="immutable field"):
        store.update(replace(spawned, experiment_id="other"))
    with pytest.raises(ValueError, match="immutable field"):
        store.update(replace(spawned, visible_params_at_launch="all"))
    with pytest.raises(ValueError, match="launching-to-spawned"):
        store.update(replace(spawned, pid=os.getppid(), pgid=os.getppid()))


def test_legacy_handle_missing_launch_authority_loads_fail_closed(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    payload = asdict(
        make_run_handle(
            run_id="exp-legacy",
            allow_cancel=True,
            visible_params_at_launch="all",
        )
    )
    payload.pop("allow_cancel")
    payload.pop("visible_params_at_launch")
    (tmp_path / "state" / "runs" / "exp-legacy.json").write_text(json.dumps(payload))

    loaded = store.get("exp-legacy")

    assert loaded is not None
    assert loaded.allow_cancel is False
    assert loaded.visible_params_at_launch is None


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
        store.create(handle)
        write_status_file(store.status_path("exp-1"), {"run_id": "exp-1", "returncode": 0})
        store.mark_cleanup_uncertain(handle)
    finally:
        os.umask(old_umask)

    assert file_mode(tmp_path / "state") == 0o700
    assert file_mode(tmp_path / "state" / "runs") == 0o700
    assert file_mode(tmp_path / "state" / "logs") == 0o700
    assert file_mode(tmp_path / "state" / "runs" / "exp-1.json") == 0o600
    assert file_mode(tmp_path / "state" / "logs" / "exp-1.status.json") == 0o600
    assert file_mode(tmp_path / "state" / "logs" / "exp-1.cleanup_uncertain.json") == 0o600


def test_open_existing_is_observational_and_requires_run_store_layout(tmp_path: Path) -> None:
    missing = tmp_path / "mistyped-state"

    with pytest.raises(ValueError, match="expected directories are missing"):
        RunStore.open_existing(missing)

    assert not missing.exists()

    state_dir = tmp_path / "state"
    RunStore(state_dir)
    before_modes = {
        path: file_mode(path) for path in (state_dir, state_dir / "runs", state_dir / "logs")
    }

    RunStore.open_existing(state_dir)

    assert {
        path: file_mode(path) for path in (state_dir, state_dir / "runs", state_dir / "logs")
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
    store.create(make_run_handle(run_id="exp-1"))
    assert store.get(unsafe) is None


def test_list_handles_skips_malformed(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    store.create(make_run_handle(run_id="exp-1"))
    store.create(make_run_handle(run_id="exp-2"))
    # A torn/partial handle file must not crash a read.
    (tmp_path / "state" / "runs" / "broken.json").write_text("{not valid json")
    assert {h.run_id for h in store.list_handles()} == {"exp-1", "exp-2"}


def test_launch_inventory_reports_malformed_and_orphaned_run_authority(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    store.create(make_run_handle(run_id="exp-valid"))
    (tmp_path / "state" / "runs" / "broken.json").write_text("{not valid json")
    store.log_path("broken").write_text("runner may still exist\n")
    store.config_snapshot_path("exp-orphan").write_text("experiment: orphan\n")
    store.status_path("exp-orphan").write_text("{}\n")

    handles, unreadable_records = store.launch_inventory()

    assert [handle.run_id for handle in handles] == ["exp-valid"]
    assert unreadable_records == {"run:broken", "run:exp-orphan"}


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
        ("boot_id", ""),
        ("boot_id", 12345),
        ("visible_params_at_launch", "some"),
        ("visible_params_at_launch", [""]),
        ("visible_params_at_launch", ["lr", "lr"]),
        ("visible_params_at_launch", ["lr", 1]),
        ("visible_params_at_launch", {"lr": True}),
    ],
)
def test_loaded_handle_shape_is_validated(tmp_path: Path, field: str, value: object) -> None:
    store = RunStore(tmp_path / "state")
    payload = asdict(make_run_handle(run_id="exp-1"))
    payload[field] = value
    (tmp_path / "state" / "runs" / "exp-1.json").write_text(json.dumps(payload))

    assert store.get("exp-1") is None
    assert store.list_handles() == []


@pytest.mark.parametrize(
    "field,value",
    [("pid", os.getpid()), ("pgid", os.getpid()), ("boot_id", "a-boot-id")],
)
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
        store.create(handle)

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


def test_successful_process_with_failed_result_snapshot_is_still_succeeded(tmp_path: Path) -> None:
    """The engine's exit status is the single authority on success.

    A failed *result snapshot* capture degrades the run's frozen results
    (result reads surface the finalization state explicitly), never the run
    outcome itself — the old ``returncode 0 + snapshot failed -> failed``
    rule let an optional diagnostic contradict an engine-defined,
    already-published success (review v0.5.16 / blocker 2).
    """
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1")
    write_run_status(
        store,
        handle.run_id,
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot_state="failed",
        result_snapshot_error="InterruptedFinalization",
    )

    assert store.state(handle) == "succeeded"


def test_pending_result_snapshot_keeps_run_live_until_finalized(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(
        run_id="exp-1",
        experiment_id="exp",
        pid=999999,
        starttime=111,
    )
    store.create(handle)
    write_run_status(
        store,
        "exp-1",
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot_state="pending",
    )

    assert store.state(handle) == "running"
    assert store.snapshot_recovery_required(handle)
    assert store.recovery_required(handle)
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
    assert not store.snapshot_recovery_required(handle)
    assert not store.recovery_required(handle)
    assert store.live_runs() == []
    assert store.live_run_for("exp") is None


def test_live_runner_pending_snapshot_does_not_require_recovery(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-live", experiment_id="exp")
    store.create(handle)
    write_run_status(
        store,
        handle.run_id,
        returncode=0,
        error_class=None,
        cleanup_confirmed=True,
        result_snapshot_state="pending",
    )

    assert store.state(handle) == "running"
    assert not store.snapshot_recovery_required(handle)
    assert not store.recovery_required(handle)


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
        {
            "run_id": "exp-1",
            "returncode": 0,
            "result_publication_state": "prepared",
        },
        {
            "run_id": "exp-1",
            "returncode": 0,
            "result_snapshot_state": "pending",
            "result_snapshot": {},
            "result_publication_state": "unknown",
            "result_publication_generation_id": "exp-1",
        },
        {
            "run_id": "exp-1",
            "returncode": 0,
            "result_snapshot_state": "complete",
            "result_snapshot": {},
            "result_publication_state": "prepared",
            "result_publication_generation_id": "exp-1",
        },
        {"run_id": "exp-1", "returncode": 0, "result_snapshot_error": 3},
        {"run_id": "exp-1", "returncode": 0, "recovered_attempt_ids": "attempt-1"},
        {"run_id": "exp-1", "returncode": 0, "recovered_attempt_ids": [""]},
        {"run_id": "exp-1", "returncode": 0, "recovered_attempt_generations": []},
        {
            "run_id": "exp-1",
            "returncode": 0,
            "recovered_attempt_generations": {"attempt-1": ""},
        },
        {"run_id": "exp-1", "returncode": 0, "uncertain_attempt_ids": "attempt-1"},
        {"run_id": "exp-1", "returncode": 0, "uncertain_attempt_ids": [""]},
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
    store.create(handle)

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
    store.create(handle)
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
    store.create(handle)

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
    store.create(handle)
    private_atomic_write_text(store.cleanup_uncertain_path("exp-1"), json.dumps(payload))

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
    store.create(pending)

    store.mark_cleanup_uncertain(spawned)
    store.mark_cleanup_uncertain(pending)

    marker = json.loads(store.cleanup_uncertain_path("exp-1").read_text())
    assert marker["pid"] == 4242
    assert marker["pgid"] == 4242
    assert marker["pid_starttime"] == 111
    identity = store.cleanup_identity(pending)
    assert (identity.pid, identity.pgid, identity.pid_starttime) == (4242, 4242, 111)


def test_boot_id_roundtrips_through_handle_and_cleanup_marker(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    boot_id = read_boot_id() or "test-boot-id"
    handle = replace(make_run_handle(run_id="exp-1", pid=4242, starttime=111), boot_id=boot_id)

    store.create(handle)
    store.mark_cleanup_uncertain(handle)

    loaded = store.get("exp-1")
    assert loaded == handle
    assert loaded.boot_id == boot_id
    assert json.loads(store.cleanup_uncertain_path("exp-1").read_text())["boot_id"] == boot_id
    assert store.cleanup_identity(loaded).boot_id == boot_id


@pytest.mark.parametrize("boot_id", ["", 12345, True])
def test_cleanup_marker_with_invalid_boot_id_is_rejected(tmp_path: Path, boot_id: object) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", config_sha256="a" * 64, pid=999999, starttime=111)
    store.create(handle)
    private_atomic_write_text(
        store.cleanup_uncertain_path("exp-1"),
        json.dumps(
            {
                "run_id": "exp-1",
                "config_sha256": "a" * 64,
                "cleanup_confirmed": False,
                "boot_id": boot_id,
            }
        ),
    )

    assert not store.cleanup_uncertain(handle)


def test_earlier_boot_identity_is_dead_without_cleanup_recovery(tmp_path: Path) -> None:
    # PID and starttime are this live process's, so only the boot id can prove
    # the runner is gone. It does: nothing survives a reboot, so the run derives
    # terminal instead of parking in cleanup-uncertain recovery, and no signal
    # is ever aimed at whatever inherited those numbers after the reboot.
    store = RunStore(tmp_path / "state")
    starttime = read_proc_starttime(os.getpid())
    if starttime is None:
        pytest.skip("/proc starttime unavailable (non-Linux)")
    handle = replace(
        make_run_handle(run_id="exp-1", pid=os.getpid(), starttime=starttime),
        boot_id=_earlier_boot_id(),
    )
    store.create(handle)

    assert store.state(handle) == "failed"
    assert not store.cleanup_uncertain(handle)
    assert not store.cleanup_recovery_required(handle)
    assert not store.recovery_required(handle)
    assert store.live_runs() == []


def test_handle_without_boot_id_keeps_conservative_cleanup_uncertainty(tmp_path: Path) -> None:
    # Same identity as the boot-mismatch case minus the boot id: an older
    # persisted handle cannot rule out PID reuse, so it must still fail closed.
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", pid=999999, starttime=111)
    assert handle.boot_id is None
    store.create(handle)

    assert store.state(handle) == "running"
    assert store.cleanup_uncertain(handle)
    assert store.recovery_required(handle)


def test_earlier_boot_clears_a_persisted_cleanup_uncertainty_marker(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = replace(
        make_run_handle(run_id="exp-1", pid=999999, starttime=111),
        boot_id=_earlier_boot_id(),
    )
    store.create(handle)
    store.mark_cleanup_uncertain(handle)

    assert store.cleanup_uncertain(handle)
    assert store.state(handle) == "failed"
    assert not store.cleanup_uncertain_path("exp-1").exists()


def test_earlier_boot_resolves_terminal_cleanup_uncertain_status(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = replace(
        make_run_handle(run_id="exp-1", experiment_id="exp", pid=999999, starttime=111),
        boot_id=_earlier_boot_id(),
    )
    store.create(handle)
    write_run_status(
        store,
        "exp-1",
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
    )

    assert store.state(handle) == "failed"
    assert not store.cleanup_recovery_required(handle)
    assert store.live_run_for("exp") is None


def test_earlier_boot_orphans_a_pending_terminal_snapshot(tmp_path: Path) -> None:
    # Liveness, not just cleanup: a live PID+starttime match must not be read
    # as a live runner once the recorded boot differs from this one.
    store = RunStore(tmp_path / "state")
    starttime = read_proc_starttime(os.getpid())
    if starttime is None:
        pytest.skip("/proc starttime unavailable (non-Linux)")
    handle = replace(
        make_run_handle(run_id="exp-1", pid=os.getpid(), starttime=starttime),
        boot_id=_earlier_boot_id(),
    )
    store.create(handle)
    write_run_status(store, "exp-1", returncode=0, result_snapshot_state="pending")

    assert store.snapshot_recovery_required(handle)
    assert not store.snapshot_recovery_required(replace(handle, boot_id=None))


def test_confirmed_terminal_status_overrides_stale_cleanup_marker(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", pid=999999, starttime=111)
    store.create(handle)
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
    store.create(handle)
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
    store.create(handle)
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


def test_cleanup_recovered_attempt_evidence_uses_one_authorized_snapshot(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", config_sha256="a" * 64)
    write_run_status(
        store,
        handle.run_id,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
        recovered_attempt_ids=["runner-match", "runner-other"],
        recovered_attempt_generations={
            "runner-match": handle.run_id,
            "runner-other": "other-run",
        },
    )
    store.cleanup_recovery_path(handle.run_id).write_text(
        json.dumps(
            {
                "run_id": handle.run_id,
                "config_sha256": handle.config_sha256,
                "cleanup_confirmed": True,
                "reaped_attempt_ids": ["operator-match"],
                "reaped_attempt_locations": {
                    "operator-match": {
                        "phase": "p",
                        "trial_number": 3,
                        "generation_id": handle.run_id,
                    },
                    "not-authorized": {
                        "phase": "p",
                        "trial_number": 4,
                        "generation_id": handle.run_id,
                    },
                },
            }
        )
    )

    attempt_ids, locations = store.cleanup_recovered_attempt_evidence(handle)

    assert attempt_ids == {"runner-match", "operator-match"}
    assert locations == {"operator-match": ("p", 3, handle.run_id)}

    # Optional runner evidence may be persisted as null by an older or partial
    # writer; the operator record remains independently usable.
    write_run_status(
        store,
        handle.run_id,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
        recovered_attempt_ids=None,
        recovered_attempt_generations={"runner-match": handle.run_id},
    )
    attempt_ids, locations = store.cleanup_recovered_attempt_evidence(handle)
    assert attempt_ids == {"operator-match"}
    assert locations == {"operator-match": ("p", 3, handle.run_id)}


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
    store.create(make_run_handle(run_id="exp-run", experiment_id="exp"))
    store.create(make_run_handle(run_id="exp-done", experiment_id="exp"))
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
