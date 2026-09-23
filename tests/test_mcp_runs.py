"""Run-handle store: persistence round-trip and derived run-state logic."""

from __future__ import annotations

import contextlib
import json
import os
import select
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event, Thread

import pytest

import phasesweep.mcp.runs as mcp_runs
from phasesweep.errors import OperatorAction
from phasesweep.mcp.recovery import RunRecoveryError, recover_run
from phasesweep.mcp.runs import RunStore, write_status_file
from phasesweep.runtime.files import UnsafePrivatePathError, private_atomic_write_text
from phasesweep.runtime.reaper import read_boot_id, read_proc_starttime
from tests.conftest import file_mode, is_pid_zombie, reaped_pid, requires_nonroot
from tests.mcp_helpers import make_run_handle, write_run_status


def _earlier_boot_id() -> str:
    """Return a boot id that cannot be this host's current one.

    :return str: Boot id differing from ``read_boot_id()``.
    """
    current = read_boot_id()
    if current is None:
        pytest.skip("boot id unavailable on this platform")
    other = "00000000-0000-0000-0000-000000000000"
    return other if other != current else "11111111-1111-1111-1111-111111111111"


def _state_tree_snapshot(state_dir: Path) -> dict[Path, tuple[int, bytes | None]]:
    """Capture durable state entries so refusal tests can prove no mutation.

    :param Path state_dir: MCP state root to snapshot.
    :return dict[Path, tuple[int, bytes | None]]: Relative path, mode, and file bytes.
    """
    snapshot: dict[Path, tuple[int, bytes | None]] = {}
    for path in (state_dir, *sorted(state_dir.rglob("*"))):
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
        snapshot[path.relative_to(state_dir)] = (
            mode,
            path.read_bytes() if stat.S_ISREG(info.st_mode) else None,
        )
    return snapshot


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

    assert store.get(handle.run_id) is None
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
    assert store.get(handle.run_id) is None
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
    assert store.launch_inventory() == ([], set())


@pytest.mark.integration
def test_launch_lease_distinguishes_live_child_from_abandoned_preparation(
    tmp_path: Path,
) -> None:
    """The inherited lease closes the Popen-before-receipt recovery race."""
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-lease", launch_state="launching")
    preparation = store.prepare_launch(handle, b"experiment: exp\n")
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    proc: subprocess.Popen | None = None
    child = (
        "import os, sys\n"
        "lease_fd, ready_fd, release_fd = map(int, sys.argv[1:])\n"
        "os.fstat(lease_fd)\n"
        "os.write(ready_fd, b'R')\n"
        "raise SystemExit(0 if os.read(release_fd, 1) == b'X' else 2)\n"
    )
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                child,
                str(preparation.lease_fd),
                str(ready_write),
                str(release_read),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(preparation.lease_fd, ready_write, release_read),
        )
        os.close(ready_write)
        ready_write = -1
        os.close(release_read)
        release_read = -1
        readable, _, _ = select.select([ready_read], [], [], 5)
        assert readable and os.read(ready_read, 1) == b"R"

        assert not store.is_pre_spawn_orphan(handle.run_id)
        preparation.close()
        assert not store.is_pre_spawn_orphan(handle.run_id)

        assert os.write(release_write, b"X") == 1
        os.close(release_write)
        release_write = -1
        assert proc.wait(timeout=5) == 0
    finally:
        with contextlib.suppress(OSError):
            preparation.close()
        for fd in (ready_read, ready_write, release_read, release_write):
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    assert store.is_pre_spawn_orphan(handle.run_id)
    store.clear_pre_spawn_orphan(handle.run_id)
    assert store.launch_inventory() == ([], set())


def test_free_launch_lease_cannot_override_missing_handle_with_runner_log(
    tmp_path: Path,
) -> None:
    """A stale lease cannot erase the remaining evidence of a spawned runner."""
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-stale-lease", launch_state="launching")
    preparation = store.prepare_launch(handle, b"experiment: exp\n")
    preparation.close()
    handle_path = tmp_path / "state" / "runs" / f"{handle.run_id}.json"
    mcp_runs._strict_unlink(handle_path)
    store.log_path(handle.run_id).write_text("runner may still be active\n")

    assert not store.is_pre_spawn_orphan(handle.run_id)
    with pytest.raises(ValueError, match="not a provably abandoned preparation"):
        store.clear_pre_spawn_orphan(handle.run_id)

    assert store.launch_lease_path(handle.run_id).is_file()
    assert store.config_snapshot_path(handle.run_id).is_file()
    assert store.log_path(handle.run_id).is_file()


@pytest.mark.parametrize(
    "lease_kind",
    ["directory", "fifo", "live_symlink", "dangling_symlink"],
)
def test_malformed_launch_lease_cannot_enable_legacy_orphan_recovery(
    tmp_path: Path,
    lease_kind: str,
) -> None:
    """A malformed lease entry cannot authorize deletion of a config snapshot."""
    store = RunStore(tmp_path / "state")
    run_id = f"exp-{lease_kind.replace('_', '-')}"
    snapshot = store.config_snapshot_path(run_id)
    snapshot.write_text("experiment: exp\n")
    lease = store.launch_lease_path(run_id)
    if lease_kind == "directory":
        lease.mkdir()
    elif lease_kind == "fifo":
        os.mkfifo(lease)
    elif lease_kind == "live_symlink":
        target = lease.with_name("lease-target")
        target.write_text("")
        lease.symlink_to(target.name)
    else:
        lease.symlink_to("missing-lease-target")

    assert not store.is_pre_spawn_orphan(run_id)
    with pytest.raises(ValueError, match="not a provably abandoned preparation"):
        store.clear_pre_spawn_orphan(run_id)

    assert snapshot.is_file()
    assert lease.exists() or lease.is_symlink()


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
    assert file_mode(tmp_path / "state" / ".phasesweep-format.json") == 0o600
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


def test_open_existing_blames_privacy_only_on_a_complete_layout(tmp_path: Path) -> None:
    """A shared directory without the layout is a wrong path, not damaged state."""
    project = tmp_path / "project"
    project.mkdir()
    project.chmod(0o755)

    with pytest.raises(ValueError, match="expected directories are missing") as missing:
        RunStore.open_existing(project)
    assert not isinstance(missing.value, UnsafePrivatePathError)
    assert file_mode(project) == 0o755

    state_dir = tmp_path / "state"
    RunStore(state_dir)
    (state_dir / "runs").chmod(0o755)

    with pytest.raises(UnsafePrivatePathError, match="must be owned by uid"):
        RunStore.open_existing(state_dir)
    assert file_mode(state_dir / "runs") == 0o755


def test_run_store_marks_fresh_scaffolded_state(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    state_dir.chmod(0o700)
    private_atomic_write_text(state_dir / "origin", "catalog.yaml\n")

    RunStore(state_dir)

    marker = state_dir / ".phasesweep-format.json"
    assert json.loads(marker.read_text()) == {"schema_version": mcp_runs.MCP_STATE_FORMAT_VERSION}
    assert file_mode(marker) == 0o600
    assert (state_dir / "origin").read_text() == "catalog.yaml\n"
    assert (state_dir / "runs").is_dir()
    assert (state_dir / "logs").is_dir()


@pytest.mark.parametrize(
    ("marker_text", "match"),
    [
        ("not json\n", "malformed or unsupported"),
        ('{"schema_version": 0}\n', "malformed or unsupported"),
        ('{"schema_version": 1, "unexpected": true}\n', "malformed or unsupported"),
    ],
)
def test_run_store_rejects_invalid_format_marker_without_mutation(
    tmp_path: Path,
    marker_text: str,
    match: str,
) -> None:
    state_dir = tmp_path / "state"
    RunStore(state_dir)
    marker = state_dir / ".phasesweep-format.json"
    private_atomic_write_text(marker, marker_text)
    before = _state_tree_snapshot(state_dir)

    with pytest.raises(ValueError, match=match):
        RunStore(state_dir)

    assert _state_tree_snapshot(state_dir) == before


@pytest.mark.parametrize("evidence_kind", ["handle", "status", "log", "lease", "audit"])
def test_run_store_refuses_unmarked_durable_state_before_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_kind: str,
) -> None:
    state_dir = tmp_path / "state"
    store = RunStore(state_dir)
    run_id = "exp-1"
    if evidence_kind == "handle":
        store.create(make_run_handle(run_id=run_id))
    elif evidence_kind == "status":
        write_status_file(store.status_path(run_id), {"run_id": run_id, "returncode": 0})
    elif evidence_kind == "log":
        private_atomic_write_text(store.log_path(run_id), "runner output\n")
    elif evidence_kind == "lease":
        private_atomic_write_text(store.launch_lease_path(run_id), "")
    else:
        private_atomic_write_text(state_dir / "audit.jsonl", '{"tool":"launch_run"}\n')
    marker = state_dir / ".phasesweep-format.json"
    marker.unlink()
    before = _state_tree_snapshot(state_dir)
    initialized: list[Path] = []

    def fail_if_initialized(path: Path) -> None:
        initialized.append(path)
        raise AssertionError("unmarked durable state must not be initialized")

    monkeypatch.setattr(mcp_runs, "ensure_private_dir", fail_if_initialized)

    with pytest.raises(ValueError, match="no format marker.*fresh MCP state directory.*0.3.1"):
        RunStore(state_dir)

    assert initialized == []
    assert _state_tree_snapshot(state_dir) == before


def test_open_existing_requires_supported_format_marker_without_mutation(tmp_path: Path) -> None:
    """Without a marker, run handles mark preserved-release state; no handles, a wrong path."""
    state_dir = tmp_path / "state"
    store = RunStore(state_dir)
    marker = state_dir / ".phasesweep-format.json"
    marker.unlink()
    before = _state_tree_snapshot(state_dir)

    with pytest.raises(ValueError, match="no format marker and no run handles") as wrong:
        RunStore.open_existing(state_dir)
    assert not isinstance(wrong.value, mcp_runs.UnsupportedStateFormatError)
    assert _state_tree_snapshot(state_dir) == before

    private_atomic_write_text(
        marker, json.dumps({"schema_version": mcp_runs.MCP_STATE_FORMAT_VERSION})
    )
    store.create(make_run_handle(run_id="exp-1"))
    marker.unlink()
    before = _state_tree_snapshot(state_dir)

    with pytest.raises(
        mcp_runs.UnsupportedStateFormatError,
        match="no format marker.*fresh MCP state directory.*0.3.1",
    ):
        RunStore.open_existing(state_dir)
    assert _state_tree_snapshot(state_dir) == before


def _logs_deleted(tmp_path: Path) -> Path:
    state_dir = tmp_path / "state"
    RunStore(state_dir)
    (state_dir / "logs").rmdir()
    return state_dir


def _shared_project_dir(tmp_path: Path, mode: int) -> Path:
    project = tmp_path / "project"
    for path in (project, project / "runs", project / "logs"):
        path.mkdir(exist_ok=True)
        path.chmod(mode)
    return project


def _marker_mode(mode: int) -> Callable[[Path], Path]:
    def build(tmp_path: Path) -> Path:
        state_dir = tmp_path / "state"
        RunStore(state_dir)
        (state_dir / ".phasesweep-format.json").chmod(mode)
        return state_dir

    return build


def _marker_text(text: str) -> Callable[[Path], Path]:
    def build(tmp_path: Path) -> Path:
        state_dir = tmp_path / "state"
        RunStore(state_dir)
        private_atomic_write_text(state_dir / ".phasesweep-format.json", text)
        return state_dir

    return build


def _regular_file(tmp_path: Path) -> Path:
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text("x: 1\n")
    return catalog


def _below_regular_file(tmp_path: Path) -> Path:
    return _regular_file(tmp_path) / "state"


def _symlink_to_state_dir(tmp_path: Path) -> Path:
    RunStore(tmp_path / "real")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    return tmp_path / "link"


def _under_unsearchable_dir(tmp_path: Path) -> Path:
    locked = tmp_path / "someone-elses-home"
    locked.mkdir()
    locked.chmod(0o000)
    return locked / "state"


@pytest.mark.parametrize(
    ("build", "action", "message"),
    [
        # The marker is there, so the directory is the state directory and the
        # missing, shared, or unreadable part of it is damage to restore.
        (_logs_deleted, OperatorAction.RESTORE_TREE, "has its format marker but is missing"),
        pytest.param(
            _marker_mode(0o000),
            OperatorAction.RESTORE_TREE,
            "cannot be read",
            marks=requires_nonroot,
        ),
        (_marker_mode(0o644), OperatorAction.RESTORE_TREE, "with mode 0600; found"),
        (_marker_text("not json\n"), OperatorAction.RESTORE_TREE, "is malformed"),
        (
            _marker_text('{"schema_version": 1, "unexpected": true}\n'),
            OperatorAction.RESTORE_TREE,
            "is malformed",
        ),
        # The CLI resolves a symlinked state_dir first; through the API the
        # marker is reached, so the link standing in for the directory is damage.
        (_symlink_to_state_dir, OperatorAction.RESTORE_TREE, "is not a real directory"),
        # A readable marker naming another format is another release's state.
        (
            _marker_text('{"schema_version": 0}\n'),
            OperatorAction.USE_PRIOR_RELEASE,
            "declares unsupported format 0",
        ),
        # No marker and no run handles: the path names some other directory,
        # whatever its layout or permissions, and no chmod turns it into state.
        (
            lambda tmp_path: _shared_project_dir(tmp_path, 0o755),
            OperatorAction.FIX_CONFIG,
            "no format marker and no run handles",
        ),
        (
            lambda tmp_path: _shared_project_dir(tmp_path, 0o700),
            OperatorAction.FIX_CONFIG,
            "no format marker and no run handles",
        ),
        (_regular_file, OperatorAction.FIX_CONFIG, "expected directories are missing"),
        # A walk that fails above state_dir never reached a state directory.
        (_below_regular_file, OperatorAction.FIX_CONFIG, "is not reachable through real"),
        pytest.param(
            _under_unsearchable_dir,
            OperatorAction.FIX_CONFIG,
            "is not reachable through real",
            marks=requires_nonroot,
        ),
    ],
    ids=[
        "logs-deleted",
        "marker-mode-000",
        "marker-mode-0644",
        "marker-not-json",
        "marker-extra-field",
        "symlinked-state-dir",
        "marker-other-format",
        "shared-project-dir",
        "private-project-dir",
        "regular-file",
        "below-regular-file",
        "under-unsearchable-dir",
    ],
)
def test_recovery_tells_a_wrong_state_dir_from_a_damaged_one(
    tmp_path: Path, build: Callable[[Path], Path], action: OperatorAction, message: str
) -> None:
    """recover-run routes a wrong path to fixing it and a damaged state dir to restoring it."""
    state_dir = build(tmp_path)
    locked = tmp_path / "someone-elses-home"
    try:
        with pytest.raises(RunRecoveryError) as excinfo:
            recover_run(state_dir, "exp-1", confirm=False, emit=lambda _message: None)
    finally:
        if locked.exists():
            locked.chmod(0o700)
    assert excinfo.value.actions == (action,)
    assert message in str(excinfo.value)


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
    private_atomic_write_text(tmp_path / "state" / "runs" / "broken.json", "{not valid json")
    assert {h.run_id for h in store.list_handles()} == {"exp-1", "exp-2"}


def test_launch_inventory_reports_malformed_and_orphaned_run_authority(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    store.create(make_run_handle(run_id="exp-valid"))
    private_atomic_write_text(tmp_path / "state" / "runs" / "broken.json", "{not valid json")
    store.log_path("broken").write_text("runner may still exist\n")
    store.config_snapshot_path("exp-orphan").write_text("experiment: orphan\n")
    store.status_path("exp-orphan").write_text("{}\n")
    store.status_path("exp-dangling").symlink_to("missing-status.json")
    store.log_path("exp-directory").mkdir()
    (tmp_path / "state" / "logs" / "exp-transition.transition.lock").write_text("")

    handles, unreadable_records = store.launch_inventory()

    assert [handle.run_id for handle in handles] == ["exp-valid"]
    assert unreadable_records == {
        "run:broken",
        "run:exp-dangling",
        "run:exp-directory",
        "run:exp-orphan",
        "run:exp-transition",
    }


def test_get_skips_malformed_handle(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    private_atomic_write_text(tmp_path / "state" / "runs" / "broken.json", "{not valid json")
    assert store.get("broken") is None


def test_dangling_handle_still_reserves_launch_authority(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle_path = tmp_path / "state" / "runs" / "dangling.json"
    handle_path.symlink_to("missing-handle.json")

    assert store.get("dangling") is None
    assert store.launch_inventory() == ([], {"run:dangling"})


@pytest.mark.parametrize(
    "record_kind",
    ["handle", "status", "cleanup_uncertain", "cleanup_recovery"],
)
def test_run_state_json_live_symlinks_are_rejected(tmp_path: Path, record_kind: str) -> None:
    """A symlink cannot make external JSON authoritative run state."""
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", config_sha256="a" * 64)
    store.create(handle)
    target = tmp_path / f"external-{record_kind}.json"

    if record_kind == "handle":
        path = store._runs_dir / f"{handle.run_id}.json"
        payload = asdict(handle)
        path.unlink()
    elif record_kind == "status":
        path = store.status_path(handle.run_id)
        payload = {
            "run_id": handle.run_id,
            "returncode": 0,
            "cleanup_confirmed": True,
        }
    elif record_kind == "cleanup_uncertain":
        path = store.cleanup_uncertain_path(handle.run_id)
        payload = {
            "run_id": handle.run_id,
            "config_sha256": handle.config_sha256,
            "pid": handle.pid,
            "pgid": handle.pgid,
            "pid_starttime": handle.pid_starttime,
            "boot_id": handle.boot_id,
            "cleanup_confirmed": False,
        }
    else:
        path = store.cleanup_recovery_path(handle.run_id)
        payload = {
            "run_id": handle.run_id,
            "config_sha256": handle.config_sha256,
            "cleanup_confirmed": True,
        }
    target.write_text(json.dumps(payload))
    target.chmod(0o600)
    path.symlink_to(target)

    if record_kind == "handle":
        assert store.get(handle.run_id) is None
        assert store.launch_inventory() == ([], {f"run:{handle.run_id}"})
    elif record_kind == "status":
        assert store.recorded_terminal_status(handle) is None
        assert store.state(handle) == "running"
    elif record_kind == "cleanup_uncertain":
        assert not store.cleanup_uncertain(handle)
    else:
        assert not store._cleanup_recovered(handle)


def test_loaded_handle_must_match_filename(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    payload = asdict(make_run_handle(run_id="other"))
    private_atomic_write_text(
        tmp_path / "state" / "runs" / "exp-1.json",
        json.dumps(payload),
    )

    assert store.get("exp-1") is None
    assert store.list_handles() == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("experiment_id", "../bad"),
        ("experiment_id", 7),
        ("config_sha256", []),
        ("config_sha256", "a" * 63),
        ("config_sha256", "A" * 64),
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
        ("boot_id", "malformed"),
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
    private_atomic_write_text(
        tmp_path / "state" / "runs" / "exp-1.json",
        json.dumps(payload),
    )

    assert store.get("exp-1") is None
    assert store.list_handles() == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", os.getpid()),
        ("pgid", os.getpid()),
        ("boot_id", "11111111-1111-1111-1111-111111111111"),
    ],
)
def test_launching_handle_cannot_have_process_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    store = RunStore(tmp_path / "state")
    payload = asdict(make_run_handle(run_id="exp-1", launch_state="launching"))
    payload[field] = value
    private_atomic_write_text(
        tmp_path / "state" / "runs" / "exp-1.json",
        json.dumps(payload),
    )

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
        pid=reaped_pid(),
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
        {"run_id": "exp-1", "returncode": 0},
        {"run_id": "exp-1", "returncode": 0, "cleanup_confirmed": None},
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
    private_atomic_write_text(store.status_path("exp-1"), json.dumps(payload))

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
    store.status_path("exp-1").chmod(0o600)
    assert store.recorded_terminal_status(handle) is None
    assert store.state(handle) == "running"
    assert store.live_runs() == [handle]

    private_atomic_write_text(store.status_path("exp-1"), '{"returncode": 0}')
    real_read_text = mcp_runs.read_private_text_at

    def fail_status_read(parent_fd: int, leaf: str, path: Path) -> str:
        if path == store.status_path("exp-1"):
            raise OSError("status file temporarily unreadable")
        return real_read_text(parent_fd, leaf, path)

    monkeypatch.setattr(mcp_runs, "read_private_text_at", fail_status_read)

    assert store.recorded_terminal_status(handle) is None
    assert store.state(handle) == "running"
    assert store.live_runs() == [handle]


def test_state_failed_from_ordinary_cleanup_confirmed_failure(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", pid=reaped_pid(), starttime=111)
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


def test_state_running_for_live_pid_without_status(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1")
    assert store.state(handle) == "running"


def test_dead_runner_without_status_stays_live_until_recovery_evidence(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", pid=reaped_pid(), starttime=111)
    store.create(handle)

    assert store.state(handle) == "running"
    assert store.cleanup_uncertain(handle)
    assert store.live_runs() == [handle]

    private_atomic_write_text(
        store.cleanup_recovery_path("exp-1"),
        json.dumps(
            {
                "run_id": "exp-1",
                "config_sha256": handle.config_sha256,
                "cleanup_confirmed": True,
            }
        ),
    )
    store.clear_cleanup_uncertain(handle)

    assert store.state(handle) == "failed"


def test_clear_cleanup_uncertain_uses_durable_idempotent_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", pid=reaped_pid(), starttime=111)
    store.mark_cleanup_uncertain(handle)
    marker = store.cleanup_uncertain_path(handle.run_id)
    real_unlink = mcp_runs._strict_unlink
    unlinked: list[Path] = []

    def track_unlink(path: Path) -> None:
        unlinked.append(path)
        real_unlink(path)

    monkeypatch.setattr(mcp_runs, "_strict_unlink", track_unlink)

    store.clear_cleanup_uncertain(handle)
    store.clear_cleanup_uncertain(handle)

    assert unlinked == [marker, marker]
    assert not marker.exists()


def test_state_does_not_restore_cleanup_marker_after_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", pid=reaped_pid(), starttime=111)
    store.create(handle)
    original_recovered = store._cleanup_recovered
    first_read = True

    def recovery_between_check_and_marker(saved: object) -> bool:
        nonlocal first_read
        if first_read:
            first_read = False
            with store.transition_lock(handle):
                write_run_status(
                    store,
                    handle.run_id,
                    returncode=1,
                    error_class="UnsafeProcessCleanupError",
                    cleanup_confirmed=False,
                )
                private_atomic_write_text(
                    store.cleanup_recovery_path(handle.run_id),
                    json.dumps(
                        {
                            "run_id": handle.run_id,
                            "config_sha256": handle.config_sha256,
                            "cleanup_confirmed": True,
                        }
                    ),
                )
                store.clear_cleanup_uncertain(handle)
            return False  # The state read already saw the old, unrecovered evidence.
        return original_recovered(saved)

    monkeypatch.setattr(store, "_cleanup_recovered", recovery_between_check_and_marker)

    assert store.state(handle) == "failed"
    assert not store.cleanup_uncertain(handle)
    assert not store.cleanup_recovery_required(handle)


def test_dead_runner_state_does_not_wait_for_confirmed_recovery(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", pid=reaped_pid(), starttime=111)
    store.create(handle)
    started = Event()
    finished = Event()
    observed: list[str] = []

    def read_state() -> None:
        started.set()
        observed.append(store.state(handle))
        finished.set()

    with store.transition_lock(handle):
        reader = Thread(target=read_state)
        reader.start()
        assert started.wait(1)
        completed_while_locked = finished.wait(0.5)
    reader.join(timeout=2)

    assert completed_while_locked
    assert observed == ["running"]
    assert not store.cleanup_uncertain(handle)


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
        {"run_id": "exp-1", "cleanup_confirmed": False, "pgid": 4242},
        {
            "run_id": "exp-1",
            "cleanup_confirmed": False,
            "pid_starttime": 123,
        },
        {
            "run_id": "exp-1",
            "cleanup_confirmed": False,
            "pid": 4242,
            "pid_starttime": 123,
        },
    ],
)
def test_cleanup_uncertain_marker_shape_is_validated(tmp_path: Path, payload: object) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(
        run_id="exp-1",
        config_sha256="a" * 64,
        pid=reaped_pid(),
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
    runner_pid = reaped_pid()
    spawned = make_run_handle(
        run_id="exp-1",
        config_sha256="a" * 64,
        pid=runner_pid,
        starttime=111,
    )
    store.create(pending)

    store.mark_cleanup_uncertain(spawned)
    store.mark_cleanup_uncertain(pending)

    marker = json.loads(store.cleanup_uncertain_path("exp-1").read_text())
    assert marker["pid"] == runner_pid
    assert marker["pgid"] == runner_pid
    assert marker["pid_starttime"] == 111
    identity = store.cleanup_identity(pending)
    assert (identity.pid, identity.pgid, identity.pid_starttime) == (runner_pid, runner_pid, 111)


def test_boot_id_roundtrips_through_handle_and_cleanup_marker(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    boot_id = read_boot_id()
    if boot_id is None:
        pytest.skip("boot id unavailable on this platform")
    handle = replace(
        make_run_handle(run_id="exp-1", pid=reaped_pid(), starttime=111), boot_id=boot_id
    )

    store.create(handle)
    store.mark_cleanup_uncertain(handle)

    loaded = store.get("exp-1")
    assert loaded == handle
    assert loaded.boot_id == boot_id
    assert json.loads(store.cleanup_uncertain_path("exp-1").read_text())["boot_id"] == boot_id
    assert store.cleanup_identity(loaded).boot_id == boot_id


def test_cleanup_marker_cannot_override_live_spawned_identity(tmp_path: Path) -> None:
    """A conflicting marker cannot declare a live spawned runner to be from an old boot."""
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-1", config_sha256="a" * 64)
    store.create(handle)
    private_atomic_write_text(
        store.cleanup_uncertain_path(handle.run_id),
        json.dumps(
            {
                "run_id": handle.run_id,
                "config_sha256": handle.config_sha256,
                "pid": handle.pid,
                "pgid": handle.pgid,
                "pid_starttime": handle.pid_starttime,
                "boot_id": _earlier_boot_id(),
                "cleanup_confirmed": False,
            }
        ),
    )

    assert store._runner_is_live(handle)
    assert not store.cleanup_uncertain(handle)
    assert store.cleanup_identity(handle).boot_id == handle.boot_id
    assert store.state(handle) == "running"
    assert not store.recovery_required(handle)


@pytest.mark.parametrize("boot_id", ["", "malformed", 12345, True])
def test_cleanup_marker_with_invalid_boot_id_is_rejected(tmp_path: Path, boot_id: object) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(
        run_id="exp-1", config_sha256="a" * 64, pid=reaped_pid(), starttime=111
    )
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
    handle = replace(make_run_handle(run_id="exp-1", pid=reaped_pid(), starttime=111), boot_id=None)
    assert handle.boot_id is None
    store.create(handle)

    assert store.state(handle) == "running"
    assert store.cleanup_uncertain(handle)
    assert store.recovery_required(handle)


def test_terminal_status_written_during_liveness_check_does_not_require_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(run_id="exp-racing-status", pid=reaped_pid(), starttime=111)
    store.create(handle)

    def runner_finishes(_pid: int | None, _starttime: int | None) -> bool:
        write_run_status(
            store,
            handle.run_id,
            returncode=0,
            cleanup_confirmed=True,
            result_snapshot_state="failed",
        )
        return False

    monkeypatch.setattr(mcp_runs, "is_same_live_process", runner_finishes)
    assert store.state(handle) == "succeeded"
    assert store.recovery_required(handle) is False
    assert not store.cleanup_uncertain_path(handle.run_id).exists()


def test_earlier_boot_clears_a_persisted_cleanup_uncertainty_marker(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = replace(
        make_run_handle(run_id="exp-1", pid=reaped_pid(), starttime=111),
        boot_id=_earlier_boot_id(),
    )
    store.create(handle)
    store.mark_cleanup_uncertain(handle)

    assert store.cleanup_uncertain(handle)
    assert store.state(handle) == "failed"
    assert not store.cleanup_uncertain_path("exp-1").exists()


def test_earlier_boot_settles_liveness_but_requires_trial_reconciliation(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "state")
    handle = replace(
        make_run_handle(run_id="exp-1", experiment_id="exp", pid=reaped_pid(), starttime=111),
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
    assert store.cleanup_recovery_required(handle)
    assert store.recovery_required(handle)


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
    handle = make_run_handle(run_id="exp-1", pid=reaped_pid(), starttime=111)
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
        pid=reaped_pid(),
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

    private_atomic_write_text(
        store.cleanup_recovery_path("exp-1"),
        json.dumps(
            {
                "run_id": "exp-1",
                "config_sha256": "a" * 64,
                "cleanup_confirmed": True,
            }
        ),
    )

    assert store.state(handle) == "failed"
    assert store.live_runs() == []


def test_terminal_cleanup_recovery_must_match_handle_hash(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(
        run_id="exp-1",
        config_sha256="a" * 64,
        pid=reaped_pid(),
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
    private_atomic_write_text(
        store.cleanup_recovery_path("exp-1"),
        json.dumps(
            {
                "run_id": "exp-1",
                "config_sha256": "b" * 64,
                "cleanup_confirmed": True,
            }
        ),
    )

    assert store.state(handle) == "running"


@pytest.mark.parametrize("reaped_attempt_ids", ["not-a-list", [""], [1], {}])
def test_cleanup_recovery_rejects_malformed_reaped_attempt_ids(
    tmp_path: Path,
    reaped_attempt_ids: object,
) -> None:
    """Malformed recovery details cannot release a cleanup reservation."""
    store = RunStore(tmp_path / "state")
    handle = make_run_handle(
        run_id="exp-1",
        config_sha256="a" * 64,
        pid=reaped_pid(),
        starttime=111,
    )
    store.create(handle)
    write_run_status(
        store,
        handle.run_id,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
    )
    private_atomic_write_text(
        store.cleanup_recovery_path(handle.run_id),
        json.dumps(
            {
                "run_id": handle.run_id,
                "config_sha256": handle.config_sha256,
                "cleanup_confirmed": True,
                "reaped_attempt_ids": reaped_attempt_ids,
            }
        ),
    )

    assert not store._cleanup_recovered(handle)
    assert store.state(handle) == "running"
    assert store.recovery_required(handle)


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
    private_atomic_write_text(
        store.cleanup_recovery_path(handle.run_id),
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
        ),
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


@pytest.mark.integration
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
