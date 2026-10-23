"""Same-host advisory run locks: output namespace and storage identity."""

from __future__ import annotations

import os
import pwd
import signal
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from phasesweep.config.models import Phase
from phasesweep.config.search import IntParam, Sampler
from phasesweep.engine.errors import ExperimentLockBusyError
from phasesweep.engine.locking import (
    _experiment_lock,
    _run_lock_paths,
)
from phasesweep.engine.run import run_experiment
from phasesweep.errors import PhaseSweepError
from phasesweep.runtime import files as runtime_files
from tests.conftest import (
    make_experiment,
    patch_directory_fsync_failure,
    raise_after_first_successful_call,
)


@pytest.mark.parametrize(
    "variable",
    ["HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR", "PHASESWEEP_HOME"],
)
def test_default_locks_contend_across_state_and_cache_environments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("PHASESWEEP_LOCK_DIR", raising=False)
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(home)))
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir(mode=0o700)
    second.mkdir(mode=0o700)
    monkeypatch.setenv(variable, str(first))

    path = runtime_files.lock_dir()

    assert path == home / ".cache" / "phasesweep" / "locks"
    assert path.is_dir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    experiment = make_experiment(workdir=str(tmp_path / "runs"))
    paths = _run_lock_paths(experiment)
    with _experiment_lock(experiment):
        monkeypatch.setenv(variable, str(second))
        assert runtime_files.lock_dir() == path
        assert _run_lock_paths(experiment) == paths
        with pytest.raises(ExperimentLockBusyError), _experiment_lock(experiment):
            pytest.fail("a different launch environment bypassed the held lock")
    assert not (first / "locks").exists()
    assert not (second / "locks").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="uses macOS root-owned system temp aliases")
@pytest.mark.parametrize("alias", ["var", "tmp"])
def test_macos_system_temp_aliases_share_physical_host_locks(monkeypatch, alias):
    """System temp aliases preserve lock contention without accepting user symlinks."""
    parent = Path("/private/var/tmp" if alias == "var" else "/private/tmp")
    with tempfile.TemporaryDirectory(prefix="phasesweep-lock-alias-", dir=parent) as directory:
        physical = Path(directory)
        aliased = Path("/").joinpath(*physical.parts[2:])
        monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(aliased))
        assert runtime_files.lock_dir() == physical
        held = runtime_files.try_lock_file(runtime_files.lock_dir() / "shared.lock")
        assert held is not None
        with held:
            monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(physical))
            assert runtime_files.try_lock_file(runtime_files.lock_dir() / "shared.lock") is None
        released = runtime_files.try_lock_file(runtime_files.lock_dir() / "shared.lock")
        assert released is not None
        runtime_files.unlock_file(released)

        (physical / "linked").symlink_to(physical, target_is_directory=True)
        monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(aliased / "linked"))
        with pytest.raises(runtime_files.UnsafeLockPathError, match="unsafe path component"):
            runtime_files.lock_dir()
        monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(aliased))
        monkeypatch.setattr(runtime_files, "sys", SimpleNamespace(platform="linux"))
        with pytest.raises(runtime_files.UnsafeLockPathError, match="unsafe path component"):
            runtime_files.lock_dir()


@pytest.mark.parametrize("account_home", [None, "", "relative"])
def test_default_lock_dir_requires_an_absolute_account_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, account_home: str | None
) -> None:
    def account_for(uid: int) -> SimpleNamespace:
        if account_home is None:
            raise KeyError(uid)
        return SimpleNamespace(pw_dir=account_home)

    monkeypatch.setattr(pwd, "getpwuid", account_for)
    monkeypatch.delenv("PHASESWEEP_LOCK_DIR")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(runtime_files.UnsafeLockPathError, match="PHASESWEEP_LOCK_DIR"):
        runtime_files.lock_dir()
    assert not (tmp_path / ".cache").exists()
    assert not (tmp_path / "relative").exists()

    override = tmp_path / "provisioned-locks"
    override.mkdir(mode=0o700)
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(override))
    assert runtime_files.lock_dir() == override


@pytest.mark.parametrize("invalid", ["mode", "symlink"])
def test_home_override_requires_private_existing_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    if invalid == "mode":
        root.chmod(0o755)
    else:
        link = tmp_path / "link"
        link.symlink_to(root, target_is_directory=True)
        root = link
    monkeypatch.setenv("PHASESWEEP_HOME", str(root))
    with pytest.raises(runtime_files.UnsafePrivatePathError):
        runtime_files.phasesweep_home()
    assert not (root / "locks").exists()
    if invalid == "mode":
        assert stat.S_IMODE(root.stat().st_mode) == 0o755


def test_home_override_creates_missing_root_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing root and its missing parents are created 0700, whatever the umask."""
    root = tmp_path / "state" / "phasesweep-home"
    monkeypatch.setenv("PHASESWEEP_HOME", str(root))
    previous_umask = os.umask(0)
    try:
        assert runtime_files.phasesweep_home() == root
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(root.parent.stat().st_mode) == 0o700
    assert runtime_files.phasesweep_home() == root


def test_home_override_refuses_relative_path_without_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PHASESWEEP_HOME", "relative-home")

    with pytest.raises(runtime_files.UnsafePrivatePathError, match="must be an absolute path"):
        runtime_files.phasesweep_home()

    assert not (tmp_path / "relative-home").exists()


def test_home_override_refuses_symlinked_ancestor_without_creating_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("PHASESWEEP_HOME", str(link / "phasesweep-home"))

    with pytest.raises(runtime_files.UnsafePrivatePathError):
        runtime_files.phasesweep_home()

    assert list(target.iterdir()) == []


def test_private_home_override_selects_state_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "private-root"
    root.mkdir(mode=0o700)
    monkeypatch.setenv("PHASESWEEP_HOME", str(root))
    assert runtime_files.phasesweep_home() == root
    monkeypatch.setenv("PHASESWEEP_HOME", "")
    assert runtime_files.phasesweep_home() is None


def test_lock_dir_honors_explicit_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "scheduler-shared-locks"
    override.mkdir(mode=0o700)
    override.chmod(0o700)
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(override))
    monkeypatch.setenv("PHASESWEEP_HOME", "relative-but-unused")

    path = runtime_files.lock_dir()

    assert path == override
    assert path.is_dir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o700


@pytest.mark.parametrize("override", [False, True], ids=["default", "override"])
def test_lock_directory_os_failure_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: bool
) -> None:
    lock_root = tmp_path / "locks" if override else tmp_path / ".cache/phasesweep/locks"
    denied = PermissionError(13, "Permission denied", str(lock_root))

    def refuse(*args: object, **kwargs: object) -> None:
        raise denied

    if override:
        monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(lock_root))
        monkeypatch.setattr(runtime_files, "open_directory_fd", refuse)
    else:
        monkeypatch.delenv("PHASESWEEP_LOCK_DIR", raising=False)
        monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(tmp_path)))
        monkeypatch.setattr(runtime_files, "ensure_private_dir", refuse)

    with pytest.raises(runtime_files.UnsafeLockPathError) as caught:
        runtime_files.lock_dir()

    assert caught.value.__cause__ is denied
    assert str(lock_root) in str(caught.value)
    assert "Permission denied" in str(caught.value)
    assert "PHASESWEEP_LOCK_DIR" in str(caught.value)
    assert "same writable" in str(caught.value)
    assert not lock_root.exists()


def test_lock_file_os_failure_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "blocked.lock"
    denied = PermissionError(13, "Permission denied", str(lock_path))

    def refuse(path: Path) -> None:
        raise denied

    monkeypatch.setattr(runtime_files, "_open_lock_file", refuse)

    with pytest.raises(runtime_files.UnsafeLockPathError) as caught:
        runtime_files.try_lock_file(lock_path)

    assert caught.value.__cause__ is denied
    assert str(lock_path) in str(caught.value)
    assert "PHASESWEEP_LOCK_DIR" in str(caught.value)
    assert "same writable" in str(caught.value)
    assert not lock_path.exists()


def test_lock_dir_accepts_admin_shared_directory_and_creates_group_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override = tmp_path / "scheduler-shared-locks"
    override.mkdir(mode=runtime_files.SHARED_DIR_MODE)
    override.chmod(runtime_files.SHARED_DIR_MODE)
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(override))
    real_fstat = runtime_files.os.fstat

    def root_owned_directories(fd: int) -> os.stat_result:
        info = real_fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            return info
        values = list(info)
        values[0] = (info.st_mode & ~0o7777) | runtime_files.SHARED_DIR_MODE
        values[4] = 0
        values[5] = os.getegid()
        return os.stat_result(values)

    monkeypatch.setattr(runtime_files.os, "fstat", root_owned_directories)

    assert runtime_files.lock_dir() == override
    handle = runtime_files.try_lock_file(override / "shared.lock")
    assert handle is not None
    runtime_files.unlock_file(handle)
    assert stat.S_IMODE((override / "shared.lock").stat().st_mode) == 0o660


def test_lock_dir_rejects_missing_or_unsafe_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "operator-managed-locks"
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(override))
    with pytest.raises(runtime_files.UnsafeLockPathError, match="does not exist"):
        runtime_files.lock_dir()
    assert not override.exists()

    override.mkdir()
    override.chmod(0o750)
    with pytest.raises(runtime_files.UnsafeLockPathError, match="Unsafe lock directory"):
        runtime_files.lock_dir()
    assert issubclass(runtime_files.UnsafeLockPathError, PhaseSweepError)


def test_lock_open_rejects_symlink_before_gpu_diagnostics_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    lock_root.chmod(0o700)
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(lock_root))
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    (lock_root / "gpu_0.lock").symlink_to(victim)

    from phasesweep.runtime.gpu import GpuDevice, _try_host_gpu_lease

    with pytest.raises(runtime_files.UnsafeLockPathError, match="must not be a symlink"):
        _try_host_gpu_lease(GpuDevice("0"))
    assert victim.read_text() == "keep me"


def test_lock_open_rejects_hardlinks_and_creates_private_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    lock_root.chmod(0o700)
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(lock_root))

    handle = runtime_files.try_lock_file(lock_root / "safe.lock")
    assert handle is not None
    runtime_files.unlock_file(handle)
    assert stat.S_IMODE((lock_root / "safe.lock").stat().st_mode) == 0o600

    (lock_root / "linked.lock").hardlink_to(lock_root / "safe.lock")
    with pytest.raises(runtime_files.UnsafeLockPathError, match="one link"):
        runtime_files.try_lock_file(lock_root / "linked.lock")


def test_default_lock_dir_rejects_symlink_without_chmodding_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    namespace = home / ".cache" / "phasesweep"
    namespace.mkdir(parents=True, mode=0o700)
    namespace.chmod(0o700)
    target = tmp_path / "unrelated"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    (namespace / "locks").symlink_to(target, target_is_directory=True)
    monkeypatch.delenv("PHASESWEEP_LOCK_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(home)))

    with pytest.raises(runtime_files.UnsafeLockPathError, match="unsafe"):
        runtime_files.lock_dir()

    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_lock_open_rejects_unsafe_mode_without_modifying_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    lock_root.chmod(0o700)
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(lock_root))
    lock_path = lock_root / "unsafe.lock"
    lock_path.write_text("unchanged")
    lock_path.chmod(0o644)

    with pytest.raises(runtime_files.UnsafeLockPathError, match="expected 0600"):
        runtime_files.open_lock_file(lock_path)

    assert lock_path.read_text() == "unchanged"
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o644


def test_private_directory_rejects_symlink_without_modifying_target(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    target = tmp_path / "unrelated"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    linked = root / "state"
    linked.symlink_to(target, target_is_directory=True)
    unsafe_mode = root / "unsafe-mode"
    unsafe_mode.mkdir(mode=0o755)
    unsafe_mode.chmod(0o755)

    with pytest.raises(runtime_files.UnsafePrivatePathError):
        runtime_files.ensure_private_dir(linked)
    with pytest.raises(runtime_files.UnsafePrivatePathError):
        runtime_files.ensure_private_dir(unsafe_mode)

    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert stat.S_IMODE(unsafe_mode.stat().st_mode) == 0o755


def test_private_open_rejects_symlink_hardlink_and_wrong_mode_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    victim = tmp_path / "victim.txt"
    victim.write_text("do not destroy")
    victim.chmod(0o600)
    victim_mode = stat.S_IMODE(victim.stat().st_mode)
    symlink = root / "symlink.log"
    symlink.symlink_to(victim)
    hardlink = root / "hardlink.log"
    hardlink.hardlink_to(victim)
    unsafe_mode = root / "mode.log"
    unsafe_mode.write_text("keep mode")
    unsafe_mode.chmod(0o644)

    for path in (symlink, hardlink, unsafe_mode):
        with (
            pytest.raises(runtime_files.UnsafePrivatePathError),
            runtime_files.open_private_text(path, "w") as handle,
        ):
            handle.write("replacement")

    assert victim.read_text() == "do not destroy"
    assert stat.S_IMODE(victim.stat().st_mode) == victim_mode
    assert unsafe_mode.read_text() == "keep mode"
    assert stat.S_IMODE(unsafe_mode.stat().st_mode) == 0o644


def test_private_atomic_write_rejects_symlink_and_intermediate_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    outside.chmod(0o700)
    victim = outside / "victim.txt"
    victim.write_text("unchanged")
    victim.chmod(0o600)
    final_link = root / "status.json"
    final_link.symlink_to(victim)
    parent_link = root / "linked-parent"
    parent_link.symlink_to(outside, target_is_directory=True)
    unsafe_mode = root / "unsafe-mode.json"
    unsafe_mode.write_text("unsafe mode")
    unsafe_mode.chmod(0o644)

    with pytest.raises(runtime_files.UnsafePrivatePathError):
        runtime_files.private_atomic_write_text(final_link, "replacement")
    with pytest.raises(runtime_files.UnsafePrivatePathError):
        runtime_files.private_atomic_write_text(parent_link / "new.txt", "replacement")
    with pytest.raises(runtime_files.UnsafePrivatePathError):
        runtime_files.private_atomic_write_text(unsafe_mode, "replacement")

    assert victim.read_text() == "unchanged"
    assert final_link.is_symlink()
    assert not (outside / "new.txt").exists()
    assert unsafe_mode.read_text() == "unsafe mode"
    assert stat.S_IMODE(unsafe_mode.stat().st_mode) == 0o644


def test_private_atomic_write_rejects_intermediate_symlink_inserted_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "state"
    child = state / "child"
    child.mkdir(parents=True, mode=0o700)
    state.chmod(0o700)
    child.chmod(0o700)
    moved = state / "moved-child"
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    outside.chmod(0o700)
    original_stat = runtime_files.os.stat
    swapped = False

    def insert_symlink(path: object, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal swapped
        info = original_stat(path, *args, **kwargs)
        if path == "child" and not swapped:
            swapped = True
            child.rename(moved)
            child.symlink_to(outside, target_is_directory=True)
        return info

    monkeypatch.setattr(runtime_files.os, "stat", insert_symlink)

    with pytest.raises(runtime_files.UnsafePrivatePathError):
        runtime_files.private_atomic_write_text(child / "status.json", "escaped")

    assert not (outside / "status.json").exists()
    assert not (moved / "status.json").exists()


def test_private_atomic_write_rejects_intermediate_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "state"
    child = state / "child"
    child.mkdir(parents=True, mode=0o700)
    state.chmod(0o700)
    child.chmod(0o700)
    moved = state / "moved-child"
    original_stat = runtime_files.os.stat
    swapped = False

    def swap_directory(path: object, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal swapped
        info = original_stat(path, *args, **kwargs)
        if path == "child" and not swapped:
            swapped = True
            child.rename(moved)
            child.mkdir(mode=0o700)
            child.chmod(0o700)
        return info

    monkeypatch.setattr(runtime_files.os, "stat", swap_directory)

    with pytest.raises(runtime_files.UnsafePrivatePathError, match="changed while it was opened"):
        runtime_files.private_atomic_write_text(child / "status.json", "replacement")

    assert not (child / "status.json").exists()
    assert not (moved / "status.json").exists()


@pytest.mark.parametrize("writer_name", ["private_atomic_write_text", "private_atomic_create_text"])
def test_private_atomic_write_keeps_opened_parent_during_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer_name: str,
) -> None:
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    state.chmod(0o700)
    moved = tmp_path / "moved-state"
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    outside.chmod(0o700)
    outside_status = outside / "status.json"
    outside_status.write_text("outside")
    outside_status.chmod(0o600)
    original_new_temp = runtime_files._new_exclusive_temp_fd

    def swap_parent(path: Path, mode: int, *, directory_fd: int | None = None) -> tuple[int, Path]:
        state.rename(moved)
        state.symlink_to(outside, target_is_directory=True)
        return original_new_temp(path, mode, directory_fd=directory_fd)

    monkeypatch.setattr(runtime_files, "_new_exclusive_temp_fd", swap_parent)

    getattr(runtime_files, writer_name)(state / "status.json", "inside")

    assert (moved / "status.json").read_text() == "inside"
    assert outside_status.read_text() == "outside"


@pytest.mark.parametrize(
    "writer_name",
    [
        "atomic_write_text",
        "atomic_create_text",
        "private_atomic_write_text",
        "private_atomic_create_text",
    ],
)
@pytest.mark.parametrize("failure_boundary", ["fdopen", "file_fsync"])
def test_private_atomic_writer_shutdown_during_fdopen_does_not_double_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer_name: str,
    failure_boundary: str,
) -> None:
    """Shutdown during staging closes its raw descriptor once and removes partial bytes."""

    class InjectedShutdown(BaseException):
        """Stand in for ``PhaseSweepShutdown`` at the fd-to-stream boundary."""

    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    state.chmod(0o700)
    real_new_temp = runtime_files._new_exclusive_temp_fd
    real_fdopen = os.fdopen
    real_close = os.close
    temporary_fd: int | None = None
    temporary_close_count = 0

    def capture_temp_fd(
        path: Path, mode: int, *, directory_fd: int | None = None
    ) -> tuple[int, Path]:
        nonlocal temporary_fd
        temporary_fd, temporary = real_new_temp(path, mode, directory_fd=directory_fd)
        return temporary_fd, temporary

    def shutdown_after_wrap(fd: int, *args: object, **kwargs: object) -> None:
        stream = real_fdopen(fd, *args, **kwargs)
        assert not stream.closed
        raise InjectedShutdown

    def count_temp_close(fd: int) -> None:
        nonlocal temporary_close_count
        if fd == temporary_fd:
            temporary_close_count += 1
        real_close(fd)

    def shutdown_during_fsync(_fd: int) -> None:
        raise InjectedShutdown

    monkeypatch.setattr(runtime_files, "_new_exclusive_temp_fd", capture_temp_fd)
    if failure_boundary == "fdopen":
        monkeypatch.setattr(os, "fdopen", shutdown_after_wrap)
    else:
        monkeypatch.setattr(os, "fsync", shutdown_during_fsync)
    monkeypatch.setattr(os, "close", count_temp_close)

    with pytest.raises(InjectedShutdown):
        getattr(runtime_files, writer_name)(state / "status.json", "replacement")

    assert temporary_close_count == 1
    assert list(state.iterdir()) == []


@pytest.mark.parametrize(
    ("opener_name", "filename"),
    [
        pytest.param("open_lock_file", "run.lock", id="lock-file"),
        pytest.param("open_private_text", "status.json", id="private-text"),
    ],
)
def test_private_open_closes_descriptor_on_shutdown_during_fdopen_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opener_name: str,
    filename: str,
) -> None:
    """A deferred shutdown aborting fd handoff must close the returned stream."""
    from phasesweep.runtime.shutdown import PhaseSweepShutdown, _shutdown_handler

    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    state.chmod(0o700)
    opened_fd: int | None = None
    real_fdopen = os.fdopen

    def shutdown_during_wrap(fd: int, *args: object, **kwargs: object) -> object:
        nonlocal opened_fd
        opened_fd = fd
        _shutdown_handler(signal.SIGTERM, None)
        return real_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", shutdown_during_wrap)

    with pytest.raises(PhaseSweepShutdown):
        getattr(runtime_files, opener_name)(state / filename)

    assert opened_fd is not None
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(opened_fd)


@pytest.mark.parametrize("strict", [False, True], ids=["best-effort", "strict"])
def test_private_atomic_write_directory_fsync_failure_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    strict: bool,
) -> None:
    """Replacement stays published when directory fsync raises or logs its failure."""
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    state.chmod(0o700)
    path = state / "status.json"
    patch_directory_fsync_failure(monkeypatch, "simulated directory fsync failure")
    caplog.set_level("WARNING", logger="phasesweep.runtime.files")

    if strict:
        with pytest.raises(OSError, match="simulated directory fsync failure"):
            runtime_files.private_atomic_write_text(path, "committed", strict_dir_fsync=True)
    else:
        runtime_files.private_atomic_write_text(path, "committed")

    assert path.read_text() == "committed"
    assert ("Directory fsync failed after a private atomic write" in caplog.text) is not strict


def test_directory_fsync_failure_policy_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Best-effort fsync logs the failure and closes its descriptor."""
    directory = tmp_path / "state"
    directory.mkdir(mode=0o700)
    synced_fd: int | None = None

    def fail_fsync(fd: int) -> None:
        nonlocal synced_fd
        synced_fd = fd
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    caplog.set_level("WARNING", logger="phasesweep.runtime.files")
    runtime_files.fsync_directory(directory)
    assert "the write is committed" in caplog.text
    assert synced_fd is not None
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(synced_fd)


def test_directory_fsync_addressing_policy(tmp_path: Path) -> None:
    """Best-effort directory fsync follows symlinks."""
    directory = tmp_path / "state"
    directory.mkdir(mode=0o700)
    link = tmp_path / "linked-state"
    link.symlink_to(directory, target_is_directory=True)

    runtime_files.fsync_directory(link)


def test_open_directory_fd_shutdown_mid_walk_does_not_double_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shutdown raised mid-walk must propagate, not become EBADF.

    ``_shutdown_handler`` raises ``PhaseSweepShutdown`` from a Python signal
    handler, so an exception can land on any bytecode boundary in the
    component walk. The descriptor swap must store the new fd before closing
    the old one: closing first would leave ``current_fd`` naming a closed
    descriptor, the cleanup path would close it again, and the real shutdown
    would surface as ``OSError: [Errno 9] Bad file descriptor``.
    """

    class InjectedShutdown(BaseException):
        """Stands in for the shutdown a signal handler raises at this point."""

    target = tmp_path / "nested" / "dir"
    target.mkdir(parents=True)
    close_then_shutdown, close_calls = raise_after_first_successful_call(
        os.close,
        InjectedShutdown(),
    )
    monkeypatch.setattr(os, "close", close_then_shutdown)

    with pytest.raises(InjectedShutdown):
        runtime_files.open_directory_fd(target, create=False, private_final=False)
    assert close_calls


@pytest.mark.signals_own_pid
@pytest.mark.skipif(
    sys.platform != "linux", reason="counts descriptors through Linux /proc/self/fd"
)
def test_open_directory_fd_defers_midwalk_shutdown_and_leaks_no_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SIGTERM during the walk is deferred and no descriptor leaks.

    Shutdown must not interrupt between storing the next descriptor and closing
    the previous one. It queues until the walk finishes, and the completed
    directory descriptor closes before shutdown propagates to the caller.
    """
    from phasesweep.runtime.shutdown import PhaseSweepShutdown, signal_handler_scope

    target = tmp_path / "nested" / "dir"
    target.mkdir(parents=True)
    real_open = os.open
    fired = False

    def open_then_sigterm(*args: object, **kwargs: object) -> int:
        nonlocal fired
        fd = real_open(*args, **kwargs)  # type: ignore[arg-type]
        if not fired and kwargs.get("dir_fd") is not None:
            fired = True
            os.kill(os.getpid(), signal.SIGTERM)
        return fd

    monkeypatch.setattr(os, "open", open_then_sigterm)
    before = set(os.listdir("/proc/self/fd"))
    with signal_handler_scope(), pytest.raises(PhaseSweepShutdown):
        runtime_files.open_directory_fd(target, create=False, private_final=False)
    assert fired
    after = set(os.listdir("/proc/self/fd"))
    assert after <= before


def test_run_lock_blocks_even_when_processes_target_different_phases(
    tmp_path: Path,
) -> None:
    """Two configs with the same storage but different phase orderings still
    collide on the run lock — because the lock is experiment-scoped, not
    phase-scoped.

    The interleaving scenario: process A is on phase ``lr`` while process B
    starts a top-up of phase ``arch``. A phase lock would not catch this; the
    run lock does.
    """
    storage = f"journal:///{tmp_path / 'shared.journal'}"
    seeded = Sampler(type="random", seed=0)
    depth = {"depth": IntParam(type="int", low=1, high=9)}
    phases_a = [
        Phase(name="arch", n_trials=1, sampler=seeded, search_space=depth),
        Phase(name="lr", inherits=["arch"], n_trials=1, sampler=seeded, search_space={}),
    ]
    phases_b = [Phase(name="arch", n_trials=2, sampler=seeded, search_space=depth)]  # top-up

    exp_a = make_experiment(workdir=str(tmp_path / "runs_a"), storage=storage, phases=phases_a)
    exp_b = make_experiment(workdir=str(tmp_path / "runs_b"), storage=storage, phases=phases_b)

    with (
        _experiment_lock(exp_a),
        pytest.raises(ExperimentLockBusyError, match="Another phasesweep process"),
        _experiment_lock(exp_b),
    ):
        pass


def test_run_lock_collides_for_different_storage_same_output_dir(
    tmp_path: Path,
) -> None:
    """Two configs sharing workdir + experiment but pointing at different
    storage backends collide on filesystem outputs, so the output-namespace
    lock must catch them even though their storage locks differ.
    """
    exp_a = make_experiment(
        workdir=str(tmp_path / "runs"), storage=f"journal:///{tmp_path / 'a.journal'}"
    )
    exp_b = make_experiment(
        workdir=str(tmp_path / "runs"), storage=f"journal:///{tmp_path / 'b.journal'}"
    )

    with (  # testing that the inner enter raises
        _experiment_lock(exp_a),
        pytest.raises(RuntimeError, match="output namespace|backend"),
        _experiment_lock(exp_b),
    ):
        pass


@pytest.mark.parametrize(
    ("storage_a_name", "storage_b_name"),
    [
        pytest.param("a.journal", "b.journal", id="different-storage-and-name"),
        pytest.param("shared.journal", "shared.journal", id="shared-storage-different-name"),
    ],
)
def test_run_lock_does_not_collide_for_distinct_experiment_names(
    tmp_path: Path, storage_a_name: str, storage_b_name: str
) -> None:
    """Distinct experiment namespaces do not share output or storage locks."""
    storage_a = f"journal:///{tmp_path / storage_a_name}"
    storage_b = f"journal:///{tmp_path / storage_b_name}"
    exp_a = make_experiment(workdir=str(tmp_path / "runs"), storage=storage_a)
    exp_b = make_experiment(experiment="other", workdir=str(tmp_path / "runs"), storage=storage_b)

    assert set(_run_lock_paths(exp_a)).isdisjoint(_run_lock_paths(exp_b))
    with _experiment_lock(exp_a), _experiment_lock(exp_b):
        pass  # must not raise


@pytest.mark.parametrize(
    ("workdir_a", "workdir_b", "same_lock"),
    [
        pytest.param("runs", "runs", True, id="same-workdir"),
        pytest.param("runs_a", "runs_b", False, id="different-workdirs"),
    ],
)
def test_in_memory_run_lock_is_keyed_by_workdir(
    tmp_path: Path,
    workdir_a: str,
    workdir_b: str,
    same_lock: bool,
) -> None:
    """In-memory storage uses one output lock whose identity follows workdir."""
    exp_a = make_experiment(workdir=str(tmp_path / workdir_a))
    exp_b = make_experiment(workdir=str(tmp_path / workdir_b))

    paths_a = _run_lock_paths(exp_a)
    paths_b = _run_lock_paths(exp_b)
    # In-memory storage means only the output lock is taken — single path.
    assert len(paths_a) == 1
    if same_lock:
        assert paths_a == paths_b
    else:
        assert set(paths_a).isdisjoint(paths_b)


@pytest.mark.integration
def test_output_lock_resolves_symlinked_experiment_leaf(tmp_path: Path) -> None:
    """A symlinked experiment leaf must share the target's output lock.

    ``_experiment_dir`` resolves only the workdir prefix before appending the
    experiment name, so without resolving the leaf ``runs/expA -> runs/expB``
    would mint a second lock identity for one physical namespace — the second
    orchestrator's preflight would then reap the first one's live trials.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "real").mkdir()
    (runs / "alias").symlink_to(runs / "real")

    exp_real = make_experiment(workdir=str(runs)).model_copy(update={"experiment": "real"})
    exp_alias = make_experiment(workdir=str(runs)).model_copy(update={"experiment": "alias"})

    assert set(_run_lock_paths(exp_real)) == set(_run_lock_paths(exp_alias))

    distinct = exp_real.model_copy(update={"experiment": "distinct"})
    (runs / "distinct").mkdir()

    def contender(config):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
from phasesweep.config.models import Experiment
from phasesweep.engine.locking import _experiment_lock
from phasesweep.engine.errors import ExperimentLockBusyError
try:
    with _experiment_lock(Experiment.model_validate_json(sys.argv[1])):
        print("acquired")
except ExperimentLockBusyError:
    print("busy")
    sys.exit(2)
""",
                config.model_dump_json(),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

    with _experiment_lock(exp_real):
        lock_inodes = {path: path.stat().st_ino for path in runtime_files.lock_dir().glob("*.lock")}
        blocked = contender(exp_alias)
        assert (blocked.returncode, blocked.stdout.strip()) == (2, "busy"), blocked.stderr
        control = contender(distinct)
        assert (control.returncode, control.stdout.strip()) == (0, "acquired"), control.stderr
    released = contender(exp_alias)
    assert (released.returncode, released.stdout.strip()) == (0, "acquired"), released.stderr
    assert all(path.stat().st_ino == inode for path, inode in lock_inodes.items())


def test_run_experiment_holds_experiment_lock_for_duration(tmp_path: Path) -> None:
    """A second concurrent ``run_experiment`` against the same experiment
    fails fast with the expected error while the run lock is held.
    """
    storage = f"journal:///{tmp_path / 'shared.journal'}"
    exp_a = make_experiment(
        workdir=str(tmp_path / "runs_a"),
        storage=storage,
        trial_command="true {overrides}",
        n_trials=1,
    )
    exp_b = make_experiment(
        workdir=str(tmp_path / "runs_b"),
        storage=storage,
        trial_command="true {overrides}",
        n_trials=1,
    )

    held = threading.Event()
    released = threading.Event()

    def hold() -> None:
        with _experiment_lock(exp_a):
            held.set()
            released.wait(timeout=5.0)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    assert held.wait(timeout=2.0)

    try:
        with pytest.raises(RuntimeError, match="Another phasesweep process"):
            run_experiment(exp_b, dry_run=False)
    finally:
        released.set()
        t.join(timeout=2.0)


def test_run_experiment_dry_run_does_not_take_experiment_lock(tmp_path: Path) -> None:
    """Dry-run is read-only: it must not require or take the experiment lock.
    A user inspecting an experiment's plan while a real run is in progress is
    a legitimate workflow.
    """
    storage = f"journal:///{tmp_path / 'shared.journal'}"
    exp_a = make_experiment(workdir=str(tmp_path / "runs_a"), storage=storage)
    exp_b = make_experiment(workdir=str(tmp_path / "runs_b"), storage=storage)

    with _experiment_lock(exp_a):
        # Dry-run must succeed with the run lock held by another caller.
        winners = run_experiment(exp_b, dry_run=True)
        assert "p" in winners
