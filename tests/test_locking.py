"""Same-host advisory run locks: output namespace and storage identity."""

from __future__ import annotations

import os
import signal
import stat
import threading
from pathlib import Path

import pytest

from phasesweep.config import Experiment, FloatParam, IntParam, Phase, Sampler
from phasesweep.engine import run_experiment
from phasesweep.engine.errors import ExperimentLockBusyError
from phasesweep.engine.guards import (
    _experiment_lock,
    _run_lock_paths,
)
from phasesweep.errors import LockBusyError, PhaseSweepError
from phasesweep.runtime import files as runtime_files
from tests.conftest import make_experiment


def test_missing_nofollow_is_a_platform_capability_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(runtime_files.os, "O_NOFOLLOW")

    with pytest.raises(
        runtime_files.PlatformCapabilityError,
        match="without following symlinks",
    ):
        runtime_files.nofollow_flag()


def test_lock_dir_default_is_independent_of_xdg_runtime_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    runtime_dir.chmod(0o700)
    monkeypatch.delenv("PHASESWEEP_LOCK_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))

    path = runtime_files.lock_dir()

    assert path == home / ".cache" / "phasesweep" / "locks"
    assert path.is_dir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o700


def test_lock_dir_honors_explicit_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "scheduler-shared-locks"
    override.mkdir(mode=0o700)
    override.chmod(0o700)
    monkeypatch.setenv("PHASESWEEP_LOCK_DIR", str(override))

    path = runtime_files.lock_dir()

    assert path == override
    assert path.is_dir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o700


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

    override.mkdir()
    override.chmod(0o750)
    with pytest.raises(runtime_files.UnsafeLockPathError, match="Unsafe lock directory"):
        runtime_files.lock_dir()
    assert issubclass(runtime_files.UnsafeLockPathError, PhaseSweepError)


def test_busy_generic_lock_is_an_operational_error(tmp_path: Path) -> None:
    """Suite-style lock contention belongs to the CLI's expected boundary."""
    lock_path = tmp_path / "busy.lock"
    held = runtime_files.try_lock_file(lock_path)
    assert held is not None
    try:
        with (
            pytest.raises(LockBusyError, match="already busy"),
            runtime_files.exclusive_lock(lock_path, busy_message="already busy"),
        ):
            pytest.fail("the held lock must not be reacquired")
    finally:
        runtime_files.unlock_file(held)


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
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

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


def test_private_atomic_write_keeps_opened_parent_during_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    original_new_temp = runtime_files._new_private_temp_fd

    def swap_parent(parent_fd: int, leaf: str) -> tuple[int, str]:
        state.rename(moved)
        state.symlink_to(outside, target_is_directory=True)
        return original_new_temp(parent_fd, leaf)

    monkeypatch.setattr(runtime_files, "_new_private_temp_fd", swap_parent)

    runtime_files.private_atomic_write_text(state / "status.json", "inside")

    assert (moved / "status.json").read_text() == "inside"
    assert outside_status.read_text() == "outside"


def test_private_atomic_writer_shutdown_during_fdopen_does_not_double_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shutdown after the stream is created must leave one descriptor owner."""

    class InjectedShutdown(BaseException):
        """Stand in for ``PhaseSweepShutdown`` at the fd-to-stream boundary."""

    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    state.chmod(0o700)
    real_new_temp = runtime_files._new_private_temp_fd
    real_fdopen = os.fdopen
    real_close = os.close
    temporary_fd: int | None = None
    temporary_close_count = 0

    def capture_temp_fd(parent_fd: int, leaf: str) -> tuple[int, str]:
        nonlocal temporary_fd
        temporary_fd, temporary = real_new_temp(parent_fd, leaf)
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

    monkeypatch.setattr(runtime_files, "_new_private_temp_fd", capture_temp_fd)
    monkeypatch.setattr(os, "fdopen", shutdown_after_wrap)
    monkeypatch.setattr(os, "close", count_temp_close)

    with pytest.raises(InjectedShutdown):
        runtime_files.private_atomic_write_text(state / "status.json", "replacement")

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
    from phasesweep.runtime.process import PhaseSweepShutdown, _shutdown_handler

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


def test_private_atomic_write_logs_directory_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A committed private write reports lost crash durability without failing."""
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    state.chmod(0o700)
    path = state / "status.json"
    real_fsync = os.fsync

    def fail_directory_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("simulated directory fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    caplog.set_level("WARNING", logger="phasesweep.runtime.files")

    runtime_files.private_atomic_write_text(path, "committed")

    assert path.read_text() == "committed"
    assert "Directory fsync failed after a private atomic write" in caplog.text


def test_open_directory_fd_shutdown_mid_walk_does_not_double_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shutdown raised mid-walk must propagate, not become EBADF.

    ``_shutdown_handler`` raises ``PhaseSweepShutdown`` from a Python signal
    handler, so an exception can land on any bytecode boundary in the
    component walk. The descriptor swap used to close the old fd before
    storing the new one, leaving ``current_fd`` naming a closed descriptor;
    the cleanup path then closed it again and the real shutdown surfaced as
    ``OSError: [Errno 9] Bad file descriptor``.
    """

    class InjectedShutdown(BaseException):
        """Stands in for the shutdown a signal handler raises at this point."""

    target = tmp_path / "nested" / "dir"
    target.mkdir(parents=True)
    real_close = os.close
    fired = False

    def close_then_shutdown(fd: int) -> None:
        nonlocal fired
        real_close(fd)
        if not fired:
            fired = True
            raise InjectedShutdown

    monkeypatch.setattr(os, "close", close_then_shutdown)

    with pytest.raises(InjectedShutdown):
        runtime_files.open_directory_fd(target, create=False, private_final=False)
    assert fired


def test_open_directory_fd_defers_midwalk_shutdown_and_leaks_no_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SIGTERM during the walk is deferred and no descriptor leaks.

    Without ``defer_shutdown_signals`` around the walk, the handler's
    ``PhaseSweepShutdown`` could land on the bytecode boundary between the
    descriptor-handoff store and the close of the previous descriptor,
    stranding an open fd until process exit — invisible in one-shot
    processes, an accumulating leak for a long-running embedder now that
    ``open_directory_fd`` is public API. The signal must queue until the
    walk finishes and the completed final descriptor must be closed before
    the shutdown propagates to the caller.
    """
    from phasesweep.runtime.process import PhaseSweepShutdown, signal_handler_scope

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

    This is the v0.5.5 reviewer's interleaving scenario: process A is on
    phase ``lr`` while process B starts a top-up of phase ``arch``. The phase
    lock would not catch this; the run lock does.
    """
    storage = f"journal:///{tmp_path / 'shared.journal'}"
    phases_a = [
        Phase(
            name="arch",
            n_trials=1,
            sampler=Sampler(type="random", seed=0),
            search_space={"depth": IntParam(type="int", low=1, high=2)},
        ),
        Phase(
            name="lr",
            inherits=["arch"],
            n_trials=1,
            sampler=Sampler(type="random", seed=1),
            search_space={
                "lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True),
            },
        ),
    ]
    phases_b = [
        Phase(
            name="arch",
            n_trials=2,  # top-up
            sampler=Sampler(type="random", seed=0),
            search_space={"depth": IntParam(type="int", low=1, high=2)},
        ),
    ]

    exp_a = make_experiment(workdir=str(tmp_path / "runs_a"), storage=storage, phases=phases_a)
    exp_b = make_experiment(workdir=str(tmp_path / "runs_b"), storage=storage, phases=phases_b)

    with (
        _experiment_lock(exp_a),
        pytest.raises(  # noqa: SIM117
            ExperimentLockBusyError, match="Another phasesweep process"
        ),
        _experiment_lock(exp_b),
    ):
        pass


def test_run_lock_collides_for_different_storage_same_output_dir(
    tmp_path: Path,
) -> None:
    """v0.5.6 missed this: two configs sharing workdir + experiment but pointing
    at different storage backends *would* collide on filesystem outputs, but
    not on the lock. v0.5.7 introduces an output-namespace lock that catches
    this (review v0.5.6 / blocker 1).
    """
    exp_a = make_experiment(
        workdir=str(tmp_path / "runs"), storage=f"sqlite:///{tmp_path / 'a.db'}"
    )
    exp_b = make_experiment(
        workdir=str(tmp_path / "runs"), storage=f"sqlite:///{tmp_path / 'b.db'}"
    )

    with (  # noqa: SIM117 — testing that the inner enter raises
        _experiment_lock(exp_a),
        pytest.raises(RuntimeError, match="output namespace|backend"),
        _experiment_lock(exp_b),
    ):
        pass


def test_run_lock_does_not_collide_for_different_experiment_dirs(
    tmp_path: Path,
) -> None:
    """Same workdir but different experiment names → different output dirs →
    no collision (output lock identities differ).
    """
    exp_a = make_experiment(
        workdir=str(tmp_path / "runs"), storage=f"sqlite:///{tmp_path / 'a.db'}"
    )
    exp_b = make_experiment(
        workdir=str(tmp_path / "runs"), storage=f"sqlite:///{tmp_path / 'b.db'}"
    )
    exp_b = exp_b.model_copy(update={"experiment": "other"})

    with _experiment_lock(exp_a), _experiment_lock(exp_b):
        pass  # must not raise


def test_run_lock_does_not_collide_for_different_experiment_names(
    tmp_path: Path,
) -> None:
    """Same storage, distinct experiment namespaces → no collision.

    A shared SQLite store can hold multiple independent experiments; locking
    them out of running concurrently would over-restrict the user.
    """
    storage = f"sqlite:///{tmp_path / 'shared.db'}"
    exp_a = make_experiment(workdir=str(tmp_path / "runs"), storage=storage)
    exp_b = make_experiment(workdir=str(tmp_path / "runs"), storage=storage)
    # make_experiment hardcodes experiment="t"; clone exp_b with another name.
    exp_b = exp_b.model_copy(update={"experiment": "other"})

    # Both output and storage lock identities differ — sets share no element.
    assert set(_run_lock_paths(exp_a)).isdisjoint(_run_lock_paths(exp_b))


def _rdb_experiment(workdir: Path, storage: str) -> Experiment:
    """Experiment with an external-RDB storage URL, bypassing the config ack.

    ``model_copy`` skips validation, which is what we want here: the RDB policy
    check (``allow_external_rdb_single_host``) is exercised in
    ``tests/test_storage_urls.py``; this file only cares about the lock path
    derived from the URL.
    """
    return make_experiment(workdir=str(workdir), storage="sqlite:///unused.db").model_copy(
        update={"storage": storage}
    )


@pytest.mark.parametrize(
    ("left_storage", "right_storage"),
    [
        (
            "postgresql://sweep:old-secret@DB.Internal/studies?a=1&b=2&application_name=x",
            "postgresql+psycopg2://sweep:new-secret@db.internal:5432/studies?b=2&a=1",
        ),
        (
            "postgresql://sweep@db.internal/studies?password=old-secret",
            "postgresql://sweep@db.internal/studies?password=new-secret",
        ),
        (
            "postgresql://sweep@db.internal/studies?access_token=old-token",
            "postgresql://sweep@db.internal/studies?access_token=new-token",
        ),
        (
            "mssql+pyodbc:///?odbc_connect="
            "SERVER%3Ddb.internal%3BDATABASE%3Dstudies%3BPWD%3Dold-secret",
            "mssql+pyodbc:///?odbc_connect="
            "SERVER%3Ddb.internal%3BDATABASE%3Dstudies%3BPWD%3Dnew-secret",
        ),
    ],
    ids=["authority", "query-password", "access-token", "nested-odbc"],
)
def test_run_lock_collides_for_equivalent_rdb_storage_urls(
    tmp_path: Path,
    left_storage: str,
    right_storage: str,
) -> None:
    """Equivalent external-RDB URLs must land on one storage lock.

    ``allow_external_rdb_single_host: true`` promises that host-local locking
    supplies all coordination for a shared RDB. Hashing the raw URL broke that
    promise: a rotated password or a reordered query split the lock namespace
    and let two orchestrators run the same study (review v0.5.17 / blocker 5).
    Distinct workdirs keep the output locks apart, so any shared path is the
    storage lock.
    """
    exp_a = _rdb_experiment(tmp_path / "runs_a", left_storage)
    exp_b = _rdb_experiment(tmp_path / "runs_b", right_storage)

    assert set(_run_lock_paths(exp_a)) & set(_run_lock_paths(exp_b))


def test_run_lock_does_not_collide_for_different_rdb_databases(tmp_path: Path) -> None:
    """Canonicalization must not over-collide: distinct databases stay independent."""
    exp_a = _rdb_experiment(tmp_path / "runs", "postgresql://sweep@db.internal/studies_a")
    exp_b = _rdb_experiment(tmp_path / "runs", "postgresql://sweep@db.internal/studies_b")
    exp_b = exp_b.model_copy(update={"experiment": "other"})

    assert set(_run_lock_paths(exp_a)).isdisjoint(_run_lock_paths(exp_b))


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


def test_output_lock_resolves_symlinked_experiment_leaf(tmp_path: Path) -> None:
    """A symlinked experiment leaf must share the target's output lock.

    ``_experiment_dir`` resolves only the workdir prefix before appending the
    experiment name, so ``runs/expA -> runs/expB`` previously minted a second
    lock identity for one physical namespace — the second orchestrator's
    preflight would then reap the first one's live trials (review v0.5.17 gap
    hunt).
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "real").mkdir()
    (runs / "alias").symlink_to(runs / "real")

    exp_real = make_experiment(workdir=str(runs)).model_copy(update={"experiment": "real"})
    exp_alias = make_experiment(workdir=str(runs)).model_copy(update={"experiment": "alias"})

    assert set(_run_lock_paths(exp_real)) == set(_run_lock_paths(exp_alias))


def test_in_memory_url_spellings_take_no_storage_lock(tmp_path: Path) -> None:
    """Every in-memory storage spelling yields only the output lock.

    ``sqlite:///:memory:``-style URLs previously produced a storage lock
    naming a backend that does not exist, so two unrelated in-memory runs
    sharing an experiment name (but nothing else) contended spuriously
    (review v0.5.17 gap hunt).
    """
    for storage in ("sqlite://", "sqlite:///:memory:", "sqlite+pysqlite:///:memory:"):
        exp = make_experiment(workdir=str(tmp_path / "runs"), storage="sqlite:///u.db")
        exp = exp.model_copy(update={"storage": storage})
        assert len(_run_lock_paths(exp)) == 1, storage


def test_run_experiment_holds_experiment_lock_for_duration(tmp_path: Path) -> None:
    """A second concurrent ``run_experiment`` against the same experiment
    fails fast with the expected error while the run lock is held.
    """
    storage = f"sqlite:///{tmp_path / 'shared.db'}"
    exp_a = make_experiment(
        workdir=str(tmp_path / "runs_a"),
        storage=storage,
        trial_command="true {overrides}",
        override_format="argparse",
        n_trials=1,
    )
    exp_b = make_experiment(
        workdir=str(tmp_path / "runs_b"),
        storage=storage,
        trial_command="true {overrides}",
        override_format="argparse",
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
    storage = f"sqlite:///{tmp_path / 'shared.db'}"
    exp_a = make_experiment(workdir=str(tmp_path / "runs_a"), storage=storage)
    exp_b = make_experiment(workdir=str(tmp_path / "runs_b"), storage=storage)

    with _experiment_lock(exp_a):
        # Dry-run must succeed with the run lock held by another caller.
        winners = run_experiment(exp_b, dry_run=True)
        assert "p" in winners
