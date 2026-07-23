"""Same-host advisory run locks: output namespace and storage identity."""

from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import pytest

from phasesweep.config import FloatParam, IntParam, Phase
from phasesweep.engine import run_experiment
from phasesweep.engine.errors import ExperimentLockBusyError
from phasesweep.engine.guards import (
    _experiment_lock,
    _run_lock_paths,
)
from phasesweep.runtime import files as runtime_files
from tests.conftest import make_experiment


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
            search_space={"depth": IntParam(type="int", low=1, high=2)},
        ),
        Phase(
            name="lr",
            inherits=["arch"],
            n_trials=1,
            search_space={
                "lr": FloatParam(type="float", low=1e-5, high=1e-3, log=True),
            },
        ),
    ]
    phases_b = [
        Phase(
            name="arch",
            n_trials=2,  # top-up
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


def test_in_memory_run_lock_keyed_by_workdir(tmp_path: Path) -> None:
    """In-memory storage: only the output lock applies (no shared backend).
    Two same-workdir+experiment configs share the same output lock path.
    """
    exp_a = make_experiment(workdir=str(tmp_path / "runs"))
    exp_b = make_experiment(workdir=str(tmp_path / "runs"))

    paths_a = _run_lock_paths(exp_a)
    paths_b = _run_lock_paths(exp_b)
    # In-memory storage means only the output lock is taken — single path.
    assert len(paths_a) == 1
    assert paths_a == paths_b


def test_in_memory_run_lock_does_not_collide_for_different_workdirs(
    tmp_path: Path,
) -> None:
    """In-memory storage with different workdirs = independent output dirs."""
    exp_a = make_experiment(workdir=str(tmp_path / "runs_a"))
    exp_b = make_experiment(workdir=str(tmp_path / "runs_b"))

    assert set(_run_lock_paths(exp_a)).isdisjoint(_run_lock_paths(exp_b))


def test_run_experiment_holds_experiment_lock_for_duration(tmp_path: Path) -> None:
    """A second concurrent ``run_experiment`` against the same experiment
    fails fast with the expected error while the run lock is held.
    """
    storage = f"sqlite:///{tmp_path / 'shared.db'}"
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
    storage = f"sqlite:///{tmp_path / 'shared.db'}"
    exp_a = make_experiment(workdir=str(tmp_path / "runs_a"), storage=storage)
    exp_b = make_experiment(workdir=str(tmp_path / "runs_b"), storage=storage)

    with _experiment_lock(exp_a):
        # Dry-run must succeed with the run lock held by another caller.
        winners = run_experiment(exp_b, dry_run=True)
        assert "p" in winners
