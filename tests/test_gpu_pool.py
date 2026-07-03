"""GPU pool: explicit IDs, autodetection, host locks, and no-GPU policy."""

from __future__ import annotations

import fcntl
import time

import pytest

from phasesweep.config import (
    IntParam,
    Phase,
)
from phasesweep.runtime.gpu import GpuPool, _gpu_lock_path


def test_gpu_pool_explicit_ids_from_yaml(tmp_path):
    """gpu_ids declared in YAML must reach the GpuPool, not be silently dropped."""
    pool = GpuPool.create(n_jobs=2, explicit_ids=[7, 8])
    acquired = []
    with pool.acquire() as gid:
        acquired.append(gid)
    with pool.acquire() as gid:
        acquired.append(gid)
    assert set(acquired) == {"7", "8"}


def test_gpu_pool_fails_on_missing_gpus_parallel(monkeypatch):
    """n_jobs > 1 with no GPUs and allow_no_gpu=False must raise, not silently degrade."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # Force nvidia-smi to fail
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_ids", lambda: [])
    with pytest.raises(RuntimeError, match="no GPUs detected"):
        GpuPool.create(n_jobs=4, allow_no_gpu=False)


def test_gpu_pool_allows_no_gpu_when_opted_in(monkeypatch):
    """n_jobs > 1 with allow_no_gpu=True should warn but not crash."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_ids", lambda: [])
    pool = GpuPool.create(n_jobs=4, allow_no_gpu=True)
    with pool.acquire() as gid:
        assert gid is None


def test_explicit_gpu_ids_honored_for_single_job():
    """A single-job phase with gpu_ids=[3] must isolate to GPU 3, not no-op."""
    pool = GpuPool.create(n_jobs=1, explicit_ids=[3])
    with pool.acquire() as gid:
        assert gid == "3"


def test_whole_node_policy_assigns_all_configured_devices() -> None:
    pool = GpuPool.create(n_jobs=1, explicit_ids=[0, 1, 2], policy="whole_node")

    with pool.acquire() as gid:
        assert gid == "0,1,2"


def test_whole_node_policy_waits_for_every_host_lock(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    lock_path = _gpu_lock_path(1)
    with lock_path.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        pool = GpuPool.create(n_jobs=1, explicit_ids=[0, 1], policy="whole_node")
        with pytest.raises(TimeoutError, match="Wallclock deadline"):
            with pool.acquire(deadline=time.monotonic() + 0.02):
                pass
        fcntl.flock(held, fcntl.LOCK_UN)


def test_none_policy_disables_cuda_isolation(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")

    pool = GpuPool.create(n_jobs=1, policy="none")

    with pool.acquire() as gid:
        assert gid is None


def test_gpu_acquire_respects_deadline_when_local_slot_is_busy() -> None:
    pool = GpuPool.create(n_jobs=1, explicit_ids=[3])

    with (
        pool.acquire(),
        pytest.raises(TimeoutError, match="Wallclock deadline"),
        pool.acquire(deadline=time.monotonic() + 0.02),
    ):
        pass


def test_single_job_autodetects_and_leases_visible_gpu(monkeypatch):
    """Single-job GPU work still takes a host-wide lease when a GPU is visible."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_ids", lambda: [3, 4])

    pool = GpuPool.create(n_jobs=1)

    with pool.acquire() as gid:
        assert gid == "3"


def test_single_job_without_gpus_runs_without_isolation(monkeypatch):
    """CPU-only single-job work does not need an explicit no-GPU opt-in."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_ids", lambda: [])

    pool = GpuPool.create(n_jobs=1)

    with pool.acquire() as gid:
        assert gid is None


def test_single_job_uses_numeric_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")

    pool = GpuPool.create(n_jobs=1)

    with pool.acquire() as gid:
        assert gid == "2"


def test_nonnumeric_cuda_visible_devices_is_leased_as_opaque_token(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-deadbeef")

    pool = GpuPool.create(n_jobs=1)
    with pool.acquire() as gid:
        assert gid == "GPU-deadbeef"


def test_mig_cuda_visible_devices_get_safe_lock_names(monkeypatch, tmp_path):
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    token = "MIG-GPU-deadbeef/3/0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", token)

    pool = GpuPool.create(n_jobs=1)

    with pool.acquire() as gid:
        assert gid == token
        locks = list(tmp_path.glob("gpu_*.lock"))
        assert len(locks) == 1
        assert "/" not in locks[0].name
        assert locks[0].name.startswith("gpu_MIG-GPU-deadbeef_3_0_")


def test_cuda_visible_devices_minus_one_is_no_visible_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    pool = GpuPool.create(n_jobs=1)

    with pool.acquire() as gid:
        assert gid is None


def test_explicit_gpu_devices_preserve_tokens_and_dedupe():
    pool = GpuPool.create(
        n_jobs=2,
        explicit_devices=["GPU-a", "MIG-GPU-b/1/0", "GPU-a"],
    )
    acquired = []
    with pool.acquire() as gid:
        acquired.append(gid)
    with pool.acquire() as gid:
        acquired.append(gid)
    assert acquired == ["GPU-a", "MIG-GPU-b/1/0"]


def test_empty_explicit_gpu_ids_raises():
    """gpu_ids=[] is a config error, not silent no-op."""
    with pytest.raises(RuntimeError, match="gpu_ids was provided but empty"):
        GpuPool.create(n_jobs=1, explicit_ids=[])


def test_explicit_gpu_ids_dedupe_preserves_order():
    """Duplicate IDs in YAML are deduped without reordering."""
    pool = GpuPool.create(n_jobs=2, explicit_ids=[2, 0, 2, 1, 0])
    assert pool._gpu_ids == [2, 0, 1]


def test_gpu_pool_skips_host_locked_gpu(tmp_path, monkeypatch) -> None:
    """A second phasesweep process must not double-book a host-locked GPU."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    lock_path = _gpu_lock_path(3)
    holder_marker = "holder-pid\n"
    with lock_path.open("w") as held:
        held.write(holder_marker)
        held.flush()
        fcntl.flock(held, fcntl.LOCK_EX)
        pool = GpuPool.create(n_jobs=1, explicit_ids=[3, 4])
        with pool.acquire() as gid:
            assert gid == "4"
        assert lock_path.read_text() == holder_marker
        fcntl.flock(held, fcntl.LOCK_UN)


def test_gpu_ids_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative CUDA device indices"):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
            search_space={"x": IntParam(type="int", low=0, high=1)},
            gpu_ids=[0, -1, 2],
        )


def test_gpu_ids_accepts_none_and_non_empty_values() -> None:
    Phase(  # type: ignore[arg-type]
        name="p1",
        n_trials=1,
        search_space={"x": IntParam(type="int", low=0, high=1)},
        gpu_ids=None,
    )
    Phase(  # type: ignore[arg-type]
        name="p2",
        n_trials=1,
        search_space={"x": IntParam(type="int", low=0, high=1)},
        gpu_ids=[0, 1, 2],
    )


def test_gpu_ids_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one CUDA device index"):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
            search_space={"x": IntParam(type="int", low=0, high=1)},
            gpu_ids=[],
        )


def test_gpu_devices_rejects_ambiguous_tokens() -> None:
    with pytest.raises(ValueError, match="gpu_devices entries"):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
            search_space={"x": IntParam(type="int", low=0, high=1)},
            gpu_devices=["GPU-ok", "bad,token"],
        )


def test_gpu_ids_and_gpu_devices_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
            search_space={"x": IntParam(type="int", low=0, high=1)},
            gpu_ids=[0],
            gpu_devices=["GPU-deadbeef"],
        )


def test_gpu_policy_whole_node_requires_single_job() -> None:
    with pytest.raises(ValueError, match="whole_node.*requires n_jobs=1"):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
            n_jobs=2,
            gpu_policy="whole_node",
            search_space={"x": IntParam(type="int", low=0, high=1)},
        )


def test_gpu_policy_none_parallel_requires_explicit_no_isolation_opt_in() -> None:
    with pytest.raises(ValueError, match="gpu_policy='none'.*allow_no_gpu_isolation"):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
            n_jobs=2,
            gpu_policy="none",
            search_space={"x": IntParam(type="int", low=0, high=1)},
        )

    Phase(  # type: ignore[arg-type]
        name="p",
        n_trials=1,
        n_jobs=2,
        gpu_policy="none",
        allow_no_gpu_isolation=True,
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )


def test_gpu_policy_none_rejects_explicit_gpu_lists() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
            gpu_policy="none",
            gpu_ids=[0],
            search_space={"x": IntParam(type="int", low=0, high=1)},
        )
