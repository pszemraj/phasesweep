"""GPU pool: explicit-id resolution, autodetection, parallel-no-GPU rejection, allow_no_gpu_isolation opt-in."""

from __future__ import annotations

import fcntl

import pytest

from phasesweep.config import (
    IntParam,
    Phase,
)
from phasesweep.gpu_pool import GpuPool, _gpu_lock_path


def test_gpu_pool_explicit_ids_from_yaml(tmp_path):
    """gpu_ids declared in YAML must reach the GpuPool, not be silently dropped."""
    pool = GpuPool.create(n_jobs=2, explicit_ids=[7, 8])
    acquired = []
    with pool.acquire() as gid:
        acquired.append(gid)
    with pool.acquire() as gid:
        acquired.append(gid)
    assert set(acquired) == {7, 8}


def test_gpu_pool_fails_on_missing_gpus_parallel(monkeypatch):
    """n_jobs > 1 with no GPUs and allow_no_gpu=False must raise, not silently degrade."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # Force nvidia-smi to fail
    monkeypatch.setattr("phasesweep.gpu_pool._detect_gpu_ids", lambda: [])
    with pytest.raises(RuntimeError, match="no GPUs detected"):
        GpuPool.create(n_jobs=4, allow_no_gpu=False)


def test_gpu_pool_allows_no_gpu_when_opted_in(monkeypatch):
    """n_jobs > 1 with allow_no_gpu=True should warn but not crash."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("phasesweep.gpu_pool._detect_gpu_ids", lambda: [])
    pool = GpuPool.create(n_jobs=4, allow_no_gpu=True)
    assert not pool.active
    with pool.acquire() as gid:
        assert gid is None


def test_explicit_gpu_ids_honored_for_single_job():
    """A single-job phase with gpu_ids=[3] must isolate to GPU 3, not no-op."""
    pool = GpuPool.create(n_jobs=1, explicit_ids=[3])
    assert pool.active, "Pool should be active when explicit IDs are set, even at n_jobs=1"
    with pool.acquire() as gid:
        assert gid == 3


def test_empty_explicit_gpu_ids_raises():
    """gpu_ids=[] is a config error, not silent no-op."""
    with pytest.raises(RuntimeError, match="gpu_ids was provided but empty"):
        GpuPool.create(n_jobs=1, explicit_ids=[])


def test_explicit_gpu_ids_dedupe_preserves_order():
    """Duplicate IDs in YAML are deduped without reordering."""
    pool = GpuPool.create(n_jobs=2, explicit_ids=[2, 0, 2, 1, 0])
    assert pool._gpu_ids == [2, 0, 1]


def test_gpu_pool_skips_host_locked_gpu() -> None:
    """A second phasesweep process must not double-book a host-locked GPU."""
    lock_path = _gpu_lock_path(3)
    with lock_path.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        pool = GpuPool.create(n_jobs=1, explicit_ids=[3, 4])
        with pool.acquire() as gid:
            assert gid == 4
        fcntl.flock(held, fcntl.LOCK_UN)


def test_gpu_ids_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative CUDA device indices"):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
            search_space={"x": IntParam(type="int", low=0, high=1)},
            gpu_ids=[0, -1, 2],
        )


def test_gpu_ids_accepts_empty_and_none() -> None:
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
