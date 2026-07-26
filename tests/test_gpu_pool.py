"""GPU pool: explicit IDs, autodetection, host locks, and no-GPU policy."""

from __future__ import annotations

import fcntl
import shutil
import time

import pytest

from phasesweep.config import (
    IntParam,
    Phase,
)
from phasesweep.runtime.files import open_lock_file
from phasesweep.runtime.gpu import GpuDevice, GpuPool, _detect_gpu_uuid_map, _gpu_lock_path

# Bound before any monkeypatching so hardware tests can restore the real probe.
_real_detect_gpu_uuid_map = _detect_gpu_uuid_map


@pytest.fixture(autouse=True)
def unreadable_gpu_uuid_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to an unreadable ``nvidia-smi`` index-to-UUID map.

    Canonical lock identity (review v0.5.17 / blocker 6) is resolved from that
    map, so without this stub the fixed device indices used throughout this file
    would resolve differently — or fail closed as nonexistent — depending on
    which GPUs the test host happens to have. Tests that exercise resolution
    re-patch the probe themselves.
    """
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {})


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
    with open_lock_file(lock_path) as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        pool = GpuPool.create(n_jobs=1, explicit_ids=[0, 1], policy="whole_node")
        with (
            pytest.raises(TimeoutError, match="Wallclock deadline"),
            pool.acquire(deadline=time.monotonic() + 0.02),
        ):
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


def test_explicit_gpu_ids_dedupe_preserves_order():
    """Duplicate IDs in YAML are deduped without reordering."""
    pool = GpuPool.create(n_jobs=2, explicit_ids=[2, 0, 2, 1, 0])
    acquired = []
    for _ in range(3):
        with pool.acquire() as gpu_id:
            acquired.append(gpu_id)
    assert acquired == ["2", "0", "1"]


def test_gpu_pool_skips_host_locked_gpu(tmp_path, monkeypatch) -> None:
    """A second phasesweep process must not double-book a host-locked GPU."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    lock_path = _gpu_lock_path(3)
    holder_marker = "holder-pid\n"
    with open_lock_file(lock_path) as held:
        held.write(holder_marker)
        held.flush()
        fcntl.flock(held, fcntl.LOCK_EX)
        pool = GpuPool.create(n_jobs=1, explicit_ids=[3, 4])
        with pool.acquire() as gid:
            assert gid == "4"
        assert lock_path.read_text() == holder_marker
        fcntl.flock(held, fcntl.LOCK_UN)


def test_numeric_index_and_uuid_token_lock_the_same_physical_gpu(tmp_path, monkeypatch) -> None:
    """gpu_ids=[0] and the UUID of GPU 0 must contend for one host lock, not two."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    uuid = "GPU-32ad40d6-019f-386a-321d-3901216c78ad"
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})

    by_index = GpuPool.create(n_jobs=1, explicit_ids=[0])
    by_uuid = GpuPool.create(n_jobs=1, explicit_devices=[uuid])

    assert _gpu_lock_path(by_index._devices[0]) == _gpu_lock_path(by_uuid._devices[0])

    with by_index.acquire() as gid:
        # The trainer still sees the token the operator configured.
        assert gid == "0"
        with (
            pytest.raises(TimeoutError, match="Wallclock deadline"),
            by_uuid.acquire(deadline=time.monotonic() + 0.02),
        ):
            pass
        assert len(list(tmp_path.glob("gpu_*.lock"))) == 1


def test_autodetected_indices_resolve_to_the_same_lock_as_uuid_tokens(tmp_path, monkeypatch):
    """Auto-detected numeric indices go through the same canonical resolution."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    uuid = "GPU-deadbeef"
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_ids", lambda: [0])
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})

    detected = GpuPool.create(n_jobs=1)

    assert _gpu_lock_path(detected._devices[0]) == _gpu_lock_path(GpuDevice(uuid))
    with detected.acquire() as gid:
        assert gid == "0"


def test_ambient_uuid_and_configured_index_share_one_lock(tmp_path, monkeypatch) -> None:
    """A UUID inherited via CUDA_VISIBLE_DEVICES cannot double-book a configured index."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    uuid = "GPU-deadbeef"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})

    ambient = GpuPool.create(n_jobs=1)
    configured = GpuPool.create(n_jobs=1, explicit_ids=[0])

    with ambient.acquire() as gid:
        assert gid == uuid
        with (
            pytest.raises(TimeoutError, match="Wallclock deadline"),
            configured.acquire(deadline=time.monotonic() + 0.02),
        ):
            pass


def test_duplicate_spellings_of_one_gpu_are_leased_once(monkeypatch) -> None:
    """Two tokens naming one card collapse to a single device instead of deadlocking."""
    uuid = "GPU-deadbeef"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"0,{uuid}")
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})

    pool = GpuPool.create(n_jobs=1)

    assert [device.visible_token for device in pool._devices] == ["0"]


def test_mig_token_locks_on_the_mig_instance(tmp_path, monkeypatch) -> None:
    """MIG instances lock on themselves; parent-GPU binding is a documented gap."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    uuid = "GPU-deadbeef"
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})
    mig_token = "MIG-GPU-deadbeef/3/0"

    mig = GpuPool.create(n_jobs=1, explicit_devices=[mig_token])
    parent = GpuPool.create(n_jobs=1, explicit_ids=[0])

    assert mig._devices[0].lock_identity == mig_token
    assert _gpu_lock_path(mig._devices[0]) != _gpu_lock_path(parent._devices[0])
    with mig.acquire() as mig_gid, parent.acquire() as parent_gid:
        assert mig_gid == mig_token
        assert parent_gid == "0"


def test_unreadable_uuid_map_keeps_numeric_lock_names(tmp_path, monkeypatch) -> None:
    """No nvidia-smi plus an all-numeric device set keeps the pre-existing behavior."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {})

    pool = GpuPool.create(n_jobs=2, explicit_ids=[0, 1])

    assert [_gpu_lock_path(device).name for device in pool._devices] == [
        "gpu_0.lock",
        "gpu_1.lock",
    ]


def test_unreadable_uuid_map_rejects_mixed_token_forms(monkeypatch) -> None:
    """Mixed indices and opaque tokens without nvidia-smi fail closed, not silently."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,GPU-deadbeef")
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {})

    with pytest.raises(RuntimeError, match="one convention"):
        GpuPool.create(n_jobs=1)


def test_unknown_numeric_index_fails_closed(monkeypatch) -> None:
    """A configured index nvidia-smi does not report is a nonexistent device."""
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_uuid_map",
        lambda: {"0": "GPU-deadbeef"},
    )

    with pytest.raises(RuntimeError, match="index 3 does not exist"):
        GpuPool.create(n_jobs=1, explicit_ids=[3])


def test_resolved_uuid_lock_names_stay_path_safe(tmp_path, monkeypatch) -> None:
    """A resolved identity is sanitized and hashed exactly like an opaque token."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_uuid_map",
        lambda: {"0": "GPU-dead/beef:0"},
    )

    pool = GpuPool.create(n_jobs=1, explicit_ids=[0])

    with pool.acquire():
        locks = list(tmp_path.glob("gpu_*.lock"))
        assert len(locks) == 1
        assert "/" not in locks[0].name
        assert locks[0].name.startswith("gpu_GPU-dead_beef_0_")


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="nvidia-smi is not installed")
def test_real_nvidia_smi_resolves_index_zero_to_a_uuid(monkeypatch) -> None:
    """On real GPU hardware, index 0 locks under the UUID nvidia-smi reports for it."""
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", _real_detect_gpu_uuid_map)
    uuid_map = _real_detect_gpu_uuid_map()
    if not uuid_map:
        pytest.skip("nvidia-smi is installed but reports no CUDA devices")

    assert uuid_map["0"].startswith("GPU-")

    pool = GpuPool.create(n_jobs=1, explicit_ids=[0])

    assert pool._devices[0].visible_token == "0"
    assert pool._devices[0].lock_identity == uuid_map["0"]
    assert _gpu_lock_path(pool._devices[0]) == _gpu_lock_path(GpuDevice(uuid_map["0"]))


def test_gpu_ids_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative CUDA device indices"):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
            search_space={"x": IntParam(type="int", low=0, high=1)},
            gpu_ids=[0, -1, 2],
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
