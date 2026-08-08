"""GPU pool: explicit IDs, autodetection, host locks, and no-GPU policy."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shlex
import shutil
import signal
import sys
import time
from contextlib import suppress

import pytest

from phasesweep.config import (
    IntParam,
    Phase,
)
from phasesweep.runtime.files import open_lock_file
from phasesweep.runtime.gpu import (
    GpuDevice,
    GpuLeaseTimeoutError,
    GpuPool,
    _detect_gpu_inventory,
    _detect_gpu_uuid_map,
    _gpu_lock_path,
    _try_host_gpu_lease,
)
from phasesweep.runtime.process import run_supervised

# Bound before any monkeypatching so hardware tests can restore the real probe.
_real_detect_gpu_inventory = _detect_gpu_inventory
_real_detect_gpu_uuid_map = _detect_gpu_uuid_map
_TEST_UUID_MAP = {str(index): f"GPU-test-{index}" for index in range(16)}


@pytest.fixture(autouse=True)
def deterministic_gpu_uuid_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to a deterministic ``nvidia-smi`` UUID map.

    Canonical lock identity (review v0.5.17 / blocker 6) is resolved from that
    map, so without this stub the fixed device indices used throughout this file
    would resolve differently — or fail closed as nonexistent — depending on
    which GPUs the test host happens to have. Tests that exercise resolution
    re-patch the probe themselves.
    """
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_uuid_map",
        lambda: dict(_TEST_UUID_MAP),
    )
    monkeypatch.setattr("phasesweep.runtime.gpu._nvidia_driver_reports_gpus", lambda: False)


def test_gpu_pool_fails_on_missing_gpus_parallel(monkeypatch):
    """n_jobs > 1 with no GPUs and allow_no_gpu=False must raise, not silently degrade."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # Force nvidia-smi to fail
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))
    with pytest.raises(RuntimeError, match="no GPUs detected"):
        GpuPool.create(n_jobs=4, allow_no_gpu=False)


def test_gpu_pool_allows_no_gpu_when_opted_in(monkeypatch):
    """n_jobs > 1 with allow_no_gpu=True should warn but not crash."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))
    pool = GpuPool.create(n_jobs=4, allow_no_gpu=True)
    with pool.acquire() as gid:
        assert gid.visible_devices is None
        assert gid.lease_fds == ()


def test_gpu_pool_create_normalizes_explicit_device_tokens(monkeypatch) -> None:
    uuid = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})
    pool = GpuPool.create(n_jobs=1, explicit_devices=[f" {uuid} "])

    assert [device.visible_token for device in pool._devices] == [uuid]


def test_explicit_gpu_ids_honored_for_single_job():
    """A single-job phase with gpu_ids=[3] must isolate to GPU 3, not no-op."""
    pool = GpuPool.create(n_jobs=1, explicit_ids=[3])
    with pool.acquire() as gid:
        assert gid.visible_devices == "3"
        assert len(gid.lease_fds) == 1


def test_whole_node_policy_assigns_all_configured_devices() -> None:
    pool = GpuPool.create(n_jobs=1, explicit_ids=[0, 1, 2], policy="whole_node")

    with pool.acquire() as gid:
        assert gid.visible_devices == "0,1,2"
        assert len(gid.lease_fds) == 3


def test_whole_node_policy_waits_for_every_host_lock(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    lock_path = _gpu_lock_path(GpuDevice("1", _TEST_UUID_MAP["1"]))
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
        assert gid.visible_devices is None
        assert gid.lease_fds == ()


def test_gpu_acquire_respects_deadline_when_local_slot_is_busy() -> None:
    pool = GpuPool.create(n_jobs=1, explicit_ids=[3])

    # The dedicated subclass, not just TimeoutError: phase attribution narrows
    # on it to tell budget exhaustion from infrastructure failure.
    with (
        pool.acquire(),
        pytest.raises(GpuLeaseTimeoutError, match="Wallclock deadline"),
        pool.acquire(deadline=time.monotonic() + 0.02),
    ):
        pass

    assert issubclass(GpuLeaseTimeoutError, TimeoutError)


def test_single_job_autodetects_and_leases_visible_gpu(monkeypatch):
    """Single-job GPU work still takes a host-wide lease when a GPU is visible."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_inventory",
        lambda: ([3, 4], {"3": _TEST_UUID_MAP["3"], "4": _TEST_UUID_MAP["4"]}),
    )

    pool = GpuPool.create(n_jobs=1)

    with pool.acquire() as gid:
        assert gid.visible_devices == "3"


def test_single_job_without_gpus_runs_without_isolation(monkeypatch):
    """CPU-only single-job work does not need an explicit no-GPU opt-in."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))

    pool = GpuPool.create(n_jobs=1)

    with pool.acquire() as gid:
        assert gid.visible_devices is None


def test_single_job_fails_when_nvidia_smi_is_broken_on_gpu_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))
    monkeypatch.setattr("phasesweep.runtime.gpu._nvidia_driver_reports_gpus", lambda: True)

    with pytest.raises(RuntimeError, match="could not enumerate GPUs.*double-book"):
        GpuPool.create(n_jobs=1)


def test_broken_nvidia_smi_requires_explicit_fail_open_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))
    monkeypatch.setattr("phasesweep.runtime.gpu._nvidia_driver_reports_gpus", lambda: True)

    with caplog.at_level(logging.WARNING, logger="phasesweep.runtime.gpu"):
        pool = GpuPool.create(n_jobs=1, allow_no_gpu=True)

    with pool.acquire() as assignment:
        assert assignment.visible_devices is None
    assert pool._devices == []
    assert any("allow_no_gpu_isolation: true" in record.message for record in caplog.records)


def test_single_job_uses_numeric_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")

    pool = GpuPool.create(n_jobs=1)

    with pool.acquire() as gid:
        assert gid.visible_devices == "2"


def test_nonnumeric_cuda_visible_devices_is_leased_as_opaque_token(monkeypatch):
    uuid = "GPU-deadbeef"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})

    pool = GpuPool.create(n_jobs=1)
    with pool.acquire() as gid:
        assert gid.visible_devices == uuid


def test_mig_cuda_visible_devices_get_safe_lock_names(monkeypatch, tmp_path):
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    token = "MIG-GPU-deadbeef/3/0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", token)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_mig_uuid_set", lambda: {token})

    pool = GpuPool.create(n_jobs=1)

    with pool.acquire() as gid:
        assert gid.visible_devices == token
        locks = list(tmp_path.glob("gpu_*.lock"))
        assert len(locks) == 1
        assert "/" not in locks[0].name
        assert locks[0].name.startswith("gpu_MIG-GPU-deadbeef_3_0_")


def test_cuda_visible_devices_minus_one_is_no_visible_gpu(monkeypatch, caplog):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr("phasesweep.runtime.gpu._nvidia_driver_reports_gpus", lambda: True)

    with caplog.at_level(logging.INFO, logger="phasesweep.runtime.gpu"):
        pool = GpuPool.create(n_jobs=1)

    with pool.acquire() as gid:
        # The sentinel is pinned, not dropped: a narrowed inherit_env contract
        # must not silently re-expose every host GPU to the trainer.
        assert gid.visible_devices == "-1"
    assert any("exposes no devices" in record.message for record in caplog.records)
    assert not any("nvidia-smi detected no GPUs" in record.message for record in caplog.records)


def test_configured_cuda_visibility_overrides_ambient_for_pool(monkeypatch) -> None:
    """Pool locks follow the trainer's configured environment override."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": "GPU-configured"}
    )

    pool = GpuPool.create(n_jobs=1, cuda_visible_devices="GPU-configured")

    with pool.acquire() as gid:
        assert gid.visible_devices == "GPU-configured"


def test_explicit_gpu_devices_preserve_tokens_and_dedupe(monkeypatch):
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": "GPU-a"})
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_mig_uuid_set", lambda: {"MIG-GPU-b/1/0"})
    pool = GpuPool.create(
        n_jobs=2,
        explicit_devices=["GPU-a", "MIG-GPU-b/1/0", "GPU-a"],
    )
    acquired = []
    with pool.acquire() as gid:
        acquired.append(gid.visible_devices)
    with pool.acquire() as gid:
        acquired.append(gid.visible_devices)
    assert acquired == ["GPU-a", "MIG-GPU-b/1/0"]


def test_explicit_gpu_ids_dedupe_preserves_order():
    """Duplicate IDs in YAML are deduped without reordering."""
    pool = GpuPool.create(n_jobs=2, explicit_ids=[2, 0, 2, 1, 0])
    acquired = []
    for _ in range(3):
        with pool.acquire() as gpu_id:
            acquired.append(gpu_id.visible_devices)
    assert acquired == ["2", "0", "1"]


def test_gpu_pool_skips_host_locked_gpu(tmp_path, monkeypatch) -> None:
    """A second phasesweep process must not double-book a host-locked GPU."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    lock_path = _gpu_lock_path(GpuDevice("3", _TEST_UUID_MAP["3"]))
    holder_marker = "holder-pid\n"
    with open_lock_file(lock_path) as held:
        held.write(holder_marker)
        held.flush()
        fcntl.flock(held, fcntl.LOCK_EX)
        pool = GpuPool.create(n_jobs=1, explicit_ids=[3, 4])
        with pool.acquire() as gid:
            assert gid.visible_devices == "4"
        assert lock_path.read_text() == holder_marker
        fcntl.flock(held, fcntl.LOCK_UN)


def test_gpu_lease_survives_orchestrator_hard_exit(tmp_path, monkeypatch) -> None:
    """A guardian retains the lock through FD scrubbing, parent death, and descendants."""
    locks = tmp_path / "locks"
    locks.mkdir()
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: locks)
    started = tmp_path / "trainer_started"
    worker = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    trainer = (
        "import json, os, subprocess, sys, time; "
        "os.closerange(3, 1048576); "
        f"worker = subprocess.Popen([sys.executable, '-c', {worker!r}]); "
        "state = {'root_pid': os.getpid(), 'pgid': os.getpgrp(), "
        "'worker_pid': worker.pid}; "
        f"open({str(started)!r}, 'w').write(json.dumps(state)); "
        "time.sleep(0.3)"
    )
    command = shlex.join([sys.executable, "-c", trainer])
    orchestrator_pid = os.fork()
    reaped = False
    trainer_pgid: int | None = None
    worker_pid: int | None = None
    if orchestrator_pid == 0:
        try:
            trial_dir = tmp_path / "trial"
            trial_dir.mkdir()
            pool = GpuPool.create(n_jobs=1, explicit_ids=[0])
            with (
                pool.acquire() as assignment,
                (trial_dir / "stdout.log").open("w") as stdout,
                (trial_dir / "stderr.log").open("w") as stderr,
            ):
                run_supervised(
                    command,
                    env=os.environ.copy(),
                    stdout=stdout,
                    stderr=stderr,
                    timeout=10.0,
                    trial_dir=trial_dir,
                    attempt_id="hard-exit-attempt",
                    gpu_lease_fds=assignment.lease_fds,
                )
        except BaseException:
            os._exit(1)
        os._exit(0)

    try:
        marker_deadline = time.monotonic() + 5.0
        while not started.is_file() and time.monotonic() < marker_deadline:
            exited, _ = os.waitpid(orchestrator_pid, os.WNOHANG)
            if exited:
                reaped = True
                pytest.fail("orchestrator exited before the trainer started")
            time.sleep(0.01)
        assert started.is_file()
        state = json.loads(started.read_text())
        root_pid = int(state["root_pid"])
        trainer_pgid = int(state["pgid"])
        worker_pid = int(state["worker_pid"])

        os.kill(orchestrator_pid, signal.SIGKILL)
        os.waitpid(orchestrator_pid, 0)
        reaped = True

        # The direct trainer exits, but its SIGTERM-ignoring worker remains.
        # The old guardian released the lease at this exact boundary.
        root_exit_deadline = time.monotonic() + 2.0
        while time.monotonic() < root_exit_deadline:
            try:
                os.kill(root_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            pytest.fail("trainer root did not exit before descendant lease check")
        os.kill(worker_pid, 0)

        competitor = GpuPool.create(n_jobs=1, explicit_ids=[0])
        with (
            pytest.raises(GpuLeaseTimeoutError, match="Wallclock deadline"),
            competitor.acquire(deadline=time.monotonic() + 0.1),
        ):
            pass

        os.kill(worker_pid, signal.SIGKILL)
        worker_pid = None
        with competitor.acquire(deadline=time.monotonic() + 5.0) as assignment:
            assert assignment.visible_devices == "0"
    finally:
        if not reaped:
            with suppress(ProcessLookupError):
                os.kill(orchestrator_pid, signal.SIGKILL)
            with suppress(ChildProcessError):
                os.waitpid(orchestrator_pid, 0)
        if trainer_pgid is not None and trainer_pgid != os.getpgrp():
            with suppress(ProcessLookupError):
                os.killpg(trainer_pgid, signal.SIGKILL)
        if worker_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(worker_pid, signal.SIGKILL)


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
        assert gid.visible_devices == "0"
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
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_inventory",
        lambda: ([0], {"0": uuid}),
    )

    detected = GpuPool.create(n_jobs=1)

    assert _gpu_lock_path(detected._devices[0]) == _gpu_lock_path(GpuDevice(uuid))
    with detected.acquire() as gid:
        assert gid.visible_devices == "0"


def test_ambient_uuid_and_configured_index_share_one_lock(tmp_path, monkeypatch) -> None:
    """A UUID inherited via CUDA_VISIBLE_DEVICES cannot double-book a configured index."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    uuid = "GPU-deadbeef"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})

    ambient = GpuPool.create(n_jobs=1)
    configured = GpuPool.create(n_jobs=1, explicit_ids=[0])

    with ambient.acquire() as gid:
        assert gid.visible_devices == uuid
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
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_mig_uuid_set", lambda: {mig_token})

    mig = GpuPool.create(n_jobs=1, explicit_devices=[mig_token])
    parent = GpuPool.create(n_jobs=1, explicit_ids=[0])

    assert mig._devices[0].lock_identity == mig_token
    assert _gpu_lock_path(mig._devices[0]) != _gpu_lock_path(parent._devices[0])
    with mig.acquire() as mig_gid, parent.acquire() as parent_gid:
        assert mig_gid.visible_devices == mig_token
        assert parent_gid.visible_devices == "0"


def test_unreadable_uuid_map_rejects_numeric_tokens(monkeypatch) -> None:
    """An unverified index can expose nothing and split the host lock namespace."""
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {})

    with pytest.raises(RuntimeError, match="Cannot validate configured CUDA device index"):
        GpuPool.create(n_jobs=2, explicit_ids=[0, 1])


def test_unreadable_uuid_map_rejects_mixed_token_forms(monkeypatch) -> None:
    """Mixed indices and opaque tokens without nvidia-smi fail closed, not silently."""
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {})

    with pytest.raises(RuntimeError, match="Cannot validate configured GPU UUID"):
        GpuPool.create(n_jobs=1, explicit_devices=["0", "GPU-deadbeef"])


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


def test_uuid_map_parser_rejects_placeholder_values(monkeypatch) -> None:
    """Driver placeholders like "[N/A]" must not become device identities.

    Restricted drivers (vGPU, locked-down containers) report the same
    placeholder for every index; accepting it would map ALL indices to one
    identity and silently collapse a multi-GPU pool to a single lock
    (review v0.5.17 gap hunt)."""

    class _Out:
        returncode = 0
        stdout = "0, [N/A]\n1, [Not Supported]\n2, GPU-abc123\n3, MIG-def456\n"

    monkeypatch.setattr("phasesweep.runtime.gpu.subprocess.run", lambda *args, **kwargs: _Out())

    assert _real_detect_gpu_inventory() == (
        [0, 1, 2, 3],
        {"2": "GPU-abc123", "3": "MIG-def456"},
    )


def test_abbreviated_uuid_prefix_shares_the_full_uuid_lock(tmp_path, monkeypatch) -> None:
    """CUDA accepts unambiguous UUID prefixes, so "GPU-2b23" and index 0 must
    lock the same card as the full UUID spelling (review v0.5.17 gap hunt)."""
    uuid = "GPU-2b234567-89ab-cdef-0123-456789abcdef"
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})

    prefix_pool = GpuPool.create(n_jobs=1, explicit_devices=["GPU-2b23"])
    index_pool = GpuPool.create(n_jobs=1, explicit_ids=[0])
    full_pool = GpuPool.create(n_jobs=1, explicit_devices=[uuid])

    prefix_lock = _gpu_lock_path(prefix_pool._devices[0])
    assert prefix_lock == _gpu_lock_path(index_pool._devices[0])
    assert prefix_lock == _gpu_lock_path(full_pool._devices[0])
    # CUDA receives the inventory spelling, not an abbreviated alias.
    assert prefix_pool._devices[0].visible_token == uuid


def test_abbreviated_uuid_prefix_without_map_fails_closed(tmp_path, monkeypatch) -> None:
    """An unverified UUID prefix cannot reserve a lock for a physical device."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {})

    with pytest.raises(RuntimeError, match="Cannot validate configured GPU UUID"):
        GpuPool.create(n_jobs=1, explicit_devices=["GPU-2b23"])


def test_lowercase_abbreviated_uuid_prefix_shares_the_full_uuid_lock(tmp_path, monkeypatch) -> None:
    """The resolver compares UUIDs case-insensitively, so the abbreviation gate
    must too: "gpu-2b23" locking on its own spelling would double-book the card
    against a run configured with gpu_ids: [0]."""
    uuid = "GPU-2b234567-89ab-cdef-0123-456789abcdef"
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})

    prefix_pool = GpuPool.create(n_jobs=1, explicit_devices=["gpu-2b23"])
    index_pool = GpuPool.create(n_jobs=1, explicit_ids=[0])

    assert _gpu_lock_path(prefix_pool._devices[0]) == _gpu_lock_path(index_pool._devices[0])
    # PyTorch's CUDA visibility parser rejects a lowercase prefix.
    assert prefix_pool._devices[0].visible_token == uuid


def test_lowercase_full_uuid_shares_the_numeric_index_lock(tmp_path, monkeypatch) -> None:
    """Full UUID spellings are case-insensitive for both validation and locking."""
    uuid = "GPU-2b234567-89ab-cdef-0123-456789abcdef"
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})

    lowercase = GpuPool.create(n_jobs=1, explicit_devices=[uuid.lower()])
    by_index = GpuPool.create(n_jobs=1, explicit_ids=[0])

    assert _gpu_lock_path(lowercase._devices[0]) == _gpu_lock_path(by_index._devices[0])
    assert lowercase._devices[0].visible_token == uuid


def test_lowercase_mig_uuid_rebinds_trainer_visibility_to_inventory_spelling(
    monkeypatch,
) -> None:
    """A lowercase MIG alias must not make a correctly locked device invisible."""
    mig_uuid = "MIG-2b234567-89ab-cdef-0123-456789abcdef"
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_mig_uuid_set", lambda: {mig_uuid})

    pool = GpuPool.create(n_jobs=1, explicit_devices=[mig_uuid.lower()])

    assert pool._devices[0].visible_token == mig_uuid
    assert pool._devices[0].lock_identity == mig_uuid


def test_whole_node_rejects_case_only_uuid_aliases(monkeypatch) -> None:
    uuid = "GPU-2b234567-89ab-cdef-0123-456789abcdef"
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})

    with pytest.raises(RuntimeError, match="only 1 distinct physical GPU"):
        GpuPool.create(
            n_jobs=1,
            explicit_devices=[uuid, uuid.lower()],
            policy="whole_node",
        )


@pytest.mark.parametrize("prefix", ["GPU-", "MIG-"])
def test_nonexistent_uuid_token_fails_closed(prefix: str, monkeypatch) -> None:
    token = f"{prefix}00000000-0000-0000-0000-000000000000"
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {})
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_mig_uuid_set", lambda: set())

    with pytest.raises(RuntimeError, match="no usable (?:GPU UUID|MIG)|matches no device"):
        GpuPool.create(n_jobs=1, explicit_devices=[token])


def test_unresolvable_opaque_token_fails_closed_when_explicit(monkeypatch, caplog) -> None:
    """A token that is neither numeric nor GPU-/MIG- shaped gives CUDA nothing
    to expose while the real GPUs sit unlocked — the opaque equivalent of a
    nonexistent numeric index, which already fails closed."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,bogus")

    with caplog.at_level(logging.WARNING, logger="phasesweep.runtime.gpu"):
        ambient = GpuPool.create(n_jobs=1)
    with ambient.acquire() as visible:
        assert visible.visible_devices == ""
    assert any("Ambient CUDA_VISIBLE_DEVICES" in record.message for record in caplog.records)

    with pytest.raises(RuntimeError, match="not a numeric index"):
        GpuPool.create(n_jobs=1, explicit_devices=["bogus"])


def test_empty_cuda_visible_devices_sentinel_is_pinned(monkeypatch) -> None:
    """An ambient "" is a decision that CUDA is off; trials must receive it even
    when a narrowed inherit_env contract drops the ambient variable."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    pool = GpuPool.create(n_jobs=1)

    with pool.acquire() as gid:
        assert gid.visible_devices == ""


def test_configured_disable_sentinel_is_pinned_for_parallel_cpu_sweeps(monkeypatch) -> None:
    """The opted-in parallel CPU path pins a configured disable the same way."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    pool = GpuPool.create(n_jobs=4, allow_no_gpu=True, cuda_visible_devices="-1")

    with pool.acquire() as gid:
        assert gid.visible_devices == "-1"


def test_explicit_cpu_visibility_needs_no_gpu_detection_opt_in(monkeypatch) -> None:
    """A configured disable sentinel is already an explicit, isolated CPU request."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    pool = GpuPool.create(n_jobs=4, cuda_visible_devices="-1")

    with pool.acquire() as gid:
        assert gid.visible_devices == "-1"


def test_ambient_scheduler_sentinel_warns_and_becomes_empty_visibility(monkeypatch, caplog) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "NoDevFiles")

    with caplog.at_level(logging.WARNING, logger="phasesweep.runtime.gpu"):
        pool = GpuPool.create(n_jobs=1)

    with pool.acquire() as gid:
        assert gid.visible_devices == ""
    assert any("treating it as empty visibility" in record.message for record in caplog.records)


def test_configured_junk_visibility_is_not_a_disable_sentinel(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(RuntimeError, match="not the empty string"):
        GpuPool.create(n_jobs=1, cuda_visible_devices=",")


def test_lock_layer_failure_does_not_shrink_the_pool(tmp_path, monkeypatch) -> None:
    """A raising lock layer must hand every candidate back: a silently shrunk
    pool deadlocks later acquires or relabels them as lease timeouts."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    pool = GpuPool.create(n_jobs=1, explicit_ids=[3])
    calls = {"n": 0}

    def flaky_lease(device: GpuDevice) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("lock layer failure")
        return _try_host_gpu_lease(device)

    monkeypatch.setattr("phasesweep.runtime.gpu._try_host_gpu_lease", flaky_lease)

    with pytest.raises(OSError, match="lock layer failure"), pool.acquire():
        pass

    assert [device.visible_token for device in pool._available] == ["3"]
    with pool.acquire(deadline=time.monotonic() + 2.0) as gid:
        assert gid.visible_devices == "3"


def test_whole_node_lock_layer_failure_releases_partial_leases(tmp_path, monkeypatch) -> None:
    """A mid-set lock failure must release already-taken host locks and clear
    the in-use flag, or every later whole-node acquisition deadlocks."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    pool = GpuPool.create(n_jobs=1, explicit_ids=[0, 1], policy="whole_node")
    calls = {"n": 0}

    def flaky_lease(device: GpuDevice) -> object:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("lock layer failure")
        return _try_host_gpu_lease(device)

    monkeypatch.setattr("phasesweep.runtime.gpu._try_host_gpu_lease", flaky_lease)

    with pytest.raises(OSError, match="lock layer failure"), pool.acquire():
        pass

    assert pool._whole_node_in_use is False
    monkeypatch.setattr("phasesweep.runtime.gpu._try_host_gpu_lease", _try_host_gpu_lease)
    with pool.acquire(deadline=time.monotonic() + 2.0) as gid:
        assert gid.visible_devices == "0,1"


def test_pid_stamp_failure_releases_the_flock(monkeypatch) -> None:
    """A write failure after flock succeeds must not leak a locked handle that
    excludes every other run from the GPU."""

    class _Handle:
        def seek(self, pos: int) -> None:
            pass

        def truncate(self) -> None:
            pass

        def write(self, text: str) -> None:
            raise OSError("disk full")

        def flush(self) -> None:
            pass

    released: list[object] = []
    monkeypatch.setattr("phasesweep.runtime.gpu.try_lock_file", lambda path: _Handle())
    monkeypatch.setattr("phasesweep.runtime.gpu.unlock_file", released.append)

    with pytest.raises(OSError, match="disk full"):
        _try_host_gpu_lease(GpuDevice("0"))

    assert len(released) == 1


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="nvidia-smi is not installed")
def test_real_nvidia_smi_resolves_index_zero_to_a_uuid(monkeypatch) -> None:
    """On real GPU hardware, index 0 locks under the UUID nvidia-smi reports for it."""
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_inventory",
        _real_detect_gpu_inventory,
    )
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


def test_gpu_policy_whole_node_requires_explicit_device_set() -> None:
    """whole_node's device set is the trainer's world size — a semantic input —
    so leaving it to ambient detection is rejected (review v0.5.17 gap hunt)."""
    with pytest.raises(ValueError, match="whole_node.*explicit gpu_ids or gpu_devices"):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
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


def test_gpu_policy_whole_node_rejects_duplicate_gpu_ids() -> None:
    """The fingerprint records the declared count, so a repeat is a semantic lie:
    gpu_ids=[0, 0] would fingerprint a 2-GPU world and lease one card
    (PR #5 review / reviewer 2, blocker 2)."""
    with pytest.raises(ValueError, match="whole_node.*requires unique device tokens"):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
            gpu_policy="whole_node",
            gpu_ids=[0, 0],
            search_space={"x": IntParam(type="int", low=0, high=1)},
        )


def test_gpu_policy_whole_node_accepts_distinct_gpu_ids() -> None:
    phase = Phase(  # type: ignore[arg-type]
        name="p",
        n_trials=1,
        gpu_policy="whole_node",
        gpu_ids=[0, 1],
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )

    assert phase.gpu_ids == [0, 1]


def test_gpu_policy_whole_node_rejects_duplicate_gpu_devices() -> None:
    with pytest.raises(ValueError, match="whole_node.*requires unique device tokens"):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
            gpu_policy="whole_node",
            gpu_devices=["GPU-uuid-a", "GPU-uuid-a"],
            search_space={"x": IntParam(type="int", low=0, high=1)},
        )


def test_gpu_policy_whole_node_rejects_duplicate_gpu_devices_after_stripping() -> None:
    """Whitespace is not a distinguishing feature: the runtime strips tokens
    before deduping, so the declared count must be counted post-strip too."""
    with pytest.raises(ValueError, match=r"whole_node.*requires unique device tokens"):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
            gpu_policy="whole_node",
            gpu_devices=[" GPU-a ", "GPU-a"],
            search_space={"x": IntParam(type="int", low=0, high=1)},
        )


def test_single_per_trial_still_accepts_duplicate_device_tokens() -> None:
    """Dedup stays supported pool behavior: only whole_node's count is semantic."""
    phase = Phase(  # type: ignore[arg-type]
        name="p",
        n_trials=1,
        gpu_ids=[0, 0, 1],
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )

    assert phase.gpu_ids == [0, 0, 1]


def test_whole_node_rejects_two_tokens_naming_one_physical_gpu(monkeypatch) -> None:
    """Physical aliasing is invisible to config validation (it needs nvidia-smi),
    so the pool must fail closed instead of leasing a 1-GPU world under a 2-GPU
    fingerprint (PR #5 review / reviewer 2, blocker 2)."""
    uuid = "GPU-32ad40d6-019f-386a-321d-3901216c78ad"
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})

    with pytest.raises(RuntimeError, match="whole_node.*2 CUDA device token"):
        GpuPool.create(n_jobs=1, explicit_devices=["0", uuid], policy="whole_node")


def test_whole_node_accepts_distinct_physical_gpus(monkeypatch) -> None:
    uuid_map = {"0": "GPU-aaa11111", "1": "GPU-bbb22222"}
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: uuid_map)

    pool = GpuPool.create(n_jobs=1, explicit_devices=["0", "1"], policy="whole_node")

    assert [device.visible_token for device in pool._devices] == ["0", "1"]
    with pool.acquire() as gid:
        assert gid.visible_devices == "0,1"


@pytest.mark.parametrize(
    "tokens",
    [
        ["MIG-aaa11111", "MIG-bbb22222"],
        ["MIG-aaa11111", "GPU-bbb22222"],
    ],
)
def test_whole_node_rejects_multi_device_worlds_containing_mig(
    tokens: list[str], monkeypatch
) -> None:
    """MIG enumeration cannot guarantee that token count equals CUDA device count."""
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_uuid_map",
        lambda: {"1": "GPU-bbb22222"},
    )
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_mig_uuid_set",
        lambda: {"MIG-aaa11111", "MIG-bbb22222"},
    )

    with pytest.raises(RuntimeError, match="whole_node.*MIG tokens"):
        GpuPool.create(n_jobs=1, explicit_devices=tokens, policy="whole_node")


def test_single_per_trial_still_dedupes_aliased_tokens_with_a_warning(monkeypatch, caplog) -> None:
    """The same alias pair that fails closed under whole_node stays a pool
    convenience under single_per_trial."""
    uuid = "GPU-32ad40d6-019f-386a-321d-3901216c78ad"
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {"0": uuid})

    with caplog.at_level(logging.WARNING, logger="phasesweep.runtime.gpu"):
        pool = GpuPool.create(n_jobs=1, explicit_devices=["0", uuid])

    assert [device.visible_token for device in pool._devices] == ["0"]
    assert any("same physical GPU" in record.message for record in caplog.records)


def test_whole_node_fails_closed_when_uuid_resolution_is_unavailable(monkeypatch) -> None:
    """A declared world cannot use lock identities that another run may spell as UUIDs."""
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_uuid_map", lambda: {})

    with pytest.raises(RuntimeError, match="Cannot validate configured CUDA device index"):
        GpuPool.create(n_jobs=1, explicit_ids=[0, 1], policy="whole_node")
