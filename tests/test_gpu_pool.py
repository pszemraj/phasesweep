"""GPU pool: explicit IDs, autodetection, host locks, and no-GPU policy."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from unittest.mock import Mock

import pytest

from phasesweep.config.models import GpuPolicy, Phase
from phasesweep.config.search import IntParam
from phasesweep.errors import GpuConfigurationError
from phasesweep.runtime.files import open_lock_file
from phasesweep.runtime.gpu import (
    GpuDevice,
    GpuLeaseCancelledError,
    GpuLeaseTimeoutError,
    GpuPool,
    _detect_gpu_inventory,
    _gpu_lock_path,
    _try_host_gpu_lease,
)
from phasesweep.runtime.process import run_supervised
from phasesweep.runtime.shutdown import PhaseSweepShutdown, _shutdown_handler

# Bound before any monkeypatching so live tests can restore the real probe.
_real_detect_gpu_inventory = _detect_gpu_inventory
_TEST_UUID_MAP = {str(index): f"GPU-test-{index}" for index in range(16)}


@pytest.fixture(autouse=True)
def deterministic_gpu_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep autodetection CPU-only while explicit GPU indices resolve deterministically.

    Tests that exercise inventory discovery replace this probe or restore the
    real one.
    """
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_inventory",
        lambda: ([], dict(_TEST_UUID_MAP)),
    )


def test_gpu_pool_fails_on_missing_gpus_parallel(monkeypatch):
    """n_jobs > 1 with no GPUs and allow_no_gpu=False must raise, not silently degrade."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="no GPUs detected"):
        GpuPool.create(n_jobs=4, allow_no_gpu=False)


def test_gpu_pool_allows_no_gpu_when_opted_in(monkeypatch):
    """n_jobs > 1 with allow_no_gpu=True should warn but not crash."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    pool = GpuPool.create(n_jobs=4, allow_no_gpu=True)
    with pool.acquire() as gid:
        assert gid.visible_devices is None
        assert gid.lease_fds == ()


@pytest.mark.parametrize(
    ("uuid", "ambient_tokens", "explicit_devices", "expected_tokens"),
    [
        pytest.param(
            "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            None,
            [" GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa "],
            ["GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
            id="explicit-token-is-stripped",
        ),
        pytest.param(
            "GPU-deadbeef",
            "0,GPU-deadbeef",
            None,
            ["0"],
            id="ambient-index-and-uuid-deduplicate",
        ),
    ],
)
def test_gpu_pool_normalizes_and_deduplicates_device_tokens(
    monkeypatch,
    uuid: str,
    ambient_tokens: str | None,
    explicit_devices: list[str] | None,
    expected_tokens: list[str],
) -> None:
    """Equivalent device spellings become one canonical pool device."""
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {"0": uuid}))
    if ambient_tokens is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ambient_tokens)

    pool = GpuPool.create(n_jobs=1, explicit_devices=explicit_devices)

    assert [device.visible_token for device in pool._devices] == expected_tokens


@pytest.mark.parametrize(
    ("gpu_kwargs", "visible", "lease_count"),
    [
        pytest.param({"explicit_ids": [3]}, "3", 1, id="default-single-device"),
        pytest.param(
            {"explicit_ids": [0, 1, 2], "policy": "whole_node"},
            "0,1,2",
            3,
            id="whole-node-all-devices",
        ),
    ],
)
def test_single_job_assigns_configured_gpu_ids(gpu_kwargs, visible, lease_count) -> None:
    """Default acquisition isolates one GPU; whole-node acquisition leases every configured GPU."""
    pool = GpuPool.create(n_jobs=1, **gpu_kwargs)
    with pool.acquire() as gid:
        assert gid.visible_devices == visible
        assert len(gid.lease_fds) == lease_count


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


def test_whole_node_policy_cancels_while_waiting_for_host_lock(tmp_path, monkeypatch) -> None:
    """An abort wakes a whole-node request after it releases partial leases."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    lock_path = _gpu_lock_path(GpuDevice("1", _TEST_UUID_MAP["1"]))
    attempted_locked_device = threading.Event()
    result: list[BaseException] = []
    original_try_host_lease = _try_host_gpu_lease

    def signal_locked_attempt(device: GpuDevice):
        lease = original_try_host_lease(device)
        if device.visible_token == "1":
            attempted_locked_device.set()
        return lease

    with open_lock_file(lock_path) as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        monkeypatch.setattr("phasesweep.runtime.gpu._try_host_gpu_lease", signal_locked_attempt)
        pool = GpuPool.create(n_jobs=1, explicit_ids=[0, 1], policy="whole_node")

        def acquire() -> None:
            try:
                with pool.acquire():
                    pytest.fail("the held host lock should prevent acquisition")
            except BaseException as exc:
                result.append(exc)

        worker = threading.Thread(target=acquire)
        worker.start()
        try:
            assert attempted_locked_device.wait(timeout=1.0)
        finally:
            pool.cancel_waiters()
            worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert len(result) == 1
    assert isinstance(result[0], GpuLeaseCancelledError)
    assert pool._whole_node_in_use is False

    competitor = GpuPool.create(n_jobs=1, explicit_ids=[0])
    with competitor.acquire(deadline=time.monotonic() + 1.0) as assignment:
        assert assignment.visible_devices == "0"


def test_shutdown_wakes_a_lease_wait_blocked_on_a_busy_local_slot(tmp_path, monkeypatch) -> None:
    """The shutdown handler ends a deadline-less wait at once, before the slot frees, naming its signal."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    pool = GpuPool.create(n_jobs=1, explicit_ids=[3])
    result: list[BaseException] = []
    blocked = threading.Event()

    class _ObservedCondition(threading.Condition):
        def wait(self, timeout: float | None = None) -> bool:
            blocked.set()
            return threading.Condition.wait(self, timeout)

    pool._condition = _ObservedCondition()

    def wait_for_the_slot() -> None:
        try:
            with pool.acquire():
                result.append(AssertionError("acquired a slot during a shutdown"))
        except BaseException as exc:
            result.append(exc)

    with pool.acquire():
        waiter = threading.Thread(target=wait_for_the_slot)
        waiter.start()
        assert blocked.wait(5.0)
        with pytest.raises(PhaseSweepShutdown):
            _shutdown_handler(signal.SIGINT, None)
        # Still holding the slot: only the handler's wake can end the wait.
        waiter.join(timeout=2.0)
        assert not waiter.is_alive()

    assert len(result) == 1
    assert isinstance(result[0], GpuLeaseCancelledError)
    assert "shutdown signal 2" in str(result[0])


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


@pytest.mark.parametrize(
    ("pool_kwargs", "visible"),
    [
        pytest.param({"n_jobs": 1}, None, id="single-job-no-gpus"),
        pytest.param({"n_jobs": 4, "cuda_visible_devices": "-1"}, "-1", id="explicit-cpu"),
    ],
)
def test_cpu_work_needs_no_gpu_detection_opt_in(monkeypatch, pool_kwargs, visible) -> None:
    """Single-job CPU work and explicitly disabled CUDA need no no-GPU opt-in."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    pool = GpuPool.create(**pool_kwargs)
    with pool.acquire() as gid:
        if visible is None:
            assert gid.visible_devices is None
        else:
            assert gid.visible_devices == visible


def test_single_job_fails_when_nvidia_smi_is_broken_on_gpu_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("phasesweep.runtime.gpu._nvidia_driver_reports_gpus", lambda: True)

    with pytest.raises(GpuConfigurationError, match="could not enumerate GPUs.*double-book"):
        GpuPool.create(n_jobs=1)


def test_ambient_cuda_visibility_fails_when_nvidia_smi_is_broken_on_gpu_host(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))
    monkeypatch.setattr("phasesweep.runtime.gpu._nvidia_driver_reports_gpus", lambda: True)

    with pytest.raises(GpuConfigurationError, match="could not enumerate GPUs.*double-book"):
        GpuPool.create(n_jobs=1)

    with caplog.at_level(logging.WARNING, logger="phasesweep.runtime.gpu"):
        pool = GpuPool.create(n_jobs=1, allow_no_gpu=True)
    with pool.acquire() as assignment:
        assert assignment.visible_devices == ""
    assert "pinning CUDA_VISIBLE_DEVICES='' without GPU host locks" in caplog.text
    assert "running without CUDA isolation" not in caplog.text


def test_broken_nvidia_smi_requires_explicit_fail_open_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("phasesweep.runtime.gpu._nvidia_driver_reports_gpus", lambda: True)

    with caplog.at_level(logging.WARNING, logger="phasesweep.runtime.gpu"):
        pool = GpuPool.create(n_jobs=1, allow_no_gpu=True)

    with pool.acquire() as assignment:
        assert assignment.visible_devices is None
    assert pool._devices == []
    assert any("allow_no_gpu_isolation: true" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    ("ambient", "visible"),
    [
        pytest.param("2,5", "2", id="numeric-devices"),
        pytest.param("", "", id="empty-disable-sentinel"),
    ],
)
def test_single_job_pins_ambient_cuda_visibility(monkeypatch, ambient, visible) -> None:
    """Select one numeric device or preserve the ambient empty CUDA-disable decision."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ambient)
    pool = GpuPool.create(n_jobs=1)
    with pool.acquire() as gid:
        assert gid.visible_devices == visible


def test_nonnumeric_cuda_visible_devices_is_leased_as_opaque_token(monkeypatch):
    uuid = "GPU-deadbeef"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {"0": uuid}))

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
        "phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {"0": "GPU-configured"})
    )

    pool = GpuPool.create(n_jobs=1, cuda_visible_devices="GPU-configured")

    with pool.acquire() as gid:
        assert gid.visible_devices == "GPU-configured"


def test_explicit_gpu_devices_preserve_tokens_and_dedupe(monkeypatch):
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {"0": "GPU-a"})
    )
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


@pytest.mark.integration
def test_gpu_lease_survives_orchestrator_hard_exit(tmp_path, monkeypatch) -> None:
    """A guardian retains the lock through FD scrubbing, parent death, and descendants."""
    locks = tmp_path / "locks"
    locks.mkdir()
    locks.chmod(0o700)
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
        # The lease must stay held at this exact boundary, while the worker lives.
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


@pytest.mark.parametrize(
    ("uuid", "ambient"),
    [
        ("GPU-32ad40d6-019f-386a-321d-3901216c78ad", False),
        ("GPU-deadbeef", True),
    ],
    ids=["configured-index-blocks-explicit-uuid", "ambient-uuid-blocks-configured-index"],
)
def test_index_and_uuid_share_one_host_lock(tmp_path, monkeypatch, uuid, ambient) -> None:
    """Configured indices and explicit or ambient UUIDs contend in both ownership directions."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {"0": uuid}))
    if ambient:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
        by_uuid = GpuPool.create(n_jobs=1)
    else:
        by_uuid = GpuPool.create(n_jobs=1, explicit_devices=[uuid])

    by_index = GpuPool.create(n_jobs=1, explicit_ids=[0])
    assert _gpu_lock_path(by_index._devices[0]) == _gpu_lock_path(by_uuid._devices[0])
    owner, waiter = (by_uuid, by_index) if ambient else (by_index, by_uuid)

    with owner.acquire() as gid:
        # The trainer still sees the token the operator configured.
        assert gid.visible_devices == (uuid if ambient else "0")
        with (
            pytest.raises(TimeoutError, match="Wallclock deadline"),
            waiter.acquire(deadline=time.monotonic() + 0.02),
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


def test_mig_token_locks_on_the_mig_instance(tmp_path, monkeypatch) -> None:
    """MIG instances lock on themselves; parent-GPU binding is a documented gap."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    uuid = "GPU-deadbeef"
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {"0": uuid}))
    mig_token = "MIG-GPU-deadbeef/3/0"
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_mig_uuid_set", lambda: {mig_token})

    mig = GpuPool.create(n_jobs=1, explicit_devices=[mig_token])
    parent = GpuPool.create(n_jobs=1, explicit_ids=[0])

    assert mig._devices[0].lock_identity == mig_token
    assert _gpu_lock_path(mig._devices[0]) != _gpu_lock_path(parent._devices[0])
    with mig.acquire() as mig_gid, parent.acquire() as parent_gid:
        assert mig_gid.visible_devices == mig_token
        assert parent_gid.visible_devices == "0"


@pytest.mark.parametrize(
    ("explicit_ids", "explicit_devices", "error_match", "policy"),
    [
        pytest.param(
            [0, 1],
            None,
            "Cannot validate configured CUDA device index",
            "single_per_trial",
            id="numeric-indices",
        ),
        pytest.param(
            None,
            ["0", "GPU-deadbeef"],
            "Cannot validate configured GPU UUID",
            "single_per_trial",
            id="mixed-token-forms",
        ),
        # A declared whole_node world cannot use lock identities that another run may spell as UUIDs.
        pytest.param(
            [0, 1],
            None,
            "Cannot validate configured CUDA device index",
            "whole_node",
            id="whole-node-numeric-indices",
        ),
    ],
)
def test_unreadable_uuid_map_rejects_unverified_tokens(
    monkeypatch,
    explicit_ids: list[int] | None,
    explicit_devices: list[str] | None,
    error_match: str,
    policy: GpuPolicy,
) -> None:
    """Unverified numeric and opaque GPU tokens fail closed, not silently."""
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))

    with pytest.raises(RuntimeError, match=error_match):
        GpuPool.create(
            # Whole-node acquisition requires a single-job phase.
            n_jobs=2 if explicit_ids is not None and policy == "single_per_trial" else 1,
            explicit_ids=explicit_ids,
            explicit_devices=explicit_devices,
            policy=policy,
        )


def test_unknown_numeric_index_fails_closed(monkeypatch) -> None:
    """A configured index nvidia-smi does not report is a nonexistent device."""
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_inventory",
        lambda: ([], {"0": "GPU-deadbeef"}),
    )

    with pytest.raises(RuntimeError, match="index 3 does not exist"):
        GpuPool.create(n_jobs=1, explicit_ids=[3])


def test_resolved_uuid_lock_names_stay_path_safe(tmp_path, monkeypatch) -> None:
    """A resolved identity is sanitized and hashed exactly like an opaque token."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_inventory",
        lambda: ([], {"0": "GPU-dead/beef:0"}),
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
    identity and silently collapse a multi-GPU pool to a single lock."""

    class _Out:
        returncode = 0
        stdout = "0, [N/A]\n1, [Not Supported]\n2, GPU-abc123\n3, MIG-def456\n"

    monkeypatch.setattr("phasesweep.runtime.gpu.subprocess.run", lambda *args, **kwargs: _Out())

    assert _real_detect_gpu_inventory() == (
        [0, 1, 2, 3],
        {"2": "GPU-abc123", "3": "MIG-def456"},
    )


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(FileNotFoundError("nvidia-smi"), id="missing-binary"),
        pytest.param(PermissionError("nvidia-smi"), id="not-executable"),
        pytest.param(subprocess.TimeoutExpired("nvidia-smi", 5), id="timeout"),
    ],
)
def test_inventory_probe_without_a_usable_nvidia_smi_is_empty(monkeypatch, failure) -> None:
    """A missing, unrunnable or hung nvidia-smi is an empty inventory, which callers refuse on."""

    def run(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr("phasesweep.runtime.gpu.subprocess.run", run)

    assert _real_detect_gpu_inventory() == ([], {})


def test_inventory_probe_with_a_nonzero_exit_is_empty(monkeypatch) -> None:
    """An nvidia-smi that fails reports no inventory, whatever it printed."""

    class _Out:
        returncode = 9
        stdout = "0, GPU-abc123\n"

    monkeypatch.setattr("phasesweep.runtime.gpu.subprocess.run", lambda *args, **kwargs: _Out())

    assert _real_detect_gpu_inventory() == ([], {})


@pytest.mark.parametrize(
    "defect",
    [
        pytest.param(
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"), id="undecodable"
        ),
        pytest.param(RuntimeError("probe defect"), id="internal-error"),
    ],
)
def test_inventory_probe_defect_is_not_swallowed(monkeypatch, defect) -> None:
    """Only "no usable nvidia-smi" is an empty inventory; any other failure must surface."""

    def run(*_args, **_kwargs):
        raise defect

    monkeypatch.setattr("phasesweep.runtime.gpu.subprocess.run", run)

    with pytest.raises(type(defect)):
        _real_detect_gpu_inventory()


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param("GPU-2b23", id="uppercase-prefix"),
        pytest.param("gpu-2b23", id="lowercase-prefix"),
        # Full UUID spellings are case-insensitive for both validation and locking.
        pytest.param("gpu-2b234567-89ab-cdef-0123-456789abcdef", id="lowercase-full-uuid"),
    ],
)
def test_uuid_spellings_share_the_full_uuid_lock(tmp_path, monkeypatch, spelling) -> None:
    """CUDA UUID prefixes and lowercase full UUIDs share the full UUID's lock."""
    uuid = "GPU-2b234567-89ab-cdef-0123-456789abcdef"
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {"0": uuid}))

    spelled_pool = GpuPool.create(n_jobs=1, explicit_devices=[spelling])
    index_pool = GpuPool.create(n_jobs=1, explicit_ids=[0])
    full_pool = GpuPool.create(n_jobs=1, explicit_devices=[uuid])

    spelled_lock = _gpu_lock_path(spelled_pool._devices[0])
    assert spelled_lock == _gpu_lock_path(index_pool._devices[0])
    assert spelled_lock == _gpu_lock_path(full_pool._devices[0])
    # CUDA receives the inventory spelling, not an abbreviated or re-cased alias.
    assert spelled_pool._devices[0].visible_token == uuid


def test_lowercase_mig_uuid_rebinds_trainer_visibility_to_inventory_spelling(
    monkeypatch,
) -> None:
    """A lowercase MIG alias must not make a correctly locked device invisible."""
    mig_uuid = "MIG-2b234567-89ab-cdef-0123-456789abcdef"
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_mig_uuid_set", lambda: {mig_uuid})

    pool = GpuPool.create(n_jobs=1, explicit_devices=[mig_uuid.lower()])

    assert pool._devices[0].visible_token == mig_uuid
    assert pool._devices[0].lock_identity == mig_uuid


@pytest.mark.parametrize(
    ("uuid", "tokens", "error_match"),
    [
        pytest.param(
            "GPU-2b234567-89ab-cdef-0123-456789abcdef",
            [
                "GPU-2b234567-89ab-cdef-0123-456789abcdef",
                "gpu-2b234567-89ab-cdef-0123-456789abcdef",
            ],
            "only 1 distinct physical GPU",
            id="case-only-uuid-alias",
        ),
        pytest.param(
            "GPU-32ad40d6-019f-386a-321d-3901216c78ad",
            ["0", "GPU-32ad40d6-019f-386a-321d-3901216c78ad"],
            "whole_node.*2 CUDA device token",
            id="index-and-uuid-alias",
        ),
    ],
)
def test_whole_node_rejects_physical_gpu_aliases(
    monkeypatch,
    uuid: str,
    tokens: list[str],
    error_match: str,
) -> None:
    """Whole-node mode fails closed when tokens name one physical GPU."""
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {"0": uuid}))

    with pytest.raises(RuntimeError, match=error_match):
        GpuPool.create(n_jobs=1, explicit_devices=tokens, policy="whole_node")


@pytest.mark.parametrize("prefix", ["GPU-", "MIG-"])
def test_nonexistent_uuid_token_fails_closed(prefix: str, monkeypatch) -> None:
    token = f"{prefix}00000000-0000-0000-0000-000000000000"
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))
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


@pytest.mark.parametrize("sentinel", ["-1", "-1,"])
def test_configured_disable_sentinel_is_pinned_for_parallel_cpu_sweeps(
    monkeypatch: pytest.MonkeyPatch,
    sentinel: str,
) -> None:
    """The opted-in parallel CPU path pins a configured disable the same way."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    pool = GpuPool.create(n_jobs=4, allow_no_gpu=True, cuda_visible_devices=sentinel)

    with pool.acquire() as gid:
        assert gid.visible_devices == sentinel


@pytest.mark.parametrize(
    ("ambient", "log_fragment"),
    [
        pytest.param("NoDevFiles", "treating it as empty visibility", id="scheduler-sentinel"),
        pytest.param("0,-1", "mixes the '-1' disable sentinel", id="mixed-disable-sentinel"),
    ],
)
def test_ambient_sentinel_warns_and_becomes_empty_visibility(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    ambient: str,
    log_fragment: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ambient)

    with caplog.at_level(logging.WARNING, logger="phasesweep.runtime.gpu"):
        pool = GpuPool.create(n_jobs=1)

    with pool.acquire() as gid:
        assert gid.visible_devices == ""
    assert any(log_fragment in record.message for record in caplog.records)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        pytest.param("0,-1", "cannot be mixed", id="mixed-disable-sentinel"),
        pytest.param(",", "not the empty string", id="junk-is-not-a-disable-sentinel"),
    ],
)
def test_configured_invalid_visibility_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str, match: str
) -> None:
    """A configured visibility that mixes the disable sentinel or is junk stays invalid."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(RuntimeError, match=match):
        GpuPool.create(n_jobs=1, cuda_visible_devices=value)


@pytest.mark.parametrize(
    ("explicit_ids", "policy", "fail_on_call", "expected_visible"),
    [
        pytest.param(
            [3],
            "single_per_trial",
            1,
            "3",
            id="single-device-pool-restores-candidate",
        ),
        pytest.param(
            [0, 1],
            "whole_node",
            2,
            "0,1",
            id="whole-node-pool-releases-partial-leases",
        ),
    ],
)
def test_lock_layer_failure_rolls_back_gpu_pool(
    tmp_path,
    monkeypatch,
    explicit_ids: list[int],
    policy: str,
    fail_on_call: int,
    expected_visible: str,
) -> None:
    """A lock failure restores candidates and permits a later acquisition."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    pool = GpuPool.create(n_jobs=1, explicit_ids=explicit_ids, policy=policy)
    calls = {"n": 0}

    def flaky_lease(device: GpuDevice) -> object:
        calls["n"] += 1
        if calls["n"] == fail_on_call:
            raise OSError("lock layer failure")
        return _try_host_gpu_lease(device)

    monkeypatch.setattr("phasesweep.runtime.gpu._try_host_gpu_lease", flaky_lease)

    with pytest.raises(OSError, match="lock layer failure"), pool.acquire():
        pass

    if policy == "single_per_trial":
        assert [device.visible_token for device in pool._available] == [expected_visible]
    else:
        assert pool._whole_node_in_use is False
    with pool.acquire(deadline=time.monotonic() + 2.0) as gid:
        assert gid.visible_devices == expected_visible


def test_pid_stamp_failure_releases_the_flock(monkeypatch) -> None:
    """A write failure after flock succeeds must not leak a locked handle that
    excludes every other run from the GPU."""

    handle = Mock()
    handle.write.side_effect = OSError("disk full")
    released: list[object] = []
    monkeypatch.setattr("phasesweep.runtime.gpu.try_lock_file", lambda path: handle)
    monkeypatch.setattr("phasesweep.runtime.gpu.unlock_file", released.append)

    with pytest.raises(OSError, match="disk full"):
        _try_host_gpu_lease(GpuDevice("0"))

    assert released == [handle]


@pytest.mark.live
@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="nvidia-smi is not installed")
def test_real_nvidia_smi_resolves_index_zero_to_a_uuid(monkeypatch) -> None:
    """On real GPU hardware, index 0 locks under the UUID nvidia-smi reports for it."""
    monkeypatch.setattr(
        "phasesweep.runtime.gpu._detect_gpu_inventory",
        _real_detect_gpu_inventory,
    )
    uuid_map = _real_detect_gpu_inventory()[1]
    if not uuid_map:
        pytest.skip("nvidia-smi is installed but reports no CUDA devices")

    assert uuid_map["0"].startswith("GPU-")

    pool = GpuPool.create(n_jobs=1, explicit_ids=[0])

    assert pool._devices[0].visible_token == "0"
    assert pool._devices[0].lock_identity == uuid_map["0"]
    assert _gpu_lock_path(pool._devices[0]) == _gpu_lock_path(GpuDevice(uuid_map["0"]))


@pytest.mark.parametrize(
    ("gpu_kwargs", "match"),
    [
        pytest.param(
            {"gpu_ids": [0, -1, 2]},
            "non-negative CUDA device indices",
            id="gpu-ids-negative",
        ),
        pytest.param(
            {"gpu_ids": []},
            "at least one CUDA device index",
            id="gpu-ids-empty",
        ),
        pytest.param(
            {"gpu_devices": ["GPU-ok", "bad,token"]},
            "gpu_devices entries",
            id="gpu-devices-ambiguous-token",
        ),
        pytest.param(
            {"gpu_ids": [0], "gpu_devices": ["GPU-deadbeef"]},
            "mutually exclusive",
            id="gpu-ids-and-gpu-devices-exclusive",
        ),
        pytest.param(
            {"n_jobs": 2, "gpu_policy": "whole_node"},
            "whole_node.*requires n_jobs=1",
            id="whole-node-requires-single-job",
        ),
        # whole_node's device set is the trainer's world size, a semantic input, so leaving it to ambient detection is rejected.
        pytest.param(
            {"gpu_policy": "whole_node"},
            "whole_node.*explicit gpu_ids or gpu_devices",
            id="whole-node-requires-explicit-device-set",
        ),
        pytest.param(
            {"gpu_policy": "none", "gpu_ids": [0]},
            "cannot be combined",
            id="none-policy-rejects-explicit-gpu-list",
        ),
        # The fingerprint records the declared count, so a repeat is a semantic lie: gpu_ids=[0, 0] would fingerprint a 2-GPU world and lease one card.
        pytest.param(
            {"gpu_policy": "whole_node", "gpu_ids": [0, 0]},
            "whole_node.*requires unique device tokens",
            id="whole-node-duplicate-gpu-ids",
        ),
        # Whitespace is not a distinguishing feature: the runtime strips tokens before deduping, so the declared count must be counted post-strip too.
        pytest.param(
            {"gpu_policy": "whole_node", "gpu_devices": [" GPU-a ", "GPU-a"]},
            "whole_node.*requires unique device tokens",
            id="whole-node-duplicate-gpu-devices-after-stripping",
        ),
    ],
)
def test_phase_rejects_invalid_gpu_config(gpu_kwargs: dict[str, object], match: str) -> None:
    """Each invalid GPU declaration is rejected at config validation, not at run time."""
    with pytest.raises(ValueError, match=match):
        Phase(  # type: ignore[arg-type]
            name="p",
            n_trials=1,
            search_space={"x": IntParam(type="int", low=0, high=1)},
            **gpu_kwargs,
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


def test_gpu_policy_whole_node_accepts_distinct_gpu_ids() -> None:
    phase = Phase(  # type: ignore[arg-type]
        name="p",
        n_trials=1,
        gpu_policy="whole_node",
        gpu_ids=[0, 1],
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )

    assert phase.gpu_ids == [0, 1]


def test_single_per_trial_still_accepts_duplicate_device_tokens() -> None:
    """Dedup stays supported pool behavior: only whole_node's count is semantic."""
    phase = Phase(  # type: ignore[arg-type]
        name="p",
        n_trials=1,
        gpu_ids=[0, 0, 1],
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )

    assert phase.gpu_ids == [0, 0, 1]


def test_whole_node_accepts_distinct_physical_gpus(monkeypatch) -> None:
    uuid_map = {"0": "GPU-aaa11111", "1": "GPU-bbb22222"}
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], uuid_map))

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
        "phasesweep.runtime.gpu._detect_gpu_inventory",
        lambda: ([], {"1": "GPU-bbb22222"}),
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
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {"0": uuid}))

    with caplog.at_level(logging.WARNING, logger="phasesweep.runtime.gpu"):
        pool = GpuPool.create(n_jobs=1, explicit_devices=["0", uuid])

    assert [device.visible_token for device in pool._devices] == ["0"]
    assert any("same physical GPU" in record.message for record in caplog.records)
