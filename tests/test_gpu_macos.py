"""macOS GPU discovery leases one Apple silicon device without CUDA visibility."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from phasesweep.config.models import GpuPolicy, Phase
from phasesweep.config.search import IntParam
from phasesweep.engine.fingerprints import _semantic_phase_dump
from phasesweep.errors import GpuConfigurationError
from phasesweep.runtime.gpu import GpuLeaseTimeoutError, GpuPool, _gpu_lock_path


@pytest.fixture(autouse=True)
def apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discover MPS without importing a trainer framework or probing NVIDIA."""
    monkeypatch.setattr("phasesweep.runtime.gpu.sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr("phasesweep.config.models.sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr("phasesweep.runtime.gpu.platform.machine", lambda: "arm64")

    def refuse_cuda_probe() -> None:
        pytest.fail("macOS GPU discovery must not invoke nvidia-smi")

    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", refuse_cuda_probe)
    monkeypatch.setattr("phasesweep.runtime.gpu._nvidia_driver_reports_gpus", refuse_cuda_probe)


@pytest.mark.parametrize("policy", ["single_per_trial", "whole_node"])
def test_mps_policies_share_a_host_lease_without_cuda_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: GpuPolicy
) -> None:
    """Separate pools serialize MPS work on the same host lock and release it."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    first = GpuPool.create(n_jobs=1, policy=policy)
    second = GpuPool.create(n_jobs=2)

    assert [device.visible_token for device in first._devices] == ["mps"]
    assert _gpu_lock_path(first._devices[0]) == tmp_path / "gpu_mps.lock"
    with first.acquire() as assignment:
        assert assignment.visible_devices is None
        assert len(assignment.lease_fds) == 1
        with (
            pytest.raises(GpuLeaseTimeoutError, match="Wallclock deadline"),
            second.acquire(deadline=time.monotonic() + 0.02),
        ):
            pytest.fail("MPS pools must not share the host GPU concurrently")
    with second.acquire(deadline=time.monotonic() + 1.0) as assignment:
        assert assignment.visible_devices is None
        assert len(assignment.lease_fds) == 1


@pytest.mark.parametrize("policy", ["single_per_trial", "whole_node"])
def test_mps_phase_configuration_uses_discovered_pool(policy: GpuPolicy) -> None:
    """Whole-node MPS runs need no fabricated CUDA device list."""
    phase = Phase(
        name="mps",
        n_trials=1,
        gpu_policy=policy,
        search_space={"x": IntParam(type="int", low=0, high=1)},
    )
    pool = GpuPool.for_phase(phase)
    assert [device.visible_token for device in pool._devices] == ["mps"]
    assert pool._whole_node is (policy == "whole_node")
    if policy == "whole_node":
        assert _semantic_phase_dump(phase)["whole_node_device_count"] == 1


def test_macos_refuses_explicit_cuda_device_lists() -> None:
    """macOS device leases always come from discovery, never CUDA tokens."""
    with pytest.raises(GpuConfigurationError, match="device pool is discovered"):
        GpuPool.create(n_jobs=1, explicit_ids=[0])
    with pytest.raises(GpuConfigurationError, match="device pool is discovered"):
        GpuPool.create(n_jobs=1, explicit_devices=["GPU-example"])


def test_mps_none_policy_does_not_take_a_host_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opting out of GPU isolation leaves the MPS device unlocked."""
    monkeypatch.setattr("phasesweep.runtime.gpu.lock_dir", lambda: tmp_path)
    pool = GpuPool.create(n_jobs=2, policy="none")
    with pool.acquire() as assignment:
        assert assignment.visible_devices is None
        assert assignment.lease_fds == ()
    assert not list(tmp_path.glob("gpu_*.lock"))


def test_intel_mac_has_no_gpu_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intel Macs retain the no-GPU host rules for single and parallel work."""
    monkeypatch.setattr("phasesweep.runtime.gpu.platform.machine", lambda: "x86_64")
    for pool in [GpuPool.create(n_jobs=1), GpuPool.create(n_jobs=2, allow_no_gpu=True)]:
        with pool.acquire() as assignment:
            assert assignment.visible_devices is None
            assert assignment.lease_fds == ()
    with pytest.raises(GpuConfigurationError, match="no GPUs detected on this Intel Mac"):
        GpuPool.create(n_jobs=2)
