"""Actionable diagnostics for sandbox lock access and unavailable GPU enumeration."""

from __future__ import annotations

import pytest

from phasesweep.errors import GpuConfigurationError
from phasesweep.mcp.runner import _terminal_failure_payload
from phasesweep.runtime.files import UnsafeLockPathError
from phasesweep.runtime.gpu import GpuPool


def test_mcp_lock_access_failure_has_operator_remediation() -> None:
    error = UnsafeLockPathError("Cannot access /private/host/locks: Permission denied")
    failure = _terminal_failure_payload(error, stage="execution", cleanup_confirmed=True)

    assert failure["code"] == "storage_unavailable"
    assert failure["stage"] == "preflight"
    assert failure["actor"] == "operator"
    assert failure["retryable"] is False
    assert "same writable PHASESWEEP_LOCK_DIR" in failure["remediation"]
    assert "restore the files and permissions" in failure["remediation"]
    assert "/private/host/locks" not in str(failure)


def test_gpu_enumeration_refusal_explains_cpu_only_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))
    monkeypatch.setattr("phasesweep.runtime.gpu._nvidia_driver_reports_gpus", lambda: True)

    with pytest.raises(GpuConfigurationError, match="gpu_policy: none") as failure:
        GpuPool.create(n_jobs=1)

    assert "sandbox" in str(failure.value)
    assert "host shell with GPU access" in str(failure.value)

    with GpuPool.create(n_jobs=1, policy="none").acquire() as assignment:
        assert assignment.visible_devices is None
        assert assignment.lease_fds == ()
