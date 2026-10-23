"""Actionable diagnostics for sandbox lock access and unavailable GPU enumeration."""

from __future__ import annotations

import pytest

from phasesweep.errors import GpuConfigurationError, OperatorAction
from phasesweep.mcp.runner import _terminal_error, _terminal_failure_payload
from phasesweep.runtime.files import UnsafeLockPathError
from phasesweep.runtime.gpu import GpuPool


def test_mcp_lock_access_failure_has_operator_remediation() -> None:
    error = UnsafeLockPathError(
        "Cannot access /private/host/locks: Permission denied",
        action=OperatorAction.PROVISION_LOCK_DIR,
    )
    # The lock is taken before the engine can report, so the runner has no report to read.
    primary, stage = _terminal_error(None, error)
    failure = _terminal_failure_payload(primary, stage=stage, cleanup_confirmed=True)

    assert failure["code"] == "storage_unavailable"
    assert failure["stage"] == "preflight"
    assert failure["actor"] == "operator"
    assert failure["retryable"] is False
    assert failure["remediation"] == (
        "Ask the operator to point PHASESWEEP_LOCK_DIR at an existing writable directory shared "
        "by every shell, MCP server, and sandbox on the host, or run on the host. The error in "
        "the PhaseSweep run log gives the details."
    )
    assert "/private/host/locks" not in str(failure)


def test_gpu_enumeration_refusal_explains_cpu_only_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreadable mocked CUDA hardware explains the deliberate CPU opt-out on any host."""
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
