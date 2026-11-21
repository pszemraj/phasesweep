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
    failure = _terminal_failure_payload(primary, run_id="r0", stage=stage, cleanup_confirmed=True)

    assert failure["code"] == "storage_unavailable"
    assert failure["stage"] == "preflight"
    assert failure["actor"] == "operator"
    assert failure["retryable"] is False
    assert failure["remediation"] == (
        "Ask the operator to point PHASESWEEP_LOCK_DIR at an existing writable directory shared "
        "by every shell, MCP server, and sandbox on the host, or run on the host. "
        "Details: `logs/r0.log` under the catalog's `state_dir`."
    )
    assert "/private/host/locks" not in str(failure)


def test_gpu_enumeration_refusal_names_the_fix_and_the_host_shell_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreadable mocked CUDA hardware says to fix nvidia-smi or rerun from a host shell, on any host.

    The deliberate CPU opt-out stays ``gpu_policy: none``, which this refusal no longer names.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("phasesweep.runtime.gpu._detect_gpu_inventory", lambda: ([], {}))
    monkeypatch.setattr("phasesweep.runtime.gpu._nvidia_driver_reports_gpus", lambda: True)

    with pytest.raises(GpuConfigurationError, match="Fix nvidia-smi") as failure:
        GpuPool.create(n_jobs=1)

    assert "sandboxed shell" in str(failure.value)
    assert "host shell with GPU access" in str(failure.value)
    assert "gpu_policy" not in str(failure.value)

    with GpuPool.create(n_jobs=1, policy="none").acquire() as assignment:
        assert assignment.visible_devices is None
        assert assignment.lease_fds == ()
