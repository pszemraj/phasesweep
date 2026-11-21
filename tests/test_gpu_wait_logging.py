"""Visible, bounded logging when a trial waits for a GPU lease."""

from __future__ import annotations

import logging

import pytest

from phasesweep.runtime.gpu import GpuDevice, GpuLeaseTimeoutError, GpuPool
from phasesweep.runtime.terminal import ReportBlock


@pytest.mark.parametrize("whole_node", [False, True], ids=["single", "whole-node"])
@pytest.mark.parametrize("blocked_by", ["host-lease", "local-slot"])
def test_gpu_wait_emits_one_info_message(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    whole_node: bool,
    blocked_by: str,
) -> None:
    """One line per wait: another process's lock is a report line, this run's own slots are not."""
    pool = GpuPool([GpuDevice("0", "GPU-test-0")], whole_node=whole_node)
    retries = 0

    def stop_waiting(*args: object, **kwargs: object) -> None:
        nonlocal retries
        retries += 1
        if retries == 3:
            raise GpuLeaseTimeoutError("test lease deadline")

    if blocked_by == "host-lease":
        monkeypatch.setattr("phasesweep.runtime.gpu._try_host_gpu_lease", lambda device: None)
        monkeypatch.setattr(pool, "_wait_for_retry", stop_waiting)
    else:
        pool._available.clear()
        pool._whole_node_in_use = True
        monkeypatch.setattr(pool._condition, "wait", stop_waiting)

    with (
        caplog.at_level(logging.INFO, logger="phasesweep"),
        pytest.raises(GpuLeaseTimeoutError, match="test lease deadline"),
        pool.acquire(),
    ):
        pytest.fail("a blocked GPU lease was acquired")

    records = [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and record.getMessage().startswith("Waiting for")
    ]
    # The console wraps a report line only when it is a block, so the lock wait must be one.
    assert [isinstance(record.msg, ReportBlock) for record in records] == [
        blocked_by == "host-lease"
    ]
    waiting = [(record.name, record.getMessage()) for record in records]
    assert waiting == [
        ("phasesweep.report", "Waiting for a GPU: another process holds its lock.")
        if blocked_by == "host-lease"
        else ("phasesweep.runtime.gpu", "Waiting for GPU lease: local GPU slots are in use.")
    ]
