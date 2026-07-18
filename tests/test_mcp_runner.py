"""Detached runner: exercises the real subprocess, the engine's signal teardown,
and the status.json written on the cancel path.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from phasesweep.config import load_config
from phasesweep.engine import read_status
from phasesweep.engine.state import _trial_dir_for
from phasesweep.mcp import runner as mcp_runner
from phasesweep.mcp.runs import RunStore
from phasesweep.mcp.time import utc_now_iso
from phasesweep.runtime.process import _process_group_alive
from tests.conftest import REPO
from tests.mcp_helpers import slow_mcp_config_text

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="cancel path relies on POSIX process groups + /proc liveness",
)


def _slow_config(tmp_path: Path, *, sleep: float = 30.0) -> Path:
    config = tmp_path / "exp.yaml"
    config.write_text(
        slow_mcp_config_text(
            tmp_path,
            trainer=REPO / "examples" / "fake_train.py",
            name="cancel_me",
            sleep=sleep,
        )
    )
    return config


def _wait_for_running_trial(config: Path, proc: subprocess.Popen, log_path: Path) -> Path:
    experiment = load_config(config)
    deadline = time.time() + 25
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"runner exited early ({proc.returncode}); log:\n{log_path.read_text()}"
            )
        status = read_status(experiment)
        if status["phases"][0]["running"] >= 1:
            trial_dir = _trial_dir_for(experiment, "p", 0)
            if (trial_dir / "pid").is_file() and (trial_dir / "pgid").is_file():
                return trial_dir
        time.sleep(0.2)
    raise AssertionError(f"trial never reached RUNNING; log:\n{log_path.read_text()}")


def test_runner_persists_terminal_evidence_before_snapshot_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _slow_config(tmp_path)
    status_path = tmp_path / "status.json"
    observed: dict[str, object] = {}

    def fail_snapshot(*_args: object, **_kwargs: object) -> dict[str, object]:
        observed.update(json.loads(status_path.read_text()))
        raise RuntimeError("snapshot read stalled")

    monkeypatch.setattr(mcp_runner, "capture_result_snapshot", fail_snapshot)
    mcp_runner._write_status(
        status_path,
        {
            "run_id": "r0",
            "returncode": 143,
            "error_class": "cancelled",
            "cleanup_confirmed": True,
        },
        config_path=config,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    )

    assert observed["error_class"] == "cancelled"
    assert observed["cleanup_confirmed"] is True
    assert "ended_at" in observed
    assert json.loads(status_path.read_text()) == observed


def test_runner_cancel_records_cancelled(tmp_path: Path) -> None:
    config = _slow_config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r1"
    status_path = store.status_path(run_id)
    log_path = store.log_path(run_id)
    cmd = [
        sys.executable,
        "-m",
        "phasesweep.mcp.runner",
        "--run-id",
        run_id,
        "--config",
        str(config),
        "--config-sha256",
        hashlib.sha256(config.read_bytes()).hexdigest(),
        "--status-path",
        str(status_path),
        "--state-dir",
        str(tmp_path / "state"),
        "--experiment-id",
        "cancel_me",
        "--started-at",
        utc_now_iso(),
    ]
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # runner is its own session/group leader -> pgid == pid
        )
    try:
        trial_dir = _wait_for_running_trial(config, proc, log_path)
        trial_pgid = int((trial_dir / "pgid").read_text())

        # SIGTERM the runner's process group. The trial runs in its OWN session,
        # so this does not reach it directly; the runner's installed shutdown
        # handler tears the trial group down, then exits 128+SIGTERM = 143.
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)

    assert proc.returncode == 143, (
        f"expected 143, got {proc.returncode}; log:\n{log_path.read_text()}"
    )
    status = json.loads(status_path.read_text())
    assert status["returncode"] == 143
    assert status["error_class"] == "cancelled"
    assert status["cleanup_confirmed"] is True
    assert status["result_snapshot"]["status"]["phases"][0]["trials"]["FAIL"] == 1
    assert status["result_snapshot"]["winners"] == []
    assert not _process_group_alive(trial_pgid)


def test_runner_cancelled_before_first_trial_still_records_cancelled(tmp_path: Path) -> None:
    """A cancel arriving before any trial starts must still write status.json.

    The runner installs shutdown handlers before persisting its handle, so once
    the handle file exists a SIGTERM is guaranteed to be caught — even while the
    config is still loading. Without the early install, a cancel in that window
    killed the runner with the default disposition, no status was recorded, and
    the run derived "running" behind its cleanup-uncertainty marker forever.
    """
    config = _slow_config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r2"
    status_path = store.status_path(run_id)
    log_path = store.log_path(run_id)
    cmd = [
        sys.executable,
        "-m",
        "phasesweep.mcp.runner",
        "--run-id",
        run_id,
        "--config",
        str(config),
        "--config-sha256",
        hashlib.sha256(config.read_bytes()).hexdigest(),
        "--status-path",
        str(status_path),
        "--state-dir",
        str(tmp_path / "state"),
        "--experiment-id",
        "cancel_me",
        "--started-at",
        utc_now_iso(),
    ]
    handle_path = tmp_path / "state" / "runs" / f"{run_id}.json"
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        # The self-persisted handle is written after handler installation, so
        # its appearance proves SIGTERM will be caught from here on.
        deadline = time.time() + 25
        while not handle_path.is_file():
            if proc.poll() is not None:
                raise AssertionError(
                    f"runner exited early ({proc.returncode}); log:\n{log_path.read_text()}"
                )
            if time.time() > deadline:
                raise AssertionError(f"handle never appeared; log:\n{log_path.read_text()}")
            time.sleep(0.02)
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)

    assert proc.returncode == 143, (
        f"expected 143, got {proc.returncode}; log:\n{log_path.read_text()}"
    )
    status = json.loads(status_path.read_text())
    assert status["returncode"] == 143
    assert status["error_class"] == "cancelled"
    assert status["cleanup_confirmed"] is True
