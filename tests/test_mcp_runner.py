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

from phasesweep.config import Experiment, Phase, load_config
from phasesweep.engine import read_status, run_experiment
from phasesweep.engine.guards import _experiment_lock
from phasesweep.engine.state import (
    _generation_path,
    _summary_path,
    _trial_dir_for,
    _winner_path,
)
from phasesweep.mcp import runner as mcp_runner
from phasesweep.mcp.runs import RunStore
from phasesweep.mcp.time import utc_now_iso
from phasesweep.runtime.process import _process_group_alive
from tests.conftest import REPO, make_experiment, write_constant_trainer
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
            trainer=REPO / "src" / "phasesweep" / "examples" / "fake_train.py",
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
            phase_dir = _trial_dir_for(experiment, "p", 0).parent
            for trial_dir in phase_dir.glob("trial_00000__*"):
                if (trial_dir / "pid").is_file() and (trial_dir / "pgid").is_file():
                    return trial_dir
        time.sleep(0.2)
    raise AssertionError(f"trial never reached RUNNING; log:\n{log_path.read_text()}")


def test_runner_persists_terminal_evidence_before_snapshot_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_path = tmp_path / "status.json"
    observed: dict[str, object] = {}

    def fail_finalization(*_args: object, **_kwargs: object) -> dict[str, object]:
        observed.update(json.loads(status_path.read_text()))
        raise RuntimeError("snapshot serialization failed")

    monkeypatch.setattr(mcp_runner, "finalize_result_snapshot", fail_finalization)
    mcp_runner._write_status(
        status_path,
        {
            "run_id": "r0",
            "returncode": 143,
            "error_class": "cancelled",
            "cleanup_confirmed": True,
        },
        result_snapshot={},
        result_snapshot_error=None,
    )

    assert observed["error_class"] == "cancelled"
    assert observed["cleanup_confirmed"] is True
    assert "ended_at" in observed
    assert observed["result_snapshot_state"] == "pending"
    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "failed"
    assert final["result_snapshot_error"] == "RuntimeError"
    assert "result_snapshot" not in final


def test_runner_defers_shutdown_until_snapshot_finalization_is_durable(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    child_code = """
import sys
import time
from pathlib import Path

import phasesweep.mcp.runner as runner

runner.install_signal_handlers()

def slow_finalization(snapshot, *, cleanup_confirmed):
    time.sleep(0.5)
    return snapshot

runner.finalize_result_snapshot = slow_finalization
runner._write_status(
    Path(sys.argv[1]),
    {
        "run_id": "r-signal",
        "returncode": 0,
        "error_class": None,
        "cleanup_confirmed": True,
    },
    result_snapshot={"captured": True},
    result_snapshot_error=None,
)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code, str(status_path)],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if status_path.is_file():
                status = json.loads(status_path.read_text())
                if status.get("result_snapshot_state") == "pending":
                    break
            if proc.poll() is not None:
                raise AssertionError(f"finalizer exited early with {proc.returncode}")
            time.sleep(0.01)
        else:
            raise AssertionError("runner did not persist the pending snapshot state")

        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)

    assert proc.returncode == 143
    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "complete"
    assert final["result_snapshot"] == {"captured": True}


def test_runner_finalizes_pre_captured_terminal_snapshot(tmp_path: Path) -> None:
    config = _slow_config(tmp_path)
    status_path = tmp_path / "status.json"
    experiment = load_config(config)
    assert isinstance(experiment, Experiment)
    snapshot = mcp_runner.capture_result_snapshot(experiment, cleanup_confirmed=False)

    mcp_runner._write_status(
        status_path,
        {
            "run_id": "r0",
            "returncode": 0,
            "error_class": None,
            "cleanup_confirmed": True,
        },
        result_snapshot=snapshot,
        result_snapshot_error=None,
    )

    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "complete"
    assert final["result_snapshot"]["status"]["phases"][0]["phase"] == "p"


def test_terminal_snapshot_is_captured_before_experiment_lock_release(tmp_path: Path) -> None:
    trainer = write_constant_trainer(tmp_path)
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        n_trials=1,
    )
    captured: dict[str, object] = {}

    def capture_locked(generation_id: str, error: BaseException | None) -> None:
        assert error is None
        with (
            pytest.raises(RuntimeError, match="Another phasesweep process"),
            _experiment_lock(experiment),
        ):
            pass
        captured.update(
            mcp_runner.capture_result_snapshot(
                experiment,
                cleanup_confirmed=False,
                generation_id=generation_id,
            )
        )

    run_experiment(experiment, terminal_callback=capture_locked)

    top_up = experiment.model_copy(
        update={"phases": [experiment.phases[0].model_copy(update={"n_trials": 2})]}
    )
    run_experiment(top_up)

    captured_phase = captured["status"]["phases"][0]  # type: ignore[index]
    assert captured_phase["n_trials"] == 1
    assert captured_phase["completed"] == 1
    current_phase = mcp_runner.capture_result_snapshot(
        top_up,
        cleanup_confirmed=True,
    )["status"]["phases"][0]
    assert current_phase["n_trials"] == 2
    assert current_phase["completed"] == 2


def test_terminal_snapshot_rejects_a_stale_generation_marker(tmp_path: Path) -> None:
    experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1)
    generation_path = _generation_path(experiment)
    generation_path.parent.mkdir(parents=True)
    generation_path.write_text("generation_id: prior-generation\n")

    with pytest.raises(RuntimeError, match="generation marker does not match"):
        mcp_runner.capture_result_snapshot(
            experiment,
            cleanup_confirmed=False,
            generation_id="failed-new-generation",
        )


def test_successful_terminal_snapshot_rejects_unavailable_trial_data(tmp_path: Path) -> None:
    experiment = make_experiment(workdir=tmp_path / "runs", n_trials=1)

    with pytest.raises(RuntimeError, match="terminal trial data is unavailable.*p"):
        mcp_runner.capture_result_snapshot(
            experiment,
            cleanup_confirmed=False,
            require_trial_data=True,
        )


def test_failed_resume_preflight_preserves_current_generation_and_results(
    tmp_path: Path,
) -> None:
    trainer = write_constant_trainer(tmp_path)
    phases = [
        Phase(name="a", n_trials=1, fixed_overrides={"k": 1}, search_space={}),
        Phase(name="b", n_trials=1, inherits=["a"], search_space={}),
    ]
    experiment = make_experiment(
        workdir=tmp_path / "runs",
        storage=f"sqlite:///{tmp_path / 'studies.db'}",
        trial_command=f"python {trainer} --out {{trial_dir}}/r.json {{overrides}}",
        phases=phases,
    )
    run_experiment(experiment)

    protected_paths = [
        _generation_path(experiment),
        _summary_path(experiment),
        *(_winner_path(experiment, phase.name) for phase in phases),
    ]
    before = {path: path.read_bytes() for path in protected_paths}
    changed = experiment.model_copy(
        update={
            "phases": [
                phases[0].model_copy(update={"fixed_overrides": {"k": 2}}),
                phases[1],
            ]
        }
    )
    callback_generations: list[str] = []

    def capture_failed_resume(generation_id: str, error: BaseException | None) -> None:
        assert isinstance(error, RuntimeError)
        callback_generations.append(generation_id)
        with pytest.raises(RuntimeError, match="generation marker does not match"):
            mcp_runner.capture_result_snapshot(
                changed,
                cleanup_confirmed=False,
                generation_id=generation_id,
            )

    with pytest.raises(RuntimeError, match="different phase config"):
        run_experiment(
            changed,
            from_phase="b",
            terminal_callback=capture_failed_resume,
        )

    assert len(callback_generations) == 1
    assert {path: path.read_bytes() for path in protected_paths} == before


def test_runner_records_snapshot_serialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(
        mcp_runner,
        "finalize_result_snapshot",
        lambda *_args, **_kwargs: {"not_json": object()},
    )

    mcp_runner._write_status(
        status_path,
        {
            "run_id": "r0",
            "returncode": 0,
            "error_class": None,
            "cleanup_confirmed": True,
        },
        result_snapshot={},
        result_snapshot_error=None,
    )

    final = json.loads(status_path.read_text())
    assert final["result_snapshot_state"] == "failed"
    assert final["result_snapshot_error"] == "TypeError"
    assert "result_snapshot" not in final


def test_runner_refuses_to_persist_handle_without_linux_process_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_runner, "read_proc_starttime", lambda _pid: None)

    with pytest.raises(RuntimeError, match="/proc start time is unavailable"):
        mcp_runner._persist_spawned_handle(
            state_dir=tmp_path / "state",
            run_id="r0",
            experiment_id="exp",
            config_sha256="a" * 64,
            started_at=utc_now_iso(),
            allow_cancel=True,
        )

    assert RunStore(tmp_path / "state").get("r0") is None


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
    assert status["result_snapshot_state"] == "complete"
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
