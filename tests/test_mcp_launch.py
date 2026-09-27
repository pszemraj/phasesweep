"""MCP server launch, preflight, and concurrency behavior. Logic that does not need a real detached runner."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import optuna
import pytest
import yaml

import phasesweep.engine.ledger as engine_ledger
import phasesweep.mcp.run_control as mcp_run_control
import phasesweep.mcp.runner as mcp_runner
import phasesweep.mcp.runs as mcp_runs
from phasesweep.config import (
    Experiment,
    FloatParam,
    IntParam,
    Phase,
    Sampler,
    load_experiment,
)
from phasesweep.engine import (
    TerminalReport,
    run_experiment,
)
from phasesweep.engine.errors import StudyFingerprintMismatchError, StudySchemaMismatchError
from phasesweep.engine.paths import (
    _experiment_dir,
    _generation_record_path,
)
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
)
from phasesweep.mcp.audit import AuditLogger
from phasesweep.mcp.errors import (
    ConcurrencyLimitError,
    ConfigChangedError,
    ExperimentBusyError,
    InvalidPhaseError,
    PermissionDeniedError,
    ResumeNotReadyError,
    RunCapacityUnknownError,
    UnknownExperimentError,
)
from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runs import RunHandle, RunStore
from phasesweep.mcp.server import (
    _safe_tool,
)
from phasesweep.mcp.snapshots import capture_result_snapshot
from phasesweep.mcp.tool_names import TOOL_LAUNCH_RUN
from phasesweep.mcp.tools import PhaseSweepMCP
from phasesweep.runtime.files import open_private_text
from phasesweep.runtime.reaper import (
    read_boot_id,
    read_proc_starttime,
)
from tests.conftest import (
    file_mode,
    make_experiment,
    mark_current_format,
    reaped_pid,
    write_constant_trainer,
)
from tests.ledger_fixtures import republish_as_incomplete
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    _catalog,
    _config,
    _drift_experiment,
    _write_experiment_config,
    claim_runner_handle,
    live_runs,
    make_mcp_app,
    make_run_handle,
    patch_popen_capture,
    runner_argv,
    runner_main,
    write_mcp_catalog,
    write_run_status,
)

RESUMABLE_PHASES = """\
  - name: p
    n_trials: 1
    sampler: { type: random, seed: 0 }
    search_space:
      lr: { type: float, low: 1.0e-5, high: 1.0e-2, log: true }
  - name: q
    inherits: [p]
    n_trials: 1
    sampler: { type: random, seed: 1 }
    search_space:
      wd: { type: float, low: 0.0, high: 0.1 }
"""


def _launch_with_poison_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Path, Path, dict[str, str]]:
    """Launch against a project dir seeded with import-time code and capture the spawn.

    The project directory holds every hook that runs before the runner's first
    statement: a ``phasesweep`` shadow package (found when the cwd is on
    ``sys.path``), a ``sitecustomize`` (imported by ``site`` when it is
    importable at all), and a ``PYTHONSTARTUP``/``PYTHONPATH`` pair exported
    into the server's own environment. Each writes a marker file naming itself.

    The patches are undone before returning so callers can spawn real children:
    patching ``phasesweep.mcp.run_control.subprocess.Popen`` patches the one shared
    ``subprocess`` module.

    :param Path tmp_path: Test-scoped directory.
    :param pytest.MonkeyPatch monkeypatch: Patcher for env and ``Popen``.
    :return tuple[dict[str, Any], Path, Path, dict[str, str]]: Captured spawn,
        project dir, marker dir, and the poisoned parent environment.
    """
    config = _config(tmp_path)
    project = tmp_path / "project"
    markers = tmp_path / "markers"
    markers.mkdir()
    shadow = project / "phasesweep" / "mcp"
    shadow.mkdir(parents=True)

    def poison(path: Path, name: str) -> None:
        path.write_text(
            "import pathlib\n"
            f"pathlib.Path({str(markers)!r}).joinpath({name!r}).write_text('executed')\n"
        )

    poison(project / "phasesweep" / "__init__.py", "shadow_package")
    poison(shadow / "__init__.py", "shadow_mcp")
    poison(shadow / "runner.py", "shadow_runner")
    poison(project / "sitecustomize.py", "sitecustomize")
    poison(project / "startup.py", "pythonstartup")

    # Scoped, so leaving it restores only these patches and the test's own
    # isolation fixtures stay in force.
    with monkeypatch.context() as scoped:
        scoped.setenv("PYTHONPATH", str(project))
        scoped.setenv("PYTHONSTARTUP", str(project / "startup.py"))
        scoped.setenv("PYTHONHOME", str(project))
        scoped.setenv("PYTHONEXECUTABLE", str(project / "python"))
        app, _registry, _store = make_mcp_app(
            write_mcp_catalog(
                tmp_path, {"srv": config}, allow=ALLOW_SIDE_EFFECTS, cwd={"srv": project}
            )
        )
        captured = patch_popen_capture(scoped)
        app.launch("srv")
        parent_env = dict(os.environ)
    return captured, project, markers, parent_env


def test_safe_tool_returns_safe_mcp_error() -> None:
    @_safe_tool
    def boom() -> None:
        raise UnknownExperimentError("srv")

    with pytest.raises(ValueError, match="unknown experiment id 'srv'"):
        boom()


def test_concurrency_limit_error_bounds_actionable_run_ids() -> None:
    run_ids = [
        "block-alpha",
        "block-beta",
        "block-gamma",
        "block-delta",
        "block-epsilon",
        "block-zeta",
        "block-eta",
    ]

    message = str(ConcurrencyLimitError(7, 3, run_ids))

    for run_id in run_ids[:5]:
        assert repr(run_id) in message
    for run_id in run_ids[5:]:
        assert repr(run_id) not in message
    assert "2 more active" in message
    assert "await_run" in message
    assert "only after that run is terminal" in message


def test_safe_tool_redacts_unexpected_exception() -> None:
    @_safe_tool
    def boom() -> None:
        raise OSError("/tmp/SECRET_PATH/config.yaml")

    with pytest.raises(ValueError) as excinfo:
        boom()
    assert str(excinfo.value) == (
        "internal server error in boom; report it to the operator and do not retry immediately"
    )
    assert "SECRET_PATH" not in str(excinfo.value)


def test_launch_permission_denied_before_spawn(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config))

    with pytest.raises(PermissionDeniedError, match="action 'launch' is not permitted"):
        app.launch("srv")


def test_from_phase_permission_denied_before_validation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, _registry, _store = make_mcp_app(
        _catalog(tmp_path, config, allow={"launch": True, "cancel": True, "from_phase": False}),
    )

    with pytest.raises(PermissionDeniedError, match="action 'from_phase' is not permitted"):
        app.launch("srv", from_phase="p")


def test_invalid_from_phase_rejected_before_spawn(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))

    with pytest.raises(InvalidPhaseError, match="phase 'missing' is not a phase"):
        app.launch("srv", from_phase="missing")


def test_resume_requires_prior_winner(tmp_path: Path) -> None:
    config = _config(tmp_path, phases=RESUMABLE_PHASES)
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))

    with pytest.raises(ResumeNotReadyError, match="earlier phase 'p' has no winner yet") as excinfo:
        app.launch("srv", from_phase="q")

    assert "get_latest_run('srv') first" in str(excinfo.value)
    assert "get_run_results with its run_id" in str(excinfo.value)


@pytest.mark.parametrize(
    ("lr_high", "incomplete"),
    [
        pytest.param(3, False, id="stale"),
        pytest.param(2, True, id="incomplete"),
    ],
)
@pytest.mark.integration
def test_resume_rejects_incompatible_winner_before_spawn(
    tmp_path: Path,
    lr_high: int,
    incomplete: bool,
) -> None:
    trainer = write_constant_trainer(tmp_path)
    published = _drift_experiment(tmp_path, trainer)
    run_experiment(published)
    if incomplete:
        republish_as_incomplete(published)
    # The operator appends phase q, which resumes from p's published winner; a
    # wider p search space than the one published changes p's fingerprint.
    (p,) = published.phases
    p = p.model_copy(update={"search_space": {"lr": IntParam(type="int", low=1, high=lr_high)}})
    q = Phase(
        name="q",
        inherits=["p"],
        n_trials=1,
        sampler=Sampler(type="random", seed=1),
        search_space={"wd": FloatParam(type="float", low=0.0, high=0.1)},
    )
    config = tmp_path / "srv.yaml"
    _write_experiment_config(config, _drift_experiment(tmp_path, trainer, phases=[p, q]))
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))

    with pytest.raises(ResumeNotReadyError, match="compatible winner"):
        app.launch("srv", from_phase="q")

    assert store.list_handles() == []


def test_launch_refuses_config_changed_after_registry_load(tmp_path: Path) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    config.write_text(config.read_text().replace("python train.py", "python changed.py"))

    with pytest.raises(ConfigChangedError, match="changed since server startup"):
        app.launch("srv")

    assert store.list_handles() == []


@pytest.mark.parametrize("record_kind", ["malformed_handle", "orphan_config"])
def test_launch_refuses_when_persisted_run_capacity_is_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_kind: str,
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    captured = patch_popen_capture(monkeypatch)
    if record_kind == "malformed_handle":
        (tmp_path / "state" / "runs" / "srv-broken.json").write_text("{not valid json")
    else:
        with open_private_text(store.config_snapshot_path("srv-orphan"), "x") as output:
            output.write("experiment: srv\n")

    with pytest.raises(RunCapacityUnknownError, match="cannot prove available launch capacity"):
        app.launch("srv")

    assert "cmd" not in captured
    assert store.list_handles() == []


def test_launch_passes_config_snapshot_to_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(
        _catalog(
            tmp_path,
            config,
            allow=ALLOW_SIDE_EFFECTS,
            visible_params=["lr"],
        )
    )
    captured = patch_popen_capture(monkeypatch)
    original_create = store.create
    snapshots_before_publish: list[bytes] = []

    def create_after_snapshot(handle: RunHandle) -> None:
        snapshots_before_publish.append(store.config_snapshot_path(handle.run_id).read_bytes())
        original_create(handle)

    monkeypatch.setattr(store, "create", create_after_snapshot)

    result = app.launch("srv")

    cmd = captured["cmd"]
    config_arg = Path(cmd[cmd.index("--config") + 1])
    sha_arg = cmd[cmd.index("--config-sha256") + 1]
    assert config_arg != config.resolve()
    assert config_arg.read_bytes() == config.read_bytes()
    assert snapshots_before_publish == [config.read_bytes()]
    assert sha_arg == registry.get("srv").config_sha256
    assert config_arg == registry.state_dir / "logs" / f"{result['run_id']}.config.yaml"
    assert Path(cmd[cmd.index("--state-dir") + 1]) == registry.state_dir
    assert cmd[cmd.index("--experiment-id") + 1] == "srv"
    assert list(config_arg.parent.glob("*.tmp")) == []
    assert list(config_arg.parent.glob(".*.tmp")) == []
    handle = store.get(result["run_id"])
    assert handle is not None
    assert handle.launch_state == "spawned"
    assert handle.allow_cancel is True
    assert handle.visible_params_at_launch == ["lr"]
    assert cmd[cmd.index("--started-at") + 1] == handle.started_at


def test_launch_retries_run_id_collision_without_touching_existing_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    collision_id = "srv-collision"
    fresh_id = "srv-fresh"
    existing = make_run_handle(
        run_id=collision_id,
        experiment_id="srv",
        config_sha256=registry.get("srv").config_sha256,
        pid=reaped_pid(),
        starttime=111,
    )
    store.create(existing)
    store.config_snapshot_path(collision_id).write_bytes(b"existing config\n")
    store.log_path(collision_id).write_bytes(b"existing log\n")
    write_run_status(store, collision_id, returncode=0, cleanup_confirmed=True)
    before = {
        path: path.read_bytes()
        for path in (
            registry.state_dir / "runs" / f"{collision_id}.json",
            store.config_snapshot_path(collision_id),
            store.log_path(collision_id),
            store.status_path(collision_id),
        )
    }
    minted = iter((collision_id, fresh_id))
    monkeypatch.setattr(store, "new_run_id", lambda _experiment_id: next(minted))

    result = app.launch("srv")

    assert result["run_id"] == fresh_id
    assert store.get(collision_id) == existing
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.integration
def test_runner_persists_spawned_handle_for_restart_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "srv-recover"
    started_at = "2026-06-24T00:00:00Z"
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
    )
    calls: list[tuple[str, str | None, bool]] = []

    def fake_run_experiment(
        config_obj: Experiment,
        *,
        from_phase: str | None,
        dry_run: bool,
        terminal_callback,
        generation_id: str,
        publication_hook: object,
    ) -> None:
        del publication_hook
        assert generation_id == run_id
        calls.append((config_obj.experiment, from_phase, dry_run))
        mark_current_format(config_obj)
        generation_path = _generation_record_path(config_obj, generation_id)
        generation_path.parent.mkdir(parents=True, exist_ok=True)
        generation_path.write_text(f"generation_id: {generation_id}\n")
        terminal_callback(
            TerminalReport(
                generation_id=generation_id,
                primary_error=None,
                cleanup_confirmed=True,
                recovered_attempt_ids=frozenset(),
                uncertain_attempt_ids=frozenset(),
            )
        )

    monkeypatch.setattr("phasesweep.mcp.runner.run_experiment", fake_run_experiment)

    def capture_without_fake_trial_storage(
        experiment: Experiment,
        *,
        generation_id: str,
        engine_winners: object = None,
    ) -> dict[str, object]:
        return capture_result_snapshot(
            experiment,
            generation_id=generation_id,
        )

    monkeypatch.setattr(
        "phasesweep.mcp.runner.capture_result_snapshot",
        capture_without_fake_trial_storage,
    )

    assert (
        runner_main(
            runner_argv(
                store,
                run_id=run_id,
                config=config,
                config_sha256=config_sha256,
                experiment_id="srv",
                started_at=started_at,
            )
        )
        == 0
    )

    handle = store.get(run_id)
    assert handle is not None
    assert handle.launch_state == "spawned"
    assert handle.experiment_id == "srv"
    assert handle.config_sha256 == config_sha256
    assert handle.pid == os.getpid()
    assert handle.pgid == (os.getpgrp() if hasattr(os, "getpgrp") else os.getpid())
    assert handle.pid_starttime == read_proc_starttime(os.getpid())
    assert handle.started_at == started_at
    assert store.state(handle) == "succeeded"
    assert calls == [("srv", None, False)]
    terminal = store.recorded_terminal_status(handle)
    assert terminal is not None
    assert terminal["result_snapshot_state"] == "complete"
    assert terminal["result_snapshot"]["status"]["phases"][0]["phase"] == "p"
    assert terminal["result_snapshot"]["winners"] == []


def test_launch_does_not_spawn_when_pending_handle_create_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    captured = patch_popen_capture(monkeypatch)

    def fail_create(handle: RunHandle) -> None:
        raise OSError("runs directory is not writable")

    monkeypatch.setattr(store, "create", fail_create)

    with pytest.raises(OSError, match="runs directory"):
        app.launch("srv")

    assert "cmd" not in captured
    assert store.list_handles() == []
    assert list((tmp_path / "state" / "logs").glob("*.config.yaml")) == []


def test_launch_finalizes_pending_handle_when_popen_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))

    def fail_popen(*args: object, **kwargs: object) -> None:
        raise OSError("runner executable is unavailable")

    monkeypatch.setattr("phasesweep.mcp.run_control.subprocess.Popen", fail_popen)

    with pytest.raises(OSError, match="runner executable"):
        app.launch("srv")

    (handle,) = store.list_handles()
    assert handle.launch_state == "launching"
    assert store.state(handle) == "failed"
    assert not store.recovery_required(handle)
    assert live_runs(store) == []
    terminal = store.recorded_terminal_status(handle)
    assert terminal is not None
    assert terminal["error_class"] == "OSError"
    assert terminal["cleanup_confirmed"] is True
    assert terminal["result_snapshot_state"] == "failed"
    assert terminal["failure"] == {
        "code": "internal_error",
        "stage": "preflight",
        "retryable": False,
        "actor": "operator",
        "remediation": (
            "Ask the operator to inspect the PhaseSweep server diagnostics before retrying."
        ),
    }
    latest = app.latest_run("srv")
    failure = latest["run"]["failure"]
    assert failure["code"] == "result_snapshot_unavailable"
    assert failure["cause"] == terminal["failure"]
    assert app.status(run_id=handle.run_id)["run"]["failure"] == failure
    assert not store.launch_lease_path(handle.run_id).exists()
    patch_popen_capture(monkeypatch)
    assert app.launch("srv")["state"] == "running"


def test_launch_retains_recoverable_lease_when_failure_status_cannot_persist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed status write cannot erase proof that Popen never succeeded."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))

    with monkeypatch.context() as faults:

        def fail_popen(*args: object, **kwargs: object) -> None:
            raise OSError("runner executable is unavailable")

        def fail_status(*args: object, **kwargs: object) -> None:
            raise OSError("status directory is unavailable")

        faults.setattr(mcp_run_control.subprocess, "Popen", fail_popen)
        faults.setattr(mcp_run_control, "write_status_file_if_absent", fail_status)

        with pytest.raises(OSError, match="runner executable"):
            app.launch("srv")

    (pending,) = store.list_handles()
    assert store.recorded_terminal_status(pending) is None
    assert store.launch_lease_path(pending.run_id).is_file()
    assert store.is_pre_spawn_orphan(pending.run_id)

    patch_popen_capture(monkeypatch)
    assert app.launch("srv")["state"] == "running"
    assert store.get(pending.run_id) is None


@pytest.mark.integration
def test_launch_terminates_real_runner_when_log_context_exit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child created before log close fails is owned and terminated."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    real_popen = subprocess.Popen
    real_open_private_text = mcp_run_control.open_private_text
    spawned: list[subprocess.Popen[Any]] = []

    @contextlib.contextmanager
    def fail_log_close(path: Path, mode: str) -> Iterator[Any]:
        with real_open_private_text(path, mode) as handle:
            yield handle
        if path.suffix == ".log":
            raise OSError("injected log context exit failure")

    def sleeping_popen(cmd: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
        proc = real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            **kwargs,
        )
        run_id = cmd[cmd.index("--run-id") + 1]
        pending = store.get(run_id)
        assert pending is not None
        store.update(
            replace(
                pending,
                pid=proc.pid,
                pgid=proc.pid,
                pid_starttime=read_proc_starttime(proc.pid),
                boot_id=read_boot_id(),
                launch_state="spawned",
            )
        )
        os.write(int(cmd[cmd.index("--launch-ready-fd") + 1]), b"R")
        spawned.append(proc)
        return proc

    monkeypatch.setattr(mcp_run_control, "open_private_text", fail_log_close)
    monkeypatch.setattr(mcp_run_control.subprocess, "Popen", sleeping_popen)
    try:
        with pytest.raises(OSError, match="log context exit"):
            app.launch("srv")
        assert len(spawned) == 1
        spawned[0].wait(timeout=5)

        (pending,) = store.list_handles()
        terminal = store.recorded_terminal_status(pending)
        assert terminal is not None
        assert terminal["cleanup_confirmed"] is True
        assert terminal["error_class"] == "OSError"
        assert store.state(pending) == "failed"
        assert not store.recovery_required(pending)
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)


@pytest.mark.parametrize(
    ("failing_reader", "cleanup_confirmed", "expected_state"),
    [
        ("starttime", True, "failed"),
        ("boot_id", True, "failed"),
        ("starttime", False, "running"),
    ],
)
def test_launch_records_real_cleanup_result_when_process_identity_read_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_reader: str,
    cleanup_confirmed: bool,
    expected_state: str,
) -> None:
    """A fallible identity read cannot turn a created child into a pre-spawn failure."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    cleanup_calls: list[tuple[int | None, int | None, int | None]] = []

    def fail_identity_read(*_args: object) -> None:
        raise OSError(f"injected {failing_reader} read failure")

    def fake_cleanup(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        cleanup_calls.append((pid, saved_starttime, pgid))
        return cleanup_confirmed

    monkeypatch.setattr(mcp_run_control, "kill_stale_group", fake_cleanup)
    monkeypatch.setattr(
        mcp_run_control,
        "read_proc_starttime" if failing_reader == "starttime" else "read_boot_id",
        fail_identity_read,
    )

    with pytest.raises(OSError, match=failing_reader):
        app.launch("srv")

    (pending,) = store.list_handles()
    terminal = store.recorded_terminal_status(pending)
    assert terminal is not None
    assert terminal["cleanup_confirmed"] is cleanup_confirmed
    assert store.state(pending) == expected_state
    assert store.recovery_required(pending) is (not cleanup_confirmed)
    assert cleanup_calls
    if failing_reader == "starttime":
        assert cleanup_calls[0][1] is None
    else:
        assert cleanup_calls[0][1] is not None


def test_launch_cleans_up_when_identity_bookkeeping_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KeyboardInterrupt after Popen is cleanup work, not a pre-spawn failure."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    cleanup_calls: list[tuple[int | None, int | None, int | None]] = []

    def interrupt_boot_id() -> None:
        raise KeyboardInterrupt

    def fake_cleanup(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        cleanup_calls.append((pid, saved_starttime, pgid))
        return True

    monkeypatch.setattr(mcp_run_control, "read_boot_id", interrupt_boot_id)
    monkeypatch.setattr(mcp_run_control, "kill_stale_group", fake_cleanup)

    with pytest.raises(KeyboardInterrupt):
        app.launch("srv")

    (pending,) = store.list_handles()
    terminal = store.recorded_terminal_status(pending)
    assert terminal is not None
    assert terminal["cleanup_confirmed"] is True
    assert terminal["error_class"] == "KeyboardInterrupt"
    assert store.state(pending) == "failed"
    assert cleanup_calls and cleanup_calls[0][1] is not None


def test_restarted_server_reserves_unresolved_launching_handle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _first, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    pending = make_run_handle(
        run_id="srv-launch-gap",
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        launch_state="launching",
    )
    store.create(pending)

    restarted = PhaseSweepMCP(registry, RunStore(registry.state_dir))
    with pytest.raises(ExperimentBusyError, match="already has a running sweep"):
        restarted.launch("srv")

    assert store.recovery_required(pending)
    assert live_runs(store) == [pending]


def test_restarted_server_reaps_abandoned_transaction_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A retry preserves diagnostics from an abandoned launch preparation."""
    config = _config(tmp_path)
    _first, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    abandoned_id = "srv-abandoned-preparation"
    pending = make_run_handle(
        run_id=abandoned_id,
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        launch_state="launching",
    )
    preparation = store.prepare_launch(pending, config.read_bytes())
    abandoned_log = store.log_path(abandoned_id)
    recovered_log = abandoned_log.with_suffix(".log.recovered")
    log_bytes = b"runner import failed before the launch receipt\n"
    abandoned_log.write_bytes(log_bytes)
    preparation.close()  # Simulate both server and child disappearing before a receipt.

    restarted_store = RunStore(registry.state_dir)
    restarted = PhaseSweepMCP(registry, restarted_store)
    monkeypatch.setattr(restarted_store, "new_run_id", lambda _experiment_id: "srv-retry")
    patch_popen_capture(monkeypatch)

    with caplog.at_level(logging.INFO, logger="phasesweep.mcp.server"):
        result = restarted.launch("srv")

    assert result["run_id"] == "srv-retry"
    assert set(result) == {"experiment_id", "run_id", "state"}
    assert (
        "phasesweep.mcp.server",
        logging.INFO,
        f"preserved abandoned launch log run={abandoned_id} at {recovered_log}",
    ) in caplog.record_tuples
    assert restarted_store.get(abandoned_id) is None
    assert not restarted_store.config_snapshot_path(abandoned_id).exists()
    assert not restarted_store.launch_lease_path(abandoned_id).exists()
    assert not abandoned_log.exists()
    assert recovered_log.read_bytes() == log_bytes
    handles, unreadable = restarted_store.launch_inventory()
    assert [handle.run_id for handle in handles] == ["srv-retry"]
    assert unreadable == set()


def test_missing_runner_receipt_fails_without_reserving_retry_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that never publishes a receipt never receives permission to work."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))

    class UnreadyProc:
        pid = os.getppid()

    monkeypatch.setattr(
        mcp_run_control.subprocess, "Popen", lambda *_args, **_kwargs: UnreadyProc()
    )
    monkeypatch.setattr(mcp_run_control, "kill_stale_group", lambda *_args, **_kwargs: True)

    with pytest.raises(RuntimeError, match="did not persist its launch receipt"):
        app.launch("srv")

    (failed,) = store.list_handles()
    assert store.state(failed) == "failed"
    assert not store.launch_lease_path(failed.run_id).exists()
    patch_popen_capture(monkeypatch)
    assert app.launch("srv")["state"] == "running"


def test_acknowledgement_write_failure_keeps_spawn_cleanup_reserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted ACK write cannot prove the child stayed behind the barrier."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    cleanup_calls: list[tuple[int | None, int | None, int | None]] = []
    real_write = os.write

    def fail_ack(fd: int, data: bytes) -> int:
        if data == b"A":
            raise OSError("injected acknowledgement failure")
        return real_write(fd, data)

    def confirm_cleanup(
        pid: int | None,
        starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        cleanup_calls.append((pid, starttime, pgid))
        return True

    monkeypatch.setattr(mcp_run_control.os, "write", fail_ack)
    monkeypatch.setattr(mcp_run_control, "kill_stale_group", confirm_cleanup)

    with pytest.raises(OSError, match="acknowledgement"):
        app.launch("srv")

    (handle,) = store.list_handles()
    assert handle.launch_state == "spawned"
    assert cleanup_calls == [(handle.pid, handle.pid_starttime, handle.pgid)]
    assert store.state(handle) == "running"
    assert store.cleanup_uncertain(handle)
    assert store.recovery_required(handle)
    status = store.recorded_terminal_status(handle)
    assert status is not None
    assert status["cleanup_confirmed"] is False
    assert not store.launch_lease_path(handle.run_id).exists()


def test_interruption_after_ack_keeps_spawn_cleanup_reserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    cleanup_calls: list[tuple[int | None, int | None, int | None]] = []
    real_write = os.write

    def interrupt_after_ack(fd: int, data: bytes) -> int:
        written = real_write(fd, data)
        if data == b"A":
            raise KeyboardInterrupt
        return written

    def confirm_runner_group_cleanup(
        pid: int | None,
        starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        cleanup_calls.append((pid, starttime, pgid))
        return True

    monkeypatch.setattr(mcp_run_control.os, "write", interrupt_after_ack)
    monkeypatch.setattr(mcp_run_control, "kill_stale_group", confirm_runner_group_cleanup)

    with pytest.raises(KeyboardInterrupt):
        app.launch("srv")

    (handle,) = store.list_handles()
    assert cleanup_calls == [(handle.pid, handle.pid_starttime, handle.pgid)]
    assert store.cleanup_uncertain(handle)
    assert store.recovery_required(handle)
    status = store.recorded_terminal_status(handle)
    assert status is not None
    assert status["cleanup_confirmed"] is False


def test_completed_lease_cleanup_failure_cannot_replace_launch_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lease sidecar cleanup is non-authoritative after an acknowledged receipt."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    real_unlink = mcp_runs._strict_unlink

    def fail_lease_unlink(path: Path) -> None:
        if path.suffix == ".lock":
            raise OSError("injected lease cleanup failure")
        real_unlink(path)

    monkeypatch.setattr(mcp_runs, "_strict_unlink", fail_lease_unlink)
    caplog.set_level(logging.WARNING, logger="phasesweep.mcp.runs")

    result = app.launch("srv")

    assert result["state"] == "running"
    assert "could not remove completed launch lease" in caplog.text
    assert store.launch_lease_path(result["run_id"]).is_file()


@pytest.mark.parametrize(
    ("cleanup_confirmed", "expected_state"),
    [(True, "failed"), (False, "running")],
)
def test_launch_refuses_runner_without_linux_process_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_confirmed: bool,
    expected_state: str,
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    cleanup_calls: list[tuple[int | None, int | None, int | None]] = []

    def fake_cleanup(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        cleanup_calls.append((pid, saved_starttime, pgid))
        return cleanup_confirmed

    monkeypatch.setattr("phasesweep.mcp.run_control.read_proc_starttime", lambda _pid: None)
    monkeypatch.setattr("phasesweep.mcp.run_control.kill_stale_group", fake_cleanup)

    with pytest.raises(RuntimeError, match="has no Linux /proc start time"):
        app.launch("srv")

    handles = store.list_handles()
    assert len(handles) == 1
    assert handles[0].launch_state == "spawned"
    assert store.state(handles[0]) == expected_state
    assert store.recovery_required(handles[0]) is (not cleanup_confirmed)
    assert bool(store.cleanup_uncertain(handles[0])) is (not cleanup_confirmed)
    assert cleanup_calls and cleanup_calls[0][1] is None


def test_launch_terminates_spawned_runner_when_handle_update_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    updated: list[RunHandle] = []
    terminated: list[tuple[int | None, int | None, int | None]] = []

    def fail_update(handle: RunHandle) -> None:
        updated.append(handle)
        raise OSError("runs directory is not writable")

    def fake_kill_stale_group(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        terminated.append((pid, saved_starttime, pgid))
        return True

    monkeypatch.setattr(store, "update", fail_update)
    monkeypatch.setattr("phasesweep.mcp.run_control.kill_stale_group", fake_kill_stale_group)

    with pytest.raises(OSError, match="runs directory"):
        app.launch("srv")

    assert [handle.launch_state for handle in updated] == ["spawned"]
    spawned = updated[0]
    assert terminated == [(spawned.pid, spawned.pid_starttime, spawned.pgid)]
    pending = store.get(spawned.run_id)
    assert pending is not None
    assert pending.launch_state == "spawned"
    assert store.state(pending) == "running"
    assert store.recovery_required(pending)
    assert store.cleanup_uncertain(pending)


def test_launch_interrupt_during_handle_update_terminates_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shutdown interrupt cannot escape after spawn but before durable identity."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    terminated: list[tuple[int | None, int | None, int | None]] = []

    def interrupt_update(_handle: RunHandle) -> None:
        raise KeyboardInterrupt

    def fake_cleanup(
        pid: int | None,
        saved_starttime: int | None,
        *,
        pgid: int | None = None,
    ) -> bool:
        terminated.append((pid, saved_starttime, pgid))
        return True

    monkeypatch.setattr(store, "update", interrupt_update)
    monkeypatch.setattr(mcp_run_control, "kill_stale_group", fake_cleanup)

    with pytest.raises(KeyboardInterrupt):
        app.launch("srv")

    (pending,) = store.list_handles()
    terminal = store.recorded_terminal_status(pending)
    assert terminal is not None
    assert terminal["cleanup_confirmed"] is False
    assert terminal["error_class"] == "KeyboardInterrupt"
    assert store.state(pending) == "running"
    assert store.recovery_required(pending)
    assert terminated and terminated[0][1] is not None


def test_launch_logs_when_cleanup_marker_write_fails_after_update_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    updated: list[RunHandle] = []

    def fail_update(handle: RunHandle) -> None:
        updated.append(handle)
        raise OSError("runs directory is not writable")

    def fail_marker(_handle: RunHandle) -> None:
        raise OSError("logs directory is not writable")

    monkeypatch.setattr(store, "update", fail_update)
    monkeypatch.setattr(store, "mark_cleanup_uncertain", fail_marker)
    monkeypatch.setattr(
        "phasesweep.mcp.run_control.kill_stale_group", lambda *args, **kwargs: False
    )
    caplog.set_level(logging.ERROR, logger="phasesweep.mcp.server")

    with pytest.raises(OSError, match="runs directory"):
        app.launch("srv")

    assert [handle.launch_state for handle in updated] == ["spawned"]
    assert "failed to persist cleanup uncertainty marker" in caplog.text
    assert "original error" in caplog.text
    assert "cleanup uncertain after failed runner launch bookkeeping" in caplog.text


def test_post_ack_launch_failure_does_not_clear_reservation_from_runner_group_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead runner group does not prove separate trial groups have stopped."""
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)

    def fail_update(_handle: RunHandle) -> None:
        raise OSError("runs directory is not writable")

    clear_calls: list[RunHandle] = []

    def track_clear(handle: RunHandle) -> None:
        clear_calls.append(handle)
        raise OSError("cleanup marker cannot be removed")

    monkeypatch.setattr(store, "update", fail_update)
    monkeypatch.setattr(store, "clear_cleanup_uncertain", track_clear)
    monkeypatch.setattr(mcp_run_control, "kill_stale_group", lambda *args, **kwargs: True)

    with pytest.raises(OSError, match="runs directory"):
        app.launch("srv")

    (pending,) = store.list_handles()
    terminal = store.recorded_terminal_status(pending)
    assert terminal is not None
    assert terminal["cleanup_confirmed"] is False
    assert store.cleanup_uncertain(pending)
    assert store.state(pending) == "running"
    assert store.recovery_required(pending)
    assert clear_calls == []


def test_launch_spawns_runner_in_neutral_dir_and_passes_project_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child must not start inside the project it is about to run.

    Interpreter startup from the experiment's directory executes project-local
    code before the server can name the process it created (review v0.5.17 /
    blocker 9), so ``Popen`` uses the server-owned state directory and the
    project directory travels as an explicit argument the runner applies later.
    """
    config = _config(tmp_path)
    runner_cwd = tmp_path / "runner-cwd"
    runner_cwd.mkdir()
    app, registry, _store = make_mcp_app(
        write_mcp_catalog(
            tmp_path,
            {"srv": config},
            allow=ALLOW_SIDE_EFFECTS,
            cwd={"srv": runner_cwd},
        )
    )
    captured = patch_popen_capture(monkeypatch)

    app.launch("srv")

    assert captured["cwd"] == str(registry.state_dir)
    assert captured["cwd"] != str(runner_cwd.resolve())
    cmd = captured["cmd"]
    assert cmd[cmd.index("--cwd") + 1] == str(runner_cwd.resolve())


def test_launch_hardens_runner_interpreter_flags_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "attacker"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "attacker"))
    monkeypatch.setenv("PYTHONSTARTUP", str(tmp_path / "attacker" / "startup.py"))
    monkeypatch.setenv("PYTHONEXECUTABLE", str(tmp_path / "attacker" / "python"))
    monkeypatch.setenv("PHASESWEEP_KEEP_ME", "kept")
    app, _registry, _store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    captured = patch_popen_capture(monkeypatch)

    app.launch("srv")

    cmd = captured["cmd"]
    assert cmd[0] == sys.executable
    # The flags must precede -m, or the interpreter treats them as module args.
    assert cmd[1:3] == ["-P", "-s"]
    assert cmd[3:5] == ["-m", "phasesweep.mcp.runner"]
    env = captured["env"]
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "PYTHONSTARTUP" not in env
    assert "PYTHONEXECUTABLE" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PHASESWEEP_KEEP_ME"] == "kept"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX spawn contract")
@pytest.mark.integration
def test_spawned_runner_cannot_execute_project_local_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spawn a real child under the server's contract; no project code may run.

    The runner module itself is not started (its import graph is irrelevant to
    the question); the child instead reports which ``phasesweep`` the hardened
    interpreter resolves and whether any poisoned hook left a marker.
    """
    captured, project, markers, _parent_env = _launch_with_poison_project(tmp_path, monkeypatch)
    resolved = tmp_path / "resolved.txt"
    probe = (
        "import pathlib, sys\n"
        "import phasesweep\n"
        "pathlib.Path(sys.argv[1]).write_text(phasesweep.__file__ + '\\n' + repr(sys.path))\n"
    )
    cmd = captured["cmd"]
    flags = cmd[1 : cmd.index("-m")]

    completed = subprocess.run(
        [cmd[0], *flags, "-c", probe, str(resolved)],
        cwd=captured["cwd"],
        env=captured["env"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in markers.iterdir()) == []
    # Neither the resolved module nor any sys.path entry comes from the project.
    assert str(project) not in resolved.read_text()


def test_launch_records_the_current_boot_id_in_the_spawned_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)

    run_id = app.launch("srv")["run_id"]

    handle = store.get(run_id)
    assert handle is not None
    assert handle.boot_id == read_boot_id()


def test_launch_refuses_a_runner_receipt_without_boot_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)
    monkeypatch.setattr(mcp_run_control, "read_boot_id", lambda: None)
    monkeypatch.setattr("tests.mcp_helpers.read_boot_id", lambda: None)
    monkeypatch.setattr(mcp_run_control, "kill_stale_group", lambda *_args, **_kwargs: True)

    with pytest.raises(RuntimeError, match="boot id"):
        app.launch("srv")

    (pending,) = store.list_handles()
    assert store.state(pending) == "failed"


def test_runner_does_not_persist_a_receipt_without_boot_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "state")
    run_id = "srv-no-boot"
    started_at = "2026-06-24T00:00:00Z"
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256="0" * 64,
        started_at=started_at,
    )
    monkeypatch.setattr(mcp_runner, "read_boot_id", lambda: None)

    with pytest.raises(RuntimeError, match="boot id"):
        mcp_runner._persist_spawned_handle(
            state_dir=tmp_path / "state",
            run_id=run_id,
            experiment_id="srv",
            config_sha256="0" * 64,
            started_at=started_at,
            allow_cancel=False,
        )

    pending = store.get(run_id)
    assert pending is not None
    assert pending.launch_state == "launching"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX spawn contract")
@pytest.mark.integration
def test_project_local_shadow_package_would_run_under_the_old_spawn_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control for the test above: the poison is genuinely executable.

    Reproducing the pre-fix contract - project cwd, no ``-P``/``-s``, inherited
    environment - must import the shadow package, so a passing hardened case is
    evidence about the fix rather than about an inert fixture.
    """
    _captured, project, markers, parent_env = _launch_with_poison_project(tmp_path, monkeypatch)
    # PYTHONHOME alone would break the interpreter before it could import
    # anything; drop it so the control isolates the cwd and PYTHONPATH hooks.
    env = {name: value for name, value in parent_env.items() if name != "PYTHONHOME"}

    completed = subprocess.run(
        [sys.executable, "-c", "import phasesweep.mcp.runner"],
        cwd=str(project),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in markers.iterdir()) == [
        "shadow_mcp",
        "shadow_package",
        "shadow_runner",
        "sitecustomize",
    ]


def test_audit_log_records_side_effects_without_sensitive_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    registry = Registry.load(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    store = RunStore(registry.state_dir)
    audit_path = registry.state_dir / "audit.jsonl"
    app = PhaseSweepMCP(registry, store, audit=AuditLogger(audit_path))
    captured = patch_popen_capture(monkeypatch)

    app.validate("srv")
    launched = app.launch("srv")
    with pytest.raises(ExperimentBusyError, match="already has a running sweep") as exc_info:
        app.launch("srv")
    busy_message = str(exc_info.value)
    assert launched["run_id"] in busy_message
    assert "lost the response" in busy_message
    assert "await_run" in busy_message
    assert "Cancel it only if the user wants" in busy_message

    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert [record["tool"] for record in records] == [
        TOOL_LAUNCH_RUN,
        TOOL_LAUNCH_RUN,
    ]
    assert {record["session_id"] for record in records}
    assert all(record["actor"] == "local-stdio" for record in records)
    assert all(record["transport"] == "stdio" for record in records)

    launch_record = records[0]
    assert launch_record["outcome"] == "success"
    assert launch_record["args"] == {"experiment_id": "srv"}
    assert launch_record["resolved"] == {"experiment_id": "srv", "run_id": launched["run_id"]}
    assert launch_record["state_before"] == {"live_runs": 0}
    assert launch_record["state_after"] == {"live_runs": 1, "run_state": "running"}
    assert launch_record["result_counts"] == {"runs": 1}

    busy_record = records[1]
    assert busy_record["outcome"] == "error"
    assert busy_record["args"] == {"experiment_id": "srv"}
    assert busy_record["resolved"] == {"experiment_id": "srv"}
    assert busy_record["state_before"] == {"live_runs": 1}
    assert busy_record["error_type"] == "ExperimentBusyError"
    assert "already has a running sweep" in busy_record["error"]

    blob = audit_path.read_text()
    for needle in ("train.py", "journal", str(config), str(tmp_path / "runs")):
        assert needle not in blob
    assert captured["cmd"]  # sanity: the launch path really reached Popen


def test_audit_log_caps_agent_supplied_string_values(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLogger(audit_path)

    audit.record(tool=TOOL_LAUNCH_RUN, args={"cursor": "x" * 500}, outcome="success")

    record = json.loads(audit_path.read_text())
    assert record["args"]["cursor"] == ("x" * 253) + "..."


def test_launch_artifacts_and_audit_are_private_under_permissive_umask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    catalog = _catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS)
    registry = Registry.load(catalog)
    store = RunStore(registry.state_dir)
    audit_path = registry.state_dir / "audit.jsonl"
    app = PhaseSweepMCP(registry, store, audit=AuditLogger(audit_path))
    captured = patch_popen_capture(monkeypatch)

    old_umask = os.umask(0)
    try:
        result = app.launch("srv")
    finally:
        os.umask(old_umask)

    run_id = result["run_id"]
    assert result["state"] == "running"
    assert captured["cmd"]
    assert file_mode(registry.state_dir) == 0o700
    assert file_mode(registry.state_dir / "runs") == 0o700
    assert file_mode(registry.state_dir / "logs") == 0o700
    assert file_mode(registry.state_dir / "runs" / f"{run_id}.json") == 0o600
    assert file_mode(store.log_path(run_id)) == 0o600
    assert file_mode(store.config_snapshot_path(run_id)) == 0o600
    assert file_mode(audit_path) == 0o600


@pytest.mark.integration
def test_runner_rejects_config_snapshot_hash_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = RunStore(tmp_path / "state")
    run_id = "r1"
    status_path = store.status_path(run_id)
    started_at = "2026-06-24T00:00:00Z"
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256="0" * 64,
        started_at=started_at,
    )

    with pytest.raises(RuntimeError, match="hash mismatch"):
        runner_main(
            runner_argv(
                store,
                run_id=run_id,
                config=config,
                config_sha256="0" * 64,
                experiment_id="srv",
                started_at=started_at,
            )
        )

    status = json.loads(status_path.read_text())
    assert status["returncode"] == 1
    assert status["error_class"] == "RuntimeError"
    assert status["cleanup_confirmed"] is True


@pytest.mark.integration
def test_preflight_failure_is_actionable_through_run_reads(tmp_path: Path) -> None:
    trainer = write_constant_trainer(tmp_path)
    config = tmp_path / "srv.yaml"
    experiment = make_experiment(
        experiment="srv",
        persistent=tmp_path,
        trainer=trainer,
        phases=[
            Phase(
                name="p",
                n_trials=1,
                fixed_overrides={"k": 1},
                sampler=Sampler(type="random", seed=0),
                search_space={},
            )
        ],
    )
    run_experiment(experiment)
    changed = experiment.model_copy(
        update={"phases": [experiment.phases[0].model_copy(update={"fixed_overrides": {"k": 2}})]}
    )
    config.write_text(yaml.safe_dump(changed.model_dump(mode="json"), sort_keys=False))
    registry = Registry.load(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    store = RunStore(registry.state_dir)
    run_id = "srv-preflight-failure"
    snapshot_path = store.config_snapshot_path(run_id)
    snapshot_path.write_bytes(config.read_bytes())
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    started_at = "2026-06-24T00:00:00Z"
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
    )

    with pytest.raises(StudyFingerprintMismatchError, match="different phase config"):
        runner_main(
            runner_argv(
                store,
                run_id=run_id,
                config=snapshot_path,
                config_sha256=config_sha256,
                experiment_id="srv",
                started_at=started_at,
            )
        )

    handle = store.get(run_id)
    assert handle is not None
    terminal = store.recorded_terminal_status(handle)
    assert terminal is not None
    assert terminal["result_snapshot_state"] == "complete"
    assert terminal["failure"]["code"] == "fingerprint_mismatch"
    app = PhaseSweepMCP(registry, store)
    latest = app.latest_run("srv")
    status = app.status(run_id=run_id)
    awaited = asyncio.run(app.await_run(run_id))
    winners = app.winners(run_id=run_id)

    for payload in (latest["run"], status["run"], awaited["run"]):
        assert payload["failure"]["code"] == "fingerprint_mismatch"
        assert payload["failure"]["retryable"] is False
        assert payload["failure"]["actor"] == "operator"
    assert status["result_source"] == "frozen_run_snapshot"
    assert status["summary_present"] is False
    assert status["phases"][0]["attempts_launched_this_run"] == 0
    assert awaited["reason"] == "terminal"
    assert winners["winner_count"] == 0
    assert winners["failure"]["code"] == "fingerprint_mismatch"


@pytest.mark.integration
def test_aggregated_schema_preflight_preserves_actionable_failure_category(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        phases="""\
  - name: a
    n_trials: 1
    sampler: { type: random, seed: 0 }
    search_space: {}
  - name: b
    n_trials: 1
    sampler: { type: random, seed: 1 }
    search_space: {}
""",
    )
    experiment = load_experiment(config)
    assert isinstance(experiment, Experiment)
    for phase in experiment.phases:
        study = optuna.create_study(
            study_name=f"{experiment.experiment}::{phase.name}",
            storage=engine_ledger._resolve_storage(experiment.resolved_storage),
            direction="minimize",
        )
        # Two populated, unmarked studies model a pre-cutover local ledger.
        study.set_user_attr(ARTIFACT_ROOT_ATTR, str(_experiment_dir(experiment)))
        study.add_trial(
            optuna.trial.create_trial(
                value=0.5,
                state=optuna.trial.TrialState.COMPLETE,
            )
        )

    store = RunStore(tmp_path / "state")
    run_id = "srv-schema-mismatch"
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    started_at = "2026-06-24T00:00:00Z"
    claim_runner_handle(
        store,
        run_id=run_id,
        config_sha256=config_sha256,
        started_at=started_at,
    )

    with pytest.raises(StudySchemaMismatchError, match="pre-cutover or unsupported"):
        runner_main(
            runner_argv(
                store,
                run_id=run_id,
                config=config,
                config_sha256=config_sha256,
                experiment_id="srv",
                started_at=started_at,
            )
        )

    terminal = json.loads(store.status_path(run_id).read_text())
    assert terminal["result_snapshot_state"] == "complete"
    assert terminal["failure"]["code"] == "study_schema_mismatch"
    assert terminal["failure"]["retryable"] is False
    # Pre-cutover state routes to the release that wrote it, which keeps the
    # study, rather than to archiving it for a fresh one.
    assert "preserved PhaseSweep release" in terminal["failure"]["remediation"]


def test_launch_bookkeeping_failure_preserves_runner_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    app, _registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    patch_popen_capture(monkeypatch)

    def interrupted_update(handle: RunHandle) -> None:
        experiment = load_experiment(config)
        assert isinstance(experiment, Experiment)
        write_run_status(
            store,
            handle.run_id,
            returncode=0,
            error_class=None,
            cleanup_confirmed=True,
            result_snapshot_state="complete",
            result_snapshot=capture_result_snapshot(experiment, generation_id=handle.run_id),
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "update", interrupted_update)
    monkeypatch.setattr(mcp_run_control, "kill_stale_group", lambda *_args, **_kwargs: True)
    with pytest.raises(KeyboardInterrupt):
        app.launch("srv")
    (handle,) = store.list_handles()
    terminal = store.recorded_terminal_status(handle)
    assert terminal is not None
    assert terminal["returncode"] == 0
    assert terminal["result_snapshot_state"] == "complete"


def test_launch_failure_cannot_replace_status_published_during_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    app, registry, store = make_mcp_app(_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS))
    reg = registry.get("srv")
    pending = make_run_handle(
        run_id="srv-racing-terminal",
        experiment_id=reg.id,
        config_sha256=reg.config_sha256,
        launch_state="launching",
    )
    store.create(pending)
    runner_status = {
        "run_id": pending.run_id,
        "returncode": 0,
        "cleanup_confirmed": True,
        "result_snapshot_state": "failed",
        "ended_at": mcp_run_control.utc_now_iso(),
    }
    original_write = mcp_runs.write_status_file
    original_link = mcp_runs.os.link

    # The launch-failure record links its status into place, so the runner's
    # status lands in the window just before that link.
    def racing_link(src: str, dst: str, **kwargs: object) -> None:
        if dst == store.status_path(pending.run_id).name:
            original_write(store.status_path(pending.run_id), runner_status)
        original_link(src, dst, **kwargs)

    monkeypatch.setattr(mcp_runs.os, "link", racing_link)
    app._record_launch_failure(pending, cleanup_confirmed=True, error_class="Injected")

    terminal = store.recorded_terminal_status(pending)
    assert terminal is not None
    assert terminal["returncode"] == 0
    assert terminal.get("error_class") != "Injected"
