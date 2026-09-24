"""Shared MCP test setup helpers."""

from __future__ import annotations

import contextlib
import hashlib
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from phasesweep.config import (
    ExecutionContext,
    Experiment,
    IntParam,
    LogRegexExtractor,
    Metric,
    Phase,
    Sampler,
)
from phasesweep.mcp import runner as mcp_runner
from phasesweep.mcp.registry import Registry, VisibleParamsPolicy
from phasesweep.mcp.run_control import _runner_protocol_argv
from phasesweep.mcp.runs import RunHandle, RunLaunchState, RunStore, write_status_file
from phasesweep.mcp.tools import PhaseSweepMCP
from phasesweep.runtime.reaper import read_boot_id, read_proc_starttime
from phasesweep.runtime.time import utc_now_iso
from tests.conftest import make_experiment, reaped_pid, restored_signal_ownership


def write_mcp_catalog(
    tmp_path: Path,
    entries: Mapping[str, Path],
    *,
    allow: Mapping[str, bool] | None = None,
    cwd: Mapping[str, Path] | None = None,
    visible_params: Mapping[str, object] | None = None,
    max_concurrent_runs: int | None = None,
    filename: str = "catalog.yaml",
) -> Path:
    lines = [f"state_dir: {tmp_path}/state"]
    if max_concurrent_runs is not None:
        lines.append(f"max_concurrent_runs: {max_concurrent_runs}")
    lines.append("experiments:")
    for entry_id, config in entries.items():
        lines += [f"  - id: {entry_id}", f"    config: {config}"]
        if cwd is not None and entry_id in cwd:
            lines.append(f"    cwd: {cwd[entry_id]}")
        if visible_params is not None and entry_id in visible_params:
            value = visible_params[entry_id]
            if isinstance(value, list):
                lines.append("    visible_params:")
                lines.extend(f"      - {item}" for item in value)
            else:
                lines.append(f"    visible_params: {value}")
        if allow is not None:
            lines.append("    allow:")
            lines.extend(f"      {key}: {str(value).lower()}" for key, value in allow.items())
    catalog = tmp_path / filename
    catalog.write_text("\n".join(lines) + "\n")
    return catalog


def write_mcp_config_catalog(
    tmp_path: Path,
    configs: Mapping[str, str],
    *,
    allow: Mapping[str, bool] | None = None,
    cwd: Mapping[str, Path] | None = None,
    visible_params: Mapping[str, object] | None = None,
    max_concurrent_runs: int | None = None,
    filename: str = "catalog.yaml",
) -> Path:
    entries = {}
    for entry_id, body in configs.items():
        config = tmp_path / f"{entry_id}.yaml"
        config.write_text(body)
        entries[entry_id] = config
    return write_mcp_catalog(
        tmp_path,
        entries,
        allow=allow,
        cwd=cwd,
        visible_params=visible_params,
        max_concurrent_runs=max_concurrent_runs,
        filename=filename,
    )


def mcp_experiment_config_text(
    tmp_path: Path,
    *,
    name: str = "srv",
    phases: str | None = None,
    with_storage: bool = True,
) -> str:
    if phases is None:
        # Seeded random: reproducible and resumable, so it satisfies the
        # persistent-storage sampler policy without an acknowledgement.
        phases = """\
  - name: p
    n_trials: 1
    sampler: { type: random, seed: 0 }
    search_space:
      lr: { type: float, low: 1.0e-5, high: 1.0e-2, log: true }
"""
    storage = (
        f"storage: journal:///{tmp_path}/{name}.journal\nprovenance: {{revision: test-fixture-v1}}\n"
        if with_storage
        else ""
    )
    return f"""\
experiment: {name}
{storage}workdir: {tmp_path}/runs/{name}
trial_command: "python train.py --out {{trial_dir}}/r.json {{overrides}}"
override_format: argparse
metric:
  name: loss
  goal: minimize
  extractor: {{ type: json_envelope, path: r.json, objective_name: loss, split: test, policy: test }}
phases:
{phases}"""


def slow_mcp_config_text(
    tmp_path: Path,
    *,
    trainer: Path,
    name: str = "slow",
    sleep: float = 30.0,
) -> str:
    return f"""\
experiment: {name}
storage: journal:///{tmp_path}/{name}.journal
provenance: {{revision: test-fixture-v1}}
workdir: {tmp_path}/runs/{name}
trial_command: "{sys.executable} {trainer} --sleep {sleep} {{overrides}}"
override_format: argparse
metric:
  name: eval_loss
  goal: minimize
  extractor: {{ type: json_envelope, path: result.json, objective_name: eval_loss, split: validation, policy: synthetic }}
phases:
  - name: p
    n_trials: 1
    sampler: {{ type: random, seed: 0 }}
    search_space:
      lr: {{ type: float, low: 1.0e-5, high: 1.0e-2, log: true }}
"""


def make_mcp_app(catalog: Path) -> tuple[PhaseSweepMCP, Registry, RunStore]:
    registry = Registry.load(catalog)
    store = RunStore(registry.state_dir)
    return PhaseSweepMCP(registry, store), registry, store


def wait_for_mcp_running_trial(app: PhaseSweepMCP, run_id: str, *, timeout: float) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = app.status(run_id=run_id)
        if status["run"]["state"] in {"succeeded", "failed", "cancelled"}:
            return status["run"]["state"]
        if status["phases"][0]["running_trials_total"] >= 1:
            return "running"
        time.sleep(0.2)
    return "timeout"


def cancel_mcp_run_quietly(app: PhaseSweepMCP, run_id: str) -> None:
    with contextlib.suppress(Exception):
        app.cancel(run_id)


def assert_no_sensitive(payload: Any, sensitive: Iterable[str]) -> None:
    """Raise ``AssertionError`` if any string leaf contains a sensitive value."""
    needles = [s for s in sensitive if s]

    def walk(node: Any) -> None:
        if isinstance(node, str):
            for needle in needles:
                assert needle not in node, f"sensitive value leaked into payload: {needle!r}"
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(key)
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(payload)


def make_run_handle(
    *,
    run_id: str,
    experiment_id: str = "exp",
    config_sha256: str = "0" * 64,
    pid: int | None = None,
    starttime: int | None = None,
    launch_state: RunLaunchState = "spawned",
    allow_cancel: bool = False,
    visible_params_at_launch: VisibleParamsPolicy | None = "none",
) -> RunHandle:
    if launch_state == "launching":
        process_id = None
        process_group_id = None
        process_starttime = None
    else:
        process_id = os.getpid() if pid is None else pid
        # Default fixtures need a genuinely live PID so RunStore state checks
        # see a running handle, so the default runner is pytest itself. The
        # unused PGID keeps code that reads ``handle.pgid`` off real groups, but
        # it is no protection from ``kill_stale_group``: a live PID with a
        # matching start time makes it signal ``os.getpgid(pid)``, pytest's real
        # group. The autouse ``guard_runner_signals`` fixture in conftest is
        # what fails a test that reaches it. Explicit PID fixtures keep their
        # matching PGID for process-lifecycle tests.
        process_group_id = 2_000_000_000 if pid is None else process_id
        process_starttime = read_proc_starttime(process_id) if starttime is None else starttime
    return RunHandle(
        run_id=run_id,
        experiment_id=experiment_id,
        config_sha256=config_sha256,
        pid=process_id,
        pgid=process_group_id,
        pid_starttime=process_starttime,
        started_at=utc_now_iso(),
        launch_state=launch_state,
        allow_cancel=allow_cancel,
        visible_params_at_launch=visible_params_at_launch,
        boot_id=read_boot_id() if launch_state == "spawned" else None,
    )


def claim_runner_handle(
    store: RunStore,
    *,
    run_id: str,
    config_sha256: str,
    started_at: str,
    experiment_id: str = "srv",
    visible_params_at_launch: VisibleParamsPolicy | None = "none",
) -> None:
    """Create the launch reservation a real MCP server owns before spawning."""
    store.create(
        RunHandle(
            run_id=run_id,
            experiment_id=experiment_id,
            config_sha256=config_sha256,
            pid=None,
            pgid=None,
            pid_starttime=None,
            started_at=started_at,
            launch_state="launching",
            visible_params_at_launch=visible_params_at_launch,
        )
    )


def runner_main(argv: list[str], *, cwd: Path | None = None) -> int:
    """Invoke the detached runner in-process without stealing pytest's process state.

    The real runner is a process entry point and therefore owns shutdown
    signals for its lifetime. Tests call it in pytest's process, where that
    ownership must end with this helper just like the temporary cwd does.

    :param list[str] argv: Detached runner arguments.
    :param Path | None cwd: Runner working directory, or the current directory.
    :return int: Runner process exit code.
    """
    original = Path.cwd()
    with restored_signal_ownership():
        try:
            return mcp_runner.main([*argv, "--cwd", str(original if cwd is None else cwd)])
        finally:
            os.chdir(original)


def runner_argv(
    store: RunStore,
    *,
    run_id: str,
    config: Path,
    config_sha256: str,
    experiment_id: str,
    started_at: str,
) -> list[str]:
    """Build the detached-runner arguments shared by MCP tests.

    :param RunStore store: Run store supplying status and state paths.
    :param str run_id: Claimed run identifier.
    :param Path config: Snapshotted experiment configuration.
    :param str config_sha256: Expected configuration digest.
    :param str experiment_id: Catalog experiment identifier.
    :param str started_at: Claimed launch timestamp.
    :return list[str]: Runner arguments without the Python module prefix or cwd.
    """
    return _runner_protocol_argv(
        run_id=run_id,
        config_snapshot_path=config,
        config_sha256=config_sha256,
        status_path=store.status_path(run_id),
        state_dir=store.log_path(run_id).parent.parent,
        experiment_id=experiment_id,
        started_at=started_at,
    )


def write_run_status(store: RunStore, run_id: str, **payload: object) -> None:
    full_payload = {"run_id": run_id, "cleanup_confirmed": True, **payload}
    write_status_file(store.status_path(run_id), full_payload)


#: Start time recorded for a runner that is gone; no live process has it.
DEAD_RUNNER_STARTTIME = 111


def stage_dead_run(
    store: RunStore,
    run_id: str,
    config: Path,
    experiment_id: str,
    *,
    cleanup_uncertain: bool,
    allow_cancel: bool = False,
) -> RunHandle:
    """Persist a spawned run of ``config`` whose runner has exited, as recover-run finds it."""
    snapshot = config.read_bytes()
    handle = make_run_handle(
        run_id=run_id,
        experiment_id=experiment_id,
        config_sha256=hashlib.sha256(snapshot).hexdigest(),
        pid=reaped_pid(),
        starttime=DEAD_RUNNER_STARTTIME,
        allow_cancel=allow_cancel,
    )
    store.create(handle)
    store.config_snapshot_path(run_id).write_bytes(snapshot)
    if cleanup_uncertain:
        store.mark_cleanup_uncertain(handle)
    return handle


def write_unsafe_cleanup_status(store: RunStore, run_id: str, **extra: object) -> None:
    """Record the terminal status of a run whose process-group cleanup was never confirmed."""
    write_run_status(
        store,
        run_id,
        returncode=1,
        error_class="UnsafeProcessCleanupError",
        cleanup_confirmed=False,
        **extra,
    )


def patch_popen_capture(monkeypatch: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class DummyProc:
        pid = os.getppid()

    def fake_popen(cmd: list[str], **kwargs: object) -> DummyProc:
        stdout = kwargs.get("stdout")
        assert kwargs.get("stdin") is subprocess.DEVNULL
        assert kwargs.get("stderr") is subprocess.STDOUT
        assert kwargs.get("start_new_session") is True
        assert stdout is not None and not getattr(stdout, "closed", True)
        ready_fd = int(cmd[cmd.index("--launch-ready-fd") + 1])
        ack_fd = int(cmd[cmd.index("--launch-ack-fd") + 1])
        lease_fd = int(cmd[cmd.index("--launch-lease-fd") + 1])
        inherited = kwargs.get("pass_fds")
        assert isinstance(inherited, tuple)
        assert ready_fd in inherited
        assert ack_fd in inherited
        assert lease_fd in inherited
        ack_reader = os.dup(ack_fd)
        pending_store = RunStore(Path(cmd[cmd.index("--state-dir") + 1]))
        run_id = cmd[cmd.index("--run-id") + 1]
        pending = pending_store.get(run_id)
        assert pending is not None
        pending_store.update(
            RunHandle(
                run_id=run_id,
                experiment_id=pending.experiment_id,
                config_sha256=pending.config_sha256,
                pid=DummyProc.pid,
                pgid=DummyProc.pid,
                pid_starttime=read_proc_starttime(DummyProc.pid),
                started_at=pending.started_at,
                launch_state="spawned",
                allow_cancel=pending.allow_cancel,
                visible_params_at_launch=pending.visible_params_at_launch,
                boot_id=read_boot_id(),
            )
        )
        os.write(ready_fd, b"R")

        def consume_ack() -> None:
            try:
                os.read(ack_reader, 1)
            finally:
                os.close(ack_reader)

        threading.Thread(target=consume_ack, daemon=True).start()
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = kwargs.get("env")
        return DummyProc()

    monkeypatch.setattr("phasesweep.mcp.run_control.subprocess.Popen", fake_popen)
    return captured


ALLOW_SIDE_EFFECTS = {"launch": True, "cancel": True, "from_phase": True}


def _config(tmp_path: Path, *, name: str = "srv", phases: str | None = None) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(mcp_experiment_config_text(tmp_path, name=name, phases=phases))
    return path


def _catalog(
    tmp_path: Path,
    config: Path,
    allow: dict[str, bool] | None = None,
    *,
    visible_params: object | None = None,
) -> Path:
    return write_mcp_catalog(
        tmp_path,
        {"srv": config},
        allow=allow,
        visible_params=None if visible_params is None else {"srv": visible_params},
        filename="srv.catalog.yaml",
    )


def _drift_experiment(
    tmp_path: Path,
    trainer: Path,
    *,
    name: str = "srv",
    metric_name: str = "x",
    goal: str = "minimize",
    extractor: object | None = None,
    phase_name: str = "p",
    phases: list[Phase] | None = None,
) -> Experiment:
    """Build the runnable experiment the tests publish and then edit: one phase unless ``phases``."""
    return make_experiment(
        experiment=name,
        storage=f"journal:///{tmp_path / 'drift.journal'}",
        workdir=str(tmp_path / "runs"),
        execution=ExecutionContext(cwd=str(tmp_path)),
        trainer=trainer,
        metric=Metric(
            name=metric_name,
            goal=goal,
            extractor=extractor
            or LogRegexExtractor(type="log_regex", pattern=r"x=(?P<value>[0-9.eE+-]+)"),
        ),
        phases=phases
        or [
            Phase(
                name=phase_name,
                n_trials=1,
                sampler=Sampler(type="random", seed=0),
                search_space={"lr": IntParam(type="int", low=1, high=2)},
            )
        ],
    )


def _write_experiment_config(config: Path, experiment: Experiment) -> None:
    """Rewrite a cataloged config file in place, as an operator edit would."""
    config.write_text(yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=False))
