"""FastMCP adapter: the only module that imports the MCP SDK.

PhaseSweepMCP holds all logic and is SDK-free and unit-testable. build_server
wraps each method as a FastMCP tool; _safe_tool guarantees tool errors are
redacted. serve() loads the catalog, builds the store, and serves over stdio.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import yaml

from phasesweep.engine import read_status, read_winners
from phasesweep.engine.state import Winner, _load_winner
from phasesweep.mcp.errors import (
    CatalogError,
    ConcurrencyLimitError,
    ConfigChangedError,
    ExperimentBusyError,
    InvalidPhaseError,
    LaunchInProgressError,
    McpToolError,
    PermissionDeniedError,
    ResumeNotReadyError,
    UnknownRunError,
)
from phasesweep.mcp.redaction import status_payload, winners_payload
from phasesweep.mcp.registry import RegisteredExperiment, Registry
from phasesweep.mcp.runs import RunHandle, RunStore, utc_now_iso
from phasesweep.runtime.process import read_proc_starttime, terminate_group

log = logging.getLogger("phasesweep.mcp.server")


class PhaseSweepMCP:
    """SDK-free implementation of every tool. Methods raise ``McpToolError``."""

    def __init__(self, registry: Registry, runs: RunStore) -> None:
        self._registry = registry
        self._runs = runs

    def list_experiments(self) -> list[dict[str, Any]]:
        """Return the path-free catalog listing (ids, descriptions, phases, metric)."""
        return self._registry.summaries()

    def validate(self, experiment_id: str) -> dict[str, Any]:
        """Report an experiment's phase structure (never the command/env/storage)."""
        reg = self._registry.get(experiment_id)
        exp = reg.experiment
        # Already validated at startup; report the structure, never the command.
        return {
            "experiment_id": reg.id,
            "metric": {"name": exp.metric.name, "goal": exp.metric.goal},
            "phases": [
                {
                    "name": p.name,
                    "n_trials": p.n_trials,
                    "sampler": p.sampler.type,
                    "inherits": p.inherits,
                    "search_space": sorted(p.search_space),  # keys only, not ranges
                }
                for p in exp.phases
            ],
        }

    def status(self, *, experiment_id: str | None = None, run_id: str | None = None) -> dict:
        """Per-phase trial counts and winner presence plus the run process state.

        Provide either ``experiment_id`` (reports the live run, if any) or
        ``run_id`` (reports that specific run). Raises if neither is given.
        """
        if run_id is not None:
            handle = self._runs.get(run_id)
            if handle is None:
                raise UnknownRunError(run_id)
            reg = self._registry.get(handle.experiment_id)
            run: dict[str, Any] | None = {
                "run_id": run_id,
                "state": self._runs.state(handle),
                "started_at": handle.started_at,
            }
            return status_payload(reg.id, read_status(reg.experiment), run)
        if experiment_id is not None:
            reg = self._registry.get(experiment_id)
            live = self._runs.live_run_for(experiment_id)
            run = (
                {"run_id": live.run_id, "state": "running", "started_at": live.started_at}
                if live is not None
                else None
            )
            return status_payload(reg.id, read_status(reg.experiment), run)
        raise McpToolError("provide either experiment_id or run_id")

    def winners(self, experiment_id: str) -> dict[str, Any]:
        """Return the winning hyperparameters per completed phase."""
        reg = self._registry.get(experiment_id)
        return winners_payload(reg.id, read_winners(reg.experiment))

    def launch(self, experiment_id: str, from_phase: str | None = None) -> dict[str, Any]:
        """Start the sweep as a detached background run; return its run_id.

        Refuses if launch is not permitted, if a ``from_phase`` resume is not
        ready (an earlier phase has no winner), if this experiment already has a
        live run, or if the server is at its max_concurrent_runs cap.
        """
        reg = self._registry.get(experiment_id)
        if not reg.allow_launch:
            raise PermissionDeniedError("launch", experiment_id)
        if from_phase is not None:
            if not reg.allow_from_phase:
                raise PermissionDeniedError("from_phase", experiment_id)
            if from_phase not in reg.phase_names:
                raise InvalidPhaseError(experiment_id, from_phase)
            self._require_resume_ready(reg, from_phase)
        # The cap check and the spawn must be atomic, or two near-simultaneous
        # launches both pass the cap and oversubscribe the GPU it protects. Hold
        # the launch lock across the whole decision. One scan then covers both
        # guards: the same experiment can't double-launch, and no more than
        # max_concurrent_runs sweeps run at once (default 1).
        with self._runs.launch_lock() as acquired:
            if not acquired:
                raise LaunchInProgressError()
            live = self._runs.live_runs()
            busy = next((h for h in live if h.experiment_id == experiment_id), None)
            if busy is not None:
                raise ExperimentBusyError(experiment_id, busy.run_id)
            if len(live) >= self._registry.max_concurrent_runs:
                raise ConcurrencyLimitError(len(live), self._registry.max_concurrent_runs)
            handle = self._spawn(reg, from_phase)
            self._runs.save(handle)
        return {"run_id": handle.run_id, "experiment_id": experiment_id, "state": "running"}

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Stop a running sweep: SIGTERM -> grace -> SIGKILL the runner's group.

        The terminal state is reported as ``cancelled`` on both the graceful
        path (the runner's handler writes status.json(143)) and the SIGKILL
        escalation (the runner is force-killed before it can; this attributes
        the cause faithfully rather than reporting ``failed``).
        """
        handle = self._runs.get(run_id)
        if handle is None:
            raise UnknownRunError(run_id)
        reg = self._registry.get(handle.experiment_id)
        if not reg.allow_cancel:
            raise PermissionDeniedError("cancel", handle.experiment_id)
        if self._runs.state(handle) != "running":
            return {"run_id": run_id, "state": self._runs.state(handle)}  # already terminal
        # SIGTERM -> grace -> SIGKILL on the runner's process group. The runner's
        # installed shutdown handler tears down the trial process groups and writes
        # status.json(143). cleanup_confirmed reports the runner group is gone, not
        # a guarantee about trial descendants (those are handled by the runner's
        # handler, or by the next launch's stale reaper on a SIGKILL escalation).
        confirmed = terminate_group(handle.pgid)
        if confirmed:
            # If escalation to SIGKILL killed the runner before it recorded a
            # graceful 143, attribute this operator-initiated stop as cancelled
            # so the state below isn't a misleading 'failed'. No-op otherwise.
            self._runs.mark_cancelled_if_unrecorded(handle)
        return {"run_id": run_id, "state": self._runs.state(handle), "cleanup_confirmed": confirmed}

    def _require_resume_ready(self, reg: RegisteredExperiment, from_phase: str) -> None:
        names = reg.phase_names
        winners: dict[str, Winner] = {}
        for phase in reg.experiment.phases[: names.index(from_phase)]:
            inherited = {parent: winners[parent] for parent in phase.inherits}
            try:
                winners[phase.name] = _load_winner(reg.experiment, phase, inherited)
            except FileNotFoundError:
                raise ResumeNotReadyError(reg.id, from_phase, phase.name) from None
            except (
                RuntimeError,
                KeyError,
                TypeError,
                ValueError,
                AttributeError,
                OSError,
                yaml.YAMLError,
            ) as exc:
                log.info(
                    "resume preflight rejected winner for experiment=%s phase=%s: %s",
                    reg.id,
                    phase.name,
                    exc,
                )
                raise ResumeNotReadyError(
                    reg.id,
                    from_phase,
                    phase.name,
                    reason="has no compatible winner for the current config",
                ) from None

    def _snapshot_config(self, reg: RegisteredExperiment, run_id: str) -> Path:
        try:
            data = reg.config_path.read_bytes()
        except OSError as exc:
            log.info("cannot read cataloged config for experiment=%s: %s", reg.id, exc)
            raise ConfigChangedError(reg.id) from None
        if hashlib.sha256(data).hexdigest() != reg.config_sha256:
            raise ConfigChangedError(reg.id)
        snapshot_path = self._runs.config_snapshot_path(run_id)
        snapshot_path.write_bytes(data)
        return snapshot_path

    def _spawn(self, reg: RegisteredExperiment, from_phase: str | None) -> RunHandle:
        run_id = self._runs.new_run_id(reg.id)
        log_path = self._runs.log_path(run_id)
        status_path = self._runs.status_path(run_id)
        config_snapshot_path = self._snapshot_config(reg, run_id)
        cmd = [
            sys.executable,
            "-m",
            "phasesweep.mcp.runner",
            "--run-id",
            run_id,
            "--config",
            str(config_snapshot_path),  # per-run snapshot, not agent input
            "--config-sha256",
            reg.config_sha256,
            "--status-path",
            str(status_path),
        ]
        if from_phase is not None:
            cmd += ["--from-phase", from_phase]
        # Open the log here, hand the fd to the child, then close our copy. The
        # child keeps it. stdin is /dev/null so the runner never blocks on input.
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(  # noqa: S603 - argv list, no shell, server-controlled
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own session/pgid; survives restart; signal as a group
            )
        return RunHandle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
            pid=proc.pid,
            # start_new_session=True makes the child a session+group leader, so
            # pgid == pid by POSIX. Avoids a getpgid() race if the child exits fast.
            pgid=proc.pid,
            pid_starttime=read_proc_starttime(proc.pid),
            started_at=utc_now_iso(),
            log_path=str(log_path),
            status_path=str(status_path),
        )


F = TypeVar("F", bound=Callable[..., Any])


def _safe_tool(fn: F) -> F:
    """Translate exceptions into redacted tool errors.

    ``McpToolError`` -> re-raised with its safe message. Anything else -> logged
    to stderr and replaced with a generic message so an unexpected error (e.g.
    an OSError carrying a path) never reaches the agent. ``functools.wraps``
    preserves the signature so FastMCP still derives the tool schema.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except McpToolError as exc:
            raise ValueError(exc.safe_message) from None
        except Exception:
            log.exception("unhandled error in tool %s", fn.__name__)
            raise ValueError("internal error") from None

    return wrapper  # type: ignore[return-value]


def build_server(app: PhaseSweepMCP) -> Any:
    """Construct the FastMCP server.

    The SDK is imported lazily so non-server code paths (and most tests) do not
    require the ``mcp`` package.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("phasesweep")

    @mcp.tool()
    @_safe_tool
    def list_experiments() -> list[dict]:
        """List the experiments this server exposes: ids, descriptions, phase names, and the optimization metric. Use an id with the other tools."""
        return app.list_experiments()

    @mcp.tool()
    @_safe_tool
    def validate_config(experiment_id: str) -> dict:
        """Return the phase structure (names, trial counts, samplers, inherited phases, search-space keys) for an experiment. Read-only; launches nothing."""
        return app.validate(experiment_id)

    @mcp.tool()
    @_safe_tool
    def get_status(experiment_id: str | None = None, run_id: str | None = None) -> dict:
        """Per-phase trial counts and winner presence, plus the run process state. Provide either experiment_id or run_id. Read-only."""
        return app.status(experiment_id=experiment_id, run_id=run_id)

    @mcp.tool()
    @_safe_tool
    def get_winners(experiment_id: str) -> dict:
        """Return the winning hyperparameters per completed phase: trial number, metric, sampled params, and the full effective overrides. Read-only."""
        return app.winners(experiment_id)

    @mcp.tool()
    @_safe_tool
    def launch_sweep(experiment_id: str, from_phase: str | None = None) -> dict:
        """Start the sweep for an experiment as a background run. Optionally resume from a phase whose earlier winners already exist. Returns a run_id."""
        return app.launch(experiment_id, from_phase=from_phase)

    @mcp.tool()
    @_safe_tool
    def cancel_sweep(run_id: str) -> dict:
        """Stop a running sweep by run_id. Terminates the orchestrator and its training processes."""
        return app.cancel(run_id)

    return mcp


def serve(catalog: Path) -> int:
    """Load the catalog, build the run store, and serve the six tools over stdio."""
    # stdio transport owns stdout for JSON-RPC. All logging goes to stderr.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname).1s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        registry = Registry.load(catalog)
    except CatalogError as exc:
        print(f"phasesweep mcp: {exc}", file=sys.stderr)
        return 2

    app = PhaseSweepMCP(registry, RunStore(registry.state_dir))
    build_server(app).run(transport="stdio")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Serve via ``python -m phasesweep.mcp.server``."""
    parser = argparse.ArgumentParser(prog="phasesweep mcp")
    parser.add_argument("--catalog", required=True, type=Path)
    args = parser.parse_args(argv)
    return serve(args.catalog)


if __name__ == "__main__":
    raise SystemExit(main())
