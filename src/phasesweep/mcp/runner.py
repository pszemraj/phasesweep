"""Detached orchestrator entrypoint for one sweep.

Spawned by the MCP server in its own session (``start_new_session=True``) with
stdout/stderr redirected to a per-run log. Runs ``run_config`` and records the
terminal cause in ``status.json`` so the server can report succeeded / failed /
cancelled without scraping logs. The path it runs is supplied by the server
from the frozen registry; it is never agent input.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from phasesweep.engine import run_experiment
from phasesweep.engine.trial import ProcessCleanupUncertainError
from phasesweep.mcp.config_snapshot import load_experiment_snapshot
from phasesweep.mcp.runs import RunHandle, RunStore, write_status_file
from phasesweep.mcp.snapshots import capture_result_snapshot, finalize_result_snapshot
from phasesweep.mcp.time import utc_now_iso
from phasesweep.runtime.process import (
    PhaseSweepShutdown,
    _defer_shutdown_signals,
    install_signal_handlers,
    read_proc_starttime,
)


def _write_status(
    status_path: Path,
    payload: dict,
    *,
    result_snapshot: dict | None,
    result_snapshot_error: str | None,
) -> None:
    """Persist terminal evidence and its already-captured result snapshot.

    :param Path status_path: JSON file where terminal cause should be recorded.
    :param dict payload: Status payload containing run id, return code, and error class.
    :param dict | None result_snapshot: Raw snapshot captured under the experiment lock.
    :param str | None result_snapshot_error: Capture error class when no snapshot exists.
    """
    # A catchable shutdown may arrive after the durable pending write. Defer it
    # until the complete/failed replacement is durable so cancellation cannot
    # strand an otherwise terminal run in the intermediate state.
    with _defer_shutdown_signals():
        terminal = {
            **payload,
            "ended_at": utc_now_iso(),
            "result_snapshot_state": "pending",
        }
        try:
            write_status_file(status_path, terminal)
        except Exception:  # noqa: BLE001 - terminal evidence must not mask the run's exit
            logging.getLogger("phasesweep.mcp.runner").exception("failed to write status.json")
            return

        try:
            if result_snapshot is None:
                raise RuntimeError(result_snapshot_error or "terminal snapshot was not captured")
            terminal["result_snapshot"] = finalize_result_snapshot(
                result_snapshot,
                cleanup_confirmed=terminal.get("cleanup_confirmed") is True,
            )
        except Exception as exc:  # noqa: BLE001 - minimal terminal evidence is already durable
            terminal["result_snapshot_state"] = "failed"
            terminal["result_snapshot_error"] = result_snapshot_error or type(exc).__name__
            logging.getLogger("phasesweep.mcp.runner").exception(
                "failed to finalize terminal result snapshot"
            )
        else:
            terminal["result_snapshot_state"] = "complete"

        try:
            write_status_file(status_path, terminal)
        except Exception as exc:  # noqa: BLE001 - preserve a serializable failed state
            logging.getLogger("phasesweep.mcp.runner").exception(
                "failed to finalize result snapshot state in status.json"
            )
            terminal.pop("result_snapshot", None)
            terminal["result_snapshot_state"] = "failed"
            terminal["result_snapshot_error"] = type(exc).__name__
            try:
                write_status_file(status_path, terminal)
            except Exception:  # noqa: BLE001 - no further persistence fallback is available
                logging.getLogger("phasesweep.mcp.runner").exception(
                    "failed to persist result snapshot finalization failure"
                )


def _persist_spawned_handle(
    *,
    state_dir: Path,
    run_id: str,
    experiment_id: str,
    config_sha256: str,
    started_at: str,
    allow_cancel: bool,
) -> None:
    """Persist this runner's process identity before it launches any training work.

    :param Path state_dir: Server state dir whose ``runs/`` receives the handle.
    :param str run_id: Run id minted by the launching server.
    :param str experiment_id: Catalog id this run belongs to.
    :param str config_sha256: Hash of the config snapshot this runner executes.
    :param str started_at: ISO-8601 UTC launch timestamp recorded by the server.
    :param bool allow_cancel: Cancel permission frozen at launch time.
    """
    store = RunStore(state_dir)
    pid = os.getpid()
    pgid = os.getpgrp() if hasattr(os, "getpgrp") else pid
    pid_starttime = read_proc_starttime(pid)
    if pid_starttime is None:
        raise RuntimeError(
            "cannot persist a PID-reuse-safe MCP runner handle because Linux "
            "/proc start time is unavailable"
        )
    store.save(
        RunHandle(
            run_id=run_id,
            experiment_id=experiment_id,
            config_sha256=config_sha256,
            pid=pid,
            pgid=pgid,
            pid_starttime=pid_starttime,
            started_at=started_at,
            launch_state="spawned",
            allow_cancel=allow_cancel,
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run one config to completion and record its terminal cause in status.json.

    :param list[str] | None argv: Optional argument vector; defaults to ``sys.argv`` when omitted.
    :return int: Process exit code, zero on successful sweep completion.
    """
    parser = argparse.ArgumentParser(prog="phasesweep mcp runner")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True, type=Path)  # snapshot, server-supplied
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--status-path", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--allow-cancel", action="store_true")
    parser.add_argument("--from-phase", default=None)
    args = parser.parse_args(argv)

    # This process's stdout/stderr are the server-redirected run log. Log to
    # stderr; never print to stdout here. (The engine's own run.log under the
    # workdir is separate and durable.)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname).1s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Install shutdown handlers before any other work (run_config re-invokes
    # this; it is idempotent). A cancel can arrive while this runner is still
    # loading config or persisting its handle; without handlers the default
    # SIGTERM disposition kills the process before status.json is written and
    # the run derives "running" behind its cleanup-uncertainty marker forever.
    install_signal_handlers()

    status: dict = {
        "run_id": args.run_id,
        "returncode": 0,
        "error_class": None,
        "cleanup_confirmed": True,
    }
    result_snapshot: dict | None = None
    result_snapshot_error: str | None = None
    try:
        # The server also saves this handle after Popen returns. The runner's
        # self-write closes the restart-recovery window if the server dies
        # after Popen but before its own spawned-handle save reaches disk.
        _persist_spawned_handle(
            state_dir=args.state_dir,
            run_id=args.run_id,
            experiment_id=args.experiment_id,
            config_sha256=args.config_sha256,
            started_at=args.started_at,
            allow_cancel=args.allow_cancel,
        )
        try:
            config = load_experiment_snapshot(
                args.config,
                args.config_sha256,
                source=f"run snapshot {args.run_id}",
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc

        def capture_terminal(generation_id: str, _error: BaseException | None) -> None:
            """Capture immutable results while ``run_experiment`` still owns its lock.

            :param str generation_id: Identifier for the completed engine invocation.
            :param BaseException | None _error: Terminal engine error, if one occurred.
            """
            nonlocal result_snapshot, result_snapshot_error
            try:
                result_snapshot = capture_result_snapshot(
                    config,
                    cleanup_confirmed=False,
                    generation_id=generation_id,
                    require_trial_data=_error is None,
                )
            except Exception as exc:  # noqa: BLE001 - preserve the engine's terminal cause
                result_snapshot_error = type(exc).__name__
                logging.getLogger("phasesweep.mcp.runner").exception(
                    "failed to capture terminal result snapshot under the experiment lock"
                )

        run_experiment(
            config,
            from_phase=args.from_phase,
            dry_run=False,
            terminal_callback=capture_terminal,
        )
    except PhaseSweepShutdown as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        status["returncode"] = code
        status["error_class"] = "cancelled"
        status["cleanup_confirmed"] = exc.report.cleanup_confirmed
        raise
    except ProcessCleanupUncertainError as exc:
        status["returncode"] = 1
        status["error_class"] = type(exc).__name__
        status["cleanup_confirmed"] = False
        raise
    except BaseException as exc:  # noqa: BLE001 - record every terminal cause, then re-raise
        status["returncode"] = 1
        status["error_class"] = type(exc).__name__
        status["cleanup_confirmed"] = True
        raise
    finally:
        _write_status(
            args.status_path,
            status,
            result_snapshot=result_snapshot,
            result_snapshot_error=result_snapshot_error,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
