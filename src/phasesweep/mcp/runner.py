"""Detached orchestrator entrypoint for one sweep.

Spawned by the MCP server in its own session (``start_new_session=True``) with
stdout/stderr redirected to a per-run log. Runs ``run_config`` and records the
terminal cause in ``status.json`` so the server can report succeeded / failed /
cancelled without scraping logs. The path it runs is supplied by the server
from the frozen registry; it is never agent input.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

from phasesweep.engine.trial import ProcessCleanupUncertainError
from phasesweep.mcp.runs import RunHandle, RunStore, write_status_file
from phasesweep.mcp.snapshots import capture_result_snapshot
from phasesweep.mcp.time import utc_now_iso
from phasesweep.runtime.process import (
    PhaseSweepShutdown,
    install_signal_handlers,
    read_proc_starttime,
)


def _write_status(
    status_path: Path,
    payload: dict,
    *,
    config_path: Path,
    config_sha256: str,
) -> None:
    """Persist terminal evidence, then best-effort enrich it with a result snapshot.

    :param Path status_path: JSON file where terminal cause should be recorded.
    :param dict payload: Status payload containing run id, return code, and error class.
    :param Path config_path: Exact per-run config snapshot that was executed.
    :param str config_sha256: Expected hash for the per-run config snapshot.
    """
    # Persist cleanup evidence before reading storage. Snapshot capture can be
    # slow on a large or contended study, and the cancelling server may exhaust
    # its grace period and force-kill this runner while that read is in flight.
    terminal = {**payload, "ended_at": utc_now_iso()}
    try:
        write_status_file(status_path, terminal)
    except OSError:
        logging.getLogger("phasesweep.mcp.runner").exception("failed to write status.json")
        return

    try:
        from phasesweep.config import Experiment, load_config

        if _sha256_file(config_path) != config_sha256:
            return
        config = load_config(config_path)
        if not isinstance(config, Experiment):
            return
        terminal["result_snapshot"] = capture_result_snapshot(
            config,
            cleanup_confirmed=terminal.get("cleanup_confirmed") is True,
        )
    except Exception:  # noqa: BLE001 - minimal terminal evidence is already durable
        logging.getLogger("phasesweep.mcp.runner").exception(
            "failed to capture terminal result snapshot"
        )
        return

    try:
        write_status_file(status_path, terminal)
    except OSError:
        logging.getLogger("phasesweep.mcp.runner").exception(
            "failed to add result snapshot to status.json"
        )


def _sha256_file(path: Path) -> str:
    """Hash a file with SHA-256.

    :param Path path: File to hash.
    :return str: Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        from phasesweep.config import load_config
        from phasesweep.engine import run_config

        actual_sha256 = _sha256_file(args.config)
        if actual_sha256 != args.config_sha256:
            raise RuntimeError("config snapshot hash mismatch; refusing to run")
        config = load_config(args.config)
        # run_config installs signal handlers and takes the flock internally.
        run_config(config, from_phase=args.from_phase, dry_run=False)
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
    except SystemExit as exc:
        # The engine shutdown handler raises SystemExit(128+signum) on
        # SIGTERM/SIGINT - this is the cancel path.
        code = exc.code if isinstance(exc.code, int) else 1
        status["returncode"] = code
        status["error_class"] = "cancelled" if code in (143, 130) else "exited"
        status["cleanup_confirmed"] = code not in (143, 130)
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
            config_path=args.config,
            config_sha256=args.config_sha256,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
