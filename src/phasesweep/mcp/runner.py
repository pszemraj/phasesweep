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

from phasesweep.mcp.runs import RunHandle, RunStore, write_status_file
from phasesweep.runtime.process import read_proc_starttime


def _write_status(status_path: Path, payload: dict) -> None:
    """Best-effort write of the runner terminal status file.

    :param Path status_path: JSON file where terminal cause should be recorded.
    :param dict payload: Status payload containing run id, return code, and error class.
    """
    # Best-effort: a failed status write must not mask the real exit cause.
    try:
        write_status_file(status_path, payload)
    except OSError:
        logging.getLogger("phasesweep.mcp.runner").exception("failed to write status.json")


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
    status_path: Path,
) -> None:
    """Persist this runner's process identity before it launches any training work."""
    store = RunStore(state_dir)
    pid = os.getpid()
    pgid = os.getpgrp() if hasattr(os, "getpgrp") else pid
    store.save(
        RunHandle(
            run_id=run_id,
            experiment_id=experiment_id,
            config_sha256=config_sha256,
            pid=pid,
            pgid=pgid,
            pid_starttime=read_proc_starttime(pid),
            started_at=started_at,
            log_path=str(store.log_path(run_id)),
            status_path=str(status_path),
            launch_state="spawned",
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

    status: dict = {"run_id": args.run_id, "returncode": 0, "error_class": None}
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
            status_path=args.status_path,
        )
        from phasesweep.config import load_config
        from phasesweep.engine import run_config

        actual_sha256 = _sha256_file(args.config)
        if actual_sha256 != args.config_sha256:
            raise RuntimeError("config snapshot hash mismatch; refusing to run")
        config = load_config(args.config)
        # run_config installs signal handlers and takes the flock internally.
        run_config(config, from_phase=args.from_phase, dry_run=False)
    except SystemExit as exc:
        # The engine shutdown handler raises SystemExit(128+signum) on
        # SIGTERM/SIGINT - this is the cancel path.
        code = exc.code if isinstance(exc.code, int) else 1
        status["returncode"] = code
        status["error_class"] = "cancelled" if code in (143, 130) else "exited"
        _write_status(args.status_path, status)
        raise
    except BaseException as exc:  # noqa: BLE001 - record every terminal cause, then re-raise
        status["returncode"] = 1
        status["error_class"] = type(exc).__name__
        _write_status(args.status_path, status)
        raise
    _write_status(args.status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
