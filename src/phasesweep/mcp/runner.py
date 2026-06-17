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
import json
import logging
import sys
from pathlib import Path

from phasesweep.config import load_config
from phasesweep.engine import run_config


def _write_status(status_path: Path, payload: dict) -> None:
    # Best-effort: a failed status write must not mask the real exit cause.
    try:
        status_path.write_text(json.dumps(payload, indent=2))
    except OSError:
        logging.getLogger("phasesweep.mcp.runner").exception("failed to write status.json")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Run one config to completion and record its terminal cause in status.json."""
    parser = argparse.ArgumentParser(prog="phasesweep mcp runner")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True, type=Path)  # snapshot, server-supplied
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--status-path", required=True, type=Path)
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
