"""Shared supervised launch mechanics for remote SDK workers."""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phasesweep.runtime.process import ProcessResult


def worker_environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    """Retain composed credentials and the orchestrator's Python bootstrap.

    :param Mapping[str, str] | None environment: Composed trainer environment, or
        ``None`` for the orchestrator's own environment.
    :return dict[str, str]: Worker environment whose ``PYTHON*`` variables match
        the orchestrator's exactly.
    """
    worker_environment = dict(os.environ if environment is None else environment)
    for name in set(worker_environment) | set(os.environ):
        if name.startswith("PYTHON"):
            if name in os.environ:
                worker_environment[name] = os.environ[name]
            else:
                worker_environment.pop(name, None)
    return worker_environment


def run_worker(
    request: Mapping[str, Any],
    worker_path: Path,
    backend: str,
    *,
    statuses: frozenset[str],
    run_id: str,
    trial_dir: Path,
    environment: Mapping[str, str] | None,
    deadline: float,
) -> tuple[ProcessResult, dict[str, Any], Path | None]:
    """Run one supervised worker request in the attempt's process slot.

    :param Mapping[str, Any] request: Backend request payload.
    :param Path worker_path: Backend worker entry point.
    :param str backend: Backend name for scratch files and diagnostics.
    :param frozenset[str] statuses: Response statuses the task can report; any
        other outcome preserves the worker's stderr for diagnosis.
    :param str run_id: Immutable attempt ID, also used by process recovery.
    :param Path trial_dir: Existing attempt directory and durable process slot.
    :param Mapping[str, str] | None environment: Actual composed trainer environment.
    :param float deadline: Absolute ``time.monotonic()`` time the worker is stopped at.
    :return tuple[ProcessResult, dict[str, Any], Path | None]: The worker's
        process result, its response (empty when it wrote none), and the
        preserved stderr path, if any.
    :raises UnsafeProcessCleanupError: Worker cleanup is uncertain.
    """
    from phasesweep.errors import UnsafeProcessCleanupError
    from phasesweep.runtime.files import cache_dir, private_atomic_write_text
    from phasesweep.runtime.process import run_supervised
    from phasesweep.runtime.reaper import PROCESS_IDENTITY_FILE

    diagnostic_path: Path | None = None
    worker_cache = cache_dir() / backend / "workers"
    worker_cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="operation-", dir=worker_cache) as directory:
        worker_dir = Path(directory)
        request_path = worker_dir / "request.json"
        response_path = worker_dir / "response.json"
        stderr_path = worker_dir / "stderr.log"
        private_atomic_write_text(request_path, json.dumps(dict(request)))
        with (
            (worker_dir / "stdout.log").open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            # Never let a previous safe trainer identity hide a failed worker launch.
            (trial_dir / PROCESS_IDENTITY_FILE).unlink(missing_ok=True)
            result = run_supervised(
                shlex.join(
                    [
                        sys.executable,
                        "-P",
                        str(worker_path),
                        str(request_path),
                        str(response_path),
                    ]
                ),
                env=worker_environment(environment),
                stdout=stdout,
                stderr=stderr,
                timeout=max(0.0, deadline - time.monotonic()),
                wallclock_deadline=deadline,
                trial_dir=trial_dir,
                attempt_id=run_id,
            )
        if not result.cleanup_confirmed:
            raise UnsafeProcessCleanupError(
                f"{backend} worker cleanup is uncertain for attempt {run_id!r}; recover {trial_dir}."
            )
        response = (
            json.loads(response_path.read_text(encoding="utf-8")) if response_path.is_file() else {}
        )
        if not result.timed_out and (
            result.return_code != 0
            or result.failure_reason is not None
            or response.get("status") not in statuses
        ):
            diagnostic = stderr_path.read_text(encoding="utf-8", errors="replace")
            if diagnostic:
                diagnostic_path = trial_dir / f"{backend}-worker.stderr.log"
                private_atomic_write_text(
                    diagnostic_path,
                    diagnostic,
                    require_private_dir=False,
                )
    return result, response, diagnostic_path
