"""On-disk run-handle store for detached sweeps.

A launched sweep is a detached process; its identity must outlive the server
process. Each run is one JSON handle under ``<state_dir>/runs/`` plus a log and
a ``status.json`` under ``<state_dir>/logs/``. Run state is *derived* on read
(live PID check + status.json), never stored mutably, so a server crash mid-run
loses nothing and there is no stale-state write race.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from phasesweep.runtime.process import is_pid_zombie, is_same_process, reap_child

RunState = Literal["running", "succeeded", "failed", "cancelled"]

# Run ids are minted by ``new_run_id`` from this same character class. A lookup
# id, however, arrives from the (untrusted) agent and is interpolated into a
# handle path, so re-validate it here: this is the one place an id becomes a
# filesystem path, and the class excludes ``/`` and ``.`` so ``..`` traversal
# cannot escape the runs dir. Mirrors the engine's name rule
# (config.common._validate_safe_name) and the catalog id rule.
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")

# 128 + SIGTERM(15); 128 + SIGINT(2). The engine shutdown handler exits
# 128+signum, so the runner records these as the "cancelled" terminal cause.
_SIGNALLED_EXIT_CODES = frozenset({143, 130})


@dataclass(frozen=True)
class RunHandle:
    """Immutable, on-disk identity of one detached sweep.

    ``log_path`` and ``status_path`` are server-internal and never returned to
    the agent.
    """

    run_id: str
    experiment_id: str
    config_sha256: str
    pid: int
    pgid: int
    pid_starttime: int | None  # /proc start time for PID-reuse-safe liveness; None off-Linux
    started_at: str  # ISO-8601 UTC
    log_path: str  # server-internal; never returned to the agent
    status_path: str  # server-internal

    @classmethod
    def from_json(cls, data: dict) -> RunHandle:
        """Rehydrate a handle from its JSON dict (the inverse of ``asdict``)."""
        return cls(**data)


class RunStore:
    """Filesystem store for run handles, logs, and status files under a state dir."""

    def __init__(self, state_dir: Path) -> None:
        self._runs_dir = state_dir / "runs"
        self._logs_dir = state_dir / "logs"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)

    def new_run_id(self, experiment_id: str) -> str:
        """Mint a fresh, collision-resistant run id prefixed with the experiment id."""
        return f"{experiment_id}-{uuid4().hex[:12]}"

    def log_path(self, run_id: str) -> Path:
        """Path to the captured stdout/stderr log for a run."""
        return self._logs_dir / f"{run_id}.log"

    def status_path(self, run_id: str) -> Path:
        """Path to the runner-written terminal-cause ``status.json`` for a run."""
        return self._logs_dir / f"{run_id}.status.json"

    def save(self, handle: RunHandle) -> None:
        """Persist a run handle as JSON under the runs dir."""
        (self._runs_dir / f"{handle.run_id}.json").write_text(json.dumps(asdict(handle), indent=2))

    def get(self, run_id: str) -> RunHandle | None:
        """Load a run handle by id, or ``None`` if there is no such handle.

        An id that is not of the minted shape (``[A-Za-z0-9_-]+``) can never
        match a stored handle, so it is reported as absent rather than used to
        build a path - this keeps an agent-supplied id from traversing out of
        the runs dir (e.g. ``../../etc/foo``).
        """
        if not _SAFE_RUN_ID.match(run_id):
            return None
        path = self._runs_dir / f"{run_id}.json"
        if not path.is_file():
            return None
        return RunHandle.from_json(json.loads(path.read_text()))

    def list_handles(self) -> list[RunHandle]:
        """Load every persisted handle, skipping any that are malformed or partial."""
        handles = []
        for path in self._runs_dir.glob("*.json"):
            try:
                handles.append(RunHandle.from_json(json.loads(path.read_text())))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue  # ignore a malformed/partial handle rather than crash a read
        return handles

    def state(self, handle: RunHandle) -> RunState:
        """Derive the current state from status.json and a live PID check.

        status.json (written by the runner on every exit) is authoritative when
        present: returncode 0 -> succeeded, a signalled code -> cancelled, any
        other -> failed. With no status.json, a live (non-zombie) PID means
        running; a dead or zombie PID with no status means the process died
        without recording a cause (e.g. SIGKILL or OOM) and is reported as failed.

        Args:
            handle: The run handle to evaluate.

        Returns:
            One of ``running`` / ``succeeded`` / ``failed`` / ``cancelled``.

        """
        # Reap the runner if it has exited. The runner is our child; without
        # this it lingers as a zombie until the server dies, and every state
        # query is a natural place to clean it up (no signal handler needed).
        reap_child(handle.pid)
        status = self._read_status(handle)
        if status is not None:
            rc = status.get("returncode")
            if rc == 0:
                return "succeeded"
            if rc in _SIGNALLED_EXIT_CODES or status.get("error_class") == "cancelled":
                return "cancelled"
            return "failed"
        # No status.json: the run is live only if its process is genuinely alive.
        # A zombie (exited without recording a cause - SIGKILL/OOM, or an early
        # SIGTERM before the engine installed its handlers) still answers
        # kill(pid, 0), so it must be filtered out or a dead run would report
        # "running" forever and block relaunch.
        if is_same_process(handle.pid, handle.pid_starttime) and not is_pid_zombie(handle.pid):
            return "running"
        return "failed"

    def live_run_for(self, experiment_id: str) -> RunHandle | None:
        """Return the currently-running handle for an experiment, if any.

        Used to reject a second launch before the engine's flock would.
        """
        for handle in self.list_handles():
            if handle.experiment_id == experiment_id and self.state(handle) == "running":
                return handle
        return None

    def live_runs(self) -> list[RunHandle]:
        """Every currently-running handle across all experiments.

        Scanning calls ``state`` on each handle, which also reaps any runner
        that has since exited - so this doubles as the cleanup sweep.
        """
        return [handle for handle in self.list_handles() if self.state(handle) == "running"]

    def _read_status(self, handle: RunHandle) -> dict | None:
        path = Path(handle.status_path)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
