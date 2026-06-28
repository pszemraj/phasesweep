"""On-disk run-handle store for detached sweeps.

A launched sweep is a detached process; its identity must outlive the server
process. Each run is one JSON handle under ``<state_dir>/runs/`` plus a log and
a ``status.json`` under ``<state_dir>/logs/``. Run state is *derived* on read
(live PID check + status.json), never stored mutably, so a server crash mid-run
loses nothing and there is no stale-state write race.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.runtime.files import atomic_write_text, try_lock_file, unlock_file
from phasesweep.runtime.process import is_pid_zombie, is_same_process, reap_child

RunState = Literal["running", "succeeded", "failed", "cancelled"]
RunLaunchState = Literal["launching", "spawned"]

__all__ = ["RunHandle", "RunLaunchState", "RunState", "RunStore", "write_status_file"]

# Run ids are minted by ``new_run_id`` from this same character class. A lookup
# id, however, arrives from the (untrusted) agent and is interpolated into a
# handle path, so re-validate it here: this is the one place an id becomes a
# filesystem path, and the class excludes ``/`` and ``.`` so ``..`` traversal
# cannot escape the runs dir.

# 128 + SIGTERM(15); 128 + SIGINT(2). The engine shutdown handler exits
# 128+signum, so the runner records these as the "cancelled" terminal cause.
_SIGNALLED_EXIT_CODES = frozenset({143, 130})


def write_status_file(status_path: Path, payload: dict) -> None:
    """Atomically write a detached-run terminal status payload.

    :param Path status_path: Destination ``status.json`` path for the run.
    :param dict payload: JSON-serializable terminal status payload.
    """
    atomic_write_text(status_path, json.dumps(payload, indent=2))


@dataclass(frozen=True)
class RunHandle:
    """Immutable, on-disk identity of one detached sweep.

    ``log_path`` and ``status_path`` are server-internal and never returned to
    the agent.
    """

    run_id: str
    experiment_id: str
    config_sha256: str
    pid: int | None
    pgid: int | None
    pid_starttime: int | None  # /proc start time for PID-reuse-safe liveness; None off-Linux
    started_at: str  # ISO-8601 UTC
    log_path: str  # server-internal; never returned to the agent
    status_path: str  # server-internal
    launch_state: RunLaunchState = "spawned"

    @classmethod
    def from_json(cls, data: dict) -> RunHandle:
        """Rehydrate a handle from its JSON dict (the inverse of ``asdict``).

        :param dict data: JSON-decoded run handle payload.
        :return RunHandle: Reconstructed immutable run handle.
        """
        return cls(**{**data, "launch_state": data.get("launch_state", "spawned")})


class RunStore:
    """Filesystem store for run handles, logs, and status files under a state dir."""

    def __init__(self, state_dir: Path) -> None:
        """Create the run-handle store under ``state_dir``.

        :param Path state_dir: Root directory for runs, logs, config snapshots, and launch lock.
        """
        self._runs_dir = state_dir / "runs"
        self._logs_dir = state_dir / "logs"
        self._launch_lock_path = state_dir / ".launch.lock"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def launch_lock(self) -> Iterator[bool]:
        """Hold the launch lock for the context, yielding whether it was acquired.

        The concurrency cap is enforced by counting live runs and then
        spawning; that check-then-spawn must be atomic or two near-simultaneous
        launches can both pass the cap and oversubscribe the GPU. This is an
        ``flock`` on a file under ``state_dir``, so it serializes launches both
        across threads in one server and across servers that share a state dir
        (which scan the same runs dir). It is non-blocking: the manager yields
        ``False`` when another launch already holds it, leaving the caller to
        surface a retryable error rather than block a request handler.

        :return Iterator[bool]: Context manager yielding True when the lock was acquired.
        """
        handle = try_lock_file(self._launch_lock_path)
        try:
            yield handle is not None
        finally:
            if handle is not None:
                unlock_file(handle)

    def new_run_id(self, experiment_id: str) -> str:
        """Mint a fresh, collision-resistant run id prefixed with the experiment id.

        :param str experiment_id: Catalog id to include in the run id prefix.
        :return str: Newly generated safe run id.
        """
        return f"{experiment_id}-{uuid4().hex[:12]}"

    def log_path(self, run_id: str) -> Path:
        """Path to the captured stdout/stderr log for a run.

        :param str run_id: Run id whose log path should be returned.
        :return Path: Operator-only runner log path.
        """
        return self._logs_dir / f"{run_id}.log"

    def status_path(self, run_id: str) -> Path:
        """Path to the runner-written terminal-cause ``status.json`` for a run.

        :param str run_id: Run id whose terminal status path should be returned.
        :return Path: Operator-only status JSON path.
        """
        return self._logs_dir / f"{run_id}.status.json"

    def config_snapshot_path(self, run_id: str) -> Path:
        """Path to the per-run config snapshot consumed by the detached runner.

        :param str run_id: Run id whose config snapshot path should be returned.
        :return Path: Operator-only config snapshot path.
        """
        return self._logs_dir / f"{run_id}.config.yaml"

    def cleanup_uncertain_path(self, run_id: str) -> Path:
        """Path to the server-owned marker for unconfirmed process cleanup.

        :param str run_id: Run id whose cleanup marker path should be returned.
        :return Path: Operator-only cleanup uncertainty marker path.
        """
        return self._logs_dir / f"{run_id}.cleanup_uncertain.json"

    def cleanup_recovery_path(self, run_id: str) -> Path:
        """Path to the operator recovery evidence file for a run.

        :param str run_id: Run id whose recovery evidence path should be returned.
        :return Path: Operator-only cleanup recovery JSON path.
        """
        return self._logs_dir / f"{run_id}.cleanup_recovery.json"

    def save(self, handle: RunHandle) -> None:
        """Persist a run handle as JSON under the runs dir.

        Writes through a same-directory temp file and then replaces the target so
        readers never observe a partially-written handle.
        """
        target = self._runs_dir / f"{handle.run_id}.json"
        payload = json.dumps(asdict(handle), indent=2)
        atomic_write_text(target, payload)

    def get(self, run_id: str) -> RunHandle | None:
        """Load a run handle by id, or ``None`` if there is no such handle.

        An id that is not of the minted shape (``[A-Za-z0-9_-]+``) can never
        match a stored handle, so it is reported as absent rather than used to
        build a path - this keeps an agent-supplied id from traversing out of
        the runs dir (e.g. ``../../etc/foo``).

        :param str run_id: Agent-supplied run id to look up.
        :return RunHandle | None: Persisted handle when present and well-shaped, otherwise ``None``.
        """
        if not SAFE_NAME_PATTERN.fullmatch(run_id):
            return None
        path = self._runs_dir / f"{run_id}.json"
        if not path.is_file():
            return None
        return self._load_handle(path, expected_run_id=run_id)

    def list_handles(self) -> list[RunHandle]:
        """Load every persisted handle, skipping any that are malformed or partial.

        :return list[RunHandle]: Successfully decoded run handles.
        """
        handles = []
        for path in self._runs_dir.glob("*.json"):
            handle = self._load_handle(path, expected_run_id=path.stem)
            if handle is not None:
                handles.append(handle)
        return handles

    def _load_handle(self, path: Path, *, expected_run_id: str) -> RunHandle | None:
        """Load and normalize one run handle, returning ``None`` when malformed.

        :param Path path: Persisted run-handle JSON path to read.
        :param str expected_run_id: Run id implied by the filename.
        :return RunHandle | None: Normalized handle, or ``None`` when validation fails.
        """
        if not SAFE_NAME_PATTERN.fullmatch(expected_run_id):
            return None
        try:
            handle = RunHandle.from_json(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError):
            return None
        if handle.run_id != expected_run_id:
            return None
        if not SAFE_NAME_PATTERN.fullmatch(handle.experiment_id):
            return None
        if handle.launch_state not in {"launching", "spawned"}:
            return None
        if handle.launch_state == "launching":
            if (
                handle.pid is not None
                or handle.pgid is not None
                or handle.pid_starttime is not None
            ):
                return None
        else:
            if type(handle.pid) is not int or handle.pid <= 0:
                return None
            if type(handle.pgid) is not int or handle.pgid <= 0:
                return None
            if handle.pid_starttime is not None and (
                type(handle.pid_starttime) is not int or handle.pid_starttime <= 0
            ):
                return None
        return replace(
            handle,
            log_path=str(self.log_path(handle.run_id)),
            status_path=str(self.status_path(handle.run_id)),
        )

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
        if handle.pid is not None:
            reap_child(handle.pid)
        if self._cleanup_uncertain(handle):
            return "running"
        status = self._read_status(handle)
        if status is not None:
            if self._terminal_cleanup_uncertain(handle, status):
                return "running"
            rc = status.get("returncode")
            if rc == 0:
                return "succeeded"
            if rc in _SIGNALLED_EXIT_CODES or status.get("error_class") == "cancelled":
                return "cancelled"
            return "failed"
        if handle.launch_state == "launching" or handle.pid is None:
            return "failed"
        # No status.json: the run is live only if its process is genuinely alive.
        # A zombie (exited without recording a cause - SIGKILL/OOM, or an early
        # SIGTERM before the engine installed its handlers) still answers
        # kill(pid, 0), so it must be filtered out or a dead run would report
        # "running" forever and block relaunch.
        if is_same_process(handle.pid, handle.pid_starttime) and not is_pid_zombie(handle.pid):
            return "running"
        return "failed"

    def has_recorded_status(self, handle: RunHandle) -> bool:
        """Return whether the runner wrote a readable terminal status file.

        :param RunHandle handle: Run handle whose status file should be checked.
        :return bool: ``True`` when a valid runner-written status is present.
        """
        return self._read_status(handle) is not None

    def recorded_terminal_status(self, handle: RunHandle) -> dict | None:
        """Return the runner-written terminal status payload, if readable.

        :param RunHandle handle: Run handle whose terminal status should be returned.
        :return dict | None: Decoded status payload, or ``None`` when absent or malformed.
        """
        return self._read_status(handle)

    def mark_cleanup_uncertain(self, handle: RunHandle) -> None:
        """Persist that a cancel attempt could not confirm process-group cleanup.

        A dead root PID normally derives to ``failed`` and frees the MCP
        concurrency slot. When cancellation could not prove the process group is
        gone, descendants may still hold resources, so this server-owned marker
        keeps the run in the live set until a later cleanup attempt succeeds.

        :param RunHandle handle: Run handle whose cleanup is uncertain.
        """
        payload = {
            "run_id": handle.run_id,
            "pid": handle.pid,
            "pgid": handle.pgid,
            "pid_starttime": handle.pid_starttime,
            "cleanup_confirmed": False,
        }
        atomic_write_text(self.cleanup_uncertain_path(handle.run_id), json.dumps(payload, indent=2))

    def clear_cleanup_uncertain(self, handle: RunHandle) -> None:
        """Clear a previously persisted cleanup uncertainty marker.

        :param RunHandle handle: Run handle whose process group is now confirmed gone.
        """
        self.cleanup_uncertain_path(handle.run_id).unlink(missing_ok=True)

    def live_run_for(self, experiment_id: str) -> RunHandle | None:
        """Return the currently-running handle for an experiment, if any.

        Used to reject a second launch before the engine's flock would.

        :param str experiment_id: Catalog id whose live run should be found.
        :return RunHandle | None: Currently-running handle, if one exists.
        """
        for handle in self.list_handles():
            if handle.experiment_id == experiment_id and self.state(handle) == "running":
                return handle
        return None

    def live_runs(self) -> list[RunHandle]:
        """Every currently-running handle across all experiments.

        Scanning calls ``state`` on each handle, which also reaps any runner
        that has since exited - so this doubles as the cleanup sweep.

        :return list[RunHandle]: All handles whose derived state is currently ``running``.
        """
        return [handle for handle in self.list_handles() if self.state(handle) == "running"]

    def _read_status(self, handle: RunHandle) -> dict | None:
        """Read the runner-written terminal status payload.

        :param RunHandle handle: Run handle whose status file should be read.
        :return dict | None: Decoded JSON payload, or ``None`` when absent or malformed.
        """
        path = self.status_path(handle.run_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def _cleanup_uncertain(self, handle: RunHandle) -> bool:
        """Return whether this run has a server-owned cleanup uncertainty marker.

        :param RunHandle handle: Run handle whose cleanup marker should be checked.
        :return bool: True when a prior cancel could not confirm cleanup.
        """
        return self.cleanup_uncertain_path(handle.run_id).is_file()

    def _terminal_cleanup_uncertain(self, handle: RunHandle, status: Mapping[str, object]) -> bool:
        """Return whether a terminal status still needs operator cleanup recovery."""
        return status.get("cleanup_confirmed") is False and not self._cleanup_recovered(handle)

    def _cleanup_recovered(self, handle: RunHandle) -> bool:
        """Return whether operator recovery evidence confirms cleanup for this run."""
        path = self.cleanup_recovery_path(handle.run_id)
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return (
            payload.get("run_id") == handle.run_id
            and payload.get("config_sha256") == handle.config_sha256
            and payload.get("cleanup_confirmed") is True
        )
