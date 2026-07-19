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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.mcp.time import parse_utc_iso
from phasesweep.runtime.files import (
    ensure_private_dir,
    private_atomic_write_text,
    try_lock_file,
    unlock_file,
)
from phasesweep.runtime.process import is_pid_zombie, is_same_process, reap_child

RunState = Literal["running", "succeeded", "failed", "cancelled"]
RunLaunchState = Literal["launching", "spawned"]

__all__ = [
    "ProcessIdentity",
    "RunHandle",
    "RunLaunchState",
    "RunState",
    "RunStore",
    "write_status_file",
]

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
    private_atomic_write_text(status_path, json.dumps(payload, indent=2) + "\n")


@dataclass(frozen=True)
class ProcessIdentity:
    """Persisted process identity used for PID-reuse-safe runner cleanup."""

    pid: int | None
    pgid: int | None
    pid_starttime: int | None


@dataclass(frozen=True)
class RunHandle:
    """Immutable, on-disk identity of one detached sweep."""

    run_id: str
    experiment_id: str
    config_sha256: str
    pid: int | None
    pgid: int | None
    pid_starttime: int | None  # /proc start time for PID-reuse-safe liveness; None off-Linux
    started_at: str  # ISO-8601 UTC
    launch_state: RunLaunchState = "spawned"
    allow_cancel: bool = False


class RunStore:
    """Filesystem store for run handles, logs, and status files under a state dir."""

    def __init__(self, state_dir: Path) -> None:
        """Create the run-handle store under ``state_dir``.

        :param Path state_dir: Root directory for runs, logs, config snapshots, and launch lock.
        """
        self._set_paths(state_dir)
        ensure_private_dir(state_dir)
        ensure_private_dir(self._runs_dir)
        ensure_private_dir(self._logs_dir)

    @classmethod
    def open_existing(cls, state_dir: Path) -> RunStore:
        """Open an existing run store without creating or chmodding any path.

        This is the operator-recovery entry point. Recovery accepts a path on
        the command line, so opening it must be observational: a typo must not
        create a new state tree or change permissions on an unrelated
        directory.

        :param Path state_dir: Existing MCP state directory containing ``runs/``
            and ``logs/`` subdirectories.
        :return RunStore: Store bound to the recognized existing layout.
        :raises ValueError: If ``state_dir`` is not an MCP run-store layout.
        """
        store = cls.__new__(cls)
        store._set_paths(state_dir)
        missing = [
            str(path) for path in (state_dir, store._runs_dir, store._logs_dir) if not path.is_dir()
        ]
        if missing:
            raise ValueError(
                "not an existing MCP state directory; expected directories are missing: "
                + ", ".join(missing)
            )
        return store

    def _set_paths(self, state_dir: Path) -> None:
        """Bind store paths without touching the filesystem.

        :param Path state_dir: Root directory for the MCP run store.
        """
        self._runs_dir = state_dir / "runs"
        self._logs_dir = state_dir / "logs"
        self._launch_lock_path = state_dir / ".launch.lock"

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
        private_atomic_write_text(target, payload + "\n")

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
            handle = RunHandle(**json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError):
            return None
        if handle.run_id != expected_run_id:
            return None
        if not SAFE_NAME_PATTERN.fullmatch(handle.experiment_id):
            return None
        if type(handle.allow_cancel) is not bool:
            return None
        if handle.launch_state not in {"launching", "spawned"}:
            return None
        if parse_utc_iso(handle.started_at) is None:
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
        return handle

    def state(self, handle: RunHandle) -> RunState:
        """Derive the current state from status.json and a live PID check.

        status.json (written by the runner on every exit) is authoritative when
        present: returncode 0 -> succeeded, a signalled code -> cancelled, any
        other -> failed. With no status.json, a live (non-zombie) PID means
        running. A dead or unverifiable spawned runner with no status cannot
        prove that its separately-sessioned trial descendants are gone, so it
        is marked cleanup-uncertain and remains ``running`` until operator
        recovery records cleanup evidence.

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
        status = self._read_status(handle)
        if self.cleanup_uncertain(handle):
            if status is not None and status.get("cleanup_confirmed") is True:
                with contextlib.suppress(OSError):
                    self.clear_cleanup_uncertain(handle)
            else:
                return "running"
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
        if handle.pid_starttime is None:
            if self._cleanup_recovered(handle):
                return "failed"
            self.mark_cleanup_uncertain(handle)
            return "running"
        # No status.json: the run is live only if its process is genuinely alive.
        # A zombie (exited without recording a cause - SIGKILL/OOM, or an early
        # SIGTERM before the engine installed its handlers) still answers
        # kill(pid, 0), so filter it out of the genuinely-live path and enter
        # cleanup-uncertain recovery below.
        if is_same_process(handle.pid, handle.pid_starttime) and not is_pid_zombie(handle.pid):
            return "running"
        if self._cleanup_recovered(handle):
            return "failed"
        self.mark_cleanup_uncertain(handle)
        return "running"

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
        existing = self._read_cleanup_identity(handle)
        candidate = ProcessIdentity(
            pid=handle.pid,
            pgid=handle.pgid,
            pid_starttime=handle.pid_starttime,
        )
        candidate_has_identity = candidate.pid is not None or candidate.pgid is not None
        identity = existing if existing is not None and not candidate_has_identity else candidate
        payload = {
            "run_id": handle.run_id,
            "config_sha256": handle.config_sha256,
            "pid": identity.pid,
            "pgid": identity.pgid,
            "pid_starttime": identity.pid_starttime,
            "cleanup_confirmed": False,
        }
        private_atomic_write_text(
            self.cleanup_uncertain_path(handle.run_id),
            json.dumps(payload, indent=2) + "\n",
        )

    def clear_cleanup_uncertain(self, handle: RunHandle) -> None:
        """Clear a previously persisted cleanup uncertainty marker.

        :param RunHandle handle: Run handle whose process group is now confirmed gone.
        """
        self.cleanup_uncertain_path(handle.run_id).unlink(missing_ok=True)

    def cleanup_uncertain(self, handle: RunHandle) -> bool:
        """Return whether a valid cleanup uncertainty marker exists for ``handle``.

        :param RunHandle handle: Run handle whose cleanup marker should be checked.
        :return bool: Whether a valid cleanup uncertainty marker exists.
        """
        return self._read_cleanup_identity(handle) is not None

    def cleanup_identity(self, handle: RunHandle) -> ProcessIdentity:
        """Return the strongest runner identity available for cleanup.

        If a pending handle was the only handle durably saved, a cleanup marker
        written from the spawned handle may carry the only usable PID/PGID. Keep
        that marker identity authoritative when present and valid.

        :param RunHandle handle: Run handle whose runner identity is needed.
        :return ProcessIdentity: Marker identity when stronger, otherwise handle identity.
        """
        marker_identity = self._read_cleanup_identity(handle)
        if marker_identity is not None and (
            marker_identity.pid is not None or marker_identity.pgid is not None
        ):
            return marker_identity
        return ProcessIdentity(
            pid=handle.pid,
            pgid=handle.pgid,
            pid_starttime=handle.pid_starttime,
        )

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

    def latest_run_for(self, experiment_id: str) -> RunHandle | None:
        """Return the newest persisted handle for an experiment deterministically.

        The result is computed from validated durable launch timestamps, with
        ``run_id`` as a stable tie-breaker. This intentionally returns one
        semantic answer rather than a list an agent would need to scan.

        :param str experiment_id: Catalog id whose latest run should be found.
        :return RunHandle | None: Newest persisted handle, or ``None`` when the
            experiment has no recorded MCP runs.
        """
        matching = [
            handle for handle in self.list_handles() if handle.experiment_id == experiment_id
        ]
        if not matching:
            return None
        return max(
            matching,
            key=lambda handle: (datetime.fromisoformat(handle.started_at), handle.run_id),
        )

    def live_runs(self) -> list[RunHandle]:
        """Every currently-running handle across all experiments.

        Scanning calls ``state`` on each handle, which also reaps any runner
        that has since exited - so this doubles as the cleanup sweep.

        :return list[RunHandle]: All handles whose derived state is currently ``running``.
        """
        return [handle for handle in self.list_handles() if self.state(handle) == "running"]

    def recovery_required(self, handle: RunHandle) -> bool:
        """Return whether operator cleanup recovery is required for a run.

        :param RunHandle handle: Persisted run whose cleanup evidence is checked.
        :return bool: True when a server marker or terminal runner status still
            records cleanup uncertainty without matching recovery evidence.
        """
        if self.cleanup_uncertain(handle):
            return True
        status = self._read_status(handle)
        return status is not None and self._terminal_cleanup_uncertain(handle, status)

    def _read_status(self, handle: RunHandle) -> dict | None:
        """Read the runner-written terminal status payload.

        :param RunHandle handle: Run handle whose status file should be read.
        :return dict | None: Decoded JSON payload, or ``None`` when absent or malformed.
        """
        path = self.status_path(handle.run_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("run_id") != handle.run_id:
            return None
        if type(payload.get("returncode")) is not int:
            return None
        cleanup_confirmed = payload.get("cleanup_confirmed")
        if cleanup_confirmed is not None and type(cleanup_confirmed) is not bool:
            return None
        error_class = payload.get("error_class")
        if error_class is not None and not isinstance(error_class, str):
            return None
        ended_at = payload.get("ended_at")
        if ended_at is not None and not isinstance(ended_at, str):
            return None
        result_snapshot_state = payload.get("result_snapshot_state")
        if result_snapshot_state is not None and result_snapshot_state not in {
            "pending",
            "complete",
            "failed",
        }:
            return None
        if result_snapshot_state == "complete" and not isinstance(
            payload.get("result_snapshot"), dict
        ):
            return None
        result_snapshot_error = payload.get("result_snapshot_error")
        if result_snapshot_error is not None and not isinstance(result_snapshot_error, str):
            return None
        return payload

    def _terminal_cleanup_uncertain(self, handle: RunHandle, status: Mapping[str, object]) -> bool:
        """Return whether a terminal status still needs operator cleanup recovery.

        :param RunHandle handle: Run handle whose recovery evidence should be checked.
        :param Mapping[str, object] status: Terminal runner status payload.
        :return bool: Whether cleanup is uncertain and lacks recovery evidence.
        """
        return status.get("cleanup_confirmed") is False and not self._cleanup_recovered(handle)

    def _cleanup_recovered(self, handle: RunHandle) -> bool:
        """Return whether operator recovery evidence confirms cleanup for this run.

        :param RunHandle handle: Run handle whose recovery evidence should be checked.
        :return bool: Whether valid recovery evidence confirms cleanup.
        """
        path = self.cleanup_recovery_path(handle.run_id)
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(payload, dict):
            return False
        return (
            payload.get("run_id") == handle.run_id
            and payload.get("config_sha256") == handle.config_sha256
            and payload.get("cleanup_confirmed") is True
        )

    def _read_cleanup_identity(
        self,
        handle: RunHandle,
    ) -> ProcessIdentity | None:
        """Read and validate this run's cleanup uncertainty marker.

        :param RunHandle handle: Run handle whose cleanup marker should be read.
        :return ProcessIdentity | None: Valid marker identity, or ``None`` if unavailable.
        """
        path = self.cleanup_uncertain_path(handle.run_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("run_id") != handle.run_id:
            return None
        marker_hash = payload.get("config_sha256")
        if marker_hash is not None and marker_hash != handle.config_sha256:
            return None
        if payload.get("cleanup_confirmed") is not False:
            return None

        pid = payload.get("pid")
        pgid = payload.get("pgid")
        pid_starttime = payload.get("pid_starttime")
        if not _valid_positive_optional_int(pid):
            return None
        if not _valid_positive_optional_int(pgid):
            return None
        if not _valid_positive_optional_int(pid_starttime):
            return None

        return ProcessIdentity(
            pid=pid,
            pgid=pgid,
            pid_starttime=pid_starttime,
        )


def _valid_positive_optional_int(value: object) -> bool:
    """Return whether ``value`` is ``None`` or a positive non-bool ``int``.

    :param object value: Value to validate.
    :return bool: Whether the value is ``None`` or a positive non-bool integer.
    """
    return value is None or (type(value) is int and value > 0)
