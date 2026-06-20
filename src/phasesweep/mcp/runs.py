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
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.mcp.time import utc_now_iso as utc_now_iso
from phasesweep.runtime.files import try_lock_file, unlock_file
from phasesweep.runtime.process import is_pid_zombie, is_same_process, reap_child

RunState = Literal["running", "succeeded", "failed", "cancelled"]

__all__ = ["RunHandle", "RunState", "RunStore", "utc_now_iso", "write_status_file"]

# Run ids are minted by ``new_run_id`` from this same character class. A lookup
# id, however, arrives from the (untrusted) agent and is interpolated into a
# handle path, so re-validate it here: this is the one place an id becomes a
# filesystem path, and the class excludes ``/`` and ``.`` so ``..`` traversal
# cannot escape the runs dir.

# 128 + SIGTERM(15); 128 + SIGINT(2). The engine shutdown handler exits
# 128+signum, so the runner records these as the "cancelled" terminal cause.
_SIGNALLED_EXIT_CODES = frozenset({143, 130})


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync for a directory after an atomic replace."""
    try:
        dir_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text through a same-directory temp file and atomically replace ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    replaced = False
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        replaced = True
        _fsync_directory(path.parent)
    finally:
        if not replaced:
            tmp.unlink(missing_ok=True)


def write_status_file(status_path: Path, payload: dict) -> None:
    """Atomically write a detached-run terminal status payload."""
    _atomic_write_text(status_path, json.dumps(payload, indent=2))


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
        """Rehydrate a handle from its JSON dict (the inverse of ``asdict``).

        :param dict data: JSON-decoded run handle payload.
        :return RunHandle: Reconstructed immutable run handle.
        """
        return cls(**data)


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

    def save(self, handle: RunHandle) -> None:
        """Persist a run handle as JSON under the runs dir.

        Writes through a same-directory temp file and then replaces the target so
        readers never observe a partially-written handle.
        """
        target = self._runs_dir / f"{handle.run_id}.json"
        payload = json.dumps(asdict(handle), indent=2)
        _atomic_write_text(target, payload)

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
        return RunHandle.from_json(json.loads(path.read_text()))

    def list_handles(self) -> list[RunHandle]:
        """Load every persisted handle, skipping any that are malformed or partial.

        :return list[RunHandle]: Successfully decoded run handles.
        """
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

    def mark_cancelled_if_unrecorded(self, handle: RunHandle) -> None:
        """Record a ``cancelled`` cause when a force-killed runner left none.

        The runner writes ``status.json`` (returncode 143) from its SIGTERM
        handler. If the server had to escalate to SIGKILL - the runner did not
        exit within the grace window - it is killed before recording anything,
        and a later read would derive ``failed`` from the now-dead PID. The
        canceller is the authority on a termination it performed, so attribute
        it faithfully. No-op when a status file already exists, so a graceful
        143 (or a genuine failure) is never clobbered.

        Args:
            handle: The run handle whose terminal cause is being recorded.

        """
        status_path = Path(handle.status_path)
        if self._read_status(handle) is not None:
            return
        # 137 == 128 + SIGKILL(9). state() keys "cancelled" off error_class, not
        # the code, so an OOM/other SIGKILL - which never writes status - still
        # reads as failed; only a cancel we performed records this cause.
        payload = {"run_id": handle.run_id, "returncode": 137, "error_class": "cancelled"}
        with contextlib.suppress(OSError):
            write_status_file(status_path, payload)

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
        path = Path(handle.status_path)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
