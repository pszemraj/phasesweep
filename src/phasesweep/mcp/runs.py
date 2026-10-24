"""Persist detached run identities, logs and terminal status.

Run state is derived from process visibility and recorded evidence; reads write nothing.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Annotated, Literal, NotRequired, TypeAlias, TypeGuard, cast
from uuid import UUID, uuid4

from pydantic import Field, TypeAdapter, ValidationError
from typing_extensions import TypedDict

from phasesweep.config.common import SAFE_NAME_PATTERN, is_sha256_hex
from phasesweep.runtime.files import (
    UnsafePrivatePathError,
    absolute_path,
    ensure_private_dir,
    open_directory_fd,
    open_lock_file,
    private_atomic_create_text,
    private_atomic_write_text,
    read_private_text,
    try_lock_file,
    unlock_file,
    validate_private_dir,
)
from phasesweep.runtime.reaper import (
    foreign_pid_namespace_reason,
    identity_from_earlier_boot,
    is_pid_namespace,
    is_same_live_process,
    read_boot_id,
    reap_child,
    recorded_process_blocker,
    same_pid_namespace,
)
from phasesweep.runtime.time import parse_utc_iso

RunState = Literal["running", "succeeded", "failed", "cancelled"]
RunLaunchState = Literal["launching", "spawned"]
#: How this process can see a run's runner. ``observable``: its PID and start
#: time can be probed here. ``absent``: it was recorded in another PID
#: namespace, and the host's initial namespace found none of its processes
#: left; it is dead, but its trials may not be, so only recovery settles it.
#: ``gone``: its boot ended, so neither it nor any trial it started remains.
#: ``unobservable``: it was recorded in another PID namespace, whose PID
#: numbers name unrelated processes here, so liveness is unknown rather than
#: dead.
RunnerVisibility: TypeAlias = Literal["observable", "absent", "gone", "unobservable"]

# phasesweep.mcp.runner writes
# LAUNCH_READY_BYTE once its process identity is durable, and
# phasesweep.mcp.tools writes LAUNCH_ACK_BYTE once it has recorded that
# receipt, settling ownership of the spawned process.
LAUNCH_READY_BYTE = b"R"
LAUNCH_ACK_BYTE = b"A"
#: Seconds between the detached runner's heartbeat refreshes.
RUNNER_HEARTBEAT_INTERVAL_SECONDS = 15.0
#: Heartbeat age, several missed refreshes, past which a reader that cannot
#: observe the runner reports that it has probably stopped.
RUNNER_HEARTBEAT_STALE_SECONDS = 120.0

log = logging.getLogger("phasesweep.mcp.runs")


# 128 + SIGTERM(15); 128 + SIGINT(2). The engine shutdown handler exits
# 128+signum, so the runner records these as the "cancelled" terminal cause.
_SIGNALLED_EXIT_CODES = frozenset({143, 130})
_RUN_EVIDENCE_SUFFIXES = (
    ".config.yaml",
    ".status.json",
    ".log",
    ".cleanup_uncertain.json",
    ".cleanup_recovery.json",
    ".launch.lock",
    ".transition.lock",
    ".heartbeat",
)
MCP_STATE_FORMAT_VERSION = 7
_STATE_FORMAT_MARKER_NAME = ".phasesweep-format.json"
_STATE_FORMAT_MARKER_PAYLOAD = {"schema_version": MCP_STATE_FORMAT_VERSION}


class UnsupportedStateFormatError(ValueError):
    """An MCP state directory uses an unsupported format and requires a fresh state directory."""


def _strict_unlink(path: Path) -> None:
    """Unlink one private file and durably record its removal.

    :param Path path: File to unlink.
    """
    directory_fd = open_directory_fd(path.parent, create=False, private_final=True)
    try:
        os.unlink(path.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_json_object(path: Path) -> dict[str, object] | None:
    """Read a JSON object, returning ``None`` for missing or malformed files.

    :param Path path: JSON file to read.
    :return dict[str, object] | None: Parsed object, or ``None`` when unavailable or invalid.
    """
    try:
        payload = json.loads(read_private_text(path))
    except (OSError, UnsafePrivatePathError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def write_status_file(status_path: Path, payload: dict[str, object]) -> None:
    """Atomically write a detached-run terminal status payload.

    :param Path status_path: Destination ``status.json`` path for the run.
    :param dict[str, object] payload: JSON-serializable terminal status payload.
    """
    private_atomic_write_text(status_path, json.dumps(payload, indent=2) + "\n")


def write_status_file_if_absent(status_path: Path, payload: dict[str, object]) -> bool:
    """Atomically create a server failure only while no runner status exists.

    A post-link durability failure removes the newly created status file.

    :param Path status_path: Destination ``status.json`` path for the run.
    :param dict[str, object] payload: JSON-serializable server failure payload.
    :return bool: Whether this call created the status; ``False`` if one already exists.
    """
    try:
        private_atomic_create_text(status_path, json.dumps(payload, indent=2) + "\n")
    except FileExistsError:
        return False
    return True


@dataclass(frozen=True)
class ProcessIdentity:
    """Persisted process identity used for PID-reuse-safe runner cleanup."""

    pid: int | None
    pgid: int | None
    pid_starttime: int | None
    boot_id: str | None = None
    pid_namespace: str | None = None


@dataclass(frozen=True)
class RunnerView:
    """Runner visibility derived once and reused within one state decision."""

    visibility: RunnerVisibility
    #: For an ``unobservable`` runner, why it cannot be acted on from here,
    #: naming any host process that stopped the proof; ``None`` otherwise.
    reason: str | None = None


class TerminalStatus(TypedDict):
    """Validated current-format terminal fields; unrelated JSON fields remain intact."""

    run_id: str
    returncode: int
    cleanup_confirmed: bool
    error_class: NotRequired[str | None]
    failure: NotRequired[dict[str, object] | None]
    ended_at: NotRequired[str | None]
    result_snapshot_state: NotRequired[Literal["pending", "complete", "failed"] | None]
    result_snapshot: NotRequired[object]
    result_publication_state: NotRequired[Literal["prepared", "committed"] | None]
    result_publication_generation_id: NotRequired[str | None]
    result_snapshot_error: NotRequired[str | None]
    recovered_attempt_ids: NotRequired[list[Annotated[str, Field(min_length=1)]] | None]
    recovered_attempt_generations: NotRequired[
        dict[Annotated[str, Field(min_length=1)], Annotated[str, Field(min_length=1)]] | None
    ]
    uncertain_attempt_ids: NotRequired[list[Annotated[str, Field(min_length=1)]] | None]


class RecoveredAttemptLocation(TypedDict):
    """Validated trial location named by operator cleanup evidence."""

    phase: Annotated[str, Field(min_length=1)]
    trial_number: Annotated[int, Field(ge=0)]
    generation_id: Annotated[str, Field(min_length=1)]


class CleanupRecovery(TypedDict):
    """Validated recovery evidence bound to one run and configuration digest."""

    run_id: str
    config_sha256: str
    cleanup_confirmed: Literal[True]
    reaped_attempt_ids: list[Annotated[str, Field(min_length=1)]]
    reaped_attempt_locations: dict[Annotated[str, Field(min_length=1)], RecoveredAttemptLocation]


_TERMINAL_STATUS_ADAPTER: TypeAdapter[TerminalStatus] = TypeAdapter(TerminalStatus)
_CLEANUP_RECOVERY_ADAPTER: TypeAdapter[CleanupRecovery] = TypeAdapter(CleanupRecovery)


@dataclass(frozen=True)
class RunObservation:
    """One coherent run decision from terminal, cleanup and process evidence.

    The terminal status is reread after a dead-runner probe. Reuse this
    observation within one payload, and obtain a new one after acquiring a
    transition lock, performing cleanup or reading mutable result artifacts.
    """

    state: RunState
    recovery_required: bool
    cleanup_recovery_required: bool
    snapshot_recovery_required: bool
    terminal_status: TerminalStatus | None
    #: The runner is ``unobservable``, so it was taken as possibly alive, and its
    #: heartbeat is missing or older than ``RUNNER_HEARTBEAT_STALE_SECONDS``.
    #: Only agent-facing reports use this guess, to ask for operator recovery;
    #: it never changes ``state``, drives a write or signal, or authorizes recovery.
    runner_heartbeat_stale: bool


@dataclass(frozen=True)
class RunHandle:
    """Immutable, on-disk identity of one detached sweep."""

    run_id: str
    experiment_id: str
    config_sha256: str
    # Absolute directory of the cataloged config file. The run's config
    # snapshot lives in the state directory, so its relative paths resolve
    # against this recorded directory, never the snapshot's own location.
    config_dir: str
    pid: int | None
    pgid: int | None
    # Native process start time; None only before spawn.
    pid_starttime: int | None
    started_at: str  # ISO-8601 UTC
    visible_params_at_launch: Literal["none", "all"] | list[str]
    launch_state: RunLaunchState = "spawned"
    allow_cancel: bool = False
    # Boot that numbered the PID; None only before spawn.
    boot_id: str | None = None
    # Platform PID-space identity; None only before spawn.
    pid_namespace: str | None = None

    @property
    def process_identity(self) -> ProcessIdentity:
        """Return the recorded runner identity, including unset pre-spawn fields."""
        return ProcessIdentity(
            pid=self.pid,
            pgid=self.pgid,
            pid_starttime=self.pid_starttime,
            boot_id=self.boot_id,
            pid_namespace=self.pid_namespace,
        )


@dataclass
class PreparedRun:
    """Owned pre-spawn launch preparation and its inherited kernel lease."""

    handle: RunHandle
    config_snapshot_path: Path
    lease: IO[str]

    @property
    def lease_fd(self) -> int:
        """Return the open launch lease descriptor inherited by the blocked runner."""
        return self.lease.fileno()

    def close(self) -> None:
        """Release this server process's copy of the launch lease."""
        self.lease.close()


class RunStore:
    """Filesystem store for run handles, logs, and status files under a state dir."""

    def __init__(self, state_dir: Path) -> None:
        """Create the run-handle store under ``state_dir``.

        :param Path state_dir: Root directory for runs, logs, config snapshots, and launch lock.
        """
        self._set_paths(state_dir)
        needs_marker = self._inspect_format_boundary()
        ensure_private_dir(state_dir)
        ensure_private_dir(self._runs_dir)
        ensure_private_dir(self._logs_dir)
        if needs_marker:
            self._create_format_marker()

    @classmethod
    def open_existing(cls, state_dir: Path) -> RunStore:
        """Open an existing run store without creating paths or changing permissions.

        The marker distinguishes state damage from an unsupported format. Without a marker,
        run handles identify unmarked state; otherwise the path is an unrelated directory.
        Unwalkable ancestors refuse opening.

        :param Path state_dir: Existing MCP state directory containing ``runs/``
            and ``logs/`` subdirectories.
        :return RunStore: Store bound to the recognized existing layout.
        :raises UnsupportedStateFormatError: The marker names another format,
            or run handles exist without a marker.
        :raises UnsafePrivatePathError: The marker is present but it, or a
            directory of the layout, is missing, unreadable, malformed, shared,
            symlinked, or not a real directory.
        :raises ValueError: If ``state_dir`` is not an MCP run-store layout.
        """
        store = cls.__new__(cls)
        store._set_paths(state_dir)
        wrong_path = ". Pass the state_dir from the catalog the MCP server runs with."
        parent = absolute_path(state_dir).parent
        if parent != parent.parent:
            try:
                os.close(open_directory_fd(parent, create=False, private_final=False))
            except (OSError, UnsafePrivatePathError) as exc:
                raise ValueError(
                    f"not an existing MCP state directory; {state_dir} is not reachable "
                    f"through real directories: {exc}{wrong_path}"
                ) from None
        try:
            os.lstat(store._format_marker_path)
        except (FileNotFoundError, NotADirectoryError):
            if store._has_run_handles():
                raise store._format_refusal("durable run data has no format marker") from None
            missing = [
                str(path)
                for path in (state_dir, store._runs_dir, store._logs_dir)
                if not path.is_dir()
            ]
            detail = (
                "expected directories are missing: " + ", ".join(missing)
                if missing
                else f"it has no format marker and no run handles under {store._runs_dir}"
            )
            raise ValueError(f"not an existing MCP state directory; {detail}{wrong_path}") from None
        except OSError:
            # Marker presence is unknown until directory validation succeeds.
            pass
        missing = []
        for path in (state_dir, store._runs_dir, store._logs_dir):
            try:
                validate_private_dir(path)
            except FileNotFoundError:
                missing.append(str(path))
        if missing:
            raise UnsafePrivatePathError(
                f"MCP state directory {state_dir} has its format marker but is missing "
                f"{', '.join(missing)}. Restore the state directory before retrying recovery."
            )
        store._require_current_marker_for_recovery()
        return store

    def _has_run_handles(self) -> bool:
        """Return whether ``runs/`` holds any run handle, without validating it.

        :return bool: Whether a ``*.json`` handle entry is present.
        """
        try:
            with os.scandir(self._runs_dir) as entries:
                return any(entry.name.endswith(".json") for entry in entries)
        except OSError:
            return False

    def _require_current_marker_for_recovery(self) -> None:
        """Require the supported format marker; report unreadable or malformed markers as damage.

        :raises UnsupportedStateFormatError: The marker names another format.
        :raises UnsafePrivatePathError: The marker is unreadable, unsafe, or malformed.
        """
        marker = self._format_marker_path
        try:
            raw = read_private_text(marker)
        except (OSError, UnicodeError) as exc:
            raise UnsafePrivatePathError(
                f"MCP state directory {marker.parent} format marker {marker} cannot be "
                f"read: {exc}. Restore it and access to it before retrying recovery."
            ) from exc
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
        version = payload.get("schema_version") if isinstance(payload, dict) else None
        if type(version) is int and version != MCP_STATE_FORMAT_VERSION:
            raise self._format_refusal(f"format marker declares unsupported format {version}")
        if type(version) is not int or set(cast(dict[str, object], payload)) != {"schema_version"}:
            raise UnsafePrivatePathError(
                f"MCP state directory {marker.parent} format marker {marker} is malformed. "
                "Restore it before retrying recovery."
            )

    def _set_paths(self, state_dir: Path) -> None:
        """Bind store paths without touching the filesystem.

        :param Path state_dir: Root directory for the MCP run store.
        """
        self._runs_dir = state_dir / "runs"
        self._logs_dir = state_dir / "logs"
        self._launch_lock_path = state_dir / ".launch.lock"
        self._format_marker_path = state_dir / _STATE_FORMAT_MARKER_NAME

    def _inspect_format_boundary(self) -> bool:
        """Inspect an existing namespace before initialization can mutate it.

        A marker is written only for a namespace with no durable run-store
        evidence. A namespace that already contains handles, logs, leases,
        or audit records without a marker is refused as unmarked PhaseSweep state.

        :return bool: Whether this fresh namespace needs its first marker.
        :raises ValueError: The namespace contains unsupported or unmarked
            durable state.
        :raises UnsafePrivatePathError: An existing state path is unsafe.
        """
        try:
            self._format_marker_path.parent.lstat()
        except FileNotFoundError:
            return True

        validate_private_dir(self._format_marker_path.parent)
        if self._path_entry_exists(self._format_marker_path):
            self._require_supported_format_marker()
            return False
        if self._has_durable_state_evidence():
            raise self._format_refusal("durable run data has no format marker")
        return True

    @staticmethod
    def _path_entry_exists(path: Path) -> bool:
        """Return whether a path entry exists without ignoring dangling links.

        :param Path path: Entry to inspect without following it.
        :return bool: Whether any filesystem entry is present at ``path``.
        """
        try:
            path.lstat()
        except FileNotFoundError:
            return False
        return True

    def _has_durable_state_evidence(self) -> bool:
        """Return whether an unmarked namespace contains run-store evidence.

        Empty run/log directories and the scaffold's root ``origin`` record remain fresh.
        Any run/log entry, including interrupted writes or unknown records, counts as evidence.

        :return bool: Whether durable unmarked run-store evidence is present.
        :raises UnsafePrivatePathError: A present run-store directory is unsafe.
        """
        if self._path_entry_exists(self._format_marker_path.parent / "audit.jsonl"):
            return True
        for directory in (self._runs_dir, self._logs_dir):
            if not self._path_entry_exists(directory):
                continue
            validate_private_dir(directory)
            try:
                with os.scandir(directory) as entries:
                    if next(entries, None) is not None:
                        return True
            except OSError as exc:
                raise self._format_refusal("durable run data could not be inspected") from exc
        return False

    def _require_supported_format_marker(self) -> None:
        """Require the exact current private namespace marker without writing it.

        :raises ValueError: The marker is malformed or declares another format.
        """
        payload = _read_json_object(self._format_marker_path)
        if (
            payload is None
            or set(payload) != {"schema_version"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != MCP_STATE_FORMAT_VERSION
        ):
            raise self._format_refusal("format marker is malformed or unsupported")

    def _create_format_marker(self) -> None:
        """Atomically claim the current format for a verified fresh namespace.

        A concurrent fresh initializer can win the exclusive create; in that
        case its marker is re-read and must be the exact supported record.
        """
        payload = json.dumps(_STATE_FORMAT_MARKER_PAYLOAD) + "\n"
        try:
            private_atomic_create_text(self._format_marker_path, payload)
        except FileExistsError:
            self._require_supported_format_marker()

    def _format_refusal(self, detail: str) -> UnsupportedStateFormatError:
        """Build the actionable refusal for unmarked or unsupported MCP state.

        :param str detail: Specific format-boundary failure observed.
        :return UnsupportedStateFormatError: Refusal that directs the operator to a safe path.
        """
        return UnsupportedStateFormatError(
            f"MCP state directory {self._format_marker_path.parent} {detail}; "
            "use a fresh MCP state directory."
        )

    @contextlib.contextmanager
    def launch_lock(self) -> Iterator[bool]:
        """Try to hold the shared, nonblocking launch lock across capacity checks and spawn.

        The file lock serializes threads and servers sharing this state directory.

        :return Iterator[bool]: Context manager yielding True when the lock was acquired.
        """
        handle = try_lock_file(self._launch_lock_path)
        try:
            yield handle is not None
        finally:
            if handle is not None:
                unlock_file(handle)

    @contextlib.contextmanager
    def transition_lock(self, handle: RunHandle | str) -> Iterator[None]:
        """Serialize cancellation markers with confirmed recovery for one run.

        The lock is shared across MCP servers and the operator CLI. Cancellation
        holds it only while rereading state and writing its marker; recovery
        holds it until cleanup evidence, terminal results, and marker removal
        have all been persisted. Only writers take it: deriving state never does.

        :param RunHandle | str handle: Run or run id whose cleanup transition is protected.
        :return Iterator[None]: Context manager holding the lock until exit.
        """
        import fcntl

        run_id = handle if isinstance(handle, str) else handle.run_id
        lock_handle = open_lock_file(self._logs_dir / f"{run_id}.transition.lock")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            yield
        finally:
            unlock_file(lock_handle)

    def new_run_id(self, experiment_id: str) -> str:
        """Mint a fresh, collision-resistant run id prefixed with the experiment id.

        :param str experiment_id: Catalog id to include in the run id prefix.
        :return str: Newly generated safe run id.
        """
        return f"{experiment_id}-{uuid4().hex}"

    def log_path(self, run_id: str) -> Path:
        """Return the operator-only captured stdout/stderr log path.

        :param str run_id: Run whose log is located.
        :return Path: Captured output path in the store's log directory.
        """
        return self._logs_dir / f"{run_id}.log"

    def status_path(self, run_id: str) -> Path:
        """Return the operator-only runner terminal-status path.

        :param str run_id: Run whose terminal status is located.
        :return Path: Terminal-status JSON path in the store's log directory.
        """
        return self._logs_dir / f"{run_id}.status.json"

    def config_snapshot_path(self, run_id: str) -> Path:
        """Return the operator-only immutable runner config path.

        :param str run_id: Run whose immutable config is located.
        :return Path: Config snapshot YAML path in the store's log directory.
        """
        return self._logs_dir / f"{run_id}.config.yaml"

    def cleanup_uncertain_path(self, run_id: str) -> Path:
        """Return the operator-only unconfirmed-cleanup marker path.

        :param str run_id: Run whose cleanup marker is located.
        :return Path: Unconfirmed-cleanup JSON path in the store's log directory.
        """
        return self._logs_dir / f"{run_id}.cleanup_uncertain.json"

    def cleanup_recovery_path(self, run_id: str) -> Path:
        """Return the operator-only cleanup recovery evidence path.

        :param str run_id: Run whose cleanup recovery evidence is located.
        :return Path: Cleanup recovery JSON path in the store's log directory.
        """
        return self._logs_dir / f"{run_id}.cleanup_recovery.json"

    def launch_lease_path(self, run_id: str) -> Path:
        """Return the private kernel-backed launch lease path.

        :param str run_id: Run whose launch lease is located.
        :return Path: Launch lock path in the store's log directory.
        """
        return self._logs_dir / f"{run_id}.launch.lock"

    def heartbeat_path(self, run_id: str) -> Path:
        """Return the private heartbeat path whose mtime the live runner refreshes.

        :param str run_id: Run whose heartbeat is located.
        :return Path: Heartbeat file path in the store's log directory.
        """
        return self._logs_dir / f"{run_id}.heartbeat"

    def prepare_launch(self, handle: RunHandle, config_bytes: bytes) -> PreparedRun:
        """Reserve and durably prepare one run without spawning a process.

        :param RunHandle handle: Launching-state handle to prepare.
        :param bytes config_bytes: Exact verified configuration bytes for the runner.
        :return PreparedRun: Owned preparation whose lease must cross ``Popen``.
        :raises ValueError: If ``handle`` is not a pre-spawn handle.
        """
        if handle.launch_state != "launching":
            raise ValueError("a launch preparation requires a launching-state handle")
        config_text = config_bytes.decode("utf-8")
        lease_path = self.launch_lease_path(handle.run_id)
        snapshot_path = self.config_snapshot_path(handle.run_id)
        created: list[Path] = []
        lease: IO[str] | None = None
        try:
            private_atomic_create_text(lease_path, "")
            created.append(lease_path)
            lease = try_lock_file(lease_path)
            if lease is None:
                raise RuntimeError(f"new launch lease for run {handle.run_id!r} is already held")
            private_atomic_create_text(snapshot_path, config_text)
            created.append(snapshot_path)
            self.create(handle)
            created.append(self._runs_dir / f"{handle.run_id}.json")
            return PreparedRun(handle=handle, config_snapshot_path=snapshot_path, lease=lease)
        except BaseException as exc:
            if lease is not None:
                lease.close()
            cleanup_failed = False
            for path in reversed([path for path in created if path != lease_path]):
                try:
                    _strict_unlink(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    cleanup_failed = True
                    log.exception("failed to roll back pre-spawn artifact %s", path)
            unpublished_survivor = not isinstance(exc, FileExistsError) and any(
                path.exists() or path.is_symlink()
                for path in (
                    snapshot_path,
                    self._runs_dir / f"{handle.run_id}.json",
                )
            )
            if lease_path in created and not cleanup_failed and not unpublished_survivor:
                try:
                    _strict_unlink(lease_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    log.exception("failed to roll back pre-spawn artifact %s", lease_path)
            raise

    def finish_launch_preparation(self, preparation: PreparedRun) -> None:
        """Release a preparation and remove its lease after a durable outcome.

        :param PreparedRun preparation: Preparation returned by :meth:`prepare_launch`.
        """
        try:
            preparation.close()
        except OSError:
            log.warning(
                "could not close completed launch lease for run %s",
                preparation.handle.run_id,
                exc_info=True,
            )
        handle = self.get(preparation.handle.run_id)
        if handle is None:
            return
        if handle.launch_state != "spawned" and self.recorded_terminal_status(handle) is None:
            # Retain the lease for orphan recovery when terminal bookkeeping failed.
            return
        self.remove_launch_lease(preparation.handle.run_id)

    def remove_launch_lease(self, run_id: str) -> None:
        """Best-effort remove a completed launch lease sidecar.

        :param str run_id: Run whose durable spawned identity supersedes the lease.
        """
        try:
            _strict_unlink(self.launch_lease_path(run_id))
        except FileNotFoundError:
            pass
        except OSError:
            log.warning(
                "could not remove completed launch lease for run %s",
                run_id,
                exc_info=True,
            )

    def create(self, handle: RunHandle) -> None:
        """Atomically claim and strictly persist a new run identity.

        :param RunHandle handle: Complete handle to publish without replacement.
        """
        target = self._runs_dir / f"{handle.run_id}.json"
        payload = json.dumps(asdict(handle), indent=2) + "\n"
        private_atomic_create_text(target, payload)

    def update(self, handle: RunHandle) -> None:
        """Persist the one legal launching-to-spawned identity transition.

        :param RunHandle handle: Updated identity for an already-created run.
        :raises FileNotFoundError: If no handle was created for ``handle.run_id``.
        :raises ValueError: If the update changes an immutable field, or is not
            a launching-to-spawned transition.
        """
        current = self.get(handle.run_id)
        if current is None:
            raise FileNotFoundError(f"Run {handle.run_id!r} has not been created.")
        immutable_fields = (
            "run_id",
            "experiment_id",
            "config_sha256",
            "config_dir",
            "started_at",
            "allow_cancel",
            "visible_params_at_launch",
        )
        changed = [
            name for name in immutable_fields if getattr(current, name) != getattr(handle, name)
        ]
        if changed:
            raise ValueError(f"Run {handle.run_id!r} update changes immutable field(s): {changed}.")
        if current == handle:
            return
        if current.launch_state != "launching" or handle.launch_state != "spawned":
            raise ValueError(f"Run {handle.run_id!r} permits only a launching-to-spawned update.")
        target = self._runs_dir / f"{handle.run_id}.json"
        payload = json.dumps(asdict(handle), indent=2)
        private_atomic_write_text(target, payload + "\n", strict_dir_fsync=True)

    def get(self, run_id: str) -> RunHandle | None:
        """Load a run handle, returning ``None`` for absent or path-unsafe IDs.

        :param str run_id: Agent-supplied run id to look up.
        :return RunHandle | None: Persisted handle when present and well-shaped, otherwise ``None``.
        """
        if not SAFE_NAME_PATTERN.fullmatch(run_id):
            return None
        path = self._runs_dir / f"{run_id}.json"
        if not path.is_file():
            return None
        return self._load_handle(path, expected_run_id=run_id)

    def _scan_handles(self) -> tuple[list[RunHandle], set[str]]:
        """Load persisted handles and identify malformed handle records.

        :return tuple: Readable handles and stable identities for malformed
            handle files. Orphaned sibling evidence is outside
            this scan and is added only by :meth:`launch_inventory`.
        """
        handles: list[RunHandle] = []
        unreadable: set[str] = set()
        for path in self._runs_dir.glob("*.json"):
            handle = self._load_handle(path, expected_run_id=path.stem)
            if handle is None:
                unreadable.add(
                    f"run:{path.stem}"
                    if SAFE_NAME_PATTERN.fullmatch(path.stem)
                    else f"handle:{path.name}"
                )
                continue
            handles.append(handle)
        return handles, unreadable

    def list_handles(self) -> list[RunHandle]:
        """Load every persisted handle, skipping any that are malformed or partial.

        :return list[RunHandle]: Successfully decoded run handles.
        """
        handles, _unreadable = self._scan_handles()
        return handles

    def launch_inventory(self) -> tuple[list[RunHandle], set[str]]:
        """Load handles and identify unreadable records that reserve launch capacity.

        Read tools skip malformed handles; launch cannot assume those records are terminal.
        The caller must hold :meth:`launch_lock` across this scan and spawn.

        :return tuple[list[RunHandle], set[str]]: Readable handles and stable
            identities for malformed handles or orphan evidence.
        """
        handles, unreadable = self._scan_handles()
        readable_ids = {handle.run_id for handle in handles}

        for path in self._logs_dir.iterdir():
            for suffix in _RUN_EVIDENCE_SUFFIXES:
                if not path.name.endswith(suffix):
                    continue
                run_id = path.name[: -len(suffix)]
                if SAFE_NAME_PATTERN.fullmatch(run_id) and run_id not in readable_ids:
                    unreadable.add(f"run:{run_id}")
                break
        return handles, unreadable

    def is_pre_spawn_orphan(self, run_id: str) -> bool:
        """Return whether a free inherited lease proves this preparation abandoned.

        The child holds the lease until its process identity is durable or it exits.

        :param str run_id: Candidate orphan run identity.
        :return bool: Whether the preparation is safe to remove automatically.
        """
        lease = self._claim_abandoned_launch_lease(run_id)
        if lease is None:
            return False
        lease.close()
        return True

    def _claim_abandoned_launch_lease(self, run_id: str) -> IO[str] | None:
        """Lock and revalidate one transactional pre-spawn preparation.

        :param str run_id: Candidate prepared run identity.
        :return IO[str] | None: Held lease when the preparation is abandoned.
        """
        if not SAFE_NAME_PATTERN.fullmatch(run_id):
            return None
        lease_path = self.launch_lease_path(run_id)
        if not lease_path.is_file() or lease_path.is_symlink():
            return None
        try:
            lease = try_lock_file(lease_path)
        except (OSError, UnsafePrivatePathError):
            return None
        if lease is None:
            return None
        handle_path = self._runs_dir / f"{run_id}.json"
        handle = self.get(run_id)
        terminal_evidence = (
            self.status_path(run_id),
            self.cleanup_uncertain_path(run_id),
            self.cleanup_recovery_path(run_id),
        )
        invalid = any(path.exists() or path.is_symlink() for path in terminal_evidence)
        if handle_path.exists() or handle_path.is_symlink():
            invalid = invalid or handle is None or handle.launch_state != "launching"
        elif self.log_path(run_id).exists() or self.log_path(run_id).is_symlink():
            # A log without its handle prevents a stale lease from proving abandonment.
            invalid = True
        if invalid:
            lease.close()
            return None
        return lease

    def clear_pre_spawn_orphan(self, run_id: str) -> Path | None:
        """Preserve any runner log and remove a revalidated preparation durably.

        :param str run_id: Run id of the abandoned launch preparation.
        :raises ValueError: The evidence no longer has the provably pre-spawn shape.
        :return Path | None: Preserved runner log path, including an archive from an
            interrupted recovery, or None when no log existed.
        """
        recovered_log: Path | None = None
        with self.transition_lock(run_id):
            lease = self._claim_abandoned_launch_lease(run_id)
            if lease is None:
                raise ValueError(f"run {run_id!r} is not a provably abandoned preparation")
            try:
                log_path = self.log_path(run_id)
                archive_path = log_path.with_suffix(".log.recovered")
                if log_path.exists() or log_path.is_symlink():
                    recovered_log = archive_path
                    # Archive diagnostics durably before removing the reservation.
                    # UUID-based run IDs are not reused.
                    directory_fd = open_directory_fd(
                        self._logs_dir,
                        create=False,
                        private_final=True,
                    )
                    try:
                        os.rename(
                            log_path.name,
                            recovered_log.name,
                            src_dir_fd=directory_fd,
                            dst_dir_fd=directory_fd,
                        )
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                elif archive_path.exists() or archive_path.is_symlink():
                    recovered_log = archive_path
                paths = (
                    self._runs_dir / f"{run_id}.json",
                    self._logs_dir / f"{run_id}.transition.lock",
                    self.heartbeat_path(run_id),
                    self.config_snapshot_path(run_id),
                    self.launch_lease_path(run_id),
                )
                for path in paths:
                    with contextlib.suppress(FileNotFoundError):
                        _strict_unlink(path)
            finally:
                lease.close()
        return recovered_log

    def _load_handle(self, path: Path, *, expected_run_id: str) -> RunHandle | None:
        """Load and normalize one run handle, returning ``None`` when malformed.

        :param Path path: Persisted run-handle JSON path to read.
        :param str expected_run_id: Run id implied by the filename.
        :return RunHandle | None: Normalized handle, or ``None`` when validation fails.
        """
        if not SAFE_NAME_PATTERN.fullmatch(expected_run_id):
            return None
        payload = _read_json_object(path)
        if payload is None:
            return None
        try:
            handle = RunHandle(**payload)  # type: ignore[arg-type]
        except TypeError:
            return None
        if handle.run_id != expected_run_id:
            return None
        if not isinstance(handle.experiment_id, str) or not SAFE_NAME_PATTERN.fullmatch(
            handle.experiment_id
        ):
            return None
        if not is_sha256_hex(handle.config_sha256):
            return None
        if not _valid_config_dir(handle.config_dir):
            return None
        if type(handle.allow_cancel) is not bool:
            return None
        visible_params = handle.visible_params_at_launch
        if isinstance(visible_params, str):
            if visible_params not in {"none", "all"}:
                return None
        elif isinstance(visible_params, list):
            if not all(
                type(key) is str and key and key == key.strip() for key in visible_params
            ) or len(set(visible_params)) != len(visible_params):
                return None
        else:
            return None
        if handle.launch_state not in {"launching", "spawned"}:
            return None
        if parse_utc_iso(handle.started_at) is None:
            return None
        if not _valid_optional_boot_id(handle.boot_id):
            return None
        if handle.pid_namespace is not None and not is_pid_namespace(handle.pid_namespace):
            return None
        if handle.launch_state == "launching":
            if (
                handle.pid is not None
                or handle.pgid is not None
                or handle.pid_starttime is not None
                or handle.boot_id is not None
                or handle.pid_namespace is not None
            ):
                return None
        else:
            if type(handle.pid) is not int or handle.pid <= 0:
                return None
            if type(handle.pgid) is not int or handle.pgid <= 0:
                return None
            if type(handle.pid_starttime) is not int or handle.pid_starttime <= 0:
                return None
            if handle.boot_id is None or handle.pid_namespace is None:
                return None
        return handle

    def observe(self, handle: RunHandle, *, view: RunnerView | None = None) -> RunObservation:
        """Read and classify one run without mutating its persisted evidence.

        Finalized return codes determine terminal state independently of snapshot capture.
        A dead or unverifiable runner without terminal evidence stays running until
        recovery reconciles its trial cleanup.

        A valid uncertainty marker reserves capacity unless terminal cleanup
        or an ended boot supersedes it. An ended boot settles processes but
        leaves explicit uncertain trial cleanup subject to reconciliation.
        Pending snapshots reserve capacity until finalization; only a dead
        runner makes that pending state require operator recovery.

        Only an ``unobservable`` runner's heartbeat is read, and its staleness
        changes neither the state nor any recovery requirement.

        :param RunHandle handle: Run whose state and recovery facts are needed.
        :param RunnerView | None view: Runner visibility already derived for this decision.
        :return RunObservation: State, recovery requirements, the same terminal
            status, and whether an unobservable runner's heartbeat is stale.
        """
        if view is None:
            view = self.runner_view(handle)
        if handle.pid is not None and view.visibility == "observable":
            reap_child(handle.pid)
        status = self._read_status(handle)
        gone = view.visibility == "gone"
        runner_alive = False
        if (status is None or status.get("result_snapshot_state") == "pending") and (
            handle.launch_state == "spawned" and not gone
        ):
            runner_alive = self.runner_may_be_alive(handle, view=view)
            if not runner_alive:
                # The runner can publish its terminal status while the probe observes
                # its exit. Pending and absent status must both be checked again.
                latest_status = self._read_status(handle)
                if latest_status is not None:
                    status = latest_status
        cleanup_confirmed = None if status is None else status["cleanup_confirmed"]
        marker_reserves = (
            self.cleanup_uncertain(handle)
            and cleanup_confirmed is not True
            and not (gone and cleanup_confirmed is not False)
        )
        needs_recovery_evidence = cleanup_confirmed is False or (
            status is None and handle.launch_state == "spawned" and not runner_alive and not gone
        )
        cleanup_recovered = needs_recovery_evidence and self._cleanup_recovered(handle)
        terminal_cleanup_uncertain = cleanup_confirmed is False and not cleanup_recovered
        cleanup_recovery = marker_reserves or terminal_cleanup_uncertain
        snapshot_recovery = (
            status is not None
            and status.get("result_snapshot_state") == "pending"
            and not runner_alive
        )
        launch_outcome_unknown = handle.launch_state == "launching" and status is None
        dead_without_status = (
            handle.launch_state == "spawned"
            and status is None
            and view.visibility in {"observable", "absent"}
            and not runner_alive
        )
        if (not gone and marker_reserves) or (terminal_cleanup_uncertain and not gone):
            state: RunState = "running"
        elif status is not None:
            if status.get("result_snapshot_state") == "pending":
                state = "running"
            elif status["returncode"] == 0:
                state = "succeeded"
            elif (
                status["returncode"] in _SIGNALLED_EXIT_CODES
                or status.get("error_class") == "cancelled"
            ):
                state = "cancelled"
            else:
                state = "failed"
        elif (
            handle.pid is None
            or launch_outcome_unknown
            or runner_alive
            or view.visibility == "unobservable"
        ):
            state = "running"
        else:
            state = "failed" if gone or cleanup_recovered else "running"
        # Probes cannot reach an unobservable runner, so the decision above took it
        # as possibly alive; its heartbeat is the only hint left that it stopped.
        heartbeat_stale = (
            runner_alive
            and view.visibility == "unobservable"
            and self._runner_heartbeat_stale(handle)
        )
        return RunObservation(
            state,
            launch_outcome_unknown or dead_without_status or cleanup_recovery or snapshot_recovery,
            cleanup_recovery,
            snapshot_recovery,
            status,
            heartbeat_stale,
        )

    def recorded_terminal_status(self, handle: RunHandle) -> dict[str, object] | None:
        """Return the runner-written terminal status payload, if readable.

        :param RunHandle handle: Run handle whose terminal status should be returned.
        :return dict[str, object] | None: Decoded status payload, or ``None`` when absent or malformed.
        """
        status = self._read_status(handle)
        return None if status is None else dict(status)

    def mark_cleanup_uncertain(self, handle: RunHandle) -> None:
        """Persist unconfirmed process-group cleanup to reserve capacity until cleanup is settled.

        :param RunHandle handle: Run handle whose cleanup is uncertain.
        """
        existing = self._read_cleanup_identity(handle)
        candidate = handle.process_identity
        candidate_has_identity = candidate.pid is not None or candidate.pgid is not None
        identity = existing if existing is not None and not candidate_has_identity else candidate
        payload = {
            "run_id": handle.run_id,
            "config_sha256": handle.config_sha256,
            **asdict(identity),
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
        with contextlib.suppress(FileNotFoundError):
            _strict_unlink(self.cleanup_uncertain_path(handle.run_id))

    def cleanup_uncertain(self, handle: RunHandle) -> bool:
        """Return whether a valid cleanup uncertainty marker exists for ``handle``.

        :param RunHandle handle: Run handle whose cleanup marker should be checked.
        :return bool: Whether a valid cleanup uncertainty marker exists.
        """
        return self._read_cleanup_identity(handle) is not None

    def cleanup_identity(self, handle: RunHandle) -> ProcessIdentity:
        """Return a valid cleanup marker's process identity, or the durable handle identity.

        A marker may hold the only PID/PGID when the saved handle is still pre-spawn.

        :param RunHandle handle: Run handle whose runner identity is needed.
        :return ProcessIdentity: Marker identity when stronger, otherwise handle identity.
        """
        marker_identity = self._read_cleanup_identity(handle)
        if (
            handle.launch_state == "launching"
            and marker_identity is not None
            and (marker_identity.pid is not None or marker_identity.pgid is not None)
        ):
            return marker_identity
        return handle.process_identity

    def runner_view(self, handle: RunHandle) -> RunnerView:
        """Classify the strongest runner identity in this boot and PID namespace.

        ``gone`` proves its boot ended, including descendants. ``absent`` proves only its runner
        group absent from another namespace. ``unobservable`` forbids local probes or signals.

        :param RunHandle handle: Run handle whose recorded boot and namespace are checked.
        :return RunnerView: ``gone``, ``absent``, ``unobservable`` (with the
            reason), or ``observable``.
        """
        identity = self.cleanup_identity(handle)
        if identity_from_earlier_boot(identity.boot_id, read_boot_id()):
            return RunnerView("gone")
        namespace = identity.pid_namespace
        # Validation records a namespace with every PID, so only a run that
        # never had a runner identity lacks both.
        if identity.pid is None or namespace is None or same_pid_namespace(namespace):
            return RunnerView("observable")
        blocker = recorded_process_blocker(
            namespace,
            pids=(identity.pid,),
            pgids=() if identity.pgid is None else (identity.pgid,),
        )
        if blocker is None:
            return RunnerView("absent")
        return RunnerView("unobservable", foreign_pid_namespace_reason(namespace, blocker))

    def latest_run_for(self, experiment_id: str) -> RunHandle | None:
        """Return the newest persisted handle for an experiment deterministically.

        The result is computed from validated durable launch timestamps, with
        ``run_id`` as a stable tie-breaker.

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

    def cleanup_recovered_attempt_evidence(
        self,
        handle: RunHandle,
    ) -> tuple[set[str], dict[str, tuple[str, int, str]]]:
        """Return recovered attempt IDs and locations from one evidence snapshot.

        Runner status contributes attempt IDs only when its generation map binds
        them to this run. Operator evidence contributes IDs and authorizes only
        the locations whose IDs appear in that same recovery record.

        :param RunHandle handle: Run whose cleanup recovery evidence should be read.
        :return tuple: Reaped attempt identities and operator-authorized locations.
        """
        attempt_ids: set[str] = set()
        status = self._read_status(handle)
        if status is not None:
            generations = status.get("recovered_attempt_generations")
            recovered_ids = status.get("recovered_attempt_ids")
            if generations is not None and recovered_ids is not None:
                attempt_ids.update(
                    attempt_id
                    for attempt_id in recovered_ids
                    if generations.get(attempt_id) == handle.run_id
                )
        payload = self._read_cleanup_recovery(handle)
        if payload is None:
            return attempt_ids, {}
        authorized = set(payload["reaped_attempt_ids"])
        locations = payload["reaped_attempt_locations"]
        attempt_ids.update(authorized)
        return attempt_ids, {
            attempt_id: (raw["phase"], raw["trial_number"], raw["generation_id"])
            for attempt_id, raw in locations.items()
            if attempt_id in authorized
        }

    def cleanup_uncertain_attempt_ids(self, handle: RunHandle) -> set[str]:
        """Return the exact attempts named cleanup-uncertain by this run's terminal report.

        :param RunHandle handle: Run whose terminal cleanup cause is read.
        :return set[str]: Persisted uncertain attempt identities.
        """
        status = self._read_status(handle)
        if status is None:
            return set()
        return set(status.get("uncertain_attempt_ids") or [])

    def runner_may_be_alive(self, handle: RunHandle, *, view: RunnerView | None = None) -> bool:
        """Return whether a spawned handle may still identify a live runner.

        :param RunHandle handle: Persisted runner identity.
        :param RunnerView | None view: The caller's derivation of the runner,
            or ``None`` to derive it here.
        :return bool: True when PID and starttime still identify a non-zombie
            process of the boot and PID namespace the handle was written under,
            or when the runner lives in a PID namespace this process cannot
            observe. False when it is provably gone or absent, or observably dead.
        """
        if handle.launch_state != "spawned" or handle.pid is None or handle.pid_starttime is None:
            return False
        if view is None:
            view = self.runner_view(handle)
        if view.visibility != "observable":
            return view.visibility == "unobservable"
        return is_same_live_process(handle.pid, handle.pid_starttime)

    def _runner_heartbeat_stale(self, handle: RunHandle) -> bool:
        """Return whether a spawned runner's heartbeat is missing or too old.

        The runner creates its heartbeat before its spawned identity is durable,
        so a missing one counts as stale. A host suspend pauses refreshes, so a
        resumed runner can read stale until its next refresh.

        :param RunHandle handle: Spawned run whose heartbeat is checked.
        :return bool: Whether no regular heartbeat file exists or its
            modification time is more than ``RUNNER_HEARTBEAT_STALE_SECONDS`` old.
        """
        try:
            info = os.stat(self.heartbeat_path(handle.run_id), follow_symlinks=False)
        except OSError:
            return True
        if not stat.S_ISREG(info.st_mode):
            return True
        return time.time() - info.st_mtime > RUNNER_HEARTBEAT_STALE_SECONDS

    def _read_status(self, handle: RunHandle) -> TerminalStatus | None:
        """Read the runner-written terminal status payload.

        :param RunHandle handle: Run handle whose status file should be read.
        :return TerminalStatus | None: Validated JSON fields, or ``None`` when absent or malformed.
        """
        payload = _read_json_object(self.status_path(handle.run_id))
        return payload if payload is not None and _valid_terminal_status(payload, handle) else None

    def _cleanup_recovered(self, handle: RunHandle) -> bool:
        """Return whether operator recovery evidence confirms cleanup for this run.

        :param RunHandle handle: Run handle whose recovery evidence should be checked.
        :return bool: Whether valid recovery evidence confirms cleanup.
        """
        return self._read_cleanup_recovery(handle) is not None

    def _read_cleanup_recovery(self, handle: RunHandle) -> CleanupRecovery | None:
        """Read matching operator cleanup evidence for a run handle.

        :param RunHandle handle: Run handle whose recovery evidence should be read.
        :return CleanupRecovery | None: The recovery payload when the file
            exists, parses as a JSON object, and its ``run_id``,
            ``config_sha256``, and ``cleanup_confirmed`` fields match this
            handle; ``None`` otherwise.
        """
        payload = _read_json_object(self.cleanup_recovery_path(handle.run_id))
        return payload if payload is not None and _valid_cleanup_recovery(payload, handle) else None

    def _read_cleanup_identity(
        self,
        handle: RunHandle,
    ) -> ProcessIdentity | None:
        """Read and validate this run's cleanup uncertainty marker.

        :param RunHandle handle: Run handle whose cleanup marker should be read.
        :return ProcessIdentity | None: Valid marker identity, or ``None`` if unavailable.
        """
        payload = _read_json_object(self.cleanup_uncertain_path(handle.run_id))
        if payload is None:
            return None
        if payload.get("run_id") != handle.run_id:
            return None
        if payload.get("config_sha256") != handle.config_sha256:
            return None
        if payload.get("cleanup_confirmed") is not False:
            return None

        pid = payload.get("pid")
        pgid = payload.get("pgid")
        pid_starttime = payload.get("pid_starttime")
        boot_id = payload.get("boot_id")
        pid_namespace = payload.get("pid_namespace")
        if not _valid_positive_optional_int(pid):
            return None
        if not _valid_positive_optional_int(pgid):
            return None
        if not _valid_positive_optional_int(pid_starttime):
            return None
        if not _valid_optional_boot_id(boot_id):
            return None
        if pid_namespace is not None and not is_pid_namespace(pid_namespace):
            return None
        # A PID means nothing without the namespace that numbers it.
        if (pid is None) != (pid_namespace is None):
            return None
        if pid is None:
            if pgid is not None or pid_starttime is not None:
                return None
        elif pgid is None:
            return None

        identity = ProcessIdentity(
            pid=pid,
            pgid=pgid,
            pid_starttime=pid_starttime,
            boot_id=boot_id,
            pid_namespace=pid_namespace,
        )
        if handle.launch_state == "spawned":
            durable_identity = asdict(handle.process_identity)
            if any(
                value is not None and value != durable_identity[name]
                for name, value in asdict(identity).items()
            ):
                return None
        return identity


def _valid_terminal_status(
    payload: dict[str, object], handle: RunHandle
) -> TypeGuard[TerminalStatus]:
    """Validate current terminal fields and their publication/finalization relationships.

    :param dict[str, object] payload: Decoded terminal JSON, including unrelated extra fields.
    :param RunHandle handle: Run identity that the terminal record must name.
    :return bool: Whether current terminal fields and cross-field relationships are valid.
    """
    if payload.get("run_id") != handle.run_id:
        return False
    try:
        _TERMINAL_STATUS_ADAPTER.validate_python(payload, strict=True)
    except ValidationError:
        return False
    result_snapshot_state = payload.get("result_snapshot_state")
    if result_snapshot_state == "complete" and not isinstance(payload.get("result_snapshot"), dict):
        return False
    result_publication_state = payload.get("result_publication_state")
    result_publication_generation_id = payload.get("result_publication_generation_id")
    if (result_publication_state is None) != (result_publication_generation_id is None):
        return False
    if result_publication_state is not None:
        if (
            not isinstance(result_publication_generation_id, str)
            or not SAFE_NAME_PATTERN.fullmatch(result_publication_generation_id)
            or not isinstance(payload.get("result_snapshot"), dict)
        ):
            return False
        if result_publication_state == "prepared" and result_snapshot_state != "pending":
            return False
    return True


def _valid_cleanup_recovery(
    payload: dict[str, object], handle: RunHandle
) -> TypeGuard[CleanupRecovery]:
    """Validate run binding and every operator-authorized recovered trial location.

    :param dict[str, object] payload: Decoded operator cleanup recovery JSON.
    :param RunHandle handle: Run and configuration identity that the record must name.
    :return bool: Whether the binding and every recovered attempt location are valid.
    """
    if not (
        payload.get("run_id") == handle.run_id
        and payload.get("config_sha256") == handle.config_sha256
        and payload.get("cleanup_confirmed") is True
    ):
        return False
    try:
        _CLEANUP_RECOVERY_ADAPTER.validate_python(payload, strict=True)
    except ValidationError:
        return False
    return True


def _valid_positive_optional_int(value: object) -> TypeGuard[int | None]:
    """Accept None or a positive integer, excluding booleans.

    :param object value: Candidate optional integer.
    :return TypeGuard[int | None]: Whether the value is None or a positive built-in integer.
    """
    return value is None or (type(value) is int and value > 0)


def _valid_config_dir(value: object) -> TypeGuard[str]:
    """Accept absolute path strings without NUL.

    :param object value: Candidate configuration-directory path.
    :return TypeGuard[str]: Whether the value is a built-in string naming an absolute path without NUL.
    """
    return type(value) is str and "\0" not in value and Path(value).is_absolute()


def _valid_optional_boot_id(value: object) -> TypeGuard[str | None]:
    """Accept None or canonical lowercase boot UUID text.

    :param object value: Candidate optional boot identifier.
    :return TypeGuard[str | None]: Whether the value is None or canonical UUID text.
    """
    if value is None:
        return True
    if type(value) is not str:
        return False
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False
