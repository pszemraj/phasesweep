"""SDK-free MCP tools over the catalog, detached runs, and frozen results.

Launch and cancellation write audit records and retain process ownership
through durable handles and the runner handshake. Read tools expose catalog
metadata and run-specific status or winners under the visibility policy.
The FastMCP adapter in :mod:`phasesweep.mcp.server` wraps each method as a tool.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NoReturn

from pydantic import ValidationError

from phasesweep.config.models import Experiment
from phasesweep.config.reference import (
    ReferenceEntry,
    closest,
    first_sentence,
    lookup,
    plain_path,
    reference_notes,
    top_level_entries,
)
from phasesweep.engine.errors import (
    ArtifactRootConflictError,
    PublicationAccessError,
    PublicationIntegrityError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
)
from phasesweep.engine.fingerprints import _experiment_semantic_fingerprint
from phasesweep.engine.read import EngineStatus, PhaseWinnerView, read_result
from phasesweep.engine.resume import _preflight_skipped_winners
from phasesweep.mcp.audit import AUDIT_LOG_NAME, AuditLogger
from phasesweep.mcp.errors import (
    ConcurrencyLimitError,
    ConfigChangedError,
    ExperimentBusyError,
    InvalidPhaseError,
    LaunchInProgressError,
    McpToolError,
    PermissionDeniedError,
    ResumeNotReadyError,
    ResumePublicationUnusableError,
    RunCapacityUnknownError,
    RunLaunchUnsettledError,
    RunNamespaceMismatchError,
    RunPersistentStateUnavailableError,
    RunResultSnapshotUnavailableError,
    RunSnapshotUnavailableError,
    UnknownExperimentError,
    UnknownRunError,
)
from phasesweep.mcp.redaction import (
    ResultSource,
    intersect_visible_params,
    status_payload,
    winners_payload,
)
from phasesweep.mcp.registry import RegisteredExperiment, Registry, VisibleParamsPolicy
from phasesweep.mcp.runner import FailurePayload
from phasesweep.mcp.runs import (
    LAUNCH_ACK_BYTE,
    LAUNCH_READY_BYTE,
    PreparedRun,
    RunHandle,
    RunnerView,
    RunObservation,
    RunState,
    RunStore,
    load_experiment_snapshot,
    write_status_file_if_absent,
)
from phasesweep.mcp.snapshots import (
    McpPublicationState,
    RunResultSnapshot,
    capture_pre_generation_result_snapshot,
    parse_result_snapshot,
)
from phasesweep.mcp.tool_names import TOOL_CANCEL_RUN, TOOL_LAUNCH_RUN
from phasesweep.runtime.files import ensure_private_dir, open_private_text
from phasesweep.runtime.json import JsonObject, JsonValue
from phasesweep.runtime.process import fd_ready
from phasesweep.runtime.reaper import (
    kill_stale_group,
    read_boot_id,
    read_pid_namespace,
    read_proc_starttime,
)
from phasesweep.runtime.time import parse_utc_iso, utc_now_iso

log = logging.getLogger("phasesweep.mcp.server")

DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 100
# await_run blocks server-side so one call replaces many status polls. Keep the
# default below common 30-second MCP client deadlines; callers with a longer
# tool-call budget can opt into the larger supported range.
AWAIT_DEFAULT_TIMEOUT_SECONDS = 20
AWAIT_MIN_TIMEOUT_SECONDS = 5
AWAIT_MAX_TIMEOUT_SECONDS = 600
# Recheck cadence inside one await. Kept at the minimum timeout so a
# default-length wait rechecks run state several times mid-wait instead of only
# at entry and at the deadline.
AWAIT_RECHECK_SECONDS = 5


_RUNNER_READY_TIMEOUT_SECONDS = 10.0


class _SpawnBookkeepingError(Exception):
    """Carry cleanup evidence for a failure after a runner process was created."""

    def __init__(self, original_error: BaseException, *, cleanup_confirmed: bool) -> None:
        """Create an internal post-spawn ownership failure.

        :param BaseException original_error: Failure raised after ``Popen`` succeeded.
        :param bool cleanup_confirmed: Whether the spawned process group is confirmed gone.
        """
        super().__init__("runner bookkeeping failed after process creation")
        self.original_error = original_error
        self.cleanup_confirmed = cleanup_confirmed


# Environment variables that make the interpreter execute code before the
# detached runner's first statement. They are dropped from the child's
# environment; PYTHONNOUSERSITE backs up the ``-s`` flag for the same reason.
_PRE_IMPORT_CODE_ENV_VARS = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONEXECUTABLE",
)


def _runner_env() -> dict[str, str]:
    """Build the detached runner's environment without pre-import code hooks.

    Everything else the operator exported is preserved: the runner's trials
    legitimately need ``PATH``, ``CUDA_*``, credentials, and the rest of the
    ambient environment, so this strips exactly the variables that can run
    code before ``phasesweep.mcp.runner`` gets control.

    :return dict[str, str]: Copy of this server's environment with the
        pre-import code hooks removed and ``PYTHONNOUSERSITE`` forced on.
    """
    env = {
        name: value for name, value in os.environ.items() if name not in _PRE_IMPORT_CODE_ENV_VARS
    }
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _runner_protocol_argv(
    *,
    run_id: str,
    config_snapshot_path: Path,
    config_sha256: str,
    status_path: Path,
    state_dir: Path,
    experiment_id: str,
    started_at: str,
    launch_ready_fd: int | None = None,
    launch_ack_fd: int | None = None,
    launch_lease_fd: int | None = None,
) -> list[str]:
    """Build the required detached-runner protocol arguments.

    Interpreter hardening flags and the project cwd stay at the spawn boundary:
    they govern how the child starts, while this list is the stable identity and
    persistence contract consumed by :mod:`phasesweep.mcp.runner`.

    :param str run_id: Claimed run identifier.
    :param Path config_snapshot_path: Immutable config snapshot for the run.
    :param str config_sha256: Expected digest of the config snapshot.
    :param Path status_path: Terminal-status destination.
    :param Path state_dir: MCP state directory containing the run store.
    :param str experiment_id: Catalog experiment identifier.
    :param str started_at: Claimed launch timestamp.
    :param int | None launch_ready_fd: Child pipe used to report a durable process identity.
    :param int | None launch_ack_fd: Child pipe blocking work until server acknowledgement.
    :param int | None launch_lease_fd: Inherited preparation lease closed after acknowledgement.
    :return list[str]: Runner arguments without interpreter prefix, cwd, or optional grants.
    """
    argv = [
        "--run-id",
        run_id,
        "--config",
        str(config_snapshot_path),
        "--config-sha256",
        config_sha256,
        "--status-path",
        str(status_path),
        "--state-dir",
        str(state_dir),
        "--experiment-id",
        experiment_id,
        "--started-at",
        started_at,
    ]
    if launch_ready_fd is not None:
        argv += ["--launch-ready-fd", str(launch_ready_fd)]
    if launch_ack_fd is not None:
        argv += ["--launch-ack-fd", str(launch_ack_fd)]
    if launch_lease_fd is not None:
        argv += ["--launch-lease-fd", str(launch_lease_fd)]
    return argv


def _config_entry_payload(entry: ReferenceEntry, description: str) -> JsonObject:
    """Build the agent-visible record of one documented config key.

    :param ReferenceEntry entry: Documented key from the packaged reference.
    :param str description: Description text to report for the key.
    :return JsonObject: The key's path, signature, section and description.
    """
    return {
        "path": entry.path,
        "signature": entry.signature,
        "section": entry.section,
        "description": description,
    }


def _cursor_offset(cursor: str | None) -> int:
    """Decode the opaque v1 cursor into a list offset.

    :param str | None cursor: Cursor supplied by the agent.
    :return int: Zero-based catalog offset.
    :raises McpToolError: If the cursor is not a non-negative integer produced
        by ``list_experiments``.
    """
    if cursor is None:
        return 0
    try:
        offset = int(cursor)
    except ValueError:
        raise McpToolError("invalid cursor; use next_cursor returned by list_experiments") from None
    if offset < 0:
        raise McpToolError("invalid cursor; use next_cursor returned by list_experiments")
    return offset


def _run_elapsed_seconds(store: RunStore, handle: RunHandle, state: str) -> int | None:
    """Compute wall seconds for a run: launch-to-now while running, total when terminal.

    Terminal runs use the runner-stamped ``ended_at`` in status.json. A
    terminal run with no readable endpoint reports ``None`` rather than a guess.

    :param RunStore store: Run store used to read the terminal status.
    :param RunHandle handle: Persisted run whose duration is measured.
    :param str state: Derived run state for ``handle``.
    :return int | None: Non-negative whole seconds, or ``None`` when the
        endpoints cannot be established.
    """
    started = parse_utc_iso(handle.started_at)
    if started is None:
        return None
    ended: datetime | None
    if state == "running":
        ended = datetime.now(UTC)
    else:
        status = store.recorded_terminal_status(handle)
        ended = parse_utc_iso(status.get("ended_at")) if status is not None else None
    if ended is None:
        return None
    return max(0, round((ended - started).total_seconds()))


def _await_snapshot(
    state: str,
    recovery_required: bool,
    status: dict[str, Any],
) -> tuple[Any, ...]:
    """Reduce a status read to the comparable facts await_run watches.

    :param str state: Derived run state at read time.
    :param bool recovery_required: Whether the run needs operator recovery.
    :param dict[str, Any] status: Path-free ``read_status`` payload.
    :return tuple[Any, ...]: Hashable snapshot of run state, recovery requirement,
        and per-phase winner presence and dense trial-state counts.
    """
    return (
        state,
        recovery_required,
        tuple(
            (
                phase["phase"],
                phase["winner_present"],
                tuple(sorted(phase["trials"].items())),
            )
            for phase in status["phases"]
        ),
    )


def _phase_gained_winner(baseline: tuple[Any, ...], snapshot: tuple[Any, ...]) -> bool:
    """Return whether any phase gained a winner between two await snapshots.

    :param tuple[Any, ...] baseline: Snapshot taken when the wait began.
    :param tuple[Any, ...] snapshot: Snapshot from the latest recheck.
    :return bool: ``True`` when a phase's winner artifact appeared mid-wait.
    """
    had_winner = {phase: winner for phase, winner, _counts in baseline[2]}
    return any(
        winner and not had_winner.get(phase, False) for phase, winner, _counts in snapshot[2]
    )


class PhaseSweepMCP:
    """SDK-free implementation of every tool. Methods raise ``McpToolError``."""

    def __init__(
        self, registry: Registry, runs: RunStore, audit: AuditLogger | None = None
    ) -> None:
        """Create the SDK-free MCP implementation.

        :param Registry registry: Validated catalog registry.
        :param RunStore runs: Persistent detached-run store.
        :param AuditLogger | None audit: Optional audit sink override; defaults to the
            ``stdio`` log under ``state_dir``.
        """
        self._registry = registry
        self._runs = runs
        self._audit = audit or AuditLogger(registry.state_dir / AUDIT_LOG_NAME)

    def _audit_success(
        self,
        tool: str,
        args: JsonObject | None = None,
        *,
        resolved: JsonObject | None = None,
        state_before: JsonObject | None = None,
        state_after: JsonObject | None = None,
        result_counts: dict[str, int] | None = None,
    ) -> None:
        """Record a successful state-changing tool call.

        :param str tool: MCP tool name.
        :param JsonObject | None args: Safe agent-supplied arguments.
        :param JsonObject | None resolved: Safe server-resolved identifiers.
        :param JsonObject | None state_before: Safe state summary before the call.
        :param JsonObject | None state_after: Safe state summary after the call.
        :param dict[str, int] | None result_counts: Result counts that avoid copying full payloads.
        """
        self._audit.record(
            tool=tool,
            args=args,
            outcome="success",
            resolved=resolved,
            state_before=state_before,
            state_after=state_after,
            result_counts=result_counts,
        )

    def _audit_error(
        self,
        tool: str,
        args: JsonObject | None,
        exc: Exception,
        *,
        resolved: JsonObject | None = None,
        state_before: JsonObject | None = None,
    ) -> None:
        """Record a failed state-changing tool call.

        :param str tool: MCP tool name.
        :param JsonObject | None args: Safe agent-supplied arguments.
        :param Exception exc: Exception that will be surfaced through the safe tool wrapper.
        :param JsonObject | None resolved: Safe server-resolved identifiers known before failure.
        :param JsonObject | None state_before: Safe state summary before the failure.
        """
        message = str(exc) if isinstance(exc, McpToolError) else "internal error"
        self._audit.record(
            tool=tool,
            args=args,
            outcome="error",
            resolved=resolved,
            state_before=state_before,
            error_type=type(exc).__name__,
            error=message,
        )

    def list_experiments(
        self,
        *,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Return a path-free catalog page (ids, descriptions, phases, metric).

        :param int limit: Maximum catalog entries to return.
        :param str | None cursor: Optional pagination cursor returned by a prior call.
        :return dict[str, Any]: Catalog page safe for the agent.
        :raises McpToolError: If ``limit`` is outside ``1..MAX_LIST_LIMIT``, or
            the cursor is not one this tool returned.
        """
        if limit < 1 or limit > MAX_LIST_LIMIT:
            raise McpToolError(f"limit must be between 1 and {MAX_LIST_LIMIT}")
        offset = _cursor_offset(cursor)
        experiments = self._registry.summaries()
        total_count = len(experiments)
        page = experiments[offset : offset + limit]
        next_offset = offset + len(page)
        return {
            "experiments": page,
            "total_count": total_count,
            "next_cursor": str(next_offset) if next_offset < total_count else None,
        }

    @staticmethod
    def explain_config(key: str | None = None) -> JsonObject:
        """Describe configuration keys from the packaged reference, independent of the catalog.

        With no key, or a blank one, the entries are the top-level keys with the first sentence
        of their description, and the notes are the reference's general notes. With a key, they
        are that key and every key below it with the full description, one entry per
        discriminator variant, and the notes are empty. Nothing here touches the catalog,
        a config file or the filesystem.

        :param str | None key: Config key as the reference spells it, such as ``phases[].sampler``,
            ``phases.sampler`` or ``phases.0.sampler``.
        :return JsonObject: The echoed key, whether it matched, the entries, the closest
            documented keys when it did not, and the general notes when no key was given.
        """
        entries: list[JsonValue]
        if key is None or not key.strip():
            entries = [
                _config_entry_payload(entry, first_sentence(entry, None))
                for entry in top_level_entries()
            ]
            return {
                "key": key,
                "found": True,
                "entries": entries,
                "closest": [],
                "notes": "\n".join(reference_notes()),
            }
        matches = lookup(key) if plain_path(key) else ()
        entries = [_config_entry_payload(entry, "\n".join(entry.comments)) for entry in matches]
        suggestions: list[JsonValue] = [] if matches else list(closest(key))
        return {
            "key": key,
            "found": bool(matches),
            "entries": entries,
            "closest": suggestions,
            "notes": "",
        }

    def validate(self, experiment_id: str) -> dict[str, Any]:
        """Report an experiment's phase structure (never the command/env/storage).

        :param str experiment_id: Catalog experiment id to inspect.
        :return dict[str, Any]: Path-free validation payload for the agent.
        """
        reg = self._registry.get(experiment_id)
        self._current_config_bytes(reg)
        exp = reg.experiment
        phases = [
            {
                "name": p.name,
                "n_trials": p.n_trials,
                "sampler": p.sampler.type,
                "inherits": p.inherits,
                "search_space": sorted(p.search_space),  # keys only, not ranges
                # The count only: the repeat key and values are config-authored.
                "repeats": None if p.repeats is None else len(p.repeats.values),
            }
            for p in exp.phases
        ]
        return {
            "experiment_id": reg.id,
            "metric": reg.metric_payload,
            "capabilities": reg.capabilities,
            "phases": phases,
        }

    def latest_run(self, experiment_id: str) -> dict[str, Any]:
        """Return the newest durable MCP run for one catalog experiment.

        :param str experiment_id: Catalog id whose latest run should be resolved.
        :return dict[str, Any]: Path-free result with one computed run or ``found=false``.
        """
        reg = self._registry.get(experiment_id)
        handle = self._runs.latest_run_for(reg.id)
        run = self._run_payload(handle) if handle is not None else None
        return {
            "experiment_id": reg.id,
            "found": run is not None,
            "run": run,
        }

    def _run_payload(
        self,
        handle: RunHandle,
        view: RunnerView | None = None,
        *,
        force_snapshot_unavailable: bool = False,
    ) -> JsonObject:
        """Build one agent-visible run state with explicit recovery metadata.

        :param RunHandle handle: Persisted run handle to derive.
        :param RunnerView | None view: The read's derivation of the runner, or
            ``None`` to derive it here; every field below uses the same one.
        :param bool force_snapshot_unavailable: The result response has already
            reported its terminal snapshot unavailable, so the failure says so too.
        :return JsonObject: Run id, state, launch timestamp, and recovery flag.
            ``runner_observable`` is False for a running run whose runner is in
            another PID namespace and cannot be proven gone.
        """
        if view is None:
            view = self._runs.runner_view(handle)
        observation = self._runs.observe(handle, view=view)
        state = observation.state
        return {
            "run_id": handle.run_id,
            "state": state,
            "started_at": handle.started_at,
            "recovery_required": observation.recovery_required,
            "runner_observable": not (state == "running" and view.visibility == "unobservable"),
            "failure": self._run_failure_payload(
                handle,
                observation=observation,
                force_snapshot_unavailable=force_snapshot_unavailable,
            ),
        }

    def _run_failure_payload(
        self,
        handle: RunHandle,
        *,
        view: RunnerView | None = None,
        observation: RunObservation | None = None,
        force_snapshot_unavailable: bool = False,
    ) -> dict[str, Any] | None:
        """Return a validated safe terminal failure, never the raw exception text.

        :param RunHandle handle: Run handle whose recorded terminal status
            should be inspected.
        :param RunnerView | None view: The read's derivation of the runner, used
            when a fresh observation must be derived here.
        :param RunObservation | None observation: State and terminal evidence already read
            together for this payload; otherwise read a fresh decision here.
        :param bool force_snapshot_unavailable: The result response has already
            refused a non-finalized or missing snapshot even if run state stays live.
        :return dict[str, Any] | None: The run's ``failure`` payload validated
            against :class:`FailurePayload` and dumped to JSON-safe types. A
            terminal run with no usable result snapshot receives a generated
            snapshot-unavailable failure; otherwise returns ``None`` when no
            valid failure was recorded.
        """
        if observation is None:
            observation = self._runs.observe(handle, view=view)
        terminal: Mapping[str, object] = observation.terminal_status or {}
        persisted: dict[str, Any] | None = None
        try:
            if terminal.get("failure") is not None:
                persisted = FailurePayload.model_validate(terminal["failure"]).model_dump(
                    mode="json", exclude_none=True
                )
        except ValidationError:
            pass
        if force_snapshot_unavailable or (
            observation.state != "running" and parse_result_snapshot(terminal) is None
        ):
            failure: dict[str, Any] = {
                "code": "result_snapshot_unavailable",
                "stage": "cleanup",
                "retryable": False,
                "actor": "operator",
                "remediation": (
                    "Report that this run's historical results are unavailable and ask "
                    "the operator to inspect the PhaseSweep run or server diagnostics; "
                    "do not substitute mutable experiment-level results."
                ),
            }
            if persisted is not None:
                failure["cause"] = persisted
            return failure
        # Preserve the terminal diagnosis and remediation after operator recovery;
        # the live recovery_required field says whether recovery is still needed.
        return persisted

    def status(self, run_id: str) -> dict[str, Any]:
        """Per-phase trial counts and winner presence plus the run process state.

        Reads are always scoped to one persisted ``run_id``. A live run reads
        its immutable configuration snapshot against current local storage;
        a terminal run reads its frozen result snapshot.

        :param str run_id: Detached run id returned by ``launch`` or ``latest_run``.
        :return dict: Path-free status payload for the agent.
        """
        target_id, status, run, handle, result_source = self._read_status_target(run_id=run_id)
        return self._status_result(target_id, status, run, handle, result_source)

    async def await_run(
        self,
        run_id: str,
        timeout_seconds: int = AWAIT_DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Wait until a launched run changes, then return its full status.

        A bounded sleep-and-recheck loop over the same reads ``status`` does:
        no new state, no side effects. Returns early when the run reaches a
        terminal state, operator recovery is required, or a phase gains a
        winner; otherwise returns the current status when the (clamped)
        timeout elapses. A blocking status read already in progress cannot be
        preempted safely and may finish after that deadline. Rechecks run every
        ``AWAIT_RECHECK_SECONDS`` (or as late as one estimated read duration
        before the deadline) and re-resolve the run from disk, so state written
        while this request is active is observed. After a server restart, a
        fresh call resumes from the same durable run.

        :param str run_id: Detached run id to wait on.
        :param int timeout_seconds: Requested wait, clamped to
            [``AWAIT_MIN_TIMEOUT_SECONDS``, ``AWAIT_MAX_TIMEOUT_SECONDS``].
        :return dict[str, Any]: ``status`` payload plus ``changed`` and
            ``reason`` (``recovery_required`` / ``terminal`` /
            ``phase_completed`` / ``timeout``).
        """
        timeout = min(
            AWAIT_MAX_TIMEOUT_SECONDS, max(AWAIT_MIN_TIMEOUT_SECONDS, int(timeout_seconds))
        )
        deadline = time.monotonic() + timeout
        baseline: tuple[Any, ...] | None = None
        read_seconds = 0.0
        while True:
            read_started = time.monotonic()
            target_id, status, run, handle, result_source = await asyncio.to_thread(
                self._read_status_target,
                run_id=run_id,
            )
            read_seconds = time.monotonic() - read_started
            snapshot = _await_snapshot(
                run["state"],
                run["recovery_required"],
                dict(status),
            )
            if baseline is None:
                baseline = snapshot
            if run["recovery_required"]:
                reason = "recovery_required"
            elif run["state"] in ("succeeded", "failed", "cancelled"):
                reason = "terminal"
            elif _phase_gained_winner(baseline, snapshot):
                reason = "phase_completed"
            elif time.monotonic() >= deadline:
                reason = "timeout"
            elif time.monotonic() + read_seconds >= deadline:
                # Another read as slow as the one just finished would end past
                # the deadline. Do not start it, but wait out the caller's
                # remaining budget before returning the fresh snapshot in hand.
                # The status read itself is blocking filesystem/storage work in
                # a worker thread and cannot be preempted safely; if that read
                # already crossed the deadline, the branch above returns as
                # soon as it finishes.
                await asyncio.sleep(max(0.0, deadline - time.monotonic()))
                reason = "timeout"
            else:
                remaining_before_next_read = max(
                    0.0,
                    deadline - time.monotonic() - read_seconds,
                )
                await asyncio.sleep(min(AWAIT_RECHECK_SECONDS, remaining_before_next_read))
                continue
            break
        result = self._status_result(target_id, status, run, handle, result_source)
        result["changed"] = snapshot != baseline
        result["reason"] = reason
        return result

    def _read_status_target(
        self,
        *,
        run_id: str,
    ) -> tuple[
        str,
        EngineStatus,
        dict[str, Any],
        RunHandle,
        ResultSource,
    ]:
        """Resolve one run-specific status read to its current or frozen payload.

        :param str run_id: Persisted run id for the read.
        :return tuple: Target id, status data, run state, handle and result provenance.
        """
        target_id, experiment, status, _winners, handle, view, result_source = (
            self._read_result_target(run_id)
        )
        # Keep the captured results while pairing them with the latest cleanup
        # and recovery state: snapshot finalization, recovery, or the runner's
        # exit can change the run's files after target resolution, so the run
        # is derived again from its files as they are now.
        run = self._run_payload(
            handle,
            view,
            force_snapshot_unavailable=result_source == "terminal_snapshot_unavailable",
        )
        delivery = self._tracking_delivery(handle, experiment)
        if delivery is not None:
            status["tracking_delivery"] = delivery
        return target_id, status, run, handle, result_source

    def _read_result_target(
        self, run_id: str
    ) -> tuple[
        str,
        Experiment | None,
        EngineStatus,
        list[PhaseWinnerView],
        RunHandle,
        RunnerView,
        ResultSource,
    ]:
        """Read one run's status and winners from the same live or frozen result.

        :param str run_id: Persisted run whose immutable configuration selects the read.
        :raises UnknownRunError: The run id has no persisted handle.
        :return tuple: Target, configuration, status, winners, handle, runner view and source.
        """
        handle = self._runs.get(run_id)
        if handle is None:
            raise UnknownRunError(run_id)
        target_id = handle.experiment_id
        view = self._runs.runner_view(handle)
        try:
            snapshot = self._terminal_result_snapshot(handle, view)
        except RunResultSnapshotUnavailableError:
            snapshot = None
        # Complete result snapshots are self-contained; only live reads and
        # unavailable-result placeholders need the sibling configuration.
        experiment = None if snapshot is not None else self._load_run_experiment(handle)
        snapshot, result_source = self._result_snapshot_view(experiment, handle, snapshot, view)
        if snapshot is None:
            assert experiment is not None
            status, winners = self._live_result(target_id, experiment, handle)
            # A live read may span the runner's final snapshot write. Once
            # durable, the frozen status and winners replace that live view together.
            snapshot, result_source = self._result_snapshot_view(experiment, handle, None, view)
        if snapshot is not None:
            status = self._snapshot_status_payload(target_id, snapshot, result_source=result_source)
            winners = (
                []
                if status["publication_integrity"] in {"failed", "permission_denied", "unknown"}
                else snapshot.winner_views()
            )
        elif status["publication_integrity"] in {"failed", "permission_denied", "unknown"}:
            winners = []
        return target_id, experiment, status, winners, handle, view, result_source

    def _tracking_delivery(
        self, handle: RunHandle, experiment: Experiment | None
    ) -> dict[str, int] | None:
        """Count this run's own MLflow result uploads from its launch config.

        Delivery records are mutable, so only status reads them; frozen result
        reads never depend on them. A terminal read loads the run's config
        snapshot just for this and reports no counts when it is unavailable.

        :param RunHandle handle: Run whose generation owns the uploads.
        :param Experiment | None experiment: Launch config, when already loaded.
        :return dict[str, int] | None: Pending and delivered counts, or ``None``
            when the run does not use MLflow or its config is unavailable.
        """
        if experiment is None:
            try:
                experiment = self._load_run_experiment(handle)
            except RunSnapshotUnavailableError:
                return None
        if experiment.mlflow is None:
            return None
        from phasesweep.tracking.publication import delivery_status

        return delivery_status(experiment, handle.run_id)

    def _catalog_comparison_experiment(self, experiment_id: str) -> Experiment | None:
        """Return the config a future catalog launch would execute, if available.

        :param str experiment_id: Catalog id associated with a current or frozen read.
        :return Experiment | None: Current catalog config, or ``None`` after removal.
        """
        try:
            return self._registry.get(experiment_id).experiment
        except UnknownExperimentError:
            return None

    def _live_result(
        self,
        experiment_id: str,
        experiment: Experiment,
        handle: RunHandle,
    ) -> tuple[EngineStatus, list[PhaseWinnerView]]:
        """Read live status and winners, leaving catalog drift unknown after decataloging.

        Both come from one pointer resolution, so the winners always belong to
        the publication the status describes. ``read_result`` normally
        compares against its read config when no comparison config is
        supplied. A run snapshot remains the correct artifact locator after its
        catalog entry is removed, but it is not a current catalog config and
        therefore cannot support a drift verdict.

        :param str experiment_id: Catalog id associated with the read target.
        :param Experiment experiment: Live run snapshot config.
        :param RunHandle handle: Live run pinning the represented generation.
        :return tuple[EngineStatus, list[PhaseWinnerView]]: Path-free live
            status payload and the represented generation's winners.
        """
        comparison = self._catalog_comparison_experiment(experiment_id)
        try:
            status, winners = read_result(
                experiment,
                generation_id=handle.run_id,
                comparison_experiment=comparison,
            )
        except (
            ArtifactRootConflictError,
            StudySchemaMismatchError,
            StudyStorageUnavailableError,
        ) as exc:
            self._raise_persistent_read_error(handle, exc)
        if comparison is None:
            status["published_config_matches_current"] = None
        return status, winners

    @staticmethod
    def _raise_persistent_read_error(
        handle: RunHandle,
        exc: ArtifactRootConflictError | StudySchemaMismatchError | StudyStorageUnavailableError,
    ) -> NoReturn:
        """Replace a path-bearing live-study read failure with a safe tool error.

        :param RunHandle handle: Run whose immutable config selected the failed state.
        :param ArtifactRootConflictError | StudySchemaMismatchError | StudyStorageUnavailableError exc:
            Explicit persistent-state failure to redact.
        :raises RunPersistentStateUnavailableError: Always, without storage details.
        """
        log.info("persistent state read failed for run=%s: %s", handle.run_id, exc)
        raise RunPersistentStateUnavailableError(handle.run_id) from None

    def _snapshot_status_payload(
        self,
        experiment_id: str,
        snapshot: RunResultSnapshot,
        *,
        result_source: ResultSource,
    ) -> EngineStatus:
        """Return frozen run facts with a config-only live drift comparison.

        A terminal run snapshot is the historical read authority for that run.
        Publication pointers and integrity stay as captured: consulting the
        live artifact tree here would make an intact snapshot unreadable, and
        a later generation's damage could erase this run's
        already-validated winners. Config drift is the sole live field and
        compares the frozen represented-config fingerprint with the current
        catalog.

        :param str experiment_id: Catalog id associated with the run.
        :param RunResultSnapshot snapshot: Validated frozen result snapshot.
        :param ResultSource result_source: Snapshot provenance, including the
            unavailable placeholder case.
        :return EngineStatus: Frozen status with a current config-drift verdict.
        """
        status = snapshot.status_payload()
        comparison = self._catalog_comparison_experiment(experiment_id)
        represented_fingerprint = snapshot.represented_config_fingerprint
        status["published_config_matches_current"] = (
            represented_fingerprint == _experiment_semantic_fingerprint(comparison)
            if comparison is not None
            and represented_fingerprint is not None
            and result_source != "terminal_snapshot_unavailable"
            else None
        )
        return status

    def _status_result(
        self,
        target_id: str,
        status: EngineStatus,
        run: dict[str, Any],
        handle: RunHandle,
        result_source: ResultSource,
    ) -> dict[str, Any]:
        """Build the common path-free response for status and await_run.

        :param str target_id: Experiment id represented by ``status``.
        :param EngineStatus status: Current or frozen engine status payload.
        :param dict[str, Any] run: Derived detached-run state.
        :param RunHandle handle: Already-resolved detached-run handle.
        :param ResultSource result_source: Current shared state or frozen run snapshot.
        :return dict[str, Any]: Agent-safe status response.
        """
        elapsed_seconds = _run_elapsed_seconds(self._runs, handle, run["state"])
        return status_payload(
            target_id,
            dict(status),
            run,
            result_source=result_source,
            elapsed_seconds=elapsed_seconds,
        )

    def _load_run_experiment(self, handle: RunHandle) -> Experiment:
        """Load and verify the immutable config snapshot for a persisted run.

        Run-specific monitoring must follow the exact config the detached
        runner received, even if the cataloged config has since been edited and
        the MCP server restarted. The saved handle's hash is the guardrail: a
        missing, corrupted, or non-experiment snapshot is rejected instead of
        silently reporting status for the wrong storage or workdir.

        :param RunHandle handle: Persisted run identity whose snapshot should be loaded.
        :return Experiment: Parsed experiment from the launched per-run snapshot.
        :raises RunSnapshotUnavailableError: If the snapshot is missing,
            unreadable, not an experiment, or no longer matches the hash
            recorded in the handle.
        """
        snapshot_path = self._runs.config_snapshot_path(handle.run_id)
        try:
            return load_experiment_snapshot(
                snapshot_path,
                handle.config_sha256,
                source=f"run snapshot {handle.run_id}",
                config_dir=Path(handle.config_dir),
            )
        except (OSError, ValueError) as exc:
            log.info("invalid config snapshot for run=%s: %s", handle.run_id, exc)
            raise RunSnapshotUnavailableError(handle.run_id) from None

    def winners(self, run_id: str) -> dict[str, Any]:
        """Return the winning hyperparameters per completed phase.

        Reads are scoped to the immutable config and terminal result snapshots
        recorded for ``run_id``. Live reads use only that run's saved config;
        neither path reconstructs results from a catalog experiment.

        The payload carries ``publication_integrity`` so an empty winner list
        is never ambiguous: ``"absent"`` means nothing has published yet,
        ``"failed"`` means a recorded publication no longer validates;
        ``"permission_denied"`` means this user cannot validate it without
        implying corruption. Both expose no winners and require operator action.

        Every label describing the winners -- metric name and goal, objective
        evidence, and the declared phase plan the completeness fields are
        measured against -- comes from the represented generation's own
        recorded semantics, never from the current catalog config. Reading
        them from the current config would relabel historical numbers with a
        metric they were never optimized for, assert evidence guarantees the
        run never had, and, after a phase rename, hide the published winner
        while reporting the new name as missing. ``result_context`` and
        ``published_config_matches_current`` report which config supplied the
        labels and whether it still matches the one a further run would
        execute. Visibility is unchanged: the current policy still decides
        which historical *values* are redacted.

        :param str run_id: Detached run id whose snapshot should be read.
        :return dict[str, Any]: Path-free winners payload for the agent.
        """
        target_id, experiment, status, winner_views, handle, view, result_source = (
            self._read_result_target(run_id)
        )
        if result_source == "current_shared_study" and (
            self._runs.recovery_required(handle, view=view)
            or not self._runs.runner_may_be_alive(handle, view=view)
        ):
            assert experiment is not None
            # No frozen result is available after the runner stopped or
            # cleanup became uncertain. Discard the earlier mutable winner view.
            unavailable = capture_pre_generation_result_snapshot(experiment)
            status = self._snapshot_status_payload(
                target_id, unavailable, result_source="terminal_snapshot_unavailable"
            )
            winner_views = []
            result_source = "terminal_snapshot_unavailable"
        represented_generation_id: str | None = status["represented_generation_id"]
        publication_integrity: McpPublicationState = status["publication_integrity"]
        visible_params = self._effective_visible_params(target_id, handle)
        result = winners_payload(
            target_id,
            winner_views,
            metric=status["metric"],
            declared_phases=status["result_phase_plan"],
            result_source=result_source,
            publication_integrity=publication_integrity,
            run_id=run_id,
            represented_generation_id=represented_generation_id,
            visible_params=visible_params,
            result_context=status["result_context"],
            published_config_matches_current=status["published_config_matches_current"],
        )
        result["failure"] = self._run_failure_payload(
            handle,
            view=view,
            force_snapshot_unavailable=result_source == "terminal_snapshot_unavailable",
        )
        return result

    def _effective_visible_params(
        self,
        experiment_id: str,
        handle: RunHandle,
    ) -> VisibleParamsPolicy:
        """Resolve current or launch/current-intersected winner visibility.

        :param str experiment_id: Catalog id associated with the represented results.
        :param RunHandle handle: MCP run whose generation is represented.
        :return VisibleParamsPolicy: Effective sampled-parameter visibility.
        """
        try:
            current_policy = self._registry.get(experiment_id).visible_params
        except UnknownExperimentError:
            return "none"
        launch_policy = handle.visible_params_at_launch or "none"
        return intersect_visible_params(launch_policy, current_policy)

    def _result_snapshot_view(
        self,
        experiment: Experiment | None,
        handle: RunHandle,
        snapshot: RunResultSnapshot | None,
        view: RunnerView,
    ) -> tuple[RunResultSnapshot | None, ResultSource]:
        """Resolve one run result without falling back to mutable terminal state.

        A failed terminal snapshot is represented by the same config-only
        unavailable-data shape used for pre-generation failures. This lets
        monitoring return the terminal run and an actionable failure while
        every phase count remains explicitly untrusted.

        :param Experiment | None experiment: Exact catalog or saved run configuration when a
            current-state read or unavailable-result placeholder needs one. A complete terminal
            snapshot is self-contained and does not require it.
        :param RunHandle handle: Detached run being read.
        :param RunResultSnapshot | None snapshot: Terminal snapshot already captured during
            target resolution, reused without rereading mutable finalization state.
        :param RunnerView view: The read's derivation of the runner.
        :return tuple: Optional result view and its agent-visible provenance.
        """
        if snapshot is not None:
            return snapshot, "frozen_run_snapshot"
        try:
            snapshot = self._terminal_result_snapshot(handle, view)
        except RunResultSnapshotUnavailableError:
            assert experiment is not None
            placeholder = capture_pre_generation_result_snapshot(experiment)
            return (
                placeholder,
                "terminal_snapshot_unavailable",
            )
        return (
            snapshot,
            "frozen_run_snapshot" if snapshot is not None else "current_shared_study",
        )

    def _terminal_result_snapshot(
        self, handle: RunHandle, view: RunnerView
    ) -> RunResultSnapshot | None:
        """Return a live run's current view or require its frozen terminal snapshot.

        :param RunHandle handle: Resolved run whose terminal status should be inspected.
        :param RunnerView view: The read's derivation of the runner.
        :return RunResultSnapshot | None: Validated snapshot, or ``None`` while
            the run remains non-terminal.
        :raises RunResultSnapshotUnavailableError: A terminal run has no
            terminal status, or its immutable result snapshot is absent or invalid.
        """
        terminal_status = self._runs.recorded_terminal_status(handle)
        if terminal_status is None:
            # A reboot proves a spawned runner and every descendant are gone,
            # so it can release capacity even when a hard exit prevented the
            # runner from writing status.json. It cannot, however, make the
            # mutable shared study a historical result for that run. Re-read
            # after the liveness probe: the runner may have completed its status
            # write between the first read and the probe.
            if self._runs.runner_may_be_alive(handle, view=view):
                return None
            terminal_status = self._runs.recorded_terminal_status(handle)
            if terminal_status is None:
                raise RunResultSnapshotUnavailableError(handle.run_id)
        if terminal_status.get("result_snapshot_state") == "pending":
            if self._runs.runner_may_be_alive(handle, view=view):
                return None
            # The runner can publish the complete snapshot and exit between
            # the status read and its liveness probe. Re-read before treating
            # the pending state as orphaned.
            latest_status = self._runs.recorded_terminal_status(handle)
            if latest_status is None or latest_status.get("result_snapshot_state") == "pending":
                raise RunResultSnapshotUnavailableError(handle.run_id, "pending")
            terminal_status = latest_status
        snapshot = parse_result_snapshot(terminal_status)
        if snapshot is not None:
            return snapshot
        if self._runs.state(handle, view=view) == "running":
            return None
        # The runner may have completed the second atomic status write between
        # the first read and the state check. Re-read before declaring a
        # terminal snapshot unavailable.
        latest_status = self._runs.recorded_terminal_status(handle)
        if latest_status is not None:
            terminal_status = latest_status
            snapshot = (
                None
                if terminal_status.get("result_snapshot_state") == "pending"
                else parse_result_snapshot(terminal_status)
            )
            if snapshot is not None:
                return snapshot
        finalization_state = terminal_status.get("result_snapshot_state")
        raise RunResultSnapshotUnavailableError(
            handle.run_id,
            finalization_state if isinstance(finalization_state, str) else None,
        )

    def launch(self, experiment_id: str, from_phase: str | None = None) -> JsonObject:
        """Start the sweep as a detached background run; return its run_id.

        Refuses if launch is not permitted, if a ``from_phase`` resume is not
        ready (an earlier phase has no winner), if this experiment already has a
        live run, or if the server is at its max_concurrent_runs cap.

        :param str experiment_id: Catalog experiment id to launch.
        :param str | None from_phase: Optional phase to resume from after earlier winners exist.
        :return JsonObject: Launch result containing run id, experiment id, and running state.
        :raises UnknownExperimentError: If ``experiment_id`` is not cataloged.
        :raises PermissionDeniedError: If the entry does not allow launching, or
            does not allow ``from_phase`` resumes.
        :raises InvalidPhaseError: If ``from_phase`` names no phase of this experiment.
        :raises ResumeNotReadyError: If an earlier phase has no compatible winner.
        :raises ResumePublicationUnusableError: If the recorded publication no longer
            validates or this server cannot read it.
        :raises LaunchInProgressError: If another launch currently holds the launch lock.
        :raises ExperimentBusyError: If this experiment already has a live run.
        :raises ConcurrencyLimitError: If the server is at ``max_concurrent_runs``.
        :raises ConfigChangedError: If the cataloged config changed since startup.
        :raises RuntimeError: If no unused run id could be minted, or the spawned
            runner has no Linux ``/proc`` start time to make cancellation PID-reuse safe.
        """
        args: JsonObject = {"experiment_id": experiment_id, "from_phase": from_phase}
        resolved: JsonObject = {}
        state_before: JsonObject | None = None
        try:
            reg = self._registry.get(experiment_id)
            resolved["experiment_id"] = reg.id
            if not reg.allow_launch:
                raise PermissionDeniedError("launch", experiment_id)
            if from_phase is not None:
                if not reg.allow_from_phase:
                    raise PermissionDeniedError("from_phase", experiment_id)
                if from_phase not in reg.phase_names:
                    raise InvalidPhaseError(experiment_id, from_phase)
                self._require_resume_ready(reg, from_phase)
            # The cap check and the spawn must be atomic, or two near-simultaneous
            # launches both pass the cap and oversubscribe the GPU it protects. Hold
            # the launch lock across the whole decision. One scan then covers both
            # guards: the same experiment can't double-launch, and no more than
            # max_concurrent_runs sweeps run at once (default 1).
            with self._runs.launch_lock() as acquired:
                if not acquired:
                    raise LaunchInProgressError()
                handles, unreadable = self._runs.launch_inventory()
                abandoned = {
                    handle.run_id
                    for handle in handles
                    if self._runs.launch_lease_path(handle.run_id).is_file()
                    and self._runs.is_pre_spawn_orphan(handle.run_id)
                }
                abandoned.update(
                    identity.removeprefix("run:")
                    for identity in unreadable
                    if identity.startswith("run:")
                    and self._runs.launch_lease_path(identity.removeprefix("run:")).is_file()
                    and self._runs.is_pre_spawn_orphan(identity.removeprefix("run:"))
                )
                for abandoned_run_id in sorted(abandoned):
                    recovered_log = self._runs.clear_pre_spawn_orphan(abandoned_run_id)
                    if recovered_log is not None:
                        log.info(
                            "preserved abandoned launch log run=%s at %s",
                            abandoned_run_id,
                            recovered_log,
                        )
                if abandoned:
                    handles, unreadable = self._runs.launch_inventory()
                if unreadable:
                    recoverable = sorted(
                        identity.removeprefix("run:")
                        for identity in unreadable
                        if identity.startswith("run:")
                        and self._runs.is_pre_spawn_orphan(identity.removeprefix("run:"))
                    )
                    raise RunCapacityUnknownError(len(unreadable), recoverable)
                live = [handle for handle in handles if self._runs.state(handle) == "running"]
                state_before = {"live_runs": len(live)}
                busy = next((h for h in live if h.experiment_id == experiment_id), None)
                if busy is not None:
                    raise ExperimentBusyError(experiment_id, busy.run_id)
                if len(live) >= self._registry.max_concurrent_runs:
                    blocking_run_ids = [
                        handle.run_id
                        for handle in sorted(live, key=lambda item: (item.started_at, item.run_id))
                    ]
                    raise ConcurrencyLimitError(
                        len(live),
                        self._registry.max_concurrent_runs,
                        blocking_run_ids,
                    )
                config_bytes = self._current_config_bytes(reg)
                preparation: PreparedRun | None = None
                for _ in range(10):
                    run_id = self._runs.new_run_id(reg.id)
                    pending = self._pending_handle(reg, run_id)
                    try:
                        preparation = self._runs.prepare_launch(pending, config_bytes)
                    except FileExistsError:
                        continue
                    break
                else:
                    raise RuntimeError("failed to mint an unused MCP run id")
                assert preparation is not None
                resolved["run_id"] = run_id
                handle: RunHandle | None = None
                try:
                    handle = self._spawn(reg, from_phase, preparation)
                    self._runs.update(handle)
                except _SpawnBookkeepingError as spawn_exc:
                    self._record_launch_failure(
                        pending,
                        cleanup_confirmed=spawn_exc.cleanup_confirmed,
                        error_class=type(spawn_exc.original_error).__name__,
                    )
                    raise spawn_exc.original_error from None
                except BaseException as launch_exc:
                    cleanup_confirmed = (
                        True
                        if handle is None
                        else self._terminate_failed_spawn(handle, launch_exc, acknowledged=True)
                    )
                    self._record_launch_failure(
                        pending,
                        cleanup_confirmed=cleanup_confirmed,
                        error_class=type(launch_exc).__name__,
                    )
                    raise
                finally:
                    self._runs.finish_launch_preparation(preparation)
                assert handle is not None
            result: JsonObject = {
                "run_id": handle.run_id,
                "experiment_id": experiment_id,
                "state": "running",
            }
        except Exception as exc:
            self._audit_error(
                TOOL_LAUNCH_RUN,
                args,
                exc,
                resolved=resolved,
                state_before=state_before,
            )
            raise
        self._audit_success(
            TOOL_LAUNCH_RUN,
            args,
            resolved={"experiment_id": experiment_id, "run_id": handle.run_id},
            state_before=state_before,
            state_after={
                "run_state": "running",
                "live_runs": len(live) + 1,
            },
            result_counts={"runs": 1},
        )
        return result

    def cancel(self, run_id: str) -> JsonObject:
        """Stop a running sweep: SIGTERM -> grace -> SIGKILL the runner's group.

        The terminal state is reported as ``cancelled`` only when the runner
        records its cancellation status. If the runner group is gone but no
        status was written, cleanup remains uncertain because trial process
        groups may still be alive.

        :param str run_id: Detached run id to cancel.
        :return JsonObject: Cancellation result containing final state and optional cleanup confirmation.
        :raises UnknownRunError: If ``run_id`` names no persisted run.
        :raises PermissionDeniedError: If the run's entry did not allow cancel at launch time.
        :raises RunLaunchUnsettledError: If the launch has not yet persisted a
            process identity, so there is nothing safe to signal.
        :raises RunNamespaceMismatchError: If the runner was recorded in another
            PID namespace, where its PID and process group cannot be signalled.
        """
        args: JsonObject = {"run_id": run_id}
        resolved: JsonObject = {}
        state_before: JsonObject | None = None
        result: JsonObject
        try:
            handle = self._runs.get(run_id)
            if handle is None:
                raise UnknownRunError(run_id)
            resolved = {"experiment_id": handle.experiment_id, "run_id": run_id}
            if not self._cancel_allowed(handle):
                raise PermissionDeniedError("cancel", handle.experiment_id)
            view = self._runs.runner_view(handle)
            observation = self._runs.observe(handle, view=view)
            before = observation.state
            recovery_required = observation.recovery_required
            state_before = {
                "run_state": before,
                "recovery_required": recovery_required,
            }
            if before == "running":
                with self._runs.transition_lock(handle):
                    # Launch may have durably replaced a pending handle with
                    # its runner identity, and recovery may have finalized the
                    # run after the first read. Use the latest handle, and one
                    # derivation of its runner, for both the state decision and
                    # any subsequent signal.
                    refreshed = self._runs.get(run_id)
                    if refreshed is None:
                        raise UnknownRunError(run_id)
                    handle = refreshed
                    view = self._runs.runner_view(handle)
                    observation = self._runs.observe(handle, view=view)
                    before = observation.state
                    recovery_required = observation.recovery_required
                    state_before = {
                        "run_state": before,
                        "recovery_required": recovery_required,
                    }
                    if before == "running" and handle.launch_state == "launching":
                        # No PID/PGID is durable yet. Signalling an empty identity
                        # could let launch continue after a false cancel response.
                        raise RunLaunchUnsettledError(run_id)
                    if before == "running" and view.visibility == "unobservable":
                        # The recorded PID and process group are numbered in
                        # another PID namespace, where they name unrelated
                        # processes. Refuse before reserving or signalling.
                        raise RunNamespaceMismatchError(run_id)
                    if before == "running":
                        # Reserve capacity before signalling; recovery cannot
                        # clear this marker between the reread and its write.
                        self._runs.mark_cleanup_uncertain(handle)
            after: RunState
            if before != "running":
                after = before
                confirmed: bool | None = None
            else:
                # Concurrent callers still signal outside the short transition
                # lock. kill_stale_group treats an already-gone group as confirmed,
                # terminal status is runner-authoritative, and marker removal
                # uses missing_ok.
                # SIGTERM -> grace -> SIGKILL on the runner's process group. A
                # runner-written status is useful only when it includes explicit
                # cleanup evidence from the engine shutdown handler. If the server
                # had to force-kill the runner first, or the handler reported
                # uncertainty, child trial PGIDs may still live, so keep the run
                # counted as live and fail closed.
                identity = self._runs.cleanup_identity(handle)
                gone = view.visibility == "gone"
                same_boot = identity.boot_id is not None and identity.boot_id == read_boot_id()
                # A prior boot proves the runner and its trials gone without a
                # signal; a runner proven absent from another PID namespace
                # proves only its own group gone. An unknown boot cannot make
                # saved PID/starttime safe to signal after reboot, and another
                # PID namespace cannot be signalled from here at all.
                runner_group_gone = (
                    gone
                    or view.visibility == "absent"
                    or (
                        same_boot
                        and view.visibility == "observable"
                        and kill_stale_group(
                            identity.pid,
                            identity.pid_starttime,
                            pgid=identity.pgid,
                            grace_seconds=30.0,
                        )
                    )
                )
                with self._runs.transition_lock(handle):
                    terminal_status = self._runs.recorded_terminal_status(handle)
                    cleanup_recovered = self._runs._cleanup_recovered(handle)
                    confirmed = runner_group_gone and (
                        gone
                        or (
                            terminal_status is not None
                            and terminal_status.get("cleanup_confirmed") is True
                        )
                        or cleanup_recovered
                    )
                    if confirmed and not cleanup_recovered:
                        # Recovery owns any remaining marker after recording
                        # cleanup; it may still be repairing the frozen result.
                        self._runs.clear_cleanup_uncertain(handle)
                    observation = self._runs.observe(handle, view=view)
                    after = observation.state
                    recovery_required = observation.recovery_required
            result = {
                "run_id": run_id,
                "state": after,
                "cleanup_confirmed": confirmed,
                "recovery_required": recovery_required,
            }
        except Exception as exc:
            self._audit_error(
                TOOL_CANCEL_RUN,
                args,
                exc,
                resolved=resolved,
                state_before=state_before,
            )
            raise
        state_after: JsonObject = {
            "run_state": after,
            "recovery_required": recovery_required,
        }
        if confirmed is not None:
            state_after["cleanup_confirmed"] = confirmed
        self._audit_success(
            TOOL_CANCEL_RUN,
            args,
            resolved=resolved,
            state_before=state_before,
            state_after=state_after,
            result_counts={"runs": 1},
        )
        return result

    def _require_resume_ready(self, reg: RegisteredExperiment, from_phase: str) -> None:
        """Verify that every earlier phase has a winner the resume may build on.

        Applies the run preflight's own acceptance rule to each skipped phase,
        against one resolution of the publication, so a launch is refused
        before spawning for anything the runner's preflight would refuse after.

        :param RegisteredExperiment reg: Registered experiment being resumed.
        :param str from_phase: Requested phase to resume from.
        :raises ResumeNotReadyError: If an earlier phase has no published
            winner, or its winner or source evidence is unusable or
            incompatible with the current config.
        :raises ResumePublicationUnusableError: If the recorded publication no
            longer validates or this server cannot read it.
        """
        if reg.phase_names.index(from_phase) == 0:
            return

        def refusal(phase_name: str, exc: Exception) -> Exception:
            """Translate a resume refusal without exposing its internal diagnostic.

            :param str phase_name: Earlier phase whose winner was being checked.
            :param Exception exc: Engine refusal from publication or winner validation.
            :return Exception: The MCP refusal, or an unexpected exception unchanged.
            """
            if isinstance(exc, (PublicationIntegrityError, PublicationAccessError)):
                log.info("resume refused for experiment=%s: %s", reg.id, exc)
                state: Literal["failed", "permission_denied"] = (
                    "failed" if isinstance(exc, PublicationIntegrityError) else "permission_denied"
                )
                return ResumePublicationUnusableError(reg.id, from_phase, state)
            if isinstance(exc, FileNotFoundError):
                return ResumeNotReadyError(reg.id, from_phase, phase_name)
            if isinstance(
                exc, (RuntimeError, KeyError, TypeError, ValueError, AttributeError, OSError)
            ):
                log.info(
                    "resume preflight rejected winner for experiment=%s phase=%s: %s",
                    reg.id,
                    phase_name,
                    exc,
                )
                return ResumeNotReadyError(
                    reg.id,
                    from_phase,
                    phase_name,
                    reason="has no compatible winner for the current config",
                )
            return exc

        _preflight_skipped_winners(
            reg.experiment, from_phase=from_phase, run_deadline=None, error_mapper=refusal
        )

    @staticmethod
    def _current_config_bytes(reg: RegisteredExperiment) -> bytes:
        """Read a cataloged config only when it still matches startup validation.

        :param RegisteredExperiment reg: Frozen catalog entry to verify.
        :return bytes: Current config bytes when their SHA-256 matches startup.
        :raises ConfigChangedError: If the file is unreadable or changed.
        """
        try:
            data = reg.config_path.read_bytes()
        except OSError as exc:
            log.info("cannot read cataloged config for experiment=%s: %s", reg.id, exc)
            raise ConfigChangedError(reg.id) from None
        if hashlib.sha256(data).hexdigest() != reg.config_sha256:
            raise ConfigChangedError(reg.id)
        return data

    def _pending_handle(self, reg: RegisteredExperiment, run_id: str) -> RunHandle:
        """Build the pre-spawn handle persisted before ``Popen``.

        :param RegisteredExperiment reg: Catalog entry being launched.
        :param str run_id: Server-minted run id for the pending launch.
        :return RunHandle: Launching-state handle without process identity.
        """
        return RunHandle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
            config_dir=str(reg.config_path.parent),
            pid=None,
            pgid=None,
            pid_starttime=None,
            started_at=utc_now_iso(),
            launch_state="launching",
            allow_cancel=reg.allow_cancel,
            visible_params_at_launch=(
                list(reg.visible_params)
                if isinstance(reg.visible_params, list)
                else reg.visible_params
            ),
        )

    def _record_launch_failure(
        self,
        pending: RunHandle,
        *,
        cleanup_confirmed: bool,
        error_class: str,
    ) -> None:
        """Finalize a known failed launch without masking its original error.

        :param RunHandle pending: Durable pre-spawn handle to finalize.
        :param bool cleanup_confirmed: Whether any spawned runner group is confirmed gone.
        :param str error_class: Operator-facing class of the original launch failure.
        """
        try:
            # The runner can finish before the server detects its own bookkeeping
            # failure. Its terminal result is authoritative once cleanup is done.
            if self._runs.recorded_terminal_status(pending) is not None:
                return
            write_status_file_if_absent(
                self._runs.status_path(pending.run_id),
                {
                    "run_id": pending.run_id,
                    "returncode": 1,
                    "error_class": error_class,
                    "cleanup_confirmed": cleanup_confirmed,
                    "ended_at": utc_now_iso(),
                    "result_snapshot_state": "failed",
                    "result_snapshot_error": "LaunchDidNotProduceSnapshot",
                    "failure": {
                        "code": "internal_error",
                        "stage": "preflight",
                        "retryable": False,
                        "actor": "operator",
                        "remediation": (
                            "Ask the operator to inspect the PhaseSweep server diagnostics "
                            "before retrying."
                        ),
                    },
                },
            )
        except Exception:
            # The unresolved launching handle remains a fail-closed concurrency
            # reservation when even terminal bookkeeping cannot be persisted.
            log.exception("failed to finalize launch error for run_id=%s", pending.run_id)

    def _terminate_failed_spawn(
        self,
        handle: RunHandle,
        original_error: BaseException,
        *,
        acknowledged: bool = False,
    ) -> bool:
        """Terminate a spawned runner whose durable bookkeeping failed.

        :param RunHandle handle: Spawned runner identity available in memory.
        :param BaseException original_error: Launch failure preserved for diagnostics.
        :param bool acknowledged: Whether the runner may already have launched trials.
        :return bool: Whether runner and possible trial groups are confirmed gone.
        """
        marker_written = False
        try:
            self._runs.mark_cleanup_uncertain(handle)
            marker_written = True
        except BaseException as marker_exc:
            log.error(
                "cleanup uncertain after failed runner launch bookkeeping for "
                "run_id=%s pgid=%s, but failed to persist cleanup uncertainty marker; "
                "original error: %r",
                handle.run_id,
                handle.pgid,
                original_error,
                exc_info=(type(marker_exc), marker_exc, marker_exc.__traceback__),
            )
        try:
            assert handle.pgid is not None
            cleanup_confirmed = kill_stale_group(
                handle.pid,
                handle.pid_starttime,
                pgid=handle.pgid,
            )
        except BaseException:
            log.exception(
                "failed to terminate untracked runner run_id=%s pgid=%s",
                handle.run_id,
                handle.pgid,
            )
            return False
        if cleanup_confirmed and acknowledged:
            # After acknowledgement the runner can start trials in separate
            # process groups. Its own terminal cleanup report is the only
            # evidence that those groups were also stopped.
            terminal = self._runs.recorded_terminal_status(handle)
            cleanup_confirmed = terminal is not None and terminal.get("cleanup_confirmed") is True
        if cleanup_confirmed:
            if marker_written:
                try:
                    self._runs.clear_cleanup_uncertain(handle)
                except BaseException:
                    log.exception(
                        "runner cleanup succeeded but its uncertainty marker could not be "
                        "cleared for run_id=%s; retaining the recovery reservation",
                        handle.run_id,
                    )
                    return False
        else:
            log.error(
                "cleanup uncertain after failed runner launch bookkeeping for run_id=%s pgid=%s",
                handle.run_id,
                handle.pgid,
            )
        return cleanup_confirmed

    def _spawn(
        self,
        reg: RegisteredExperiment,
        from_phase: str | None,
        preparation: PreparedRun,
    ) -> RunHandle:
        """Spawn a blocked runner and acknowledge its durable process receipt.

        :param RegisteredExperiment reg: Registered experiment to run.
        :param str | None from_phase: Optional phase to resume from.
        :param PreparedRun preparation: Durable preparation and inherited launch lease.
        :raises OSError: The log or detached runner cannot be opened before spawn.
        :raises RuntimeError: If this process's PID namespace is unreadable, so
            the child's PID could not be tied to the namespace that numbers it.
        :raises _SpawnBookkeepingError: Post-spawn identity bookkeeping fails; the
            exception records whether cleanup of the spawned process group was confirmed.
        :return RunHandle: Runner-validated spawned handle.
        """
        pending = preparation.handle
        run_id = pending.run_id
        # The child inherits this namespace, so it is read before any pipe or
        # child exists: an unreadable namespace refuses the launch with nothing
        # to clean up.
        pid_namespace = read_pid_namespace()
        if pid_namespace is None:
            raise RuntimeError(
                "cannot read this process's PID namespace; refused launch because later "
                "cancellation could not tell whether the runner's PID is visible"
            )
        config_snapshot_path = preparation.config_snapshot_path
        log_path = self._runs.log_path(run_id)
        status_path = self._runs.status_path(run_id)
        ready_read, ready_write = os.pipe()
        ack_read, ack_write = os.pipe()
        cmd = [
            sys.executable,
            # -P: never prepend the cwd or script dir to sys.path. -s: no
            # per-user site dir. Both close the pre-identity window described
            # below; keep them together with the sanitized env.
            "-P",
            "-s",
            "-m",
            "phasesweep.mcp.runner",
            *_runner_protocol_argv(
                run_id=run_id,
                config_snapshot_path=config_snapshot_path,
                config_sha256=reg.config_sha256,
                status_path=status_path,
                state_dir=self._registry.state_dir,
                experiment_id=reg.id,
                started_at=pending.started_at,
                launch_ready_fd=ready_write,
                launch_ack_fd=ack_read,
                launch_lease_fd=preparation.lease_fd,
            ),
            # The runner chdirs here itself once its identity is durable; see
            # the trust-boundary note below for why Popen must not do it.
            "--cwd",
            str(reg.cwd),
        ]
        if reg.allow_cancel:
            cmd.append("--allow-cancel")
        if from_phase is not None:
            cmd += ["--from-phase", from_phase]
        # Pre-identity trust boundary. Everything between exec and the runner's
        # own durable handle write runs before this server can name the process
        # it just created. Spawning with the experiment's project directory as
        # cwd would put that window inside the project's reach: interpreter
        # startup would import a project-local `phasesweep/` shadow package or
        # `sitecustomize.py`, and a PYTHONPATH/PYTHONHOME/PYTHONSTARTUP-injected
        # module would run earlier still. Any of it could fork+setsid out of the
        # process group recorded below, after which "cleanup confirmed" would be
        # a claim about an empty group. So the child starts in the server-owned
        # state directory, with -P/-s and a sanitized environment, and receives
        # the project directory as an explicit argument it applies only after its
        # identity is on disk.
        spawn_cwd = self._neutral_spawn_cwd()
        # Open the log here, hand the fd to the child, then close our copy. The
        # child keeps it. stdin is /dev/null so the runner never blocks on input.
        # Once Popen returns, every later operation is inside the BaseException
        # boundary: shutdown interrupts are ownership failures too, not proof
        # that no child was created.
        proc: subprocess.Popen[bytes] | None = None
        handle: RunHandle | None = None
        pid_starttime: int | None = None
        boot_id: str | None = None
        runner_acknowledged = False
        try:
            with open_private_text(log_path, "w") as log_file:
                proc = subprocess.Popen(  # noqa: S603 - argv list, no shell, server-controlled
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # own session/pgid; survives restart; signal as a group
                    cwd=str(spawn_cwd),
                    env=_runner_env(),
                    pass_fds=(ready_write, ack_read, preparation.lease_fd),
                )
                os.close(ready_write)
                ready_write = -1
                os.close(ack_read)
                ack_read = -1
                # Build the minimum usable identity before log close, /proc,
                # boot-id, or handle enrichment can fail. start_new_session=True
                # makes the child the group leader, so pgid == pid without a
                # getpgid() race if it exits quickly.
                handle = replace(
                    pending,
                    pid=proc.pid,
                    pgid=proc.pid,
                    pid_starttime=None,
                    launch_state="spawned",
                    boot_id=None,
                    pid_namespace=pid_namespace,
                )
                pid_starttime = read_proc_starttime(proc.pid)
                handle = replace(handle, pid_starttime=pid_starttime)
                boot_id = read_boot_id()
                handle = replace(handle, boot_id=boot_id)
                if handle.pid_starttime is None:
                    raise RuntimeError(
                        "spawned runner has no Linux /proc start time; refused launch because "
                        "later cancellation could not distinguish PID reuse"
                    )
                if handle.boot_id is None:
                    raise RuntimeError(
                        "spawned runner has no Linux boot id; refused launch because later "
                        "cancellation could not distinguish PID reuse after reboot"
                    )
                readable = fd_ready(ready_read, timeout=_RUNNER_READY_TIMEOUT_SECONDS)
                ready = os.read(ready_read, 1) if readable else b""
                if ready != LAUNCH_READY_BYTE:
                    raise RuntimeError(
                        "detached runner did not persist its launch receipt before launch"
                    )
                persisted = self._runs.get(run_id)
                if persisted != handle:
                    raise RuntimeError("detached runner launch receipt did not match its process")
            # An asynchronous exception may arrive after the byte reaches the
            # runner but before os.write returns. From this point onward,
            # cleanup must assume separately-sessioned trials could start.
            runner_acknowledged = True
            if os.write(ack_write, LAUNCH_ACK_BYTE) != len(LAUNCH_ACK_BYTE):
                raise RuntimeError("could not acknowledge the detached runner launch")
            acknowledged_fd = ack_write
            ack_write = -1
            with contextlib.suppress(OSError):
                os.close(acknowledged_fd)
        except BaseException as exc:
            if proc is None:
                # Opening the log or Popen itself failed: no child exists.
                raise
            cleanup_handle = replace(
                pending,
                pid=proc.pid,
                pgid=proc.pid,
                pid_starttime=pid_starttime,
                launch_state="spawned",
                boot_id=boot_id,
                pid_namespace=pid_namespace,
            )
            cleanup_confirmed = self._terminate_failed_spawn(
                cleanup_handle, exc, acknowledged=runner_acknowledged
            )
            raise _SpawnBookkeepingError(
                exc,
                cleanup_confirmed=cleanup_confirmed,
            ) from exc
        finally:
            for fd in (ready_read, ready_write, ack_read, ack_write):
                if fd >= 0:
                    with contextlib.suppress(OSError):
                        os.close(fd)
        assert handle is not None
        return handle

    def _neutral_spawn_cwd(self) -> Path:
        """Return the server-owned directory the detached runner is spawned in.

        The runner must not start in a directory the experiment's project can
        write to, so use the operator-owned MCP state directory that already
        holds run handles, logs, and config snapshots. It is re-validated here
        because it must exist and stay owner-only at the moment of the spawn.

        :return Path: Private state directory used as the child's initial cwd.
        """
        state_dir = self._registry.state_dir
        ensure_private_dir(state_dir)
        return state_dir

    def _cancel_allowed(self, handle: RunHandle) -> bool:
        """Return whether launch-time and current policy permit cancellation.

        A catalog edit may revoke cancellation but never grant it, so a still
        cataloged id is intersected with the permission frozen at launch.

        When the id is gone from the catalog there is no current policy to
        intersect with, and the launch-time permission stands alone. Refusing
        there would strand a live detached runner: ``status``
        still reports it as ``running`` from its snapshot, and
        ``phasesweep mcp recover-run`` refuses while the runner is alive, so
        the operator would be left hunting the PGID by hand. Cancellation is
        risk-reducing and was authorized when this run started.

        :param RunHandle handle: Run handle whose cancellation permission should be checked.
        :return bool: Whether both launch-time and current permission are true,
            or the launch-time permission alone when the id is no longer cataloged.
        """
        if not handle.allow_cancel:
            # A launch-time denial is permanent for this run.
            return False
        try:
            return self._registry.get(handle.experiment_id).allow_cancel
        except UnknownExperimentError:
            return True
