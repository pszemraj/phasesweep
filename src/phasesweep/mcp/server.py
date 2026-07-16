"""FastMCP adapter: the only module that imports the MCP SDK.

PhaseSweepMCP holds all logic and is SDK-free and unit-testable. build_server
wraps each method as a FastMCP tool; _safe_tool guarantees tool errors are
redacted. serve() loads the catalog, builds the store, and serves over stdio.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib.resources
import importlib.util
import logging
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

from phasesweep.config import Experiment
from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.config.io import load_config_bytes
from phasesweep.engine import read_status, read_winners
from phasesweep.engine.state import Winner, _load_winner
from phasesweep.mcp.audit import AuditLogger
from phasesweep.mcp.errors import (
    CatalogError,
    ConcurrencyLimitError,
    ConfigChangedError,
    ExperimentBusyError,
    InvalidPhaseError,
    LaunchInProgressError,
    McpToolError,
    PermissionDeniedError,
    ResumeNotReadyError,
    RunSnapshotUnavailableError,
    UnknownExperimentError,
    UnknownRunError,
)
from phasesweep.mcp.redaction import status_payload, winners_payload
from phasesweep.mcp.registry import RegisteredExperiment, Registry
from phasesweep.mcp.runs import RunHandle, RunState, RunStore
from phasesweep.mcp.time import utc_now_iso
from phasesweep.runtime.files import open_private_text, private_atomic_write_bytes
from phasesweep.runtime.process import kill_stale_group, read_proc_starttime

log = logging.getLogger("phasesweep.mcp.server")

SAFE_NAME_JSON_PATTERN = SAFE_NAME_PATTERN.pattern
TOOL_LIST_EXPERIMENTS = "phasesweep_list_experiments"
TOOL_VALIDATE_CONFIG = "phasesweep_validate_config"
TOOL_GET_STATUS = "phasesweep_get_status"
TOOL_GET_WINNERS = "phasesweep_get_winners"
TOOL_LAUNCH_SWEEP = "phasesweep_launch_sweep"
TOOL_CANCEL_SWEEP = "phasesweep_cancel_sweep"
TOOL_AWAIT_RUN = "phasesweep_await_run"
CATALOG_RESOURCE_URI = "phasesweep://catalog"
PROMPT_RUN_AND_MONITOR = "phasesweep_run_and_monitor"
DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 100
# Adaptive poll-interval bounds for get_status: the suggested wait tracks the
# median completed-trial duration so agents polling minute-long trials back
# off, clamped so pathological durations can neither hammer the storage nor
# park the agent for an hour.
POLL_DEFAULT_SECONDS = 30
POLL_MIN_SECONDS = 15
POLL_MAX_SECONDS = 600
# await_run blocks server-side so one call replaces dozens of polls; the cap
# keeps a response inside common client tool timeouts, and the recheck interval
# bounds how stale a change notification can be.
AWAIT_DEFAULT_TIMEOUT_SECONDS = 120
AWAIT_MIN_TIMEOUT_SECONDS = 5
AWAIT_MAX_TIMEOUT_SECONDS = 600
AWAIT_RECHECK_SECONDS = 5.0

# Agent-facing tool descriptions. Descriptions are the one instruction channel
# present on every call even when the user loads no prompt, so each one chains
# to the next tool in the workflow by literal name.
DESCRIPTION_LIST_EXPERIMENTS = (
    "List the human-curated experiments this server can run: ids, descriptions, "
    "phase names, and the optimization metric. Read-only. Start here. If "
    "next_cursor is non-null, call again with it. Then call "
    f"{TOOL_VALIDATE_CONFIG} on the id you plan to use."
)
DESCRIPTION_VALIDATE_CONFIG = (
    "Inspect an experiment before launching it: per-phase names, trial counts, "
    "samplers, inherited phases, and search-space keys (never ranges). Read-only. "
    f"Call this before every {TOOL_LAUNCH_SWEEP}."
)
DESCRIPTION_LAUNCH_SWEEP = (
    "Start an experiment's sweep as a detached background run that survives this "
    f"session. Returns a run_id: save it, then wait on {TOOL_AWAIT_RUN} (or poll "
    f"{TOOL_GET_STATUS}) with it until the state is terminal. Pass from_phase "
    "only to resume when earlier phase winners already exist. A refusal such as "
    "\"action 'launch' is not permitted\" or a concurrency limit is deliberate "
    "catalog policy: report it to the user; do not retry or work around it."
)
DESCRIPTION_GET_STATUS = (
    "Per-phase trial progress and the run process state (running / succeeded / "
    "failed / cancelled). Provide exactly one of experiment_id or run_id; after a "
    "launch, always use the run_id so catalog edits cannot redirect monitoring. "
    f"Read-only. Prefer {TOOL_AWAIT_RUN} for monitoring; when polling this "
    "instead, wait poll_after_seconds between calls. When terminal, call "
    f"{TOOL_GET_WINNERS} with the same run_id."
)
DESCRIPTION_GET_WINNERS = (
    "The end of the workflow: per completed phase, the winning trial number, "
    "metric value, policy-filtered sampled params, gate status, and completeness. "
    "Phases that completed still report winners when the run later failed or was "
    "cancelled. Values shown as <redacted> are intentional catalog policy, not "
    "errors. Provide exactly one of experiment_id or run_id (prefer the launched "
    "run_id). Read-only."
)
DESCRIPTION_CANCEL_SWEEP = (
    "Stop a launched run by run_id. Use only when the user asks or to prevent an "
    "unwanted active sweep. If cleanup_confirmed is false, report it to the user; "
    "recovery is operator-only and no MCP tool can clear it."
)
DESCRIPTION_AWAIT_RUN = (
    "Wait for a launched run to change. Blocks up to timeout_seconds (default "
    f"{AWAIT_DEFAULT_TIMEOUT_SECONDS}, max {AWAIT_MAX_TIMEOUT_SECONDS}) and returns "
    "early when the run reaches a terminal state or a phase gains a winner; "
    "otherwise returns the current status at timeout — if the state is still "
    "running, call again with the same run_id. Returns the same payload as "
    f"{TOOL_GET_STATUS} plus changed and reason (terminal / phase_completed / "
    f"timeout). Read-only. Prefer this over polling {TOOL_GET_STATUS} in a loop."
)

ExperimentId = Annotated[
    str,
    Field(
        description=f"Catalog experiment id exposed by {TOOL_LIST_EXPERIMENTS}.",
        pattern=SAFE_NAME_JSON_PATTERN,
    ),
]
MaybeExperimentId = Annotated[
    str | None,
    Field(
        description="Catalog experiment id. Provide exactly one of experiment_id or run_id.",
        pattern=SAFE_NAME_JSON_PATTERN,
    ),
]
RunId = Annotated[
    str,
    Field(
        description=f"MCP run id returned by {TOOL_LAUNCH_SWEEP}.", pattern=SAFE_NAME_JSON_PATTERN
    ),
]
MaybeRunId = Annotated[
    str | None,
    Field(
        description=f"MCP run id returned by {TOOL_LAUNCH_SWEEP}. Provide exactly one of experiment_id or run_id.",
        pattern=SAFE_NAME_JSON_PATTERN,
    ),
]
PhaseName = Annotated[
    str,
    Field(description="Phase name from the experiment config.", pattern=SAFE_NAME_JSON_PATTERN),
]
MaybePhaseName = Annotated[
    str | None,
    Field(
        description="Optional phase name to resume from after earlier phases already have winners.",
        pattern=SAFE_NAME_JSON_PATTERN,
    ),
]
ListLimit = Annotated[
    int,
    Field(
        description="Maximum catalog entries to return. Use the next_cursor value to fetch more.",
        ge=1,
        le=MAX_LIST_LIMIT,
    ),
]
MaybeCursor = Annotated[
    str | None,
    Field(
        description="Opaque pagination cursor returned by a previous phasesweep_list_experiments call.",
    ),
]
AwaitTimeoutSeconds = Annotated[
    int,
    Field(
        description=(
            "Seconds to block waiting for the run to change before returning its current status."
        ),
        ge=AWAIT_MIN_TIMEOUT_SECONDS,
        le=AWAIT_MAX_TIMEOUT_SECONDS,
    ),
]


class _ToolPayload(BaseModel):
    """Strict base for structured MCP tool results."""

    model_config = ConfigDict(extra="forbid")


class MetricPayload(_ToolPayload):
    """Optimization metric descriptor."""

    name: str = Field(description="Metric key extracted from trial output.")
    goal: Literal["minimize", "maximize"] = Field(description="Optimization direction.")


class ExperimentSummaryPayload(_ToolPayload):
    """Path-free catalog entry summary."""

    id: ExperimentId
    description: str = Field(description="Operator-authored catalog description.")
    phases: list[PhaseName] = Field(description="Declared phases, in execution order.")
    metric: MetricPayload


class ListExperimentsResult(_ToolPayload):
    """Structured output for list_experiments."""

    experiments: list[ExperimentSummaryPayload]
    total_count: int = Field(ge=0, description="Total catalog entries exposed by this server.")
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the next page, or null when this page is complete.",
    )


class PhaseValidationPayload(_ToolPayload):
    """Agent-safe phase structure."""

    name: PhaseName
    n_trials: int = Field(ge=0, description="Number of trials configured for this phase.")
    sampler: str = Field(description="Sampler type only; sampler internals stay in the config.")
    inherits: list[PhaseName] = Field(description="Parent phases inherited by this phase.")
    search_space: list[str] = Field(description="Search-space keys only, never ranges or values.")


class ValidateConfigResult(_ToolPayload):
    """Structured output for validate_config."""

    experiment_id: ExperimentId
    metric: MetricPayload
    phases: list[PhaseValidationPayload]


class RunPayload(_ToolPayload):
    """Agent-visible run process state."""

    run_id: RunId
    state: RunState
    started_at: str = Field(description="UTC ISO-8601 launch timestamp.")


class PhaseStatusPayload(_ToolPayload):
    """Per-phase status without filesystem paths."""

    phase: PhaseName
    trials: dict[str, int] = Field(description="Optuna trial counts by state.")
    running: int = Field(ge=0, description="Number of currently running trials.")
    n_trials: int = Field(ge=0, description="Configured trial budget for this phase.")
    completed: int = Field(ge=0, description="Trials completed so far in this phase.")
    winner_present: bool = Field(description="Whether this phase has a winner artifact.")


class GetStatusResult(_ToolPayload):
    """Structured output for get_status."""

    experiment_id: ExperimentId
    metric: MetricPayload
    phases: list[PhaseStatusPayload]
    summary_present: bool
    run: RunPayload | None
    elapsed_seconds: int | None = Field(
        description=(
            "Seconds since launch while running; total run duration once terminal; "
            "null when no run is associated with this query."
        )
    )
    poll_after_seconds: int = Field(
        ge=POLL_MIN_SECONDS,
        le=POLL_MAX_SECONDS,
        description=(
            "Suggested seconds to wait before the next phasesweep_get_status call, "
            "sized from the median completed-trial duration."
        ),
    )


class AwaitRunResult(GetStatusResult):
    """Structured output for await_run: get_status plus what ended the wait."""

    changed: bool = Field(
        description=(
            "Whether run state, phase progress, or winner presence changed during the wait."
        )
    )
    reason: Literal["terminal", "phase_completed", "timeout"] = Field(
        description=(
            "Why the wait returned: the run reached a terminal state, a phase gained a "
            "winner, or the timeout elapsed with no change."
        )
    )


class WinnerPhasePayload(_ToolPayload):
    """Agent-visible phase winner."""

    phase: PhaseName
    trial_number: int = Field(ge=0)
    metric: float
    params: dict[str, Any] = Field(
        description=(
            "Sampled winning hyperparameters only; fixed/inherited overrides are omitted. "
            "Values may be redacted by catalog visible_params policy."
        )
    )
    params_redacted: bool = Field(
        description=(
            "True when any param value was withheld by catalog visible_params policy - "
            "deliberate, not missing data."
        )
    )
    gates_passed: bool | None = Field(
        description="True/false when gates were declared; null when the phase has no gates."
    )
    incomplete: bool = Field(description="Whether a wallclock timeout produced a partial winner.")


class GetWinnersResult(_ToolPayload):
    """Structured output for get_winners."""

    experiment_id: ExperimentId
    phases: list[WinnerPhasePayload]


class LaunchSweepResult(_ToolPayload):
    """Structured output for launch_sweep."""

    run_id: RunId
    experiment_id: ExperimentId
    state: Literal["running"]


class CancelSweepResult(_ToolPayload):
    """Structured output for cancel_sweep."""

    run_id: RunId
    state: RunState
    cleanup_confirmed: bool | None = Field(
        default=None,
        description=(
            "Whether runner cleanup was fully confirmed; null when the run was already terminal."
        ),
    )


def _cursor_offset(cursor: str | None) -> int:
    """Decode the opaque v1 cursor into a list offset.

    :param str | None cursor: Cursor supplied by the agent.
    :return int: Zero-based catalog offset.
    """
    if cursor is None:
        return 0
    try:
        offset = int(cursor)
    except ValueError:
        raise McpToolError(
            "invalid cursor; use next_cursor returned by phasesweep_list_experiments"
        ) from None
    if offset < 0:
        raise McpToolError(
            "invalid cursor; use next_cursor returned by phasesweep_list_experiments"
        )
    return offset


def _parse_utc_iso(value: object) -> datetime | None:
    """Parse a persisted ISO-8601 timestamp, tolerating malformed values.

    :param object value: Raw timestamp field from a handle or status payload.
    :return datetime | None: Timezone-aware datetime, or ``None`` when the
        value is missing, malformed, or naive.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _run_elapsed_seconds(store: RunStore, handle: RunHandle, state: str) -> int | None:
    """Compute wall seconds for a run: launch-to-now while running, total when terminal.

    Terminal runs prefer the runner-stamped ``ended_at`` in status.json and
    fall back to the status file's mtime (runs recorded before the stamp
    existed). A terminal run with no readable status at all - e.g. SIGKILLed
    before writing one - reports ``None`` rather than a guess.

    :param RunStore store: Run store used to read the terminal status.
    :param RunHandle handle: Persisted run whose duration is measured.
    :param str state: Derived run state for ``handle``.
    :return int | None: Non-negative whole seconds, or ``None`` when the
        endpoints cannot be established.
    """
    started = _parse_utc_iso(handle.started_at)
    if started is None:
        return None
    ended: datetime | None
    if state == "running":
        ended = datetime.now(timezone.utc)
    else:
        status = store.recorded_terminal_status(handle)
        ended = _parse_utc_iso(status.get("ended_at")) if status is not None else None
        if ended is None and status is not None:
            try:
                mtime = store.status_path(handle.run_id).stat().st_mtime
            except OSError:
                return None
            ended = datetime.fromtimestamp(mtime, tz=timezone.utc)
    if ended is None:
        return None
    return max(0, round((ended - started).total_seconds()))


def _poll_after_seconds(median_trial_seconds: float | None) -> int:
    """Suggest a status poll interval from the median completed-trial duration.

    :param float | None median_trial_seconds: Median COMPLETE-trial wall
        duration, or ``None`` while nothing has finished.
    :return int: Whole seconds clamped to the poll bounds; the default when no
        trial has completed yet.
    """
    if median_trial_seconds is None:
        return POLL_DEFAULT_SECONDS
    return int(min(POLL_MAX_SECONDS, max(POLL_MIN_SECONDS, round(median_trial_seconds))))


def _await_snapshot(state: str, status: dict[str, Any]) -> tuple[Any, ...]:
    """Reduce a status read to the comparable facts await_run watches.

    :param str state: Derived run state at read time.
    :param dict[str, Any] status: Path-free ``read_status`` payload.
    :return tuple[Any, ...]: Hashable snapshot of run state and per-phase
        winner presence and completed-trial counts.
    """
    return (
        state,
        tuple(
            (phase["phase"], phase["winner_present"], phase["completed"])
            for phase in status["phases"]
        ),
    )


def _phase_gained_winner(baseline: tuple[Any, ...], snapshot: tuple[Any, ...]) -> bool:
    """Return whether any phase gained a winner between two await snapshots.

    :param tuple[Any, ...] baseline: Snapshot taken when the wait began.
    :param tuple[Any, ...] snapshot: Snapshot from the latest recheck.
    :return bool: ``True`` when a phase's winner artifact appeared mid-wait.
    """
    had_winner = {phase: winner for phase, winner, _completed in baseline[1]}
    return any(
        winner and not had_winner.get(phase, False) for phase, winner, _completed in snapshot[1]
    )


class PhaseSweepMCP:
    """SDK-free implementation of every tool. Methods raise ``McpToolError``."""

    def __init__(
        self, registry: Registry, runs: RunStore, audit: AuditLogger | None = None
    ) -> None:
        """Create the SDK-free MCP implementation.

        :param Registry registry: Validated catalog registry.
        :param RunStore runs: Persistent detached-run store.
        :param AuditLogger | None audit: Optional structured audit logger.
        """
        self._registry = registry
        self._runs = runs
        self._audit = audit
        # await_run's recheck pause; tests replace it to make waits instant.
        self._sleep: Callable[[float], None] = time.sleep

    def _audit_success(
        self,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        resolved: dict[str, Any] | None = None,
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
        result_counts: dict[str, int] | None = None,
    ) -> None:
        """Record a successful MCP tool call when audit logging is enabled.

        :param str tool: MCP tool name.
        :param dict[str, Any] | None args: Safe agent-supplied arguments.
        :param dict[str, Any] | None resolved: Safe server-resolved identifiers.
        :param dict[str, Any] | None state_before: Safe state summary before the call.
        :param dict[str, Any] | None state_after: Safe state summary after the call.
        :param dict[str, int] | None result_counts: Result counts that avoid copying full payloads.
        """
        if self._audit is not None:
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
        args: dict[str, Any] | None,
        exc: Exception,
        *,
        resolved: dict[str, Any] | None = None,
        state_before: dict[str, Any] | None = None,
    ) -> None:
        """Record a failed MCP tool call when audit logging is enabled.

        :param str tool: MCP tool name.
        :param dict[str, Any] | None args: Safe agent-supplied arguments.
        :param Exception exc: Exception that will be surfaced through the safe tool wrapper.
        :param dict[str, Any] | None resolved: Safe server-resolved identifiers known before failure.
        :param dict[str, Any] | None state_before: Safe state summary before the failure.
        """
        if self._audit is None:
            return
        message = exc.safe_message if isinstance(exc, McpToolError) else "internal error"
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
        """
        args = {"limit": limit, "cursor": cursor}
        total_count = 0
        page: list[dict[str, Any]] = []
        try:
            if limit < 1 or limit > MAX_LIST_LIMIT:
                raise McpToolError(f"limit must be between 1 and {MAX_LIST_LIMIT}")
            offset = _cursor_offset(cursor)
            experiments = self._registry.summaries()
            total_count = len(experiments)
            page = experiments[offset : offset + limit]
            next_offset = offset + len(page)
            result = {
                "experiments": page,
                "total_count": total_count,
                "next_cursor": str(next_offset) if next_offset < total_count else None,
            }
        except Exception as exc:
            self._audit_error(TOOL_LIST_EXPERIMENTS, args, exc)
            raise
        self._audit_success(
            TOOL_LIST_EXPERIMENTS,
            args,
            result_counts={"experiments": len(page), "total_count": total_count},
        )
        return result

    def validate(self, experiment_id: str) -> dict[str, Any]:
        """Report an experiment's phase structure (never the command/env/storage).

        :param str experiment_id: Catalog experiment id to inspect.
        :return dict[str, Any]: Path-free validation payload for the agent.
        """
        args = {"experiment_id": experiment_id}
        resolved: dict[str, Any] = {}
        search_space_keys = 0
        try:
            reg = self._registry.get(experiment_id)
            resolved["experiment_id"] = reg.id
            exp = reg.experiment
            search_space_keys = sum(len(p.search_space) for p in exp.phases)
            phases = [
                {
                    "name": p.name,
                    "n_trials": p.n_trials,
                    "sampler": p.sampler.type,
                    "inherits": p.inherits,
                    "search_space": sorted(p.search_space),  # keys only, not ranges
                }
                for p in exp.phases
            ]
            # Already validated at startup; report the structure, never the command.
            result = {
                "experiment_id": reg.id,
                "metric": {"name": exp.metric.name, "goal": exp.metric.goal},
                "phases": phases,
            }
        except Exception as exc:
            self._audit_error(TOOL_VALIDATE_CONFIG, args, exc, resolved=resolved)
            raise
        self._audit_success(
            TOOL_VALIDATE_CONFIG,
            args,
            resolved=resolved,
            result_counts={
                "phases": len(phases),
                "search_space_keys": search_space_keys,
            },
        )
        return result

    def status(self, *, experiment_id: str | None = None, run_id: str | None = None) -> dict:
        """Per-phase trial counts and winner presence plus the run process state.

        Provide either ``experiment_id`` (reports the live run, if any) or
        ``run_id`` (reports that specific run). Raises if neither is given.

        :param str | None experiment_id: Optional catalog id for experiment-level status.
        :param str | None run_id: Optional detached run id for run-specific status.
        :return dict: Path-free status payload for the agent.
        """
        args = {"experiment_id": experiment_id, "run_id": run_id}
        resolved: dict[str, Any] = {}
        state_after: dict[str, Any] = {}
        try:
            target_id, experiment, run, resolved = self._resolve_read_target(
                experiment_id=experiment_id,
                run_id=run_id,
                include_run=True,
            )
            if run is not None:
                state_after = {"run_state": run["state"]}
            status = read_status(experiment)
            elapsed_seconds = None
            if run is not None:
                handle = self._runs.get(run["run_id"])
                if handle is not None:
                    elapsed_seconds = _run_elapsed_seconds(self._runs, handle, run["state"])
            result = status_payload(
                target_id,
                status,
                run,
                elapsed_seconds=elapsed_seconds,
                poll_after_seconds=_poll_after_seconds(status.get("median_trial_seconds")),
            )
        except Exception as exc:
            self._audit_error(TOOL_GET_STATUS, args, exc, resolved=resolved)
            raise
        self._audit_success(
            TOOL_GET_STATUS,
            args,
            resolved=resolved,
            state_after=state_after or None,
            result_counts={"phases": len(result["phases"]), "running_runs": int(run is not None)},
        )
        return result

    def await_run(
        self,
        run_id: str,
        timeout_seconds: int = AWAIT_DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Block until a launched run changes, then return its full status.

        A bounded sleep-and-recheck loop over the same reads ``status`` does:
        no new state, no side effects. Returns early when the run reaches a
        terminal state or a phase gains a winner; otherwise returns the
        current status when the (clamped) timeout elapses. Each recheck
        re-resolves the run from disk, so a server restart mid-wait is
        invisible to the caller.

        :param str run_id: Detached run id to wait on.
        :param int timeout_seconds: Requested wait, clamped to
            [``AWAIT_MIN_TIMEOUT_SECONDS``, ``AWAIT_MAX_TIMEOUT_SECONDS``].
        :return dict[str, Any]: ``status`` payload plus ``changed`` and
            ``reason`` (``terminal`` / ``phase_completed`` / ``timeout``).
        """
        args = {"run_id": run_id, "timeout_seconds": timeout_seconds}
        resolved: dict[str, Any] = {}
        try:
            timeout = min(
                AWAIT_MAX_TIMEOUT_SECONDS, max(AWAIT_MIN_TIMEOUT_SECONDS, int(timeout_seconds))
            )
            deadline = time.monotonic() + timeout
            baseline: tuple[Any, ...] | None = None
            while True:
                target_id, experiment, run, resolved = self._resolve_read_target(
                    experiment_id=None,
                    run_id=run_id,
                    include_run=True,
                )
                assert run is not None  # the run_id path always includes run state
                status = read_status(experiment)
                snapshot = _await_snapshot(run["state"], status)
                if baseline is None:
                    baseline = snapshot
                if run["state"] in ("succeeded", "failed", "cancelled"):
                    reason = "terminal"
                elif _phase_gained_winner(baseline, snapshot):
                    reason = "phase_completed"
                elif time.monotonic() >= deadline:
                    reason = "timeout"
                else:
                    self._sleep(min(AWAIT_RECHECK_SECONDS, max(0.0, deadline - time.monotonic())))
                    continue
                break
            elapsed_seconds = None
            handle = self._runs.get(run_id)
            if handle is not None:
                elapsed_seconds = _run_elapsed_seconds(self._runs, handle, run["state"])
            result = status_payload(
                target_id,
                status,
                run,
                elapsed_seconds=elapsed_seconds,
                poll_after_seconds=_poll_after_seconds(status.get("median_trial_seconds")),
            )
            result["changed"] = snapshot != baseline
            result["reason"] = reason
        except Exception as exc:
            self._audit_error(TOOL_AWAIT_RUN, args, exc, resolved=resolved)
            raise
        self._audit_success(
            TOOL_AWAIT_RUN,
            args,
            resolved=resolved,
            state_after={"run_state": run["state"], "await_reason": reason},
            result_counts={"phases": len(result["phases"])},
        )
        return result

    def _resolve_read_target(
        self,
        *,
        experiment_id: str | None,
        run_id: str | None,
        include_run: bool,
    ) -> tuple[str, Experiment, dict[str, Any] | None, dict[str, Any]]:
        """Resolve status/winner reads to the catalog config or immutable run snapshot.

        :param str | None experiment_id: Catalog id for current experiment-level reads.
        :param str | None run_id: Persisted run id for immutable run-specific reads.
        :param bool include_run: Whether to include live run state in the returned payload.
        :return tuple[str, Experiment, dict[str, Any] | None, dict[str, Any]]: Target id,
            parsed experiment, optional run payload, and audit-safe resolved ids.
        """
        if (experiment_id is None) == (run_id is None):
            provided = "neither" if experiment_id is None else "both"
            raise McpToolError(
                f"provide exactly one of experiment_id or run_id; you provided {provided}. "
                "After a launch, prefer the run_id."
            )
        if run_id is not None:
            handle = self._runs.get(run_id)
            if handle is None:
                raise UnknownRunError(run_id)
            resolved = {"experiment_id": handle.experiment_id, "run_id": run_id}
            experiment = self._load_run_experiment(handle)
            run = None
            if include_run:
                run = {
                    "run_id": run_id,
                    "state": self._runs.state(handle),
                    "started_at": handle.started_at,
                }
            return handle.experiment_id, experiment, run, resolved

        assert experiment_id is not None
        reg = self._registry.get(experiment_id)
        live = self._runs.live_run_for(experiment_id)
        resolved = {"experiment_id": reg.id}
        if live is not None:
            resolved["run_id"] = live.run_id
        experiment = self._load_run_experiment(live) if live is not None else reg.experiment
        run = (
            {"run_id": live.run_id, "state": "running", "started_at": live.started_at}
            if include_run and live is not None
            else None
        )
        return reg.id, experiment, run, resolved

    def _load_run_experiment(self, handle: RunHandle) -> Experiment:
        """Load and verify the immutable config snapshot for a persisted run.

        Run-specific monitoring must follow the exact config the detached
        runner received, even if the cataloged config has since been edited and
        the MCP server restarted. The saved handle's hash is the guardrail: a
        missing, corrupted, or non-experiment snapshot is rejected instead of
        silently reporting status for the wrong storage or workdir.

        :param RunHandle handle: Persisted run identity whose snapshot should be loaded.
        :return Experiment: Parsed experiment from the launched per-run snapshot.
        """
        snapshot_path = self._runs.config_snapshot_path(handle.run_id)
        try:
            data = snapshot_path.read_bytes()
        except OSError as exc:
            log.info("cannot read config snapshot for run=%s: %s", handle.run_id, exc)
            raise RunSnapshotUnavailableError(handle.run_id) from None
        if hashlib.sha256(data).hexdigest() != handle.config_sha256:
            log.info("config snapshot hash mismatch for run=%s", handle.run_id)
            raise RunSnapshotUnavailableError(handle.run_id)
        try:
            config = load_config_bytes(data, source=f"run snapshot {handle.run_id}")
        except (ValueError, yaml.YAMLError) as exc:
            log.info("invalid config snapshot for run=%s: %s", handle.run_id, exc)
            raise RunSnapshotUnavailableError(handle.run_id) from None
        if not isinstance(config, Experiment):
            log.info("config snapshot for run=%s is not a single experiment", handle.run_id)
            raise RunSnapshotUnavailableError(handle.run_id)
        return config

    def winners(
        self, experiment_id: str | None = None, *, run_id: str | None = None
    ) -> dict[str, Any]:
        """Return the winning hyperparameters per completed phase.

        Provide ``experiment_id`` for the current cataloged experiment, or ``run_id``
        to read winners from the immutable config snapshot that run was launched
        with. Run-specific reads must not drift when the cataloged config changes.

        :param str | None experiment_id: Optional catalog experiment id whose winners should be read.
        :param str | None run_id: Optional detached run id whose snapshot should be read.
        :return dict[str, Any]: Path-free winners payload for the agent.
        """
        args = {"experiment_id": experiment_id, "run_id": run_id}
        resolved: dict[str, Any] = {}
        try:
            target_id, experiment, _run, resolved = self._resolve_read_target(
                experiment_id=experiment_id,
                run_id=run_id,
                include_run=False,
            )
            try:
                visible_params = self._registry.get(target_id).visible_params
            except UnknownExperimentError:
                if run_id is None:
                    raise
                # A run snapshot remains readable after the operator removes
                # its catalog entry. Without a current visibility policy,
                # default to the strict redacted posture.
                visible_params = "none"
            result = winners_payload(
                target_id,
                read_winners(experiment),
                visible_params=visible_params,
            )
        except Exception as exc:
            self._audit_error(TOOL_GET_WINNERS, args, exc, resolved=resolved)
            raise
        self._audit_success(
            TOOL_GET_WINNERS,
            args,
            resolved=resolved,
            result_counts={"phases": len(result["phases"])},
        )
        return result

    def launch(self, experiment_id: str, from_phase: str | None = None) -> dict[str, Any]:
        """Start the sweep as a detached background run; return its run_id.

        Refuses if launch is not permitted, if a ``from_phase`` resume is not
        ready (an earlier phase has no winner), if this experiment already has a
        live run, or if the server is at its max_concurrent_runs cap.

        :param str experiment_id: Catalog experiment id to launch.
        :param str | None from_phase: Optional phase to resume from after earlier winners exist.
        :return dict[str, Any]: Launch result containing run id, experiment id, and running state.
        """
        args = {"experiment_id": experiment_id, "from_phase": from_phase}
        resolved: dict[str, Any] = {}
        state_before: dict[str, Any] | None = None
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
                live = self._runs.live_runs()
                state_before = {"live_runs": len(live)}
                busy = next((h for h in live if h.experiment_id == experiment_id), None)
                if busy is not None:
                    raise ExperimentBusyError(experiment_id, busy.run_id)
                if len(live) >= self._registry.max_concurrent_runs:
                    raise ConcurrencyLimitError(len(live), self._registry.max_concurrent_runs)
                run_id = self._runs.new_run_id(reg.id)
                config_snapshot_path = self._snapshot_config(reg, run_id)
                pending = self._pending_handle(reg, run_id)
                self._runs.save(pending)
                handle = self._spawn(reg, from_phase, pending, config_snapshot_path)
                try:
                    self._runs.save(handle)
                except Exception as save_exc:
                    marker_written = False
                    try:
                        self._runs.mark_cleanup_uncertain(handle)
                        marker_written = True
                    except Exception as marker_exc:
                        log.error(
                            "cleanup uncertain after failed handle save for run_id=%s pgid=%s, "
                            "but failed to persist cleanup uncertainty marker; original save error: %r",
                            handle.run_id,
                            handle.pgid,
                            save_exc,
                            exc_info=(type(marker_exc), marker_exc, marker_exc.__traceback__),
                        )
                    try:
                        assert handle.pgid is not None
                        cleanup_confirmed = kill_stale_group(
                            handle.pid,
                            handle.pid_starttime,
                            pgid=handle.pgid,
                        )
                    except Exception:
                        log.exception(
                            "failed to terminate unsaved runner run_id=%s pgid=%s",
                            handle.run_id,
                            handle.pgid,
                        )
                    else:
                        if cleanup_confirmed:
                            if marker_written:
                                with contextlib.suppress(Exception):
                                    self._runs.clear_cleanup_uncertain(handle)
                        else:
                            log.error(
                                "cleanup uncertain after failed handle save for run_id=%s pgid=%s",
                                handle.run_id,
                                handle.pgid,
                            )
                    raise
            result = {"run_id": handle.run_id, "experiment_id": experiment_id, "state": "running"}
        except Exception as exc:
            self._audit_error(
                TOOL_LAUNCH_SWEEP,
                args,
                exc,
                resolved=resolved,
                state_before=state_before,
            )
            raise
        self._audit_success(
            TOOL_LAUNCH_SWEEP,
            args,
            resolved={"experiment_id": experiment_id, "run_id": handle.run_id},
            state_before=state_before,
            state_after={
                "run_state": "running",
                "live_runs": (state_before or {}).get("live_runs", 0) + 1,
            },
            result_counts={"runs": 1},
        )
        return result

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Stop a running sweep: SIGTERM -> grace -> SIGKILL the runner's group.

        The terminal state is reported as ``cancelled`` only when the runner
        records its cancellation status. If the runner group is gone but no
        status was written, cleanup remains uncertain because trial process
        groups may still be alive.

        :param str run_id: Detached run id to cancel.
        :return dict[str, Any]: Cancellation result containing final state and optional cleanup confirmation.
        """
        args = {"run_id": run_id}
        resolved: dict[str, Any] = {}
        state_before: dict[str, Any] | None = None
        result: dict[str, Any]
        try:
            handle = self._runs.get(run_id)
            if handle is None:
                raise UnknownRunError(run_id)
            resolved = {"experiment_id": handle.experiment_id, "run_id": run_id}
            if not self._cancel_allowed(handle):
                raise PermissionDeniedError("cancel", handle.experiment_id)
            before = self._runs.state(handle)
            state_before = {"run_state": before}
            if before != "running":
                result = {"run_id": run_id, "state": before}  # already terminal
                self._audit_success(
                    TOOL_CANCEL_SWEEP,
                    args,
                    resolved=resolved,
                    state_before=state_before,
                    state_after={"run_state": before},
                    result_counts={"runs": 1},
                )
                return result
            # Keep the run live before signalling. In the force-kill/no-status
            # case state() could otherwise briefly derive "failed" while trial
            # descendants still hold resources.
            self._runs.mark_cleanup_uncertain(handle)
            # SIGTERM -> grace -> SIGKILL on the runner's process group. A
            # runner-written status is useful only when it includes explicit
            # cleanup evidence from the engine shutdown handler. If the server
            # had to force-kill the runner first, or the handler reported
            # uncertainty, child trial PGIDs may still live, so keep the run
            # counted as live and fail closed.
            identity = self._runs.cleanup_identity(handle)
            runner_group_gone = kill_stale_group(
                identity.pid,
                identity.pid_starttime,
                pgid=identity.pgid,
                grace_seconds=30.0,
            )
            terminal_status = self._runs.recorded_terminal_status(handle)
            confirmed = (
                runner_group_gone
                and terminal_status is not None
                and terminal_status.get("cleanup_confirmed") is True
            )
            if confirmed:
                self._runs.clear_cleanup_uncertain(handle)
            after = self._runs.state(handle)
            result = {"run_id": run_id, "state": after, "cleanup_confirmed": confirmed}
        except Exception as exc:
            self._audit_error(
                TOOL_CANCEL_SWEEP,
                args,
                exc,
                resolved=resolved,
                state_before=state_before,
            )
            raise
        self._audit_success(
            TOOL_CANCEL_SWEEP,
            args,
            resolved=resolved,
            state_before=state_before,
            state_after={"run_state": after, "cleanup_confirmed": confirmed},
            result_counts={"runs": 1},
        )
        return result

    def _require_resume_ready(self, reg: RegisteredExperiment, from_phase: str) -> None:
        """Verify that every earlier phase has a compatible persisted winner.

        :param RegisteredExperiment reg: Registered experiment being resumed.
        :param str from_phase: Requested phase to resume from.
        """
        names = reg.phase_names
        winners: dict[str, Winner] = {}
        for phase in reg.experiment.phases[: names.index(from_phase)]:
            inherited = {parent: winners[parent] for parent in phase.inherits}
            try:
                winners[phase.name] = _load_winner(reg.experiment, phase, inherited)
            except FileNotFoundError:
                raise ResumeNotReadyError(reg.id, from_phase, phase.name) from None
            except (
                RuntimeError,
                KeyError,
                TypeError,
                ValueError,
                AttributeError,
                OSError,
                yaml.YAMLError,
            ) as exc:
                log.info(
                    "resume preflight rejected winner for experiment=%s phase=%s: %s",
                    reg.id,
                    phase.name,
                    exc,
                )
                raise ResumeNotReadyError(
                    reg.id,
                    from_phase,
                    phase.name,
                    reason="has no compatible winner for the current config",
                ) from None

    def _snapshot_config(self, reg: RegisteredExperiment, run_id: str) -> Path:
        """Write the immutable per-run config snapshot after hash verification.

        :param RegisteredExperiment reg: Registered experiment whose config should be snapshotted.
        :param str run_id: Run id whose snapshot path should be used.
        :return Path: Written config snapshot path consumed by the detached runner.
        """
        try:
            data = reg.config_path.read_bytes()
        except OSError as exc:
            log.info("cannot read cataloged config for experiment=%s: %s", reg.id, exc)
            raise ConfigChangedError(reg.id) from None
        if hashlib.sha256(data).hexdigest() != reg.config_sha256:
            raise ConfigChangedError(reg.id)
        snapshot_path = self._runs.config_snapshot_path(run_id)
        try:
            private_atomic_write_bytes(snapshot_path, data)
        except OSError as exc:
            log.info("cannot snapshot config for experiment=%s run=%s: %s", reg.id, run_id, exc)
            raise RuntimeError("failed to create run config snapshot") from None
        return snapshot_path

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
            pid=None,
            pgid=None,
            pid_starttime=None,
            started_at=utc_now_iso(),
            log_path=str(self._runs.log_path(run_id)),
            status_path=str(self._runs.status_path(run_id)),
            launch_state="launching",
            allow_cancel=reg.allow_cancel,
        )

    def _spawn(
        self,
        reg: RegisteredExperiment,
        from_phase: str | None,
        pending: RunHandle,
        config_snapshot_path: Path,
    ) -> RunHandle:
        """Spawn the detached runner process for one registered experiment.

        :param RegisteredExperiment reg: Registered experiment to run.
        :param str | None from_phase: Optional phase to resume from.
        :param RunHandle pending: Pre-spawn persisted handle.
        :param Path config_snapshot_path: Config snapshot path consumed by the runner.
        :return RunHandle: Unsaved run handle for the spawned runner.
        """
        run_id = pending.run_id
        log_path = self._runs.log_path(run_id)
        status_path = self._runs.status_path(run_id)
        cmd = [
            sys.executable,
            "-m",
            "phasesweep.mcp.runner",
            "--run-id",
            run_id,
            "--config",
            str(config_snapshot_path),  # per-run snapshot, not agent input
            "--config-sha256",
            reg.config_sha256,
            "--status-path",
            str(status_path),
            "--state-dir",
            str(self._registry.state_dir),
            "--experiment-id",
            reg.id,
            "--started-at",
            pending.started_at,
        ]
        if reg.allow_cancel:
            cmd.append("--allow-cancel")
        if from_phase is not None:
            cmd += ["--from-phase", from_phase]
        # Open the log here, hand the fd to the child, then close our copy. The
        # child keeps it. stdin is /dev/null so the runner never blocks on input.
        with open_private_text(log_path, "w") as log_file:
            proc = subprocess.Popen(  # noqa: S603 - argv list, no shell, server-controlled
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own session/pgid; survives restart; signal as a group
                cwd=str(reg.cwd),
            )
        return RunHandle(
            run_id=run_id,
            experiment_id=reg.id,
            config_sha256=reg.config_sha256,
            pid=proc.pid,
            # start_new_session=True makes the child a session+group leader, so
            # pgid == pid by POSIX. Avoids a getpgid() race if the child exits fast.
            pgid=proc.pid,
            pid_starttime=read_proc_starttime(proc.pid),
            started_at=pending.started_at,
            log_path=str(log_path),
            status_path=str(status_path),
            launch_state="spawned",
            allow_cancel=reg.allow_cancel,
        )

    def _cancel_allowed(self, handle: RunHandle) -> bool:
        """Return whether MCP cancellation is permitted for this persisted run.

        :param RunHandle handle: Persisted run whose cancel permission is checked.
        :return bool: Current catalog ``allow.cancel`` for the experiment, falling
            back to the permission frozen on the handle when the experiment has
            since left the catalog.
        """
        try:
            return self._registry.get(handle.experiment_id).allow_cancel
        except UnknownExperimentError:
            return handle.allow_cancel


F = TypeVar("F", bound=Callable[..., Any])


def _safe_tool(fn: F) -> F:
    """Translate exceptions into redacted tool errors.

    ``McpToolError`` -> re-raised as ``ValueError`` with its safe message.
    FastMCP's low-level handler serializes tool exceptions as
    ``CallToolResult(isError=True)``. Anything else -> logged to stderr and
    replaced with a generic message so an unexpected ``Exception`` (e.g. an OSError
    carrying a path) never reaches the agent. ``functools.wraps`` preserves the
    signature so FastMCP still derives the tool schema.

    :param F fn: Tool implementation to wrap.
    :return F: Wrapped function that raises only safe tool errors.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped tool and redact any exception.

        :param Any args: Positional arguments passed through to the wrapped tool.
        :param Any kwargs: Keyword arguments passed through to the wrapped tool.
        :return Any: Wrapped tool result.
        """
        try:
            return fn(*args, **kwargs)
        except McpToolError as exc:
            raise ValueError(exc.safe_message) from None
        except Exception:
            log.exception("unhandled error in tool %s", fn.__name__)
            raise ValueError("internal error") from None

    return wrapper  # type: ignore[return-value]


def _read_annotations(title: str) -> Any:
    """Return MCP annotations for read-only, idempotent tools.

    :param str title: Human-readable tool title.
    :return Any: MCP ``ToolAnnotations`` instance.
    """
    from mcp.types import ToolAnnotations

    return ToolAnnotations(
        title=title,
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )


def _launch_annotations() -> Any:
    """Return MCP annotations for the side-effecting launch tool.

    :return Any: MCP ``ToolAnnotations`` instance.
    """
    from mcp.types import ToolAnnotations

    return ToolAnnotations(
        title="Launch Sweep",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )


def _cancel_annotations() -> Any:
    """Return MCP annotations for the process-terminating cancel tool.

    :return Any: MCP ``ToolAnnotations`` instance.
    """
    from mcp.types import ToolAnnotations

    return ToolAnnotations(
        title="Cancel Sweep",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    )


def _strict_tool_inputs(mcp: Any) -> None:
    """Make FastMCP's generated argument models reject undeclared keys.

    ``mcp.server.fastmcp.FastMCP`` 1.27.x does not expose a public switch for
    closed input schemas, so this intentionally patches the generated tool
    metadata. Keep the narrow dependency pin and startup verification together
    with this hook: if SDK internals move, ``_verify_strict_tool_inputs`` must
    fail before serving permissive tools.
    """
    for tool in mcp._tool_manager.list_tools():
        arg_model = tool.fn_metadata.arg_model
        arg_model.model_config["extra"] = "forbid"
        arg_model.model_rebuild(force=True)
        tool.parameters = arg_model.model_json_schema(by_alias=True)

    for tool_name in (TOOL_GET_STATUS, TOOL_GET_WINNERS):
        tool = mcp._tool_manager.get_tool(tool_name)
        if tool is not None:
            tool.parameters["oneOf"] = [
                {"required": ["experiment_id"], "not": {"required": ["run_id"]}},
                {"required": ["run_id"], "not": {"required": ["experiment_id"]}},
            ]
    _verify_strict_tool_inputs(mcp)


def _verify_strict_tool_inputs(mcp: Any) -> None:
    """Fail startup if FastMCP internals did not keep the strict schemas."""
    expected_one_of = [
        {"required": ["experiment_id"], "not": {"required": ["run_id"]}},
        {"required": ["run_id"], "not": {"required": ["experiment_id"]}},
    ]
    for tool in mcp._tool_manager.list_tools():
        if tool.parameters.get("additionalProperties") is not False:
            raise RuntimeError(f"MCP tool {tool.name!r} accepts undeclared input keys")
    for tool_name in (TOOL_GET_STATUS, TOOL_GET_WINNERS):
        tool = mcp._tool_manager.get_tool(tool_name)
        if tool is None:
            raise RuntimeError(f"MCP tool {tool_name!r} was not registered")
        if tool.parameters.get("oneOf") != expected_one_of:
            raise RuntimeError(f"MCP tool {tool_name!r} lost its exactly-one-of schema")


@functools.cache
def _run_and_monitor_prompt_text() -> str:
    """Return the reusable agent workflow prompt served over MCP.

    The text is packaged data (``agent_prompt.md``) so the served prompt, the
    setup docs, and the installer-injected instructions share one source.

    :return str: Safe run-and-monitor instructions for MCP clients that support prompts.
    """
    return (
        importlib.resources.files("phasesweep.mcp")
        .joinpath("agent_prompt.md")
        .read_text(encoding="utf-8")
        .strip()
    )


def build_server(app: PhaseSweepMCP) -> Any:
    """Construct the FastMCP server.

    The SDK is imported lazily so non-server code paths (and most tests) do not
    require the ``mcp`` package.

    :param PhaseSweepMCP app: SDK-free tool implementation to expose.
    :return Any: Configured FastMCP server.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("phasesweep")

    @mcp.tool(
        name=TOOL_LIST_EXPERIMENTS,
        description=DESCRIPTION_LIST_EXPERIMENTS,
        annotations=_read_annotations("List Experiments"),
        structured_output=True,
    )
    @_safe_tool
    def list_experiments(
        limit: ListLimit = DEFAULT_LIST_LIMIT,
        cursor: MaybeCursor = None,
    ) -> ListExperimentsResult:
        """List cataloged experiments: ids, descriptions, phase names, and optimization metric. Use next_cursor to fetch more.

        :param ListLimit limit: Maximum catalog entries to return.
        :param MaybeCursor cursor: Optional pagination cursor from a prior result.
        :return ListExperimentsResult: Structured catalog listing.
        """
        return ListExperimentsResult.model_validate(
            app.list_experiments(limit=limit, cursor=cursor)
        )

    @mcp.tool(
        name=TOOL_VALIDATE_CONFIG,
        description=DESCRIPTION_VALIDATE_CONFIG,
        annotations=_read_annotations("Validate Config"),
        structured_output=True,
    )
    @_safe_tool
    def validate_config(experiment_id: ExperimentId) -> ValidateConfigResult:
        """Return the phase structure (names, trial counts, samplers, inherited phases, search-space keys) for an experiment. Read-only; launches nothing.

        :param ExperimentId experiment_id: Catalog experiment id to inspect.
        :return ValidateConfigResult: Structured validation payload.
        """
        return ValidateConfigResult.model_validate(app.validate(experiment_id))

    @mcp.tool(
        name=TOOL_GET_STATUS,
        description=DESCRIPTION_GET_STATUS,
        annotations=_read_annotations("Get Status"),
        structured_output=True,
    )
    @_safe_tool
    def get_status(
        experiment_id: MaybeExperimentId = None,
        run_id: MaybeRunId = None,
    ) -> GetStatusResult:
        """Per-phase trial counts and winner presence, plus the run process state. Provide exactly one of experiment_id or run_id. Read-only.

        :param MaybeExperimentId experiment_id: Optional catalog experiment id for experiment-level status.
        :param MaybeRunId run_id: Optional detached run id for run-specific status.
        :return GetStatusResult: Structured status payload.
        """
        return GetStatusResult.model_validate(
            app.status(experiment_id=experiment_id, run_id=run_id)
        )

    @mcp.tool(
        name=TOOL_AWAIT_RUN,
        description=DESCRIPTION_AWAIT_RUN,
        annotations=_read_annotations("Await Run"),
        structured_output=True,
    )
    @_safe_tool
    def await_run(
        run_id: RunId,
        timeout_seconds: AwaitTimeoutSeconds = AWAIT_DEFAULT_TIMEOUT_SECONDS,
    ) -> AwaitRunResult:
        """Block until a launched run changes or the timeout elapses, then return its status. Read-only.

        :param RunId run_id: Detached run id to wait on.
        :param AwaitTimeoutSeconds timeout_seconds: Seconds to wait before returning current status.
        :return AwaitRunResult: Structured status payload plus changed and reason.
        """
        return AwaitRunResult.model_validate(app.await_run(run_id, timeout_seconds=timeout_seconds))

    @mcp.tool(
        name=TOOL_GET_WINNERS,
        description=DESCRIPTION_GET_WINNERS,
        annotations=_read_annotations("Get Winners"),
        structured_output=True,
    )
    @_safe_tool
    def get_winners(
        experiment_id: MaybeExperimentId = None,
        run_id: MaybeRunId = None,
    ) -> GetWinnersResult:
        """Return policy-filtered winning sampled hyperparameters per completed phase: trial number, metric, params, gate status, and completeness. Provide exactly one of experiment_id or run_id. Read-only.

        :param MaybeExperimentId experiment_id: Optional catalog experiment id whose winners should be read.
        :param MaybeRunId run_id: Optional detached run id whose snapshot should be read.
        :return GetWinnersResult: Structured winners payload.
        """
        return GetWinnersResult.model_validate(
            app.winners(experiment_id=experiment_id, run_id=run_id)
        )

    @mcp.tool(
        name=TOOL_LAUNCH_SWEEP,
        description=DESCRIPTION_LAUNCH_SWEEP,
        annotations=_launch_annotations(),
        structured_output=True,
    )
    @_safe_tool
    def launch_sweep(
        experiment_id: ExperimentId,
        from_phase: MaybePhaseName = None,
    ) -> LaunchSweepResult:
        """Start the sweep for an experiment as a background run. Optionally resume from a phase whose earlier winners already exist. Returns a run_id.

        :param ExperimentId experiment_id: Catalog experiment id to launch.
        :param MaybePhaseName from_phase: Optional phase to resume from.
        :return LaunchSweepResult: Structured launch result.
        """
        return LaunchSweepResult.model_validate(app.launch(experiment_id, from_phase=from_phase))

    @mcp.tool(
        name=TOOL_CANCEL_SWEEP,
        description=DESCRIPTION_CANCEL_SWEEP,
        annotations=_cancel_annotations(),
        structured_output=True,
    )
    @_safe_tool
    def cancel_sweep(run_id: RunId) -> CancelSweepResult:
        """Stop a running sweep by run_id. Terminates the orchestrator and its training processes.

        :param RunId run_id: Detached run id to cancel.
        :return CancelSweepResult: Structured cancellation result.
        """
        return CancelSweepResult.model_validate(app.cancel(run_id))

    @mcp.resource(
        CATALOG_RESOURCE_URI,
        name="phasesweep_catalog",
        title="PhaseSweep Catalog",
        description="Read-only first page of the human-curated experiment catalog.",
        mime_type="application/json",
    )
    @_safe_tool
    def catalog_resource() -> str:
        """Return the first catalog page for clients that attach MCP resources.

        :return str: Compact JSON catalog page.
        """
        result = ListExperimentsResult.model_validate(
            app.list_experiments(limit=DEFAULT_LIST_LIMIT, cursor=None)
        )
        return result.model_dump_json(exclude_none=True)

    @mcp.prompt(
        name=PROMPT_RUN_AND_MONITOR,
        title="Run and Monitor Sweep",
        description="Safe workflow for launching, monitoring, and summarizing a phasesweep run.",
    )
    def run_and_monitor_prompt() -> str:
        """Return safe agent instructions for the normal MCP sweep workflow.

        :return str: Prompt text.
        """
        return _run_and_monitor_prompt_text()

    _strict_tool_inputs(mcp)
    return mcp


def serve(catalog: Path) -> int:
    """Load the catalog, build the run store, and serve the seven tools over stdio.

    :param Path catalog: Operator-authored catalog file.
    :return int: Process exit code, where 2 means catalog load failure.
    """
    # stdio transport owns stdout for JSON-RPC. All logging goes to stderr.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname).1s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if importlib.util.find_spec("mcp") is None:
        print(
            "phasesweep mcp: MCP support is not installed; install with `pip install 'phasesweep[mcp]'`.",
            file=sys.stderr,
        )
        return 2

    try:
        registry = Registry.load(catalog)
    except CatalogError as exc:
        print(f"phasesweep mcp: {exc}", file=sys.stderr)
        return 2

    app = PhaseSweepMCP(
        registry,
        RunStore(registry.state_dir),
        audit=AuditLogger(registry.state_dir / "audit.jsonl"),
    )
    try:
        build_server(app).run(transport="stdio")
    except ModuleNotFoundError as exc:
        if exc.name == "mcp":
            print(
                "phasesweep mcp: MCP support is not installed; install with `pip install 'phasesweep[mcp]'`.",
                file=sys.stderr,
            )
            return 2
        raise
    return 0


def main(argv: list[str] | None = None) -> int:
    """Serve via ``python -m phasesweep.mcp.server``.

    :param list[str] | None argv: Optional argument vector; defaults to ``sys.argv`` when omitted.
    :return int: Process exit code.
    """
    parser = argparse.ArgumentParser(prog="phasesweep mcp")
    parser.add_argument("--catalog", required=True, type=Path)
    args = parser.parse_args(argv)
    return serve(args.catalog)


if __name__ == "__main__":
    raise SystemExit(main())
