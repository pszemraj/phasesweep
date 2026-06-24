"""FastMCP adapter: the only module that imports the MCP SDK.

PhaseSweepMCP holds all logic and is SDK-free and unit-testable. build_server
wraps each method as a FastMCP tool; _safe_tool guarantees tool errors are
redacted. serve() loads the catalog, builds the store, and serves over stdio.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import logging
import subprocess
import sys
from collections.abc import Callable
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
    UnknownRunError,
)
from phasesweep.mcp.redaction import status_payload, winners_payload
from phasesweep.mcp.registry import RegisteredExperiment, Registry
from phasesweep.mcp.runs import RunHandle, RunState, RunStore
from phasesweep.mcp.time import utc_now_iso
from phasesweep.runtime.files import atomic_write_bytes
from phasesweep.runtime.process import kill_stale_group, read_proc_starttime

log = logging.getLogger("phasesweep.mcp.server")

SAFE_NAME_JSON_PATTERN = SAFE_NAME_PATTERN.pattern
TOOL_LIST_EXPERIMENTS = "phasesweep_list_experiments"
TOOL_VALIDATE_CONFIG = "phasesweep_validate_config"
TOOL_GET_STATUS = "phasesweep_get_status"
TOOL_GET_WINNERS = "phasesweep_get_winners"
TOOL_LAUNCH_SWEEP = "phasesweep_launch_sweep"
TOOL_CANCEL_SWEEP = "phasesweep_cancel_sweep"
CATALOG_RESOURCE_URI = "phasesweep://catalog"
PROMPT_RUN_AND_MONITOR = "phasesweep_run_and_monitor"
DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 100

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
    winner_present: bool = Field(description="Whether this phase has a winner artifact.")


class GetStatusResult(_ToolPayload):
    """Structured output for get_status."""

    experiment_id: ExperimentId
    metric: MetricPayload
    phases: list[PhaseStatusPayload]
    summary_present: bool
    run: RunPayload | None


class WinnerPhasePayload(_ToolPayload):
    """Agent-visible phase winner."""

    phase: PhaseName
    trial_number: int = Field(ge=0)
    metric: float
    params: dict[str, Any] = Field(
        description="Sampled winning hyperparameters only; fixed/inherited overrides are omitted."
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
        description="Whether the runner process group is gone; null when the run was already terminal.",
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
            result = status_payload(target_id, read_status(experiment), run)
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
            raise McpToolError("provide exactly one of experiment_id or run_id")
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
            result = winners_payload(target_id, read_winners(experiment))
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
                except Exception:
                    try:
                        assert handle.pgid is not None
                        cleanup_confirmed = kill_stale_group(
                            handle.pid,
                            handle.pid_starttime,
                            pgid=handle.pgid,
                        )
                    except Exception:
                        log.exception(
                            "failed to terminate unsaved runner run_id=%s pgid=%d",
                            handle.run_id,
                            handle.pgid,
                        )
                    else:
                        if not cleanup_confirmed:
                            log.error(
                                "cleanup uncertain after failed handle save for run_id=%s pgid=%d",
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

        The terminal state is reported as ``cancelled`` on both the graceful
        path (the runner's handler writes status.json(143)) and the SIGKILL
        escalation (the runner is force-killed before it can; this attributes
        the cause faithfully rather than reporting ``failed``).

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
            reg = self._registry.get(handle.experiment_id)
            if not reg.allow_cancel:
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
            # SIGTERM -> grace -> SIGKILL on the runner's process group. The runner's
            # installed shutdown handler tears down the trial process groups and writes
            # status.json(143). cleanup_confirmed reports the runner group is gone, not
            # a guarantee about trial descendants (those are handled by the runner's
            # handler, or by the next launch's stale reaper on a SIGKILL escalation).
            confirmed = kill_stale_group(
                handle.pid,
                handle.pid_starttime,
                pgid=handle.pgid,
            )
            if confirmed:
                # If escalation to SIGKILL killed the runner before it recorded a
                # graceful 143, attribute this operator-initiated stop as cancelled
                # so the state below isn't a misleading 'failed'. No-op otherwise.
                self._runs.mark_cancelled_if_unrecorded(handle)
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
            atomic_write_bytes(snapshot_path, data)
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
        if from_phase is not None:
            cmd += ["--from-phase", from_phase]
        # Open the log here, hand the fd to the child, then close our copy. The
        # child keeps it. stdin is /dev/null so the runner never blocks on input.
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(  # noqa: S603 - argv list, no shell, server-controlled
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own session/pgid; survives restart; signal as a group
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
        )


F = TypeVar("F", bound=Callable[..., Any])


def _safe_tool(fn: F) -> F:
    """Translate exceptions into redacted tool errors.

    ``McpToolError`` -> re-raised as ``ValueError`` with its safe message.
    FastMCP's low-level handler serializes tool exceptions as
    ``CallToolResult(isError=True)``. Anything else -> logged to stderr and
    replaced with a generic message so an unexpected error (e.g. an OSError
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

    FastMCP does not currently expose a public switch for closed input schemas,
    so this intentionally patches the generated tool metadata. Keep the version
    cap and startup verification together with this hook: if SDK internals move,
    ``_verify_strict_tool_inputs`` must fail before serving permissive tools.
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


def _run_and_monitor_prompt_text() -> str:
    """Return the reusable agent workflow prompt served over MCP.

    :return str: Safe run-and-monitor instructions for MCP clients that support prompts.
    """
    return """You have access to a local phasesweep MCP server. Use it to operate only the human-curated experiment catalog exposed by the server.

Start by calling phasesweep_list_experiments, then call phasesweep_validate_config for the experiment id you plan to use. Do not ask for config paths, storage URLs, workdirs, commands, environment variables, or run-control settings; the catalog is the authority for those.

If asked to run a sweep, call phasesweep_launch_sweep with the catalog experiment id. Use from_phase only when explicitly asked to resume from a phase or when earlier phase winners are already confirmed. After launch, poll phasesweep_get_status by run_id until the run is succeeded, failed, or cancelled.

Use phasesweep_get_winners with the same run_id to summarize completed phase winners after a launched sweep. Treat returned metric values as experiment summaries and sampled params as user-visible hyperparameters, not secrets. Do not inspect raw datasets, target/dependent-variable columns, validation labels, predictions, trainer logs, raw result files, W&B dashboards, or per-trial metric histories unless explicitly asked for that separate work.

When recommending a next manual experiment, base the recommendation on MCP outputs: catalog descriptions, phase shape, status counts, exposed winner metrics, and sampled params. Do not change the objective metric, extractor, trainer command, search space, constraints, gates, storage, workdir, environment, or safety waivers unless explicitly asked for config-authoring help.

Use phasesweep_cancel_sweep only when explicitly asked to stop a run, or when stopping is clearly necessary to prevent an unwanted active sweep."""


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
        name=TOOL_GET_WINNERS,
        annotations=_read_annotations("Get Winners"),
        structured_output=True,
    )
    @_safe_tool
    def get_winners(
        experiment_id: MaybeExperimentId = None,
        run_id: MaybeRunId = None,
    ) -> GetWinnersResult:
        """Return the winning sampled hyperparameters per completed phase: trial number, metric, params, gate status, and completeness. Provide exactly one of experiment_id or run_id. Read-only.

        :param MaybeExperimentId experiment_id: Optional catalog experiment id whose winners should be read.
        :param MaybeRunId run_id: Optional detached run id whose snapshot should be read.
        :return GetWinnersResult: Structured winners payload.
        """
        return GetWinnersResult.model_validate(
            app.winners(experiment_id=experiment_id, run_id=run_id)
        )

    @mcp.tool(
        name=TOOL_LAUNCH_SWEEP,
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
    """Load the catalog, build the run store, and serve the six tools over stdio.

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
