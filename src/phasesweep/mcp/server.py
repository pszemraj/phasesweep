"""FastMCP adapter: the only module that imports the MCP SDK.

PhaseSweepMCP holds all logic and is SDK-free and unit-testable. build_server
wraps each method as a FastMCP tool; _safe_tool guarantees tool errors are
redacted. serve() loads the catalog, builds the store, and serves over stdio.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import hashlib
import importlib.util
import inspect
import logging
import os
import select
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, TypeVar, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from phasesweep.config import Experiment
from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.engine import generation_id_source, read_status, read_winners
from phasesweep.engine.guards import _experiment_semantic_fingerprint
from phasesweep.engine.read import ResultContext as ResultContextLiteral
from phasesweep.engine.state import Winner, WinnerSourceKind, _load_winner
from phasesweep.evidence.models import _ObjectiveEvidenceFields
from phasesweep.mcp import MCP_EXTRA_INSTALL_COMMAND, agent_prompt_text
from phasesweep.mcp.audit import AuditLogger
from phasesweep.mcp.config_snapshot import load_experiment_snapshot
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
    RunCapacityUnknownError,
    RunLaunchUnsettledError,
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
    PreparedRun,
    RunHandle,
    RunState,
    RunStore,
    identity_from_earlier_boot,
    write_status_file,
)
from phasesweep.mcp.snapshots import (
    McpPublicationState,
    RunResultSnapshot,
    capture_pre_generation_result_snapshot,
    parse_result_snapshot,
)
from phasesweep.mcp.time import parse_utc_iso
from phasesweep.runtime.files import ensure_private_dir, open_private_text
from phasesweep.runtime.process import kill_stale_group, read_boot_id, read_proc_starttime
from phasesweep.runtime.time import utc_now_iso

log = logging.getLogger("phasesweep.mcp.server")

SAFE_NAME_JSON_PATTERN = SAFE_NAME_PATTERN.pattern
TOOL_INSPECT_EXPERIMENT = "inspect_experiment"
TOOL_GET_LATEST_RUN = "get_latest_run"
TOOL_GET_RUN_STATUS = "get_run_status"
TOOL_GET_RUN_RESULTS = "get_run_results"
TOOL_LAUNCH_RUN = "launch_run"
TOOL_CANCEL_RUN = "cancel_run"
TOOL_AWAIT_RUN = "await_run"
TOOL_LIST_EXPERIMENTS = "list_experiments"
CATALOG_RESOURCE_URI = "phasesweep://catalog"
PROMPT_RUN_AND_MONITOR = "phasesweep_run_and_monitor"
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
_RUNNER_READY_BYTE = b"R"
_RUNNER_ACK_BYTE = b"A"


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


# Agent-facing tool descriptions. Descriptions are the one instruction channel
# present on every call even when the user loads no prompt, so each one chains
# to the next tool in the workflow by literal name.
DESCRIPTION_LIST_EXPERIMENTS = (
    "List the operator-approved experiments and their permitted actions. Call this first for "
    "discovery; follow next_cursor until null, then call inspect_experiment. Read-only: use only "
    "ids returned by this catalog."
)
DESCRIPTION_INSPECT_EXPERIMENT = (
    "Inspect one approved experiment's metric, permissions, phases, trial targets, samplers, "
    "inheritance, and search-space keys. Call after list_experiments and before launch_run; "
    "next_action is always null because only the user may authorize a launch. "
    "Read-only: launch_run separately rechecks config identity and refuses catalog drift."
)
DESCRIPTION_GET_LATEST_RUN = (
    "Return the most recently launched run for one experiment. Call only to recover a lost "
    "run_id; next use await_run for an active run or get_run_results for a terminal run. "
    "Read-only: found=false never authorizes launching a replacement."
)
DESCRIPTION_LAUNCH_RUN = (
    "Launch one approved experiment as a detached run and return its run_id. Call only after "
    "inspection and explicit user authorization; next call await_run with the returned run_id. "
    "Never retry permission or config-identity refusals; await named blockers before retrying a "
    "capacity refusal."
)
DESCRIPTION_GET_RUN_STATUS = (
    "Read process state and per-phase progress for exactly one experiment_id or run_id. Use as a "
    "single status check when await_run is unsuitable; next await an active run or read terminal "
    "results. Read-only: after launch always use run_id, and stop if recovery_required is true."
)
DESCRIPTION_GET_RUN_RESULTS = (
    "Return terminal per-phase winners, completeness, promotion context, metrics, gates, and "
    "policy-filtered sampled parameters. Call after a run becomes terminal; this ends the normal "
    "workflow. Read-only: treat redaction as policy and never infer trends or convergence from "
    "winner-only data."
)
DESCRIPTION_CANCEL_RUN = (
    "Request bounded cancellation of one launched run. Call only when the user explicitly asks; "
    "next read get_run_results if cancellation finishes cleanly. Never cancel automatically, and "
    "stop for operator recovery when recovery_required is true."
)
DESCRIPTION_AWAIT_RUN = (
    "Wait up to timeout_seconds for a launched run to change, become terminal, or require "
    "recovery. Call after launch_run and repeat while running; omit timeout_seconds for a "
    "client-safe 20-second wait, and request longer waits only when the client permits them. "
    "Next call get_run_results when terminal. Read-only: always reuse the run_id and stop "
    "immediately for recovery_required."
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
    Field(description=f"MCP run id returned by {TOOL_LAUNCH_RUN}.", pattern=SAFE_NAME_JSON_PATTERN),
]
MaybeRunId = Annotated[
    str | None,
    Field(
        description=f"MCP run id returned by {TOOL_LAUNCH_RUN}. Provide exactly one of experiment_id or run_id.",
        pattern=SAFE_NAME_JSON_PATTERN,
    ),
]
PhaseName = Annotated[
    str,
    Field(description="Phase name from the experiment config.", pattern=SAFE_NAME_JSON_PATTERN),
]
PublicationIntegrity = Annotated[
    McpPublicationState,
    Field(
        description=(
            "Whether this experiment's recorded publication still validates. 'ok': it does. "
            "'absent': nothing has ever published here, which is normal for a new experiment "
            "and for one whose first run has not finished. 'failed': a publication WAS "
            "recorded but its artifacts no longer validate, so every result field reads as "
            "if nothing published. On 'failed', report the corruption to the operator and do "
            "not launch a run against this experiment: a successful run advances the "
            "publication pointer past the corrupt result and nothing reports it afterwards. "
            "'permission_denied': the publication may be healthy, but this user cannot "
            "validate its owner-only evidence; expose no winners and ask the operator to "
            "re-read it as the publishing user or restore read permission. 'unknown': the "
            "terminal placeholder could not inspect publication state; expose no winners "
            "and report the run's snapshot-unavailable failure."
        )
    ),
]
ResultContext = Annotated[
    ResultContextLiteral,
    Field(
        description=(
            "Which config's semantics label this result. 'represented_generation': the "
            "metric name/goal and phase plan are the ones the represented generation "
            "recorded when it ran, so they describe the numbers as produced even if the "
            "config has since been edited. 'current_config': the represented result "
            "recorded no semantics of its own - nothing has published yet, or the "
            "publication predates result manifests - so the currently loaded config "
            "describes it. Under 'represented_generation', objective_evidence still "
            "falls back to the current extractor's assurance when the recorded summary "
            "carries none: treat evidence flags as unproven for such a result."
        )
    ),
]
PublishedConfigMatchesCurrent = Annotated[
    bool | None,
    Field(
        description=(
            "Whether the config that produced this result still matches the one a further "
            "run would execute. False means the config was edited after publication: the "
            "labels here are historical, and the current phases may not correspond to the "
            "published ones. Null when undeterminable - nothing published, or a "
            "publication that recorded no config fingerprint. Never treat null as 'no drift'."
        )
    ),
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
        description=f"Opaque pagination cursor returned by a previous {TOOL_LIST_EXPERIMENTS} call.",
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
NextAction = Literal[
    "list_experiments",
    "inspect_experiment",
    "get_latest_run",
    "get_run_status",
    "await_run",
    "get_run_results",
    "launch_run",
    "cancel_run",
]

# Environment variables that make the interpreter execute code before the
# detached runner's first statement. They are dropped from the child's
# environment (review v0.5.17 / blocker 9); PYTHONNOUSERSITE backs up the
# ``-s`` flag for the same reason.
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


class _ToolPayload(BaseModel):
    """Strict base for structured MCP tool results."""

    model_config = ConfigDict(extra="forbid")


class _ResultPayload(_ToolPayload):
    """Top-level tool result with an optional normal workflow transition."""

    next_action: NextAction | None = Field(
        default=None,
        description="Normal next tool for this result, or null when operator/user input is needed.",
    )


class ObjectiveEvidencePayload(_ToolPayload, _ObjectiveEvidenceFields):
    """Assurance properties enforced by the configured objective extractor.

    See :func:`phasesweep.evidence.models.objective_evidence_assurance` for
    exactly what each flag means and which runtime checks back it.
    """


class MetricPayload(_ToolPayload):
    """Optimization metric descriptor."""

    name: str = Field(description="Metric key extracted from trial output.")
    goal: Literal["minimize", "maximize"] = Field(description="Optimization direction.")
    objective_evidence: ObjectiveEvidencePayload


class CapabilitiesPayload(_ToolPayload):
    """Catalog-authorized actions for one experiment."""

    launch: bool = Field(description="Whether the agent may launch this experiment.")
    cancel: bool = Field(description="Whether the agent may cancel its launched runs.")
    resume_from_phase: bool = Field(
        description="Whether launch may use from_phase after earlier winners exist."
    )


class ExperimentSummaryPayload(_ToolPayload):
    """Path-free catalog entry summary."""

    id: ExperimentId
    description: str = Field(description="Operator-authored catalog description.")
    phases: list[PhaseName] = Field(description="Declared phases, in execution order.")
    metric: MetricPayload
    capabilities: CapabilitiesPayload


class ListExperimentsResult(_ResultPayload):
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
    n_trials: int = Field(
        ge=1,
        description=(
            "Target number of terminal attempts; COMPLETE, FAIL, and PRUNED trials all count."
        ),
    )
    sampler: str = Field(description="Sampler type only; sampler internals stay in the config.")
    inherits: list[PhaseName] = Field(description="Parent phases inherited by this phase.")
    search_space: list[str] = Field(description="Search-space keys only, never ranges or values.")


class InspectExperimentResult(_ResultPayload):
    """Structured output for inspect_experiment."""

    experiment_id: ExperimentId
    metric: MetricPayload
    capabilities: CapabilitiesPayload
    phases: list[PhaseValidationPayload]


class RunPayload(_ToolPayload):
    """Agent-visible run process state."""

    run_id: RunId
    state: RunState
    started_at: str = Field(description="UTC ISO-8601 launch timestamp.")
    recovery_required: bool = Field(
        description=(
            "True when the agent must stop monitoring and report an unresolved launch, "
            "uncertain cleanup, or interrupted snapshot finalization to the operator. "
            "A launch handoff observed mid-transition can still settle."
        )
    )
    failure: FailurePayload | None = Field(
        default=None,
        description=(
            "Safe actionable failure category and next action; null when no actionable "
            "failure is attached."
        ),
    )


class GetLatestRunResult(_ResultPayload):
    """Newest durable run handle for one experiment."""

    experiment_id: ExperimentId
    found: bool = Field(description="Whether any MCP run is recorded for this experiment.")
    run: RunPayload | None


class PhaseStatusPayload(_ToolPayload):
    """Per-phase status without filesystem paths."""

    phase: PhaseName
    trials: dict[Literal["WAITING", "RUNNING", "COMPLETE", "PRUNED", "FAIL"], int] = Field(
        description="Dense cumulative study counts; every state key is always present."
    )
    running_trials_total: int = Field(ge=0, description="Cumulative RUNNING study rows.")
    target_terminal_trials: int = Field(ge=0, description="Configured terminal-trial target.")
    completed_trials_total: int = Field(ge=0, description="Cumulative COMPLETE study rows.")
    terminal_trials_total: int = Field(
        ge=0, description="Cumulative COMPLETE + PRUNED + FAIL study rows."
    )
    terminal_trials_before_run: int = Field(
        ge=0, description="Terminal rows that predate the represented generation."
    )
    attempts_launched_this_run: int = Field(
        ge=0, description="All attempts owned by the represented generation."
    )
    terminal_trials_this_run: int = Field(
        ge=0, description="Terminal attempts owned by the represented generation."
    )
    remaining_trials: int = Field(
        ge=0, description="Additional terminal trials needed to reach the configured target."
    )
    target_already_satisfied: bool = Field(
        description="Whether pre-run terminal history already met the configured target."
    )
    winner_present: bool = Field(description="Whether this phase has a winner artifact.")
    published_study_unavailable: bool = Field(
        description=(
            "Whether a current published phase's local trial identity could not be matched. When "
            "trial_data_available is true, its published trial is confirmed missing or replaced: "
            "restore the original ledger/study or use a new experiment identity; "
            "earlier phases may load validated saved winners via from_phase. When "
            "trial_data_available is false, inspection failed: a run can report "
            "cleanup_uncertain and require operator "
            "recovery before further MCP launches."
        )
    )
    trial_data_available: bool = Field(
        description=(
            "Whether trial counts are known, including confirmed absent or empty studies. "
            "False means inspection failed, a journal snapshot is incomplete, or no "
            "persistent observation is available; "
            "zero counts then do not establish that no trials exist."
        )
    )


class GetRunStatusResult(_ResultPayload):
    """Structured output for get_run_status."""

    experiment_id: ExperimentId
    result_source: ResultSource = Field(
        description=(
            "Where the result facts came from: the mutable shared study for a live/current "
            "read, an immutable terminal run snapshot, or an unavailable terminal-snapshot "
            "placeholder whose counts are explicitly untrusted."
        )
    )
    current_generation_id: str | None = Field(
        description=(
            "Most recent generation id known to this experiment; may be failed or "
            "in-progress. Null when no generation has ever started. Always the actual "
            "mutable pointer, never forced to a queried run_id."
        )
    )
    published_generation_id: str | None = Field(
        description=(
            "Last successfully published generation id. Null when nothing has published "
            "yet. Always the actual validated pointer, never forced to a queried run_id, "
            "and may differ from represented_generation_id."
        )
    )
    represented_generation_id: str | None = Field(
        description=(
            "The generation whose winner_present, summary_present, and this-run trial "
            "counts this payload shows: normally the queried run_id when one was given, "
            "otherwise published_generation_id. Null for a config-only unavailable placeholder."
        )
    )
    is_published: bool = Field(
        description=(
            "True only when represented_generation_id is not null and equals "
            "published_generation_id. A run_id whose own publication failed reports "
            "false here while still showing that generation's own winners. For a "
            "legacy pre-generation workdir every generation id is null and this "
            "instead reports whether its compatibility winner artifact exists."
        )
    )
    publication_integrity: PublicationIntegrity
    result_context: ResultContext
    published_config_matches_current: PublishedConfigMatchesCurrent
    result_phase_plan: list[PhaseName] = Field(
        description=(
            "Phase plan the represented generation published under, in execution order. "
            "Equals the phases list's names while the config is unchanged; after an edit "
            "it is the historical plan, and the phases list is the current one a further "
            "run would execute."
        )
    )
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


class AwaitRunResult(GetRunStatusResult):
    """Structured output for await_run: run status plus what ended the wait."""

    changed: bool = Field(
        description=(
            "Whether run state, recovery requirement, phase progress, or winner presence "
            "changed during the wait."
        )
    )
    reason: Literal["recovery_required", "terminal", "phase_completed", "timeout"] = Field(
        description=(
            "Why the wait returned: operator recovery is required, the run reached a "
            "terminal state, a phase gained a winner, or the timeout elapsed without one "
            "of those stop conditions. At timeout, changed may still be true when trial "
            "counts advanced."
        )
    )


class WinnerSourcePayload(_ToolPayload):
    """Concrete trial that supplies an exposed winner."""

    kind: WinnerSourceKind
    phase: PhaseName
    trial_number: int = Field(ge=0)
    study: str | None = None


class PromotionOutcomePayload(_ToolPayload):
    """Safe phase-promotion context for an exposed winner."""

    action: Literal["promote", "continue_baseline"]
    baseline_phase: PhaseName
    candidate_trial_number: int = Field(ge=0)
    candidate_metric: float
    baseline_trial_number: int = Field(ge=0)
    baseline_metric: float
    min_delta: float
    improvement: float | None


class WinnerPhasePayload(_ToolPayload):
    """Agent-visible phase winner."""

    phase: PhaseName
    winner_source: WinnerSourcePayload
    winner_generation: Literal["current_generation", "prior_generation", "unknown"]
    promotion: PromotionOutcomePayload | None = None
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


class GetRunResultsResult(_ResultPayload):
    """Structured output for get_run_results.

    Every result-scoped field describes the represented generation under the
    config that produced it: ``metric``, ``declared_phase_count``,
    ``missing_phases``, and ``all_phases_have_winners`` are measured against
    the historical phase plan, not the current one (review v0.5.16 /
    blocker 4). ``result_context`` and ``published_config_matches_current``
    disclose which config that was.
    """

    experiment_id: ExperimentId
    run_id: RunId | None
    result_source: ResultSource = Field(
        description=(
            "Where the result facts came from: the mutable shared study for a live/current "
            "read, an immutable terminal run snapshot, or an unavailable terminal-snapshot "
            "placeholder whose counts are explicitly untrusted."
        )
    )
    represented_generation_id: str | None = Field(
        description=(
            "Generation whose winners and completeness this payload represents. During a "
            "live experiment-scoped read this may be the in-flight run generation; null when "
            "there is no represented generation."
        )
    )
    publication_integrity: PublicationIntegrity
    result_context: ResultContext
    published_config_matches_current: PublishedConfigMatchesCurrent
    metric: MetricPayload
    declared_phase_count: int = Field(
        ge=0, description="Phases the represented generation's own plan declared."
    )
    winner_count: int = Field(ge=0)
    missing_phases: list[PhaseName] = Field(
        description="Phases of that same historical plan with no winner artifact."
    )
    all_phases_have_winners: bool
    phases: list[WinnerPhasePayload]
    failure: FailurePayload | None = None


class LaunchRunResult(_ResultPayload):
    """Structured output for launch_run."""

    run_id: RunId
    experiment_id: ExperimentId
    state: Literal["running"]


class CancelRunResult(_ResultPayload):
    """Structured output for cancel_run."""

    run_id: RunId
    state: RunState
    cleanup_confirmed: bool | None = Field(
        default=None,
        description=(
            "Whether runner cleanup was fully confirmed; null when the run was already terminal."
        ),
    )
    recovery_required: bool = Field(
        description="Whether this run requires operator attention after the cancellation call."
    )


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


def _run_next_action(run: RunPayload | None) -> NextAction | None:
    """Choose the normal follow-up for a returned run handle.

    :param RunPayload | None run: Agent-visible run state, when one exists.
    :return NextAction | None: Monitoring or result tool, or null when no automatic step is safe.
    """
    if run is None or run.recovery_required:
        return None
    if run.state == "running":
        return cast(NextAction, TOOL_AWAIT_RUN)
    return cast(NextAction, TOOL_GET_RUN_RESULTS)


def _status_next_action(result: GetRunStatusResult) -> NextAction | None:
    """Choose the normal follow-up for one status read.

    ``run`` is null only for an experiment-scoped read that found no live run:
    the experiment either never ran under this server or every run of it is
    already terminal. Deferring to :func:`_run_next_action` there would answer
    "nothing left to do" even when finished work is on disk, so a published
    winner steers the agent to the results tool instead. With nothing published
    and no winner anywhere there is genuinely nothing further to read, and null
    keeps its documented meaning.

    ``is_published`` is checked beside the per-phase winners because those
    phases are the *current* config's: after a phase rename none of them has a
    winner while the publication the results tool would return is intact, and
    answering "stop" there sends the agent away from a result it can read
    (review v0.5.16 / blocker 4).

    :param GetRunStatusResult result: Status payload whose transition is chosen.
    :return NextAction | None: Monitoring or result tool, or null when no
        automatic step is safe or useful.
    """
    if result.run is not None:
        return _run_next_action(result.run)
    if result.is_published or any(phase.winner_present for phase in result.phases):
        return cast(NextAction, TOOL_GET_RUN_RESULTS)
    return None


def _run_elapsed_seconds(store: RunStore, handle: RunHandle, state: str) -> int | None:
    """Compute wall seconds for a run: launch-to-now while running, total when terminal.

    Terminal runs use the runner-stamped ``ended_at`` in status.json. A
    terminal run with no readable endpoint reports ``None`` rather than a guess.

    ``RunStore`` refuses to load a handle whose ``started_at`` does not parse,
    so the start endpoint is established for every handle that reaches here.
    It is still re-checked rather than asserted: an assertion would be
    compiled out under ``python -O`` and the unparsed ``None`` would surface
    as a ``TypeError`` from the subtraction below instead of the documented
    "endpoints unknown" answer.

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
        :param AuditLogger | None audit: Optional audit sink override; defaults to state_dir/audit.jsonl.
        """
        self._registry = registry
        self._runs = runs
        self._audit = audit or AuditLogger(registry.state_dir / "audit.jsonl")

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
        """Record a successful state-changing tool call.

        :param str tool: MCP tool name.
        :param dict[str, Any] | None args: Safe agent-supplied arguments.
        :param dict[str, Any] | None resolved: Safe server-resolved identifiers.
        :param dict[str, Any] | None state_before: Safe state summary before the call.
        :param dict[str, Any] | None state_after: Safe state summary after the call.
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
        args: dict[str, Any] | None,
        exc: Exception,
        *,
        resolved: dict[str, Any] | None = None,
        state_before: dict[str, Any] | None = None,
    ) -> None:
        """Record a failed state-changing tool call.

        :param str tool: MCP tool name.
        :param dict[str, Any] | None args: Safe agent-supplied arguments.
        :param Exception exc: Exception that will be surfaced through the safe tool wrapper.
        :param dict[str, Any] | None resolved: Safe server-resolved identifiers known before failure.
        :param dict[str, Any] | None state_before: Safe state summary before the failure.
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

    def _run_payload(self, handle: RunHandle) -> dict[str, Any]:
        """Build one agent-visible run state with explicit recovery metadata.

        :param RunHandle handle: Persisted run handle to derive.
        :return dict[str, Any]: Run id, state, launch timestamp, and recovery flag.
        """
        state = self._runs.state(handle)
        return {
            "run_id": handle.run_id,
            "state": state,
            "started_at": handle.started_at,
            "recovery_required": self._runs.recovery_required(handle),
            "failure": self._run_failure_payload(handle, state=state),
        }

    def _run_failure_payload(
        self,
        handle: RunHandle,
        *,
        state: RunState | None = None,
    ) -> dict[str, Any] | None:
        """Return a validated safe terminal failure, never the raw exception text.

        :param RunHandle handle: Run handle whose recorded terminal status
            should be inspected.
        :param RunState | None state: Already-derived run state, when available.
        :return dict[str, Any] | None: The run's ``failure`` payload validated
            against :class:`FailurePayload` and dumped to JSON-safe types. A
            terminal run with no usable result snapshot receives a generated
            snapshot-unavailable failure; otherwise returns ``None`` when no
            valid failure was recorded.
        """
        terminal = self._runs.recorded_terminal_status(handle)
        if terminal is None:
            return None
        persisted: dict[str, Any] | None = None
        try:
            if terminal.get("failure") is not None:
                persisted = FailurePayload.model_validate(terminal["failure"]).model_dump(
                    mode="json", exclude_none=True
                )
        except ValidationError:
            pass
        if (
            state if state is not None else self._runs.state(handle)
        ) != "running" and parse_result_snapshot(terminal) is None:
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
        return persisted

    def status(self, *, experiment_id: str | None = None, run_id: str | None = None) -> dict:
        """Per-phase trial counts and winner presence plus the run process state.

        Provide either ``experiment_id`` (reports the current shared studies and
        live run, if any) or ``run_id`` (reports that specific run, using its
        frozen result snapshot once terminal). Raises if neither is given.

        :param str | None experiment_id: Optional catalog id for experiment-level status.
        :param str | None run_id: Optional detached run id for run-specific status.
        :return dict: Path-free status payload for the agent.
        """
        target_id, status, run, handle, result_source = self._read_status_target(
            experiment_id=experiment_id,
            run_id=run_id,
        )
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
                experiment_id=None,
                run_id=run_id,
            )
            read_seconds = time.monotonic() - read_started
            run = cast(dict[str, Any], run)
            snapshot = _await_snapshot(
                run["state"],
                run["recovery_required"],
                status,
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
        experiment_id: str | None,
        run_id: str | None,
    ) -> tuple[
        str,
        dict[str, Any],
        dict[str, Any] | None,
        RunHandle | None,
        ResultSource,
    ]:
        """Resolve a status target and read its current or frozen status payload.

        :param str | None experiment_id: Catalog id for a current experiment read.
        :param str | None run_id: Persisted run id for a run-specific read.
        :return tuple: Target id, status data, optional run state and handle,
            and result provenance.
        """
        target_id, experiment, run, handle = self._resolve_read_target(
            experiment_id=experiment_id,
            run_id=run_id,
            include_run=True,
        )
        snapshot, result_source = self._result_snapshot_view(experiment, handle)
        status = (
            self._snapshot_status_payload(
                target_id,
                snapshot,
                result_source=result_source,
            )
            if snapshot is not None
            else self._live_status_payload(
                target_id,
                experiment,
                handle,
            )
        )
        return target_id, status, run, handle, result_source

    def _catalog_comparison_experiment(self, experiment_id: str) -> Experiment | None:
        """Return the config a future catalog launch would execute, if available.

        :param str experiment_id: Catalog id associated with a current or frozen read.
        :return Experiment | None: Current catalog config, or ``None`` after removal.
        """
        try:
            return self._registry.get(experiment_id).experiment
        except UnknownExperimentError:
            return None

    def _live_status_payload(
        self,
        experiment_id: str,
        experiment: Experiment,
        handle: RunHandle | None,
    ) -> dict[str, Any]:
        """Read live status and leave catalog drift unknown after decataloging.

        ``read_status`` normally compares against its read config when no
        comparison config is supplied. A run snapshot remains the correct
        artifact locator after its catalog entry is removed, but it is not a
        current catalog config and therefore cannot support a drift verdict.

        :param str experiment_id: Catalog id associated with the read target.
        :param Experiment experiment: Current catalog config or live run snapshot config.
        :param RunHandle | None handle: Optional live run pinning the represented generation.
        :return dict[str, Any]: Path-free live status payload.
        """
        comparison = self._catalog_comparison_experiment(experiment_id)
        status = read_status(
            experiment,
            generation_id=handle.run_id if handle is not None else None,
            comparison_experiment=comparison,
        )
        if comparison is None:
            status["published_config_matches_current"] = None
        return status

    def _snapshot_status_payload(
        self,
        experiment_id: str,
        snapshot: RunResultSnapshot,
        *,
        result_source: ResultSource,
    ) -> dict[str, Any]:
        """Return frozen run facts with a config-only live drift comparison.

        A terminal run snapshot is the historical read authority for that run.
        Publication pointers and integrity stay as captured: consulting the
        live artifact tree here would make an intact snapshot unreadable after
        relocation, and a later generation's damage could erase this run's
        already-validated winners. Config drift is the sole live field and
        compares the frozen represented-config fingerprint with the current
        catalog.

        :param str experiment_id: Catalog id associated with the run.
        :param RunResultSnapshot snapshot: Validated frozen result snapshot.
        :param ResultSource result_source: Snapshot provenance, including the
            unavailable placeholder case.
        :return dict[str, Any]: Frozen status with a current config-drift verdict.
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
        status: dict[str, Any],
        run: dict[str, Any] | None,
        handle: RunHandle | None,
        result_source: ResultSource,
    ) -> dict[str, Any]:
        """Build the common path-free response for status and await_run.

        :param str target_id: Experiment id represented by ``status``.
        :param dict[str, Any] status: Current or frozen engine status payload.
        :param dict[str, Any] | None run: Optional derived detached-run state.
        :param RunHandle | None handle: Optional already-resolved detached-run handle.
        :param ResultSource result_source: Current shared state or frozen run snapshot.
        :return dict[str, Any]: Agent-safe status response.
        """
        elapsed_seconds = None
        if run is not None and handle is not None:
            elapsed_seconds = _run_elapsed_seconds(self._runs, handle, run["state"])
        return status_payload(
            target_id,
            status,
            run,
            result_source=result_source,
            elapsed_seconds=elapsed_seconds,
        )

    def _resolve_read_target(
        self,
        *,
        experiment_id: str | None,
        run_id: str | None,
        include_run: bool,
    ) -> tuple[str, Experiment, dict[str, Any] | None, RunHandle | None]:
        """Resolve status/winner reads to the catalog config or immutable run snapshot.

        :param str | None experiment_id: Catalog id for current experiment-level reads.
        :param str | None run_id: Persisted run id for immutable run-specific reads.
        :param bool include_run: Whether to include live run state in the returned payload.
        :return tuple: Target id, parsed experiment, optional run payload, and handle.
        :raises McpToolError: If neither or both of ``experiment_id`` and
            ``run_id`` were provided.
        :raises UnknownRunError: If ``run_id`` names no persisted run.
        :raises UnknownExperimentError: If ``experiment_id`` is not cataloged.
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
            experiment = self._load_run_experiment(handle)
            run = None
            if include_run:
                run = self._run_payload(handle)
            return handle.experiment_id, experiment, run, handle

        assert experiment_id is not None
        reg = self._registry.get(experiment_id)
        live = self._runs.live_run_for(experiment_id)
        experiment = self._load_run_experiment(live) if live is not None else reg.experiment
        run = self._run_payload(live) if include_run and live is not None else None
        return reg.id, experiment, run, live

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
            )
        except (OSError, ValueError) as exc:
            log.info("invalid config snapshot for run=%s: %s", handle.run_id, exc)
            raise RunSnapshotUnavailableError(handle.run_id) from None

    def winners(
        self, experiment_id: str | None = None, *, run_id: str | None = None
    ) -> dict[str, Any]:
        """Return the winning hyperparameters per completed phase.

        Provide ``experiment_id`` for the current cataloged experiment, or ``run_id``
        to read winners from the immutable config and terminal result snapshots
        that run recorded. Run-specific reads must not drift when the cataloged
        config changes or a later run resumes the shared studies.

        The payload carries ``publication_integrity`` so an empty winner list
        is never ambiguous: ``"absent"`` means nothing has published yet,
        ``"failed"`` means a recorded publication no longer validates;
        ``"permission_denied"`` means this user cannot validate it without
        implying corruption. Both expose no winners and require operator action.

        Every label describing the winners -- metric name and goal, objective
        evidence, and the declared phase plan the completeness fields are
        measured against -- comes from the represented generation's own
        recorded semantics, never from the current catalog config (review
        v0.5.16 / blocker 4). Reading them from the current config relabeled
        historical numbers with a metric they were never optimized for,
        asserted evidence guarantees the run never had, and, after a phase
        rename, hid the published winner while reporting the new name as
        missing. ``result_context`` and ``published_config_matches_current``
        report which config supplied the labels and whether it still matches
        the one a further run would execute. Visibility is unchanged: the
        current policy still decides which historical *values* are redacted.

        :param str | None experiment_id: Optional catalog experiment id whose winners should be read.
        :param str | None run_id: Optional detached run id whose snapshot should be read.
        :return dict[str, Any]: Path-free winners payload for the agent.
        """
        target_id, experiment, _run, handle = self._resolve_read_target(
            experiment_id=experiment_id,
            run_id=run_id,
            include_run=False,
        )
        snapshot, result_source = self._result_snapshot_view(experiment, handle)
        if snapshot is not None:
            # The frozen snapshot already carries the represented generation's
            # own metric, phase plan, and drift verdict, captured under the
            # config that run executed -- so this branch reads its labels from
            # the same place the live branch does.
            status = self._snapshot_status_payload(
                target_id,
                snapshot,
                result_source=result_source,
            )
            winner_views = (
                []
                if status["publication_integrity"] in {"failed", "permission_denied", "unknown"}
                else snapshot.winner_views()
            )
        else:
            # Resolve the represented generation once via read_status, then
            # reuse that exact id for read_winners: two independent pointer
            # resolutions here could otherwise mix identities from different
            # moments (review v0.5.15 / blocker 3). represented_generation_id
            # is the queried run_id itself when pinned, else the captured
            # published id -- never the live current pointer, which is what
            # this used to (incorrectly) label winners with.
            status = self._live_status_payload(
                target_id,
                experiment,
                handle,
            )
            # Enumerate the plan that generation published under, not the one
            # the config declares now: a phase renamed since publication used
            # to drop its winner from this payload entirely while reporting
            # the new name as missing (review v0.5.16 / blocker 4).
            winner_views = read_winners(
                experiment,
                generation_id=status["represented_generation_id"],
                phase_names=status["result_phase_plan"],
            )
            if status["publication_integrity"] in {"failed", "permission_denied", "unknown"}:
                winner_views = []
        represented_generation_id: str | None = status["represented_generation_id"]
        publication_integrity: McpPublicationState = status["publication_integrity"]
        authority_handle = handle
        authority_unreadable = False
        if authority_handle is None and represented_generation_id is not None:
            authority_handle = self._runs.get(represented_generation_id)
            authority_unreadable = authority_handle is None and (
                self._runs.handle_exists(represented_generation_id)
                or self._runs.run_evidence_exists(represented_generation_id)
                or generation_id_source(experiment, represented_generation_id) == "caller"
            )
        if authority_unreadable:
            # The represented generation WAS an MCP-launched run, but its
            # frozen launch authority cannot be read: the handle no longer
            # decodes, or only sibling per-run files survive its deletion, or
            # -- proof that outlives the state dir itself -- the generation's
            # own reproducibility record says its id was caller-granted while
            # no handle answers for it (PR #5 review / P2 missing-handle
            # authority). Fall back to the narrowest policy instead of the
            # current catalog's, which may be wider than the launch grant.
            visible_params: VisibleParamsPolicy = "none"
        else:
            visible_params = self._effective_visible_params(target_id, authority_handle)
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
        result["failure"] = self._run_failure_payload(handle) if handle is not None else None
        return result

    def _effective_visible_params(
        self,
        experiment_id: str,
        handle: RunHandle | None,
    ) -> VisibleParamsPolicy:
        """Resolve current or launch/current-intersected winner visibility.

        :param str experiment_id: Catalog id associated with the represented results.
        :param RunHandle | None handle: MCP run whose generation is represented, if any.
        :return VisibleParamsPolicy: Effective sampled-parameter visibility.
        """
        try:
            current_policy = self._registry.get(experiment_id).visible_params
        except UnknownExperimentError:
            return "none"
        if handle is None:
            return current_policy
        launch_policy = handle.visible_params_at_launch or "none"
        return intersect_visible_params(launch_policy, current_policy)

    def _result_snapshot_view(
        self,
        experiment: Experiment,
        handle: RunHandle | None,
    ) -> tuple[RunResultSnapshot | None, ResultSource]:
        """Resolve one run result without falling back to mutable terminal state.

        A failed terminal snapshot is represented by the same config-only
        unavailable-data shape used for pre-generation failures. This lets
        monitoring return the terminal run and an actionable failure while
        every phase count remains explicitly untrusted.

        :param Experiment experiment: Exact catalog or saved run configuration.
        :param RunHandle | None handle: Optional detached run being read.
        :return tuple: Optional result view and its agent-visible provenance.
        """
        if handle is None:
            return None, "current_shared_study"
        try:
            snapshot = self._terminal_result_snapshot(handle)
        except RunResultSnapshotUnavailableError:
            placeholder = capture_pre_generation_result_snapshot(experiment)
            return (
                RunResultSnapshot.model_validate(placeholder),
                "terminal_snapshot_unavailable",
            )
        return (
            snapshot,
            "frozen_run_snapshot" if snapshot is not None else "current_shared_study",
        )

    def _terminal_result_snapshot(self, handle: RunHandle) -> RunResultSnapshot | None:
        """Return a live run's current view or require its frozen terminal snapshot.

        :param RunHandle handle: Resolved run whose terminal status should be inspected.
        :return RunResultSnapshot | None: Validated snapshot, or ``None`` while
            the run remains non-terminal.
        :raises RunResultSnapshotUnavailableError: Terminal status exists but
            its immutable result snapshot is absent or invalid.
        """
        terminal_status = self._runs.recorded_terminal_status(handle)
        if terminal_status is None:
            return None
        if terminal_status.get("result_snapshot_state") == "pending":
            return None
        snapshot = parse_result_snapshot(terminal_status)
        if snapshot is not None:
            return snapshot
        if self._runs.state(handle) == "running":
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

    def launch(self, experiment_id: str, from_phase: str | None = None) -> dict[str, Any]:
        """Start the sweep as a detached background run; return its run_id.

        Refuses if launch is not permitted, if a ``from_phase`` resume is not
        ready (an earlier phase has no winner), if this experiment already has a
        live run, or if the server is at its max_concurrent_runs cap.

        :param str experiment_id: Catalog experiment id to launch.
        :param str | None from_phase: Optional phase to resume from after earlier winners exist.
        :return dict[str, Any]: Launch result containing run id, experiment id, and running state.
        :raises UnknownExperimentError: If ``experiment_id`` is not cataloged.
        :raises PermissionDeniedError: If the entry does not allow launching, or
            does not allow ``from_phase`` resumes.
        :raises InvalidPhaseError: If ``from_phase`` names no phase of this experiment.
        :raises ResumeNotReadyError: If an earlier phase has no compatible winner.
        :raises LaunchInProgressError: If another launch currently holds the launch lock.
        :raises ExperimentBusyError: If this experiment already has a live run.
        :raises ConcurrencyLimitError: If the server is at ``max_concurrent_runs``.
        :raises ConfigChangedError: If the cataloged config changed since startup.
        :raises RuntimeError: If no unused run id could be minted, or the spawned
            runner has no Linux ``/proc`` start time to make cancellation PID-reuse safe.
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
                    self._runs.clear_pre_spawn_orphan(abandoned_run_id)
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
                    if handle.pid_starttime is None:
                        raise RuntimeError(
                            "spawned runner has no Linux /proc start time; refused launch because "
                            "later cancellation could not distinguish PID reuse"
                        )
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
                        True if handle is None else self._terminate_failed_spawn(handle, launch_exc)
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
            result = {"run_id": handle.run_id, "experiment_id": experiment_id, "state": "running"}
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
        :raises UnknownRunError: If ``run_id`` names no persisted run.
        :raises PermissionDeniedError: If the run's entry did not allow cancel at launch time.
        :raises RunLaunchUnsettledError: If the launch has not yet persisted a
            process identity, so there is nothing safe to signal.
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
            recovery_required = self._runs.recovery_required(handle)
            state_before = {
                "run_state": before,
                "recovery_required": recovery_required,
            }
            if before == "running" and handle.launch_state == "launching":
                # No PID/PGID is durable yet. Signalling an empty identity and
                # returning would let the launch continue immediately after a
                # misleading cancellation response from a second server.
                raise RunLaunchUnsettledError(run_id)
            after: RunState
            if before != "running":
                after = before
                confirmed: bool | None = None
            else:
                # Concurrent callers deliberately converge without a cancel
                # lock: marker writes are idempotent, kill_stale_group verifies
                # the persisted process identity and treats an already-gone
                # group as confirmed, terminal status is runner-authoritative,
                # and marker removal uses missing_ok.
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
                # A recorded boot id from an earlier boot proves the runner and
                # its trial descendants cannot exist, so signalling the saved
                # PGID would only reach whatever inherited those numbers after
                # the reboot. Skip the signal and treat cleanup as confirmed.
                earlier_boot = identity_from_earlier_boot(identity.boot_id)
                runner_group_gone = earlier_boot or kill_stale_group(
                    identity.pid,
                    identity.pid_starttime,
                    pgid=identity.pgid,
                    grace_seconds=30.0,
                )
                terminal_status = self._runs.recorded_terminal_status(handle)
                confirmed = runner_group_gone and (
                    earlier_boot
                    or (
                        terminal_status is not None
                        and terminal_status.get("cleanup_confirmed") is True
                    )
                )
                if confirmed:
                    self._runs.clear_cleanup_uncertain(handle)
                after = self._runs.state(handle)
                recovery_required = self._runs.recovery_required(handle)
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
        state_after = {
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
        """Verify that every earlier phase has a compatible persisted winner.

        :param RegisteredExperiment reg: Registered experiment being resumed.
        :param str from_phase: Requested phase to resume from.
        :raises ResumeNotReadyError: If an earlier phase has no persisted
            winner, or its stored winner is unreadable or incompatible with the
            current config.
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
            write_status_file(
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
    ) -> bool:
        """Terminate a spawned runner whose durable bookkeeping failed.

        :param RunHandle handle: Spawned runner identity available in memory.
        :param BaseException original_error: Launch failure preserved for diagnostics.
        :return bool: Whether the runner process group is confirmed gone.
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
        :raises _SpawnBookkeepingError: Post-spawn identity bookkeeping fails; the
            exception records whether cleanup of the spawned process group was confirmed.
        :return RunHandle: Runner-validated spawned handle.
        """
        pending = preparation.handle
        run_id = pending.run_id
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
        # Pre-identity trust boundary (review v0.5.17 / blocker 9). Everything
        # between exec and the runner's own durable handle write runs before
        # this server can name the process it just created. Spawning with the
        # experiment's project directory as cwd put that window inside the
        # project's reach: interpreter startup would import a project-local
        # `phasesweep/` shadow package or `sitecustomize.py`, and a
        # PYTHONPATH/PYTHONHOME/PYTHONSTARTUP-injected module would run
        # earlier still. Any of it could fork+setsid out of the process group
        # recorded below, after which "cleanup confirmed" would be a claim
        # about an empty group. So the child starts in the server-owned state
        # directory, with -P/-s and a sanitized environment, and receives the
        # project directory as an explicit argument it applies only after its
        # identity is on disk.
        spawn_cwd = self._neutral_spawn_cwd()
        # Open the log here, hand the fd to the child, then close our copy. The
        # child keeps it. stdin is /dev/null so the runner never blocks on input.
        # Once Popen returns, every later operation is inside the BaseException
        # boundary: shutdown interrupts are ownership failures too, not proof
        # that no child was created.
        proc: subprocess.Popen | None = None
        handle: RunHandle | None = None
        pid_starttime: int | None = None
        boot_id: str | None = None
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
                handle = RunHandle(
                    run_id=run_id,
                    experiment_id=reg.id,
                    config_sha256=reg.config_sha256,
                    pid=proc.pid,
                    pgid=proc.pid,
                    pid_starttime=None,
                    started_at=pending.started_at,
                    launch_state="spawned",
                    allow_cancel=pending.allow_cancel,
                    visible_params_at_launch=pending.visible_params_at_launch,
                    boot_id=None,
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
                readable, _, _ = select.select(
                    [ready_read],
                    [],
                    [],
                    _RUNNER_READY_TIMEOUT_SECONDS,
                )
                ready = os.read(ready_read, 1) if readable else b""
                if ready != _RUNNER_READY_BYTE:
                    raise RuntimeError(
                        "detached runner did not persist its launch receipt before launch"
                    )
                persisted = self._runs.get(run_id)
                if persisted != handle:
                    raise RuntimeError("detached runner launch receipt did not match its process")
            if os.write(ack_write, _RUNNER_ACK_BYTE) != len(_RUNNER_ACK_BYTE):
                raise RuntimeError("could not acknowledge the detached runner launch")
            acknowledged_fd = ack_write
            ack_write = -1
            with contextlib.suppress(OSError):
                os.close(acknowledged_fd)
        except BaseException as exc:
            if proc is None:
                # Opening the log or Popen itself failed: no child exists.
                raise
            cleanup_handle = RunHandle(
                run_id=run_id,
                experiment_id=reg.id,
                config_sha256=reg.config_sha256,
                pid=proc.pid,
                pgid=proc.pid,
                pid_starttime=pid_starttime,
                started_at=pending.started_at,
                launch_state="spawned",
                allow_cancel=pending.allow_cancel,
                visible_params_at_launch=pending.visible_params_at_launch,
                boot_id=boot_id,
            )
            cleanup_confirmed = self._terminate_failed_spawn(cleanup_handle, exc)
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
        there would strand a live detached runner: ``_resolve_read_target``
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


F = TypeVar("F", bound=Callable[..., Any])


def _raise_safe_tool_error(tool_name: str, exc: Exception) -> NoReturn:
    """Translate one implementation exception into an agent-safe tool error.

    :param str tool_name: Tool function used in the generic error message.
    :param Exception exc: Exception raised by the implementation.
    :raises ValueError: Always, with either a domain message or generic text.
    """
    if isinstance(exc, McpToolError):
        # Immediate invocation failures use MCP's isError text channel. Durable
        # run failures are separate successful status reads with FailurePayload;
        # FastMCP 1.27 does not attach structured content to raised tool errors.
        raise ValueError(str(exc)) from None
    log.exception("unhandled error in tool %s", tool_name, exc_info=exc)
    raise ValueError(
        f"internal server error in {tool_name}; report it to the operator "
        "and do not retry immediately"
    ) from None


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
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Invoke an async tool and redact any exception.

            :param Any args: Positional arguments passed through to the wrapped tool.
            :param Any kwargs: Keyword arguments passed through to the wrapped tool.
            :return Any: Awaited tool result.
            """
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                _raise_safe_tool_error(fn.__name__, exc)

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        """Invoke a synchronous tool and redact any exception.

        :param Any args: Positional arguments passed through to the wrapped tool.
        :param Any kwargs: Keyword arguments passed through to the wrapped tool.
        :return Any: Wrapped tool result.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            _raise_safe_tool_error(fn.__name__, exc)

    return sync_wrapper  # type: ignore[return-value]


def _tool_annotations(
    title: str,
    *,
    read_only: bool = True,
    destructive: bool = False,
    idempotent: bool = True,
    open_world: bool = False,
) -> Any:
    """Return MCP annotations for one tool's side-effect contract.

    :param str title: Human-readable tool title.
    :param bool read_only: Whether the tool leaves server state unchanged.
    :param bool destructive: Whether the tool can perform destructive side effects.
    :param bool idempotent: Whether repeated calls have the same side effects.
    :param bool open_world: Whether the tool interacts beyond the local server boundary.
    :return Any: MCP ``ToolAnnotations`` instance.
    """
    from mcp.types import ToolAnnotations

    return ToolAnnotations(
        title=title,
        readOnlyHint=read_only,
        destructiveHint=destructive,
        idempotentHint=idempotent,
        openWorldHint=open_world,
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

    for tool_name in (TOOL_GET_RUN_STATUS, TOOL_GET_RUN_RESULTS):
        tool = mcp._tool_manager.get_tool(tool_name)
        if tool is not None:
            tool.parameters["oneOf"] = [
                {"required": ["experiment_id"], "not": {"required": ["run_id"]}},
                {"required": ["run_id"], "not": {"required": ["experiment_id"]}},
            ]
    _verify_strict_tool_inputs(mcp)


def _verify_strict_tool_inputs(mcp: Any) -> None:
    """Fail startup if FastMCP internals did not keep the strict schemas.

    :param Any mcp: Configured FastMCP server whose registered tools are inspected.
    :raises RuntimeError: If any tool still accepts undeclared input keys, a
        read tool is unregistered, or a read tool lost its exactly-one-of schema.
    """
    expected_one_of = [
        {"required": ["experiment_id"], "not": {"required": ["run_id"]}},
        {"required": ["run_id"], "not": {"required": ["experiment_id"]}},
    ]
    for tool in mcp._tool_manager.list_tools():
        if tool.parameters.get("additionalProperties") is not False:
            raise RuntimeError(f"MCP tool {tool.name!r} accepts undeclared input keys")
    for tool_name in (TOOL_GET_RUN_STATUS, TOOL_GET_RUN_RESULTS):
        tool = mcp._tool_manager.get_tool(tool_name)
        if tool is None:
            raise RuntimeError(f"MCP tool {tool_name!r} was not registered")
        if tool.parameters.get("oneOf") != expected_one_of:
            raise RuntimeError(f"MCP tool {tool_name!r} lost its exactly-one-of schema")


def build_server(app: PhaseSweepMCP) -> Any:
    """Construct the FastMCP server.

    The SDK is imported lazily so non-server code paths (and most tests) do not
    require the ``mcp`` package.

    :param PhaseSweepMCP app: SDK-free tool implementation to expose.
    :return Any: Configured FastMCP server.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("phasesweep", instructions=agent_prompt_text(strip=True))

    @mcp.tool(
        name=TOOL_LIST_EXPERIMENTS,
        description=DESCRIPTION_LIST_EXPERIMENTS,
        annotations=_tool_annotations("List Experiments"),
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
        result = ListExperimentsResult.model_validate(
            app.list_experiments(limit=limit, cursor=cursor)
        )
        return result.model_copy(
            update={
                "next_action": (
                    TOOL_LIST_EXPERIMENTS
                    if result.next_cursor is not None
                    else TOOL_INSPECT_EXPERIMENT
                    if result.experiments
                    else None
                )
            }
        )

    @mcp.tool(
        name=TOOL_INSPECT_EXPERIMENT,
        description=DESCRIPTION_INSPECT_EXPERIMENT,
        annotations=_tool_annotations("Inspect Experiment"),
        structured_output=True,
    )
    @_safe_tool
    async def inspect_experiment(experiment_id: ExperimentId) -> InspectExperimentResult:
        """Return the phase structure (names, trial counts, samplers, inherited phases, search-space keys) for an experiment. Read-only; launches nothing.

        :param ExperimentId experiment_id: Catalog experiment id to inspect.
        :return InspectExperimentResult: Structured inspection payload with a
            null ``next_action``; the user, not the server, authorizes a launch.
        """
        # next_action stays null (the field default): launching requires explicit
        # user authorization, so the server never proposes launch_run itself.
        return InspectExperimentResult.model_validate(
            await asyncio.to_thread(app.validate, experiment_id)
        )

    @mcp.tool(
        name=TOOL_GET_LATEST_RUN,
        description=DESCRIPTION_GET_LATEST_RUN,
        annotations=_tool_annotations("Get Latest Run"),
        structured_output=True,
    )
    @_safe_tool
    async def get_latest_run(experiment_id: ExperimentId) -> GetLatestRunResult:
        """Return the newest recorded MCP run for one experiment, already selected by launch time.

        :param ExperimentId experiment_id: Catalog experiment id whose latest run is needed.
        :return GetLatestRunResult: One computed run handle or ``found=false``.
        """
        result = GetLatestRunResult.model_validate(
            await asyncio.to_thread(app.latest_run, experiment_id)
        )
        return result.model_copy(update={"next_action": _run_next_action(result.run)})

    @mcp.tool(
        name=TOOL_GET_RUN_STATUS,
        description=DESCRIPTION_GET_RUN_STATUS,
        annotations=_tool_annotations("Get Run Status"),
        structured_output=True,
    )
    @_safe_tool
    async def get_run_status(
        experiment_id: MaybeExperimentId = None,
        run_id: MaybeRunId = None,
    ) -> GetRunStatusResult:
        """Per-phase trial counts and winner presence, plus the run process state. Provide exactly one of experiment_id or run_id. Read-only.

        :param MaybeExperimentId experiment_id: Optional catalog experiment id for experiment-level status.
        :param MaybeRunId run_id: Optional detached run id for run-specific status.
        :return GetRunStatusResult: Structured status payload.
        """
        result = GetRunStatusResult.model_validate(
            await asyncio.to_thread(app.status, experiment_id=experiment_id, run_id=run_id)
        )
        return result.model_copy(update={"next_action": _status_next_action(result)})

    @mcp.tool(
        name=TOOL_AWAIT_RUN,
        description=DESCRIPTION_AWAIT_RUN,
        annotations=_tool_annotations("Await Run"),
        structured_output=True,
    )
    @_safe_tool
    async def await_run(
        run_id: RunId,
        timeout_seconds: AwaitTimeoutSeconds = AWAIT_DEFAULT_TIMEOUT_SECONDS,
    ) -> AwaitRunResult:
        """Block until a launched run changes or the timeout elapses, then return its status. Read-only.

        :param RunId run_id: Detached run id to wait on.
        :param AwaitTimeoutSeconds timeout_seconds: Seconds to wait before returning current status.
        :return AwaitRunResult: Structured status payload plus changed and reason.
        """
        result = AwaitRunResult.model_validate(
            await app.await_run(run_id, timeout_seconds=timeout_seconds)
        )
        next_action: NextAction | None
        if result.reason == "terminal":
            next_action = cast(NextAction, TOOL_GET_RUN_RESULTS)
        elif result.reason == "recovery_required":
            next_action = None
        else:
            next_action = cast(NextAction, TOOL_AWAIT_RUN)
        return result.model_copy(update={"next_action": next_action})

    @mcp.tool(
        name=TOOL_GET_RUN_RESULTS,
        description=DESCRIPTION_GET_RUN_RESULTS,
        annotations=_tool_annotations("Get Run Results"),
        structured_output=True,
    )
    @_safe_tool
    async def get_run_results(
        experiment_id: MaybeExperimentId = None,
        run_id: MaybeRunId = None,
    ) -> GetRunResultsResult:
        """Return policy-filtered winning sampled hyperparameters per completed phase: trial number, metric, params, gate status, and completeness. Provide exactly one of experiment_id or run_id. Read-only.

        :param MaybeExperimentId experiment_id: Optional catalog experiment id whose winners should be read.
        :param MaybeRunId run_id: Optional detached run id whose snapshot should be read.
        :return GetRunResultsResult: Structured results payload.
        """
        return GetRunResultsResult.model_validate(
            await asyncio.to_thread(app.winners, experiment_id=experiment_id, run_id=run_id)
        )

    @mcp.tool(
        name=TOOL_LAUNCH_RUN,
        description=DESCRIPTION_LAUNCH_RUN,
        annotations=_tool_annotations(
            "Launch Run",
            read_only=False,
            destructive=True,
            idempotent=False,
            open_world=True,
        ),
        structured_output=True,
    )
    @_safe_tool
    async def launch_run(
        experiment_id: ExperimentId,
        from_phase: MaybePhaseName = None,
    ) -> LaunchRunResult:
        """Start the sweep for an experiment as a background run. Optionally resume from a phase whose earlier winners already exist. Returns a run_id.

        :param ExperimentId experiment_id: Catalog experiment id to launch.
        :param MaybePhaseName from_phase: Optional phase to resume from.
        :return LaunchRunResult: Structured launch result.
        """
        result = LaunchRunResult.model_validate(
            await asyncio.to_thread(app.launch, experiment_id, from_phase=from_phase)
        )
        return result.model_copy(update={"next_action": TOOL_AWAIT_RUN})

    @mcp.tool(
        name=TOOL_CANCEL_RUN,
        description=DESCRIPTION_CANCEL_RUN,
        annotations=_tool_annotations(
            "Cancel Run",
            read_only=False,
            destructive=True,
        ),
        structured_output=True,
    )
    @_safe_tool
    async def cancel_run(run_id: RunId) -> CancelRunResult:
        """Stop a running sweep by run_id. Terminates the orchestrator and its training processes.

        :param RunId run_id: Detached run id to cancel.
        :return CancelRunResult: Structured cancellation result.
        """
        # MCP 1.27 invokes synchronous tool functions on the event-loop
        # thread. Cancellation may spend its 30-second grace period waiting
        # for a process group, so isolate it from concurrent await/status calls.
        result = CancelRunResult.model_validate(await asyncio.to_thread(app.cancel, run_id))
        return result.model_copy(
            update={"next_action": None if result.recovery_required else TOOL_GET_RUN_RESULTS}
        )

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
        return agent_prompt_text(strip=True)

    _strict_tool_inputs(mcp)
    return mcp


def serve(catalog: Path) -> int:
    """Load the catalog, build the run store, and serve the eight tools over stdio.

    :param Path catalog: Operator-authored catalog file.
    :return int: Process exit code, where 2 means a startup prerequisite or
        catalog validation failed before the server could run.
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
            f"phasesweep mcp: MCP support is not installed; install with "
            f"`{MCP_EXTRA_INSTALL_COMMAND}`.",
            file=sys.stderr,
        )
        return 2

    try:
        registry = Registry.load(catalog)
    except CatalogError as exc:
        print(f"phasesweep mcp: {exc}", file=sys.stderr)
        return 2

    app = PhaseSweepMCP(registry, RunStore(registry.state_dir))
    build_server(app).run(transport="stdio")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Serve through the installed ``phasesweep-mcp`` entry point.

    :param list[str] | None argv: Optional argument vector; defaults to ``sys.argv`` when omitted.
    :return int: Process exit code.
    """
    parser = argparse.ArgumentParser(prog="phasesweep mcp")
    parser.add_argument("--catalog", required=True, type=Path)
    args = parser.parse_args(argv)
    return serve(args.catalog)


if __name__ == "__main__":
    raise SystemExit(main())
