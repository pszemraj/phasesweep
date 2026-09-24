"""FastMCP adapter: the only module that imports the MCP SDK.

PhaseSweepMCP (in :mod:`phasesweep.mcp.tools`) holds all logic and is SDK-free
and unit-testable. build_server wraps each method as a FastMCP tool; _safe_tool
guarantees tool errors are redacted. serve() loads the catalog, builds the
store, and serves over stdio.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import importlib.util
import inspect
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from phasesweep import __version__
from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.engine.read import ResultContext as ResultContextLiteral
from phasesweep.engine.state import WinnerSourceKind
from phasesweep.evidence.models import _ObjectiveEvidenceFields
from phasesweep.mcp import MCP_EXTRA_INSTALL_COMMAND, agent_prompt_text
from phasesweep.mcp.errors import CatalogError, McpToolError
from phasesweep.mcp.redaction import ResultSource
from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runner import FailurePayload
from phasesweep.mcp.runs import RunState, RunStore
from phasesweep.mcp.snapshots import McpPublicationState
from phasesweep.mcp.tool_names import (
    TOOL_AWAIT_RUN,
    TOOL_CANCEL_RUN,
    TOOL_GET_LATEST_RUN,
    TOOL_GET_RUN_RESULTS,
    TOOL_GET_RUN_STATUS,
    TOOL_INSPECT_EXPERIMENT,
    TOOL_LAUNCH_RUN,
    TOOL_LIST_EXPERIMENTS,
)
from phasesweep.mcp.tools import (
    AWAIT_DEFAULT_TIMEOUT_SECONDS,
    AWAIT_MAX_TIMEOUT_SECONDS,
    AWAIT_MIN_TIMEOUT_SECONDS,
    DEFAULT_LIST_LIMIT,
    MAX_LIST_LIMIT,
    PhaseSweepMCP,
)

log = logging.getLogger("phasesweep.mcp.server")

SAFE_NAME_JSON_PATTERN = SAFE_NAME_PATTERN.pattern
CATALOG_RESOURCE_URI = "phasesweep://catalog"
PROMPT_RUN_AND_MONITOR = "phasesweep_run_and_monitor"


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
    "inheritance, and search-space keys. Call after list_experiments and before launch_run. "
    "Read-only: launch_run separately rechecks config identity and refuses catalog drift."
)
DESCRIPTION_GET_LATEST_RUN = (
    "Return the most recently launched run for one experiment. Call only to recover a lost "
    "run_id; then use that run_id with await_run, get_run_status, or get_run_results. "
    "Read-only: found=false never authorizes launching a replacement."
)
DESCRIPTION_LAUNCH_RUN = (
    "Launch one approved experiment as a detached run and return its run_id. Call only after "
    "inspection and explicit user authorization; monitor with the returned run_id. "
    "Never retry permission or config-identity refusals; await named blockers before retrying a "
    "capacity refusal."
)
DESCRIPTION_GET_RUN_STATUS = (
    "Read process state and per-phase progress for exactly one run_id. Use as a single status "
    "check when await_run is unsuitable. State is run.state; stop if run.recovery_required is "
    "true. Read-only: reuse the run_id returned by launch_run or get_latest_run."
)
DESCRIPTION_GET_RUN_RESULTS = (
    "Return terminal per-phase winners, completeness, metrics, gates, and "
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
    "State is run.state; stop immediately if run.recovery_required is true. Call "
    "get_run_results when terminal. Read-only: always reuse "
    "the run_id, including after a client disconnect."
)

ExperimentId = Annotated[
    str,
    Field(
        description=f"Catalog experiment id exposed by {TOOL_LIST_EXPERIMENTS}.",
        pattern=SAFE_NAME_JSON_PATTERN,
    ),
]
RunId = Annotated[
    str,
    Field(description=f"MCP run id returned by {TOOL_LAUNCH_RUN}.", pattern=SAFE_NAME_JSON_PATTERN),
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


class _ToolPayload(BaseModel):
    """Strict base for structured MCP tool results."""

    model_config = ConfigDict(extra="forbid")


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
    n_trials: int = Field(
        ge=1,
        description=(
            "Target number of terminal attempts; COMPLETE, FAIL, and PRUNED trials all count."
        ),
    )
    sampler: str = Field(description="Sampler type only; sampler internals stay in the config.")
    inherits: list[PhaseName] = Field(description="Parent phases inherited by this phase.")
    search_space: list[str] = Field(description="Search-space keys only, never ranges or values.")


class InspectExperimentResult(_ToolPayload):
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


class GetLatestRunResult(_ToolPayload):
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
    published_study_unavailable: bool | None = Field(
        description=(
            "Whether a current published phase's local trial identity could not be matched. When "
            "trial_data_available is true, its published trial is confirmed missing or replaced: "
            "restore the original ledger/study or use a new experiment identity; "
            "earlier phases may load validated saved winners via from_phase. When "
            "trial_data_available is false, inspection failed: a run can report "
            "cleanup_uncertain and require operator "
            "recovery before further MCP launches. Null means availability was not checked, "
            "including older snapshots and pre-generation placeholders."
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


class GetRunStatusResult(_ToolPayload):
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
            "counts this payload shows: normally the queried run_id. Null for a "
            "config-only unavailable placeholder."
        )
    )
    is_published: bool = Field(
        description=(
            "True only when represented_generation_id is not null and equals "
            "published_generation_id. A run_id whose own publication failed reports "
            "false here while still showing that generation's own winners."
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
    run: RunPayload
    elapsed_seconds: int | None = Field(
        description=(
            "Seconds since launch while running; total run duration once terminal; "
            "null when the terminal endpoint is unavailable."
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


class WinnerPhasePayload(_ToolPayload):
    """Agent-visible phase winner."""

    phase: PhaseName
    winner_source: WinnerSourcePayload
    winner_generation: Literal["current_generation", "prior_generation", "unknown"] = Field(
        description=(
            "Whether this winner was selected in the represented generation, carried from an "
            "earlier generation, or has no recorded generation provenance."
        )
    )
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


class GetRunResultsResult(_ToolPayload):
    """Structured output for get_run_results.

    Every result-scoped field describes the represented generation under the
    config that produced it: ``metric``, ``declared_phase_count``,
    ``missing_phases``, and ``all_phases_have_winners`` are measured against
    the historical phase plan, not the current one (review v0.5.16 /
    blocker 4). ``result_context`` and ``published_config_matches_current``
    disclose which config that was.
    """

    experiment_id: ExperimentId
    run_id: RunId
    result_source: ResultSource = Field(
        description=(
            "Where the result facts came from: the mutable shared study for a live/current "
            "read, an immutable terminal run snapshot, or an unavailable terminal-snapshot "
            "placeholder whose counts are explicitly untrusted."
        )
    )
    represented_generation_id: str | None = Field(
        description=(
            "Generation whose winners and completeness this payload represents. This is the "
            "in-flight run generation during a live run; null when there is no represented "
            "generation."
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


class LaunchRunResult(_ToolPayload):
    """Structured output for launch_run."""

    run_id: RunId
    experiment_id: ExperimentId
    state: Literal["running"]


class CancelRunResult(_ToolPayload):
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

    _verify_strict_tool_inputs(mcp)


def _verify_strict_tool_inputs(mcp: Any) -> None:
    """Fail startup if FastMCP internals did not keep the strict schemas.

    :param Any mcp: Configured FastMCP server whose registered tools are inspected.
    :raises RuntimeError: If any tool still accepts undeclared input keys or a
        required run-scoped read tool is unregistered.
    """
    for tool in mcp._tool_manager.list_tools():
        if tool.parameters.get("additionalProperties") is not False:
            raise RuntimeError(f"MCP tool {tool.name!r} accepts undeclared input keys")
    for tool_name in (TOOL_GET_RUN_STATUS, TOOL_GET_RUN_RESULTS):
        tool = mcp._tool_manager.get_tool(tool_name)
        if tool is None:
            raise RuntimeError(f"MCP tool {tool_name!r} was not registered")
        if tool.parameters.get("required") != ["run_id"]:
            raise RuntimeError(f"MCP tool {tool_name!r} must require run_id")


def build_server(app: PhaseSweepMCP) -> Any:
    """Construct the FastMCP server.

    The SDK is imported lazily so non-server code paths (and most tests) do not
    require the ``mcp`` package.

    :param PhaseSweepMCP app: SDK-free tool implementation to expose.
    :return Any: Configured FastMCP server.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("phasesweep", instructions=agent_prompt_text(strip=True))
    # Pinned FastMCP 1.27 has no version constructor argument; leaving the
    # underlying version unset advertises the MCP SDK as PhaseSweep's version.
    mcp._mcp_server.version = __version__

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
        return ListExperimentsResult.model_validate(
            app.list_experiments(limit=limit, cursor=cursor)
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
        :return InspectExperimentResult: Structured inspection payload.
        """
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
        return GetLatestRunResult.model_validate(
            await asyncio.to_thread(app.latest_run, experiment_id)
        )

    @mcp.tool(
        name=TOOL_GET_RUN_STATUS,
        description=DESCRIPTION_GET_RUN_STATUS,
        annotations=_tool_annotations("Get Run Status"),
        structured_output=True,
    )
    @_safe_tool
    async def get_run_status(run_id: RunId) -> GetRunStatusResult:
        """Return per-phase trial counts, winner presence, and one run's process state.

        :param RunId run_id: Detached run id returned by launch_run or get_latest_run.
        :return GetRunStatusResult: Structured status payload.
        """
        return GetRunStatusResult.model_validate(await asyncio.to_thread(app.status, run_id))

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
        return result

    @mcp.tool(
        name=TOOL_GET_RUN_RESULTS,
        description=DESCRIPTION_GET_RUN_RESULTS,
        annotations=_tool_annotations("Get Run Results"),
        structured_output=True,
    )
    @_safe_tool
    async def get_run_results(run_id: RunId) -> GetRunResultsResult:
        """Return policy-filtered winning sampled hyperparameters for one run.

        :param RunId run_id: Detached run id returned by launch_run or get_latest_run.
        :return GetRunResultsResult: Structured results payload.
        """
        return GetRunResultsResult.model_validate(await asyncio.to_thread(app.winners, run_id))

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
        return LaunchRunResult.model_validate(
            await asyncio.to_thread(app.launch, experiment_id, from_phase=from_phase)
        )

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
        return CancelRunResult.model_validate(await asyncio.to_thread(app.cancel, run_id))

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
