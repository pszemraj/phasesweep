"""FastMCP adapter: the only module that imports the MCP SDK.

PhaseSweepMCP (in :mod:`phasesweep.mcp.tools`) holds all logic and is SDK-free.
build_server wraps each method as a FastMCP tool and adds get_guide; _safe_tool
guarantees tool errors are redacted. serve() loads the catalog, builds the
store, and serves over stdio; call() runs one tool through the same server.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import importlib.util
import inspect
import json
import logging
import sys
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, TypeVar

from pydantic import BaseModel, ConfigDict, Field

if typing.TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

from phasesweep import __version__
from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.engine.artifacts import LOG_FORMAT
from phasesweep.engine.read import CostAdvisoryStatus
from phasesweep.engine.read import ResultContext as ResultContextLiteral
from phasesweep.engine.state import TrialStateName, WinnerSourceKind
from phasesweep.mcp import MCP_EXTRA_INSTALL_COMMAND, agent_prompt_text
from phasesweep.mcp.audit import AUDIT_LOG_NAME, AuditLogger, AuditTransport
from phasesweep.mcp.errors import CatalogError, McpToolError
from phasesweep.mcp.redaction import ResultSource
from phasesweep.mcp.registry import Registry
from phasesweep.mcp.runner import FailurePayload
from phasesweep.mcp.runs import RunState, RunStore
from phasesweep.mcp.snapshots import (
    ConfoundSummary,
    McpPublicationState,
    MetricSnapshot,
    RepeatsSpread,
    TieGroupSnapshot,
)
from phasesweep.mcp.tool_names import (
    TOOL_AWAIT_RUN,
    TOOL_CANCEL_RUN,
    TOOL_EXPLAIN_CONFIG,
    TOOL_GET_GUIDE,
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
from phasesweep.runtime.json import JsonObject, json_object
from phasesweep.runtime.reaper import in_initial_pid_namespace

log = logging.getLogger("phasesweep.mcp.server")

SAFE_NAME_JSON_PATTERN = SAFE_NAME_PATTERN.pattern
CATALOG_RESOURCE_URI = "phasesweep://catalog"
PROMPT_RUN_AND_MONITOR = "phasesweep_run_and_monitor"

# Clients may load this into every session and may show only its first line, cut at 250
# characters, so it is one short line; the guide loads on demand through get_guide.
SERVER_INSTRUCTIONS = (
    "PhaseSweep runs hyperparameter sweeps on the MCP server host. "
    "Use these tools only when the user asks for PhaseSweep in this conversation. Never suggest "
    "PhaseSweep or offer it as an option otherwise. "
    f"Call {TOOL_GET_GUIDE} before any other PhaseSweep tool."
)

DESCRIPTION_GET_GUIDE = (
    "Return the PhaseSweep guide: who owns the catalog and configs, the call loop, when to stop, "
    "what to report with a winner, and where per-trial data lives. Call it once, before any "
    "other PhaseSweep tool."
)
DESCRIPTION_LIST_EXPERIMENTS = (
    "List the operator-approved experiments and their permitted actions. Call it after get_guide "
    "to discover experiments; follow next_cursor until null, then call inspect_experiment. "
    "Read-only: use only ids returned by this catalog."
)
DESCRIPTION_EXPLAIN_CONFIG = (
    "Explain the experiment configuration schema: the type, default and rules of one config key "
    "and every key below it, with the section rules and shared-block descriptions that govern "
    "them, or the top-level keys and the general notes on the config file, paths, YAML "
    "typing and the study fingerprint when key is omitted. Call only when the user "
    "asks for config-authoring help, narrowing with a key such as phases[].sampler; when "
    "found=false, retry with one of closest. Read-only: it describes the schema, not any "
    "catalog experiment, and needs no experiment id."
)
DESCRIPTION_INSPECT_EXPERIMENT = (
    "Inspect one approved experiment's metric, permissions, phases, trial targets, samplers, "
    "inheritance, and search-space keys. Call after list_experiments, before launch_run or "
    "advising on the experiment. Read-only: the server rereads an edited catalog or config "
    "before each call, so this describes what launch_run would start now."
)
DESCRIPTION_GET_LATEST_RUN = (
    "Return the most recently launched run for one experiment. Call only to recover a lost "
    "run_id; then use run.run_id with await_run, get_run_status, or get_run_results. "
    "Read-only: when found is false, stop; it never authorizes launching a replacement."
)
DESCRIPTION_LAUNCH_RUN = (
    "Launch one approved experiment as a detached run and return its run_id. Call only after "
    "inspection and explicit user authorization for that experiment, and keep the returned "
    "run_id for every later call. Never launch a replacement run on your own. "
    "Never retry permission refusals. Retry a config-changed refusal once; report an "
    "invalid-catalog or changed-state_dir refusal to the operator and wait for the fix. For "
    "capacity refusals, await healthy named blockers; report blockers requiring recovery to the "
    "operator before retrying."
)
DESCRIPTION_GET_RUN_STATUS = (
    "Read process state and per-phase progress for exactly one run_id. Use as a single status "
    "check when await_run is unsuitable, never in a polling loop. State is run.state; stop if "
    "run.recovery_required is true. Read-only: reuse the run_id returned by launch_run or "
    "get_latest_run."
)
DESCRIPTION_GET_RUN_RESULTS = (
    "Return terminal per-phase winners, completeness, metrics, gates, and "
    "policy-filtered sampled parameters. Call after a run becomes terminal; this ends the normal "
    "workflow. Say when a winner comes from an earlier generation (winner_generation "
    "prior_generation), when its labels are historical (published_config_matches_current "
    "false), and that a confound check reading n/a was not run. Read-only."
)
DESCRIPTION_CANCEL_RUN = (
    "Request bounded cancellation of one launched run. Call only when the user explicitly asks; "
    "next read get_run_results if cancellation finishes cleanly. Never cancel automatically, and "
    "stop for operator recovery when recovery_required is true."
)
DESCRIPTION_AWAIT_RUN = (
    "Wait up to timeout_seconds for a launched run to change, become terminal, or require "
    "recovery. Call after launch_run and repeat with the same run_id while run.state is "
    "running, including after reason phase_completed and after a client disconnect; omit "
    "timeout_seconds for a client-safe 20-second wait, and request longer waits only when the "
    "client permits them. Stop immediately if run.recovery_required is true. When "
    "run.runner_observable is false, monitor durable progress and notify the operator if it "
    "stops. Call get_run_results when terminal. Read-only."
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
            "Publication validation for this result view. Terminal responses preserve "
            "snapshot-time validation rather than rechecking the current publishing tree. "
            "'ok': the publication passed validation. 'absent': no publication was "
            "recorded in this view, which is normal before the first publication. "
            "'failed': a publication was recorded but failed validation, so result fields read as "
            "if nothing published. 'permission_denied': the publication may be healthy, but the "
            "server could not validate its owner-only evidence. 'unknown': the terminal "
            "placeholder could not inspect publication state. The last three withhold every "
            "winner, and the response's failure names the operator's step."
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
            "represented generation's summary is absent or records another summary "
            "schema version - so the currently loaded config describes it."
        )
    ),
]
PublishedConfigMatchesCurrent = Annotated[
    bool | None,
    Field(
        description=(
            "Whether the producing configuration matches the corresponding prefix of the "
            "current phase plan. Appended phases do not invalidate that agreement. False "
            "means the producing semantics differ and these result labels are historical. "
            "Null when no producing fingerprint is available; null does not establish a match."
        )
    ),
]
MaybePhaseName = Annotated[
    str | None,
    Field(
        description=(
            "Optional phase name to resume from once the earlier phases the remaining phases "
            "inherit have compatible published winners."
        ),
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
ConfigKey = Annotated[
    str | None,
    Field(
        description=(
            "Config key to explain, such as phases[].sampler, phases.sampler or "
            "phases.0.sampler. Omit it to list the top-level keys."
        ),
        max_length=200,
    ),
]
AwaitTimeoutSeconds = Annotated[
    int,
    Field(
        description=(
            "Seconds to block waiting for the run to change before returning its current status. "
            f"At most {AWAIT_MAX_TIMEOUT_SECONDS}; to keep waiting longer, call await_run again "
            "with the same run_id."
        ),
        ge=AWAIT_MIN_TIMEOUT_SECONDS,
        le=AWAIT_MAX_TIMEOUT_SECONDS,
    ),
]


class _ToolPayload(BaseModel):
    """Strict base for structured MCP tool results."""

    model_config = ConfigDict(extra="forbid")


class CapabilitiesPayload(_ToolPayload):
    """Catalog-authorized actions for one experiment."""

    launch: bool = Field(description="Whether the agent may launch this experiment.")
    cancel: bool = Field(description="Whether the agent may cancel its launched runs.")
    from_phase: bool = Field(
        description="Whether launch may use from_phase after earlier winners exist."
    )


class ExperimentSummaryPayload(_ToolPayload):
    """Path-free catalog entry summary."""

    id: ExperimentId
    description: str = Field(description="Operator-authored catalog description.")
    phases: list[PhaseName] = Field(description="Declared phases, in execution order.")
    metric: MetricSnapshot
    capabilities: CapabilitiesPayload


class CatalogStatusPayload(_ToolPayload):
    """Freshness of the catalog used to answer a read."""

    stale: bool
    reload_refusal: Literal["invalid", "state_dir_changed"] | None
    remediation: str | None


class ListExperimentsResult(_ToolPayload):
    """Structured output for list_experiments."""

    experiments: list[ExperimentSummaryPayload]
    catalog_status: CatalogStatusPayload
    total_count: int = Field(ge=0, description="Total catalog entries exposed by this server.")
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the next page, or null when this page is complete.",
    )


class ConfigEntryPayload(_ToolPayload):
    """One documented config key from the packaged reference."""

    path: str = Field(
        description=(
            "Documented key path. [] follows a list key, [type=...] labels one discriminator "
            "variant of the key, and <key> stands for a user-chosen map key."
        )
    )
    signature: str = Field(
        description="Type and default, or '(required)', as the reference spells it."
    )
    section: str = Field(description="Reference section the key belongs to.")
    description: str = Field(
        description=(
            "The key's rules, one line per paragraph. When key is omitted this is only the "
            "first sentence of each top-level key."
        )
    )


class ConfigRulesPayload(_ToolPayload):
    """Reference lines that govern a group of keys rather than one key."""

    kind: Literal["section", "block"] = Field(
        description=(
            "section for the rules at the head of a reference section; block for the "
            "description of a shared block whose fields several keys reuse."
        )
    )
    name: str = Field(description="Section title, matching entries' section, or block name.")
    description: str = Field(description="The rules, one line per paragraph.")


class ExplainConfigResult(_ToolPayload):
    """Structured output for explain_config."""

    key: str | None = Field(description="The requested key, echoed; null when it was omitted.")
    found: bool = Field(description="Whether the key matches a documented config key.")
    rules: list[ConfigRulesPayload] = Field(
        description=(
            "Section rules and shared-block descriptions that govern the returned entries, "
            "each once, in order of first use; empty when key is omitted or not found."
        )
    )
    entries: list[ConfigEntryPayload] = Field(
        description="The key and every key below it in reference order; empty when not found."
    )
    closest: list[str] = Field(
        description="Nearest documented keys when nothing matched; otherwise empty."
    )
    notes: str = Field(
        description=(
            "General notes on the config file, paths, YAML typing and the study fingerprint, "
            "one line per paragraph. Returned only when key is omitted; otherwise empty."
        )
    )


class PhaseValidationPayload(_ToolPayload):
    """Agent-safe phase structure."""

    name: PhaseName
    n_trials: int = Field(
        ge=1,
        description=(
            "Target number of finished attempts; COMPLETE, INFEASIBLE and FAIL trials all count. "
            "INTERRUPTED trials never count: each reruns with its parameters before any new draw."
        ),
    )
    sampler: str = Field(description="Sampler type only; sampler internals stay in the config.")
    inherits: list[PhaseName] = Field(description="Parent phases inherited by this phase.")
    search_space: list[str] = Field(description="Search-space keys only, never ranges or values.")
    repeats: int | None = Field(
        ge=2,
        description=(
            "How many times each configuration runs, once per repeat value, with the winner "
            "ranked by a reducer over them; null without repeats. The repeat key and values "
            "stay in the config."
        ),
    )


class InspectExperimentResult(_ToolPayload):
    """Structured output for inspect_experiment."""

    experiment_id: ExperimentId
    catalog_status: CatalogStatusPayload
    metric: MetricSnapshot
    capabilities: CapabilitiesPayload
    phases: list[PhaseValidationPayload]


class RunPayload(_ToolPayload):
    """Agent-visible run process state."""

    run_id: RunId
    state: RunState
    started_at: str = Field(description="UTC ISO-8601 launch timestamp.")
    recovery_required: bool = Field(
        description=(
            "True when the run needs operator recovery: an unresolved launch, uncertain "
            "cleanup, interrupted snapshot finalization, or a runner in another PID namespace "
            "whose heartbeat stopped. failure then names the operator's step."
        )
    )
    runner_observable: bool = Field(
        description=(
            "False for a running runner in another PID namespace, where liveness and "
            "cancellation are unavailable."
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
    catalog_status: CatalogStatusPayload
    found: bool = Field(description="Whether any MCP run is recorded for this experiment.")
    run: RunPayload | None


class PhaseStatusPayload(_ToolPayload):
    """Per-phase status without filesystem paths."""

    phase: PhaseName
    trials: dict[TrialStateName, int] | None = Field(
        description=(
            "Dense cumulative study counts, or null when unavailable. Every state key is "
            "present when counts are known. COMPLETE counts trials that finished inside every "
            "metric bound and constraint. INFEASIBLE counts trials that finished outside "
            "the metric bounds or constraints; they can never win. INTERRUPTED "
            "counts trials a deadline, shutdown or orchestrator death cut short; they are "
            "not finished, and WAITING counts their scheduled reruns."
        )
    )
    target_finished_trials: int = Field(ge=0, description="Configured finished-trial target.")
    finished_trials_total: int | None = Field(
        ge=0,
        description="Cumulative COMPLETE + INFEASIBLE + FAIL study rows; null when unavailable.",
    )
    finished_trials_before_run: int | None = Field(
        ge=0,
        description="Finished rows not owned by the queried run; null when unavailable.",
    )
    attempts_this_run: int | None = Field(
        ge=0,
        description=(
            "Attempts the queried run's generation started, in any state, including any that "
            "a deadline, abort or shutdown stopped before their trainer launched; zero for a "
            "run that started none; null when unavailable."
        ),
    )
    finished_trials_this_run: int | None = Field(
        ge=0,
        description="Finished attempts owned by the queried run; null when unavailable.",
    )
    remaining_trials: int | None = Field(
        ge=0,
        description=(
            "Additional finished trials needed to reach the configured target, reruns of "
            "interrupted trials included; null when counts are unavailable."
        ),
    )
    target_already_satisfied: bool | None = Field(
        description=(
            "Whether the trials finished before this run already met the configured target; "
            "null when counts are unavailable."
        )
    )
    winner_present: bool = Field(description="Whether this phase has a winner artifact.")
    published_study_unavailable: bool | None = Field(
        description=(
            "Unmatched published trial identity/history; null when unchecked. With "
            "trial_data_available=true, restore the original ledger/study or use a new "
            "experiment; from_phase may load saved winners for skipped phases. With false, "
            "inspection failed; cleanup_uncertain may require operator recovery before launches."
        )
    )
    trial_data_available: bool = Field(
        description=(
            "Whether trial counts are known, including confirmed absent or empty studies. "
            "False means inspection failed, a journal snapshot is incomplete, or no "
            "persistent observation is available; trial counts and derived progress "
            "fields are then null."
        )
    )


class TrackingDeliveryPayload(_ToolPayload):
    """Mutable path-free upload counts, separate from accepted scientific facts."""

    pending: int
    delivered: int
    remedy: str | None = Field(
        description="The operator's retry step once the run has ended with uploads pending; otherwise null."
    )


class GetRunStatusResult(_ToolPayload):
    """Structured output for get_run_status."""

    experiment_id: ExperimentId
    catalog_status: CatalogStatusPayload
    result_source: ResultSource = Field(
        description=(
            "Result source: mutable shared study, immutable terminal snapshot, or unavailable "
            "terminal-snapshot placeholder with untrusted counts."
        )
    )
    current_generation_id: str | None = Field(
        description=(
            "Current pointer at the live read or frozen snapshot capture, according to "
            "result_source; may differ from this run. Null before any generation starts."
        )
    )
    published_generation_id: str | None = Field(
        description=(
            "Validated last-success pointer at the live read or frozen snapshot capture, "
            "according to result_source; null before the first success."
        )
    )
    represented_generation_id: str | None = Field(
        description=(
            "Generation supplying this payload's winners and summary: the queried run's own "
            "generation, named by its run ID, or the published generation a run that launched "
            "nothing replayed. Null for a run refused before it wrote a generation, and for an "
            "unavailable placeholder."
        )
    )
    is_published: bool = Field(
        description=(
            "True iff the non-null represented_generation_id equals published_generation_id; "
            "a run can still expose its own winners when this is false."
        )
    )
    run_wrote_generation: bool | None = Field(
        description=(
            "Whether the queried run has written its own generation, which it does when it "
            "launches a trial or publishes. False for a run that replayed the published "
            "winners or was refused before its first trial; null for an unavailable placeholder."
        )
    )
    publication_integrity: PublicationIntegrity
    result_context: ResultContext
    published_config_matches_current: PublishedConfigMatchesCurrent
    result_phase_plan: list[PhaseName] = Field(
        description=(
            "Represented generation's phase plan in execution order. Terminal snapshots "
            "preserve both phase lists captured for that run after catalog edits. "
            "Use inspect_experiment for the current catalog plan."
        )
    )
    metric: MetricSnapshot
    phases: list[PhaseStatusPayload]
    summary_present: bool
    run: RunPayload
    tracking_delivery: TrackingDeliveryPayload | None = Field(
        default=None,
        description=(
            "This run's live MLflow upload counts, including after termination; pending uploads "
            "never change scores or winners. Null when the run's launch config has no mlflow "
            "block or can no longer be read."
        ),
    )
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
    winner_generation: Literal["represented_generation", "prior_generation", "unknown"] = Field(
        description=(
            "Whether the winner's source trial belongs to the represented generation or an "
            "earlier generation; unknown when no represented generation is available for "
            "comparison. Reselecting an earlier trial does not change its source generation."
        )
    )
    metric: float = Field(
        description=(
            "The winner's metric. For a phase with repeats it is the reducer (repeats.reduce) "
            "over every member of the winning group, not any single trial's value."
        )
    )
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
    cost_advisory: str | None = Field(
        default=None,
        description="Cheaper configuration within the winner's standard error; parameters follow visible_params policy.",
    )
    cost_advisory_status: CostAdvisoryStatus = Field(
        description="Whether cost_key is unset, comparison unavailable, no cheaper configuration qualifies, or advice is available."
    )
    tie_groups: list[TieGroupSnapshot] | None = Field(
        description="Other repeat configurations within two combined standard errors, with params, gap, combined_se and threshold; null when retained trials are unavailable. Parameter values follow visible_params."
    )
    confound: ConfoundSummary = Field(
        description="Verdicts on the population this winner was selected from, frozen at selection."
    )
    observed: dict[str, float] = Field(
        description=(
            "Report-only winning-trial values by name; these do not affect selection. Failed "
            "reads are omitted; no observations yields an empty map. With repeats, each value "
            "is the mean over members that recorded it."
        )
    )
    repeats: RepeatsSpread | None = Field(
        description=(
            "For a phase with repeats: the reducer, the spread of the member metrics and the "
            "member trial numbers of the winning group, whose last member is "
            "winner_source.trial_number. Null for a phase without repeats."
        )
    )


class GetRunResultsResult(_ToolPayload):
    """Structured output for get_run_results.

    Every result-scoped field describes the represented generation under the
    config that produced it: ``metric``, ``declared_phase_count``,
    ``missing_phases``, and ``all_phases_have_winners`` are measured against
    the historical phase plan, not the current one. ``result_context`` and
    ``published_config_matches_current`` disclose which config that was.
    """

    experiment_id: ExperimentId
    catalog_status: CatalogStatusPayload
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
    run_wrote_generation: bool | None = Field(
        description=(
            "Whether this run wrote its own generation. When false, the run launched no trials "
            "and its winners are an earlier generation's; null for an unavailable placeholder."
        )
    )
    publication_integrity: PublicationIntegrity
    result_context: ResultContext
    published_config_matches_current: PublishedConfigMatchesCurrent
    metric: MetricSnapshot
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
    """Successful cancellation output; unconfirmed cleanup returns an MCP tool error."""

    run_id: RunId
    state: RunState
    cleanup_confirmed: bool | None = Field(
        default=None,
        description=(
            "Whether runner cleanup was fully confirmed; null when the run was already terminal."
        ),
    )
    recovery_required: bool = Field(
        description="Whether successful cancellation still needs operator result repair."
    )
    failure: FailurePayload | None = Field(
        default=None,
        description="The operator's recovery step when recovery_required is true; otherwise null.",
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
    """Wrap tools with redacted ``ValueError`` failures while preserving their signatures.

    Known tool failures retain their safe messages. Unexpected exceptions are logged and
    replaced with a generic error; FastMCP returns either as ``isError`` text.

    :param F fn: Tool implementation to wrap.
    :return F: Wrapped function that raises only safe tool errors.
    """
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: object, **kwargs: object) -> object:
            """Invoke an async tool and redact any exception.

            :param object args: Positional arguments passed through to the wrapped tool.
            :param object kwargs: Keyword arguments passed through to the wrapped tool.
            :return object: Awaited tool result.
            """
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                _raise_safe_tool_error(fn.__name__, exc)

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def sync_wrapper(*args: object, **kwargs: object) -> object:
        """Invoke a synchronous tool and redact any exception.

        :param object args: Positional arguments passed through to the wrapped tool.
        :param object kwargs: Keyword arguments passed through to the wrapped tool.
        :return object: Wrapped tool result.
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
) -> ToolAnnotations:
    """Return MCP annotations for one tool's side-effect contract.

    :param str title: Human-readable tool title.
    :param bool read_only: Whether the tool leaves server state unchanged.
    :param bool destructive: Whether the tool can perform destructive side effects.
    :param bool idempotent: Whether repeated calls have the same side effects.
    :param bool open_world: Whether the tool interacts beyond the local server boundary.
    :return ToolAnnotations: MCP tool annotations.
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

    Uses pinned SDK metadata; startup verification rejects incompatible schemas.
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


def build_server(app: PhaseSweepMCP) -> FastMCP:
    """Construct the FastMCP server.

    The SDK is imported lazily so non-server code paths do not
    require the ``mcp`` package.

    :param PhaseSweepMCP app: SDK-free tool implementation to expose.
    :return FastMCP: Configured FastMCP server.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("phasesweep", instructions=SERVER_INSTRUCTIONS)
    mcp._mcp_server.version = __version__

    def get_guide() -> str:
        """Return the packaged agent guide.

        :return str: The guide text.
        """
        return agent_prompt_text()

    # The guide is prose for a model to read, so it is the one tool served as plain text:
    # clients hand a structured result to the model as JSON, which escapes every newline.
    mcp.tool(
        name=TOOL_GET_GUIDE,
        description=DESCRIPTION_GET_GUIDE,
        annotations=_tool_annotations("Get Guide"),
        structured_output=False,
    )(_safe_tool(get_guide))

    async def list_experiments(
        limit: ListLimit = DEFAULT_LIST_LIMIT,
        cursor: MaybeCursor = None,
    ) -> ListExperimentsResult:
        """Validate the paginated catalog result.

        The call may reread an edited catalog, so it runs off the event-loop thread.

        :param ListLimit limit: Maximum catalog entries to return.
        :param MaybeCursor cursor: Optional pagination cursor from a prior result.
        :return ListExperimentsResult: Structured catalog listing.
        """
        return ListExperimentsResult.model_validate(
            await asyncio.to_thread(app.list_experiments, limit=limit, cursor=cursor)
        )

    def explain_config(key: ConfigKey = None) -> ExplainConfigResult:
        """Validate the configuration-reference result.

        :return ExplainConfigResult: Matching reference entries, or the closest documented keys.
        """
        return ExplainConfigResult.model_validate(app.explain_config(key))

    async def inspect_experiment(experiment_id: ExperimentId) -> InspectExperimentResult:
        """Validate the experiment inspection result.

        :return InspectExperimentResult: Structured inspection payload.
        """
        return InspectExperimentResult.model_validate(
            await asyncio.to_thread(app.validate, experiment_id)
        )

    async def get_latest_run(experiment_id: ExperimentId) -> GetLatestRunResult:
        """Validate the latest-run result.

        :return GetLatestRunResult: One computed run handle or ``found=false``.
        """
        return GetLatestRunResult.model_validate(
            await asyncio.to_thread(app.latest_run, experiment_id)
        )

    async def get_run_status(run_id: RunId) -> GetRunStatusResult:
        """Validate the run-status result.

        :return GetRunStatusResult: Structured status payload.
        """
        return GetRunStatusResult.model_validate(await asyncio.to_thread(app.status, run_id))

    async def await_run(
        run_id: RunId,
        timeout_seconds: AwaitTimeoutSeconds = AWAIT_DEFAULT_TIMEOUT_SECONDS,
    ) -> AwaitRunResult:
        """Wait for a run update and validate the result.

        :param RunId run_id: Detached run id to wait on.
        :param AwaitTimeoutSeconds timeout_seconds: Seconds to wait before returning current
            status.
        :return AwaitRunResult: Structured status payload plus changed and reason.
        """
        result = AwaitRunResult.model_validate(
            await app.await_run(run_id, timeout_seconds=timeout_seconds)
        )
        return result

    async def get_run_results(run_id: RunId) -> GetRunResultsResult:
        """Validate the run-results payload.

        :return GetRunResultsResult: Structured results payload.
        """
        return GetRunResultsResult.model_validate(await asyncio.to_thread(app.winners, run_id))

    async def launch_run(
        experiment_id: ExperimentId,
        from_phase: MaybePhaseName = None,
    ) -> LaunchRunResult:
        """Launch a detached run and validate the result.

        :param ExperimentId experiment_id: Catalog experiment id to launch.
        :param MaybePhaseName from_phase: Optional phase to resume from.
        :return LaunchRunResult: Structured launch result.
        """
        return LaunchRunResult.model_validate(
            await asyncio.to_thread(app.launch, experiment_id, from_phase=from_phase)
        )

    async def cancel_run(run_id: RunId) -> CancelRunResult:
        """Cancel a run and validate the result.

        :return CancelRunResult: Structured cancellation result.
        """
        # MCP 1.27 invokes synchronous tool functions on the event-loop
        # thread. Cancellation may spend its 30-second grace period waiting
        # for a process group, so isolate it from concurrent await/status calls.
        return CancelRunResult.model_validate(await asyncio.to_thread(app.cancel, run_id))

    registrations: tuple[tuple[Callable[..., object], str, str, ToolAnnotations], ...] = (
        (
            list_experiments,
            TOOL_LIST_EXPERIMENTS,
            DESCRIPTION_LIST_EXPERIMENTS,
            _tool_annotations("List Experiments"),
        ),
        (
            explain_config,
            TOOL_EXPLAIN_CONFIG,
            DESCRIPTION_EXPLAIN_CONFIG,
            _tool_annotations("Explain Config"),
        ),
        (
            inspect_experiment,
            TOOL_INSPECT_EXPERIMENT,
            DESCRIPTION_INSPECT_EXPERIMENT,
            _tool_annotations("Inspect Experiment"),
        ),
        (
            get_latest_run,
            TOOL_GET_LATEST_RUN,
            DESCRIPTION_GET_LATEST_RUN,
            _tool_annotations("Get Latest Run"),
        ),
        (
            get_run_status,
            TOOL_GET_RUN_STATUS,
            DESCRIPTION_GET_RUN_STATUS,
            _tool_annotations("Get Run Status"),
        ),
        (await_run, TOOL_AWAIT_RUN, DESCRIPTION_AWAIT_RUN, _tool_annotations("Await Run")),
        (
            get_run_results,
            TOOL_GET_RUN_RESULTS,
            DESCRIPTION_GET_RUN_RESULTS,
            _tool_annotations("Get Run Results"),
        ),
        (
            launch_run,
            TOOL_LAUNCH_RUN,
            DESCRIPTION_LAUNCH_RUN,
            _tool_annotations(
                "Launch Run", read_only=False, destructive=True, idempotent=False, open_world=True
            ),
        ),
        (
            cancel_run,
            TOOL_CANCEL_RUN,
            DESCRIPTION_CANCEL_RUN,
            _tool_annotations("Cancel Run", read_only=False, destructive=True),
        ),
    )
    for adapter, name, description, tool_annotations in registrations:
        mcp.tool(
            name=name,
            description=description,
            annotations=tool_annotations,
            structured_output=True,
        )(_safe_tool(adapter))

    @mcp.resource(
        CATALOG_RESOURCE_URI,
        name="phasesweep_catalog",
        title="PhaseSweep Catalog",
        description="Read-only first page of the human-curated experiment catalog.",
        mime_type="application/json",
    )
    @_safe_tool
    async def catalog_resource() -> str:
        """Return the first catalog page for clients that attach MCP resources.

        :return str: Compact JSON catalog page.
        """
        result = ListExperimentsResult.model_validate(
            await asyncio.to_thread(app.list_experiments, limit=DEFAULT_LIST_LIMIT, cursor=None)
        )
        return result.model_dump_json(exclude_none=True)

    @mcp.prompt(
        name=PROMPT_RUN_AND_MONITOR,
        title="Run and Monitor Sweep",
        description="Safe workflow for launching, monitoring, and summarizing a phasesweep run.",
    )
    def run_and_monitor_prompt() -> str:
        """Return the packaged agent guide.

        :return str: Prompt text.
        """
        return agent_prompt_text()

    _strict_tool_inputs(mcp)
    return mcp


def _open_app(registry: Registry, transport: AuditTransport) -> PhaseSweepMCP:
    """Build the tool implementation with an audit log that records its transport.

    :param Registry registry: Validated catalog registry.
    :param AuditTransport transport: How calls reach the tools: ``stdio`` when served,
        ``cli`` for ``phasesweep mcp call``.
    :return PhaseSweepMCP: Tool implementation over the catalog's run store and audit log.
    """
    audit = AuditLogger(registry.state_dir / AUDIT_LOG_NAME, transport=transport)
    return PhaseSweepMCP(registry, RunStore(registry.state_dir), audit=audit)


def _report_startup_failure(error: str, suggestion: str | None) -> None:
    """Print one catalog failure for the operator, with its fix when there is one.

    :param str error: Failure description; may reference paths.
    :param str | None suggestion: Concrete fix, printed on its own ``fix:`` line.
    """
    print(f"phasesweep mcp: {error}", file=sys.stderr)
    if suggestion:
        print(f"fix: {suggestion}", file=sys.stderr)


def _load_server_registry(catalog: Path, *, log_level: int) -> Registry | None:
    """Configure stderr logging and load a catalog when MCP support is installed.

    An entry that fails to load does not stop startup: it is reported on stderr and the
    server serves the entries that load, with launches refused until the catalog fully
    validates. Each served entry's trainer directory is logged at info level, so ``serve``
    shows it and ``call`` stays quiet. A catalog-level failure stops startup.

    :param Path catalog: Operator-authored catalog file.
    :param int log_level: Logging level for the selected server transport.
    :return Registry | None: Loaded registry, or None after reporting a startup failure.
    """
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )

    if importlib.util.find_spec("mcp") is None:
        print(
            f"phasesweep mcp: MCP support is not installed; install with "
            f"`{MCP_EXTRA_INSTALL_COMMAND}`.",
            file=sys.stderr,
        )
        return None

    try:
        registry = Registry.load(catalog, collect_errors=True)
    except CatalogError as exc:
        _report_startup_failure(str(exc), exc.suggestion)
        return None
    for failure in registry.failures:
        _report_startup_failure(failure.error or "", failure.suggestion)
    for experiment in registry:
        log.info(
            "serving experiment %s: trainer cwd %s",
            experiment.id,
            experiment.experiment.execution.resolved_cwd(),
        )
    return registry


def serve(catalog: Path) -> int:
    """Load the catalog, build the run store, and serve the tools over stdio.

    The tools reread the catalog when it or a listed config changes; the run store
    stays on the ``state_dir`` loaded here.

    :param Path catalog: Operator-authored catalog file.
    :return int: Process exit code, where 2 means a startup prerequisite or
        catalog-level validation failed before the server could run.
    """
    registry = _load_server_registry(catalog, log_level=logging.INFO)
    if registry is None:
        return 2

    build_server(_open_app(registry, "stdio")).run(transport="stdio")
    return 0


def call(catalog: Path, tool: str, arguments: JsonObject) -> int:
    """Dispatch and print one tool through FastMCP with CLI transport audit records.

    :param Path catalog: Operator-authored catalog file.
    :param str tool: Registered tool name, such as ``list_experiments``.
    :param JsonObject arguments: Tool arguments, validated by the tool's own schema.
    :return int: Process exit code: 0 after the result is printed on stdout (JSON, or
        the text of ``get_guide``), 1 after a tool error is printed on stderr, and 2 when
        a startup prerequisite or the catalog as a whole failed before the tool could run.
    """
    registry = _load_server_registry(catalog, log_level=logging.WARNING)
    if registry is None:
        return 2

    if tool == TOOL_LAUNCH_RUN and not in_initial_pid_namespace():
        print(
            "phasesweep mcp call: warning: this process is not in the host's initial PID "
            "namespace, so a runner launched from it ends when that namespace does, such as "
            "when a sandboxed command finishes; launch from a shell outside any sandbox for "
            "a run that must outlive it.",
            file=sys.stderr,
        )

    from mcp.server.fastmcp.exceptions import ToolError
    from mcp.types import TextContent

    server = build_server(_open_app(registry, "cli"))
    try:
        result = asyncio.run(server.call_tool(tool, arguments))
    except ToolError as exc:
        # The text a client receives in its ``isError`` result.
        print(str(exc), file=sys.stderr)
        return 1
    if isinstance(result, tuple):
        # A tool with structured output returns (unstructured, structured).
        print(json.dumps(json_object(result[1]), indent=2))
    else:
        # get_guide returns text blocks.
        text = "".join(block.text for block in result if isinstance(block, TextContent))
        print(text)
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
