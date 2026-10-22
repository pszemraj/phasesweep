"""Build outbound payloads from path-free status, winner and catalog views."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

from phasesweep.engine.read import PhaseWinnerView, ResultContext
from phasesweep.engine.state import FINISHED_TRIAL_STATES, TRIAL_STATE_NAMES
from phasesweep.mcp.registry import VisibleParamsPolicy
from phasesweep.mcp.snapshots import McpPublicationState, WinnerSnapshot

ResultSource: TypeAlias = Literal[
    "current_shared_study",
    "frozen_run_snapshot",
    "terminal_snapshot_unavailable",
]


def intersect_visible_params(
    launch_policy: VisibleParamsPolicy,
    current_policy: VisibleParamsPolicy,
) -> VisibleParamsPolicy:
    """Return the non-escalating intersection of two visibility policies.

    :param VisibleParamsPolicy launch_policy: Visibility authorized when the run launched.
    :param VisibleParamsPolicy current_policy: Visibility authorized by the current catalog.
    :return VisibleParamsPolicy: Policy no broader than either input.
    """
    if launch_policy == "none" or current_policy == "none":
        return "none"
    if launch_policy == "all":
        return current_policy
    if current_policy == "all":
        return launch_policy
    current_keys = set(current_policy)
    return [key for key in launch_policy if key in current_keys]


def _visible_winner_params(
    params: dict[str, Any], policy: VisibleParamsPolicy
) -> tuple[dict[str, Any], bool]:
    """Filter sampled winner params and report whether any value was withheld.

    :param dict[str, Any] params: Sampled winner parameters to filter.
    :param VisibleParamsPolicy policy: Visibility policy or allowlist to apply.
    :return tuple[dict[str, Any], bool]: Filtered parameters and whether any value was withheld.
    """
    if policy == "all":
        return dict(params), False
    if policy == "none":
        return {key: "<redacted>" for key in params}, bool(params)
    visible = set(policy)
    return (
        {key: value if key in visible else "<redacted>" for key, value in params.items()},
        any(key not in visible for key in params),
    )


def winners_payload(
    experiment_id: str,
    views: Sequence[PhaseWinnerView | WinnerSnapshot],
    *,
    metric: dict[str, Any],
    declared_phases: list[str],
    result_source: ResultSource,
    publication_integrity: McpPublicationState,
    run_id: str | None = None,
    represented_generation_id: str | None = None,
    visible_params: VisibleParamsPolicy = "none",
    result_context: ResultContext = "current_config",
    published_config_matches_current: bool | None = None,
) -> dict[str, Any]:
    """Build the ``get_run_results`` payload from live views or frozen winners.

    Expose sampled params under the visibility policy; omit effective overrides, fixed
    values and inherited values.

    :param str experiment_id: Catalog id whose winners are being returned.
    :param Sequence[PhaseWinnerView | WinnerSnapshot] views: Live engine views or
        validated frozen winners.
    :param dict[str, Any] metric: Optimization metric and objective-evidence
        assurance the represented generation recorded.
    :param list[str] declared_phases: Phase plan the represented generation
        published under, in execution order.
    :param ResultSource result_source: Whether results came from current shared
        state or a frozen terminal run snapshot.
    :param McpPublicationState publication_integrity: Last-success validation verdict,
        or ``unknown`` when the terminal snapshot is unavailable.
    :param str | None run_id: Run id represented by a frozen snapshot, if any.
    :param str | None represented_generation_id: Generation whose results are being represented.
    :param VisibleParamsPolicy visible_params: Catalog policy for sampled param values.
    :param ResultContext result_context: Source of the metric and phase plan:
        represented generation, or current config when nothing is published.
    :param bool | None published_config_matches_current: Represented/current config
        fingerprint agreement, or ``None`` when undeterminable.
    :raises RuntimeError: A winner view carries no source provenance.
    :return dict[str, Any]: MCP-safe winners payload.
    """
    winner_phases = {view.phase for view in views}
    missing_phases = [phase for phase in declared_phases if phase not in winner_phases]
    phases: list[dict[str, Any]] = []
    for view in views:
        params, redacted = _visible_winner_params(view.params, visible_params)
        source = view.source
        if source is None:
            raise RuntimeError("Current-format winner is missing source provenance.")
        winner_source = {
            "kind": source.kind,
            "phase": source.phase,
            "trial_number": source.trial_number,
        }
        winner_generation = (
            "unknown"
            if represented_generation_id is None
            else "represented_generation"
            if source.generation_id == represented_generation_id
            else "prior_generation"
        )
        phases.append(
            {
                "phase": view.phase,
                "winner_source": winner_source,
                "winner_generation": winner_generation,
                "metric": view.metric,
                "params": params,
                "params_redacted": redacted,
                "gates_passed": view.gates_passed,
                "incomplete": view.incomplete,
                "confound": (
                    view.confound.model_dump(mode="json")
                    if isinstance(view, WinnerSnapshot)
                    else view.confound
                ),
                "observed": dict(view.observed),
                "repeats": (
                    view.repeats.model_dump(mode="json")
                    if isinstance(view, WinnerSnapshot) and view.repeats is not None
                    else view.repeats
                ),
            }
        )
    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "result_source": result_source,
        "represented_generation_id": represented_generation_id,
        "publication_integrity": publication_integrity,
        "result_context": result_context,
        "published_config_matches_current": published_config_matches_current,
        "metric": metric,
        "declared_phase_count": len(declared_phases),
        "winner_count": len(views),
        "missing_phases": missing_phases,
        "all_phases_have_winners": not missing_phases,
        "phases": phases,
    }


def status_payload(
    experiment_id: str,
    status: dict[str, Any],
    run: dict[str, Any],
    *,
    result_source: ResultSource,
    elapsed_seconds: int | None,
) -> dict[str, Any]:
    """Map ``read_status`` and detached-run state to the ``get_run_status`` payload.

    Omit operator-only free-text publication errors. The this-run counts come from
    ``generation_trials``, which counts the trials of the run's own generation.

    :param str experiment_id: Catalog id whose status is being returned.
    :param dict[str, Any] status: Path-free status payload from ``read_status``.
    :param dict[str, Any] run: Path-free detached-run state.
    :param ResultSource result_source: Whether status came from current shared
        state or a frozen terminal run snapshot.
    :param int | None elapsed_seconds: Seconds since launch (running) or total
        run duration (terminal); ``None`` without an associated run.
    :return dict[str, Any]: MCP-safe status payload.
    """
    phases = []
    for phase in status["phases"]:
        available = phase["trial_data_available"]
        raw_counts = phase["trials"]
        counts = {state: int(raw_counts.get(state, 0)) for state in TRIAL_STATE_NAMES}
        raw_generation_counts = phase.get("generation_trials") or {}
        generation_counts = {
            state: int(raw_generation_counts.get(state, 0)) for state in TRIAL_STATE_NAMES
        }
        finished_trials_total = sum(counts[state] for state in FINISHED_TRIAL_STATES)
        finished_trials_this_run = sum(generation_counts[state] for state in FINISHED_TRIAL_STATES)
        finished_trials_before_run = max(0, finished_trials_total - finished_trials_this_run)
        target_finished_trials = int(phase["n_trials"])
        phases.append(
            {
                "phase": phase["phase"],
                "trials": counts if available else None,
                "target_finished_trials": target_finished_trials,
                "finished_trials_total": finished_trials_total if available else None,
                "finished_trials_before_run": finished_trials_before_run if available else None,
                "attempts_launched_this_run": sum(generation_counts.values())
                if available
                else None,
                "finished_trials_this_run": finished_trials_this_run if available else None,
                "remaining_trials": (
                    max(0, target_finished_trials - finished_trials_total) if available else None
                ),
                "target_already_satisfied": (
                    finished_trials_before_run >= target_finished_trials if available else None
                ),
                "winner_present": phase["winner_present"],
                "trial_data_available": available,
                "published_study_unavailable": phase.get("published_study_unavailable"),
            }
        )
    return {
        "experiment_id": experiment_id,
        "result_source": result_source,
        **(
            {"tracking_delivery": status["tracking_delivery"]}
            if "tracking_delivery" in status
            else {}
        ),
        "current_generation_id": status["current_generation_id"],
        "published_generation_id": status["published_generation_id"],
        "represented_generation_id": status["represented_generation_id"],
        "is_published": status["is_published"],
        # A run that claims its generation moves the current pointer to its run ID.
        "run_wrote_generation": (
            None
            if result_source == "terminal_snapshot_unavailable"
            else status["current_generation_id"] == run["run_id"]
        ),
        "publication_integrity": status["publication_integrity"],
        "result_context": status["result_context"],
        "published_config_matches_current": status["published_config_matches_current"],
        "result_phase_plan": status["result_phase_plan"],
        "metric": status["metric"],
        "phases": phases,
        "summary_present": status["summary_present"],
        "run": run,
        "elapsed_seconds": elapsed_seconds,
    }
