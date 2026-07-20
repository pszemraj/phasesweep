"""Outbound payload construction. Whitelist, do not blacklist.

Payloads are built only from typed, path-free views (PhaseWinnerView, the
read_status dict, catalog summaries). There is no path to interpolate a
trial_command, env value, or storage URL into a result, so there is nothing to
redact after the fact.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from phasesweep.engine import PhaseWinnerView

VisibleParamsPolicy: TypeAlias = Literal["none", "all"] | list[str]
ResultSource: TypeAlias = Literal["current_shared_study", "frozen_run_snapshot"]

_TRIAL_STATES = ("WAITING", "RUNNING", "COMPLETE", "PRUNED", "FAIL")


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
    views: list[PhaseWinnerView],
    *,
    metric: dict[str, Any],
    declared_phases: list[str],
    result_source: ResultSource,
    run_id: str | None = None,
    represented_generation_id: str | None = None,
    visible_params: VisibleParamsPolicy = "none",
) -> dict[str, Any]:
    """Build the ``get_winners`` payload from path-free phase-winner views.

    MCP output exposes sampled ``params`` only, and values are redacted unless
    the catalog explicitly allows them. ``effective_overrides`` can include
    config-authored fixed or inherited values, so it is intentionally kept out
    of agent-visible tool results.

    :param str experiment_id: Catalog id whose winners are being returned.
    :param list[PhaseWinnerView] views: Path-free winner views read from engine state.
    :param dict[str, Any] metric: Optimization metric and objective-evidence assurance.
    :param list[str] declared_phases: All phase names in execution order.
    :param ResultSource result_source: Whether results came from current shared
        state or a frozen terminal run snapshot.
    :param str | None run_id: Run id represented by a frozen snapshot, if any.
    :param str | None represented_generation_id: Generation whose results are being represented.
    :param VisibleParamsPolicy visible_params: Catalog policy for sampled param values.
    :return dict[str, Any]: MCP-safe winners payload.
    """
    winner_phases = {view.phase for view in views}
    missing_phases = [phase for phase in declared_phases if phase not in winner_phases]
    phases: list[dict[str, Any]] = []
    for view in views:
        params, redacted = _visible_winner_params(view.params, visible_params)
        source = view.source
        winner_source = {
            "kind": source.kind if source is not None else "phase_trial",
            "phase": source.phase if source is not None else view.phase,
            "trial_number": (source.trial_number if source is not None else view.trial_number),
            "study": source.study if source is not None else None,
        }
        source_generation_id = source.generation_id if source is not None else view.generation_id
        winner_generation = (
            "unknown"
            if source_generation_id is None or represented_generation_id is None
            else "current_generation"
            if source_generation_id == represented_generation_id
            else "prior_generation"
        )
        promotion = None
        if view.promotion is not None and view.promotion.get("action") in (
            "promote",
            "continue_baseline",
        ):
            promotion = {
                "action": view.promotion["action"],
                "baseline_phase": view.promotion["baseline"],
                "candidate_trial_number": view.promotion["candidate_trial_number"],
                "candidate_metric": view.promotion["candidate_metric"],
                "baseline_trial_number": view.promotion["baseline_trial_number"],
                "baseline_metric": view.promotion["baseline_metric"],
                "min_delta": view.promotion["min_delta"],
                "improvement": view.promotion["improvement"],
            }
        phases.append(
            {
                "phase": view.phase,
                "winner_source": winner_source,
                "winner_generation": winner_generation,
                "promotion": promotion,
                "metric": view.metric,
                "params": params,
                "params_redacted": redacted,
                "gates_passed": view.gates_passed,
                "incomplete": view.incomplete,
            }
        )
    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "result_source": result_source,
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
    run: dict[str, Any] | None,
    *,
    result_source: ResultSource,
    elapsed_seconds: int | None,
) -> dict[str, Any]:
    """Build the ``get_status`` payload from the path-free read_status dict.

    ``status`` is the read_status output (already path-free). ``run`` is the
    process-level state for a specific run_id, or None for an experiment-level
    query with no recorded runs. Timing fields are counts of seconds computed
    by the server - durations only, never timestamps of operator activity or
    anything path-shaped.

    :param str experiment_id: Catalog id whose status is being returned.
    :param dict[str, Any] status: Path-free status payload from ``read_status``.
    :param dict[str, Any] | None run: Optional path-free detached-run state.
    :param ResultSource result_source: Whether status came from current shared
        state or a frozen terminal run snapshot.
    :param int | None elapsed_seconds: Seconds since launch (running) or total
        run duration (terminal); ``None`` without an associated run.
    :return dict[str, Any]: MCP-safe status payload.
    """
    phases = []
    for phase in status["phases"]:
        raw_counts = phase["trials"]
        counts = {state: int(raw_counts.get(state, 0)) for state in _TRIAL_STATES}
        raw_generation_counts = phase.get("generation_trials") or {}
        generation_counts = {
            state: int(raw_generation_counts.get(state, 0)) for state in _TRIAL_STATES
        }
        terminal_trials_total = counts["COMPLETE"] + counts["PRUNED"] + counts["FAIL"]
        terminal_trials_this_run = (
            generation_counts["COMPLETE"] + generation_counts["PRUNED"] + generation_counts["FAIL"]
        )
        terminal_trials_before_run = terminal_trials_total - terminal_trials_this_run
        target_terminal_trials = int(phase["n_trials"])
        phases.append(
            {
                "phase": phase["phase"],
                "trials": counts,
                "running_trials_total": counts["RUNNING"],
                "target_terminal_trials": target_terminal_trials,
                "completed_trials_total": counts["COMPLETE"],
                "terminal_trials_total": terminal_trials_total,
                "terminal_trials_before_run": terminal_trials_before_run,
                "attempts_launched_this_run": sum(generation_counts.values()),
                "terminal_trials_this_run": terminal_trials_this_run,
                "remaining_trials": max(0, target_terminal_trials - terminal_trials_total),
                "target_already_satisfied": (terminal_trials_before_run >= target_terminal_trials),
                "winner_present": phase["winner_present"],
                "trial_data_available": phase["trial_data_available"],
            }
        )
    return {
        "experiment_id": experiment_id,
        "result_source": result_source,
        "metric": status["metric"],
        "phases": phases,
        "summary_present": status["summary_present"],
        "run": run,
        "elapsed_seconds": elapsed_seconds,
    }
