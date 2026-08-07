"""Outbound payload construction. Whitelist, do not blacklist.

Payloads are built only from typed, path-free views (PhaseWinnerView, the
read_status dict, catalog summaries). There is no path to interpolate a
trial_command, env value, or storage URL into a result, so there is nothing to
redact after the fact.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from phasesweep.engine import PhaseWinnerView
from phasesweep.engine.read import ResultContext
from phasesweep.engine.state import PublicationState, _winner_source_or_default
from phasesweep.mcp.registry import VisibleParamsPolicy

ResultSource: TypeAlias = Literal[
    "current_shared_study",
    "frozen_run_snapshot",
    "terminal_snapshot_unavailable",
]

_TRIAL_STATES = ("WAITING", "RUNNING", "COMPLETE", "PRUNED", "FAIL")


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
    views: list[PhaseWinnerView],
    *,
    metric: dict[str, Any],
    declared_phases: list[str],
    result_source: ResultSource,
    publication_integrity: PublicationState,
    run_id: str | None = None,
    represented_generation_id: str | None = None,
    visible_params: VisibleParamsPolicy = "none",
    result_context: ResultContext = "current_config",
    published_config_matches_current: bool | None = None,
) -> dict[str, Any]:
    """Build the ``get_run_results`` payload from path-free phase-winner views.

    MCP output exposes sampled ``params`` only, and values are redacted unless
    the catalog explicitly allows them. ``effective_overrides`` can include
    config-authored fixed or inherited values, so it is intentionally kept out
    of agent-visible tool results.

    A published result is *historical evidence*: ``metric`` and
    ``declared_phases`` must describe the generation the winners came from, not
    whatever the config says today (review v0.5.16 / blocker 4). Callers are
    responsible for sourcing both from the represented generation;
    ``result_context`` and ``published_config_matches_current`` then tell the
    agent which config those labels came from and whether it still matches the
    one a further run would execute. Visibility policy is unaffected: the
    *current* policy still decides which historical values are redacted, it
    just never relabels them.

    :param str experiment_id: Catalog id whose winners are being returned.
    :param list[PhaseWinnerView] views: Path-free winner views read from engine state.
    :param dict[str, Any] metric: Optimization metric and objective-evidence
        assurance the represented generation recorded.
    :param list[str] declared_phases: Phase plan the represented generation
        published under, in execution order.
    :param ResultSource result_source: Whether results came from current shared
        state or a frozen terminal run snapshot.
    :param PublicationState publication_integrity: Whether the experiment's
        last-success pointer is valid, absent, or names a generation that no
        longer validates. An empty winner list means two very different things
        under ``"absent"`` and ``"failed"`` (review v0.5.18 / finding F4).
    :param str | None run_id: Run id represented by a frozen snapshot, if any.
    :param str | None represented_generation_id: Generation whose results are being represented.
    :param VisibleParamsPolicy visible_params: Catalog policy for sampled param values.
    :param ResultContext result_context: Whether ``metric``/``declared_phases``
        are the represented generation's own recorded semantics or the current
        config's (the latter when nothing is published, or for a pre-manifest
        legacy layout that recorded none).
    :param bool | None published_config_matches_current: Whether the represented
        generation's recorded config fingerprint matches the config this read
        was interpreted through; ``None`` when undeterminable.
    :return dict[str, Any]: MCP-safe winners payload.
    """
    winner_phases = {view.phase for view in views}
    missing_phases = [phase for phase in declared_phases if phase not in winner_phases]
    phases: list[dict[str, Any]] = []
    for view in views:
        params, redacted = _visible_winner_params(view.params, visible_params)
        source = _winner_source_or_default(view, view.phase)
        winner_source = {
            "kind": source.kind,
            "phase": source.phase,
            "trial_number": source.trial_number,
            "study": source.study,
        }
        source_generation_id = source.generation_id
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
    run: dict[str, Any] | None,
    *,
    result_source: ResultSource,
    elapsed_seconds: int | None,
) -> dict[str, Any]:
    """Build the ``get_run_status`` payload from the path-free read_status dict.

    ``status`` is the read_status output (already path-free). ``run`` is the
    process-level state for a specific run_id, or None for an experiment-level
    query with no recorded runs. Timing fields are counts of seconds computed
    by the server - durations only, never timestamps of operator activity or
    anything path-shaped.

    ``current_generation_id``/``published_generation_id``/
    ``represented_generation_id``/``is_published`` follow the identity
    contract :func:`phasesweep.engine.read.read_status` defines:
    ``attempts_launched_this_run``/``terminal_trials_this_run`` describe
    ``current_generation_id`` in a live (non-pinned) read, or the pinned
    generation itself for a run-scoped read; ``winner_present`` and the
    top-level ``summary_present`` describe ``represented_generation_id``.
    ``is_published`` says whether the represented generation is the actual
    published one -- ``False`` for a pinned read of a failed-publication
    generation, even though its winners still show.

    ``publication_integrity`` carries the tri-state verdict through unchanged
    (review v0.5.18 / finding F4). The accompanying ``publication_error`` is
    deliberately *not* forwarded: the enum is what an agent must branch on,
    and the free-text detail belongs to the operator-facing CLI, which is the
    surface trusted with local specifics.

    ``result_context``, ``published_config_matches_current`` and
    ``result_phase_plan`` are forwarded rather than dropped (review v0.5.16 /
    blocker 4). The ``phases`` list describes the *current* config's phases --
    what a further run would execute -- so under a config edited since
    publication it can legitimately show no winners beside ``is_published:
    true``. Without these three the agent had no way to tell that apart from an
    empty publication.

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
        terminal_trials_before_run = max(
            0,
            terminal_trials_total - terminal_trials_this_run,
        )
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
        "current_generation_id": status["current_generation_id"],
        "published_generation_id": status["published_generation_id"],
        "represented_generation_id": status["represented_generation_id"],
        "is_published": status["is_published"],
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
