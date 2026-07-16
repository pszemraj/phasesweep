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


def visible_winner_params(params: dict[str, Any], policy: VisibleParamsPolicy) -> dict[str, Any]:
    """Filter sampled winner params according to the catalog visibility policy.

    :param dict[str, Any] params: Sampled winner params keyed by parameter name.
    :param VisibleParamsPolicy policy: Catalog ``visible_params`` setting: ``"all"``
        exposes every value, ``"none"`` redacts every value, and a list of keys
        exposes only those keys and redacts the rest.
    :return dict[str, Any]: ``params`` with each value either kept as-is or
        replaced with the ``"<redacted>"`` sentinel per ``policy``.
    """
    if policy == "all":
        return dict(params)
    if policy == "none":
        return {key: "<redacted>" for key in params}
    visible = set(policy)
    return {key: value if key in visible else "<redacted>" for key, value in params.items()}


def params_redacted(params: dict[str, Any], policy: VisibleParamsPolicy) -> bool:
    """Return whether the visibility policy withholds any of ``params``.

    Computed from the policy, not by scanning for the sentinel string, so a
    literal ``"<redacted>"`` param value can never masquerade as policy.

    :param dict[str, Any] params: Sampled winner params keyed by parameter name.
    :param VisibleParamsPolicy policy: Catalog ``visible_params`` setting.
    :return bool: ``True`` when at least one value is replaced by the sentinel.
    """
    if policy == "all":
        return False
    if policy == "none":
        return bool(params)
    visible = set(policy)
    return any(key not in visible for key in params)


def winners_payload(
    experiment_id: str,
    views: list[PhaseWinnerView],
    *,
    visible_params: VisibleParamsPolicy = "none",
) -> dict[str, Any]:
    """Build the ``get_winners`` payload from path-free phase-winner views.

    MCP output exposes sampled ``params`` only, and values are redacted unless
    the catalog explicitly allows them. ``effective_overrides`` can include
    config-authored fixed or inherited values, so it is intentionally kept out
    of agent-visible tool results.

    :param str experiment_id: Catalog id whose winners are being returned.
    :param list[PhaseWinnerView] views: Path-free winner views read from engine state.
    :param VisibleParamsPolicy visible_params: Catalog policy for sampled param values.
    :return dict[str, Any]: MCP-safe winners payload.
    """
    return {
        "experiment_id": experiment_id,
        "phases": [
            {
                "phase": v.phase,
                "trial_number": v.trial_number,
                "metric": v.metric,
                "params": visible_winner_params(v.params, visible_params),
                "params_redacted": params_redacted(v.params, visible_params),
                "gates_passed": v.gates_passed,
                "incomplete": v.incomplete,
            }
            for v in views
        ],
    }


def status_payload(
    experiment_id: str,
    status: dict[str, Any],
    run: dict[str, Any] | None,
    *,
    elapsed_seconds: int | None,
    poll_after_seconds: int,
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
    :param int | None elapsed_seconds: Seconds since launch (running) or total
        run duration (terminal); ``None`` without an associated run.
    :param int poll_after_seconds: Suggested wait before the next status call.
    :return dict[str, Any]: MCP-safe status payload.
    """
    return {
        "experiment_id": experiment_id,
        "metric": status["metric"],
        "phases": status["phases"],
        "summary_present": status["summary_present"],
        "run": run,
        "elapsed_seconds": elapsed_seconds,
        "poll_after_seconds": poll_after_seconds,
    }
