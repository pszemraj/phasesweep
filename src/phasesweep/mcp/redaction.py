"""Outbound payload construction. Whitelist, do not blacklist.

Payloads are built only from typed, path-free views (PhaseWinnerView, the
read_status dict, catalog summaries). There is no path to interpolate a
trial_command, env value, or storage URL into a result, so there is nothing to
redact after the fact.
"""

from __future__ import annotations

from typing import Any

from phasesweep.engine import PhaseWinnerView


def winners_payload(experiment_id: str, views: list[PhaseWinnerView]) -> dict[str, Any]:
    """Build the ``get_winners`` payload from path-free phase-winner views.

    MCP output exposes sampled ``params`` only. ``effective_overrides`` can
    include config-authored fixed or inherited values, so it is intentionally
    kept out of agent-visible tool results.

    :param str experiment_id: Catalog id whose winners are being returned.
    :param list[PhaseWinnerView] views: Path-free winner views read from engine state.
    :return dict[str, Any]: MCP-safe winners payload.
    """
    return {
        "experiment_id": experiment_id,
        "phases": [
            {
                "phase": v.phase,
                "trial_number": v.trial_number,
                "metric": v.metric,
                "params": v.params,
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
) -> dict[str, Any]:
    """Build the ``get_status`` payload from the path-free read_status dict.

    ``status`` is the read_status output (already path-free). ``run`` is the
    process-level state for a specific run_id, or None for an experiment-level
    query with no recorded runs.

    :param str experiment_id: Catalog id whose status is being returned.
    :param dict[str, Any] status: Path-free status payload from ``read_status``.
    :param dict[str, Any] | None run: Optional path-free detached-run state.
    :return dict[str, Any]: MCP-safe status payload.
    """
    return {
        "experiment_id": experiment_id,
        "metric": status["metric"],
        "phases": status["phases"],
        "summary_present": status["summary_present"],
        "run": run,
    }
