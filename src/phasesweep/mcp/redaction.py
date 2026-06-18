"""Outbound payload construction. Whitelist, do not blacklist.

Payloads are built only from typed, path-free views (PhaseWinnerView, the
read_status dict, catalog summaries). There is no path to interpolate a
trial_command, env value, or storage URL into a result, so there is nothing to
redact after the fact. ``assert_no_sensitive`` makes that property checkable.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from phasesweep.engine import PhaseWinnerView


def winners_payload(experiment_id: str, views: list[PhaseWinnerView]) -> dict[str, Any]:
    """Build the ``get_winners`` payload from path-free phase-winner views.

    MCP output exposes sampled ``params`` only. ``effective_overrides`` can
    include config-authored fixed or inherited values, so it is intentionally
    kept out of agent-visible tool results.
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
    """
    return {
        "experiment_id": experiment_id,
        "metric": status["metric"],
        "phases": status["phases"],
        "summary_present": status["summary_present"],
        "run": run,
    }


def assert_no_sensitive(payload: Any, sensitive: Iterable[str]) -> None:
    """Raise ``AssertionError`` if any string leaf contains a sensitive needle.

    Defensive check for tests and an optional server debug mode. ``sensitive``
    is the set of values that must never appear: the trial command, the storage
    URL, and every env value for the experiment.
    """
    needles = [s for s in sensitive if s]

    def walk(node: Any) -> None:
        if isinstance(node, str):
            for needle in needles:
                assert needle not in node, f"sensitive value leaked into payload: {needle!r}"
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(key)
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(payload)
