"""Read-only views of experiment results for status and winner reporting.

This module is the single public surface for *reading* run state without
launching anything or reaching into engine-private path helpers. The CLI may
adopt it later; the MCP layer consumes it now, so winner and status shapes
have exactly one definition.

Reads here are intentionally permissive. They report whatever is on disk -
including partial runs - and never raise on a missing winner. They do NOT
re-verify phase fingerprints: that check belongs to the resume path in
``engine.state._load_winner``, not to a status read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from phasesweep.config import Experiment
from phasesweep.engine.optuna import _phase_trial_counts
from phasesweep.engine.state import _summary_path, _winner_path


@dataclass(frozen=True)
class PhaseWinnerView:
    """A phase winner reduced to the fields a caller may safely see.

    Notably absent: any filesystem path, the trial command, the environment,
    and the storage URL. ``effective_overrides`` is the composed hyperparameter
    set (inherited + fixed + sampled) - i.e. "the best parameters" - and is the
    point of this view, not a leak.
    """

    phase: str
    trial_number: int
    metric: float
    params: dict[str, Any]
    effective_overrides: dict[str, Any]
    gates_passed: bool | None  # None when the phase declared no gates
    incomplete: bool  # True when a wallclock timeout produced a partial winner


def read_winner(experiment: Experiment, phase_name: str) -> PhaseWinnerView | None:
    """Read a single phase's persisted winner, or ``None`` if not yet written.

    Args:
        experiment: Parsed experiment config; supplies the metric name used to
            pull the scalar out of the ``metric`` block of ``winner.yaml``.
        phase_name: Phase whose ``winner.yaml`` to read.

    Returns:
        A :class:`PhaseWinnerView`, or ``None`` when the phase has no winner on
        disk yet (still running, never run, or selection failed).

    """
    path = _winner_path(experiment, phase_name)
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    # winner.yaml stores metric as {<metric_name>: value, "goal": ...}; pull the
    # value by the configured metric name rather than positionally.
    metric_block = data.get("metric") or {}
    gates = [g for g in (data.get("gates") or []) if isinstance(g, dict)]
    completion = data.get("completion") or {}
    return PhaseWinnerView(
        phase=phase_name,
        trial_number=int(data["trial_number"]),
        metric=float(metric_block[experiment.metric.name]),
        params=dict(data.get("params") or {}),
        effective_overrides=dict(data.get("effective_overrides") or {}),
        gates_passed=(all(bool(g.get("passed")) for g in gates) if gates else None),
        incomplete=bool(completion.get("incomplete", False)),
    )


def read_winners(experiment: Experiment) -> list[PhaseWinnerView]:
    """Read every persisted phase winner, in declared phase order.

    Phases without a winner yet are skipped, so the list length tells the
    caller how far the chain has progressed.

    Args:
        experiment: Parsed experiment config whose phases are read in order.

    Returns:
        One :class:`PhaseWinnerView` per phase that has a winner on disk.

    """
    views = (read_winner(experiment, phase.name) for phase in experiment.phases)
    return [view for view in views if view is not None]


def read_status(experiment: Experiment) -> dict[str, Any]:
    """Per-phase trial counts and winner presence, with no paths in the output.

    Trial counts come from ``_phase_trial_counts``, which returns ``{}`` for a
    study that does not exist yet, never creates one as a side effect, and
    swallows transient backend errors (e.g. a momentary SQLite lock while the
    runner writes) by returning ``{}`` for that phase rather than raising.

    Args:
        experiment: Parsed experiment config whose phases are inspected.

    Returns:
        A path-free mapping with the experiment name, metric descriptor, a
        per-phase list of trial counts plus winner presence, and whether the
        experiment summary has been written.

    """
    phases: list[dict[str, Any]] = []
    for phase in experiment.phases:
        counts = _phase_trial_counts(experiment, phase)
        phases.append(
            {
                "phase": phase.name,
                "trials": counts,  # {"COMPLETE": n, "RUNNING": m, "FAIL": k, ...}
                "running": counts.get("RUNNING", 0),
                "winner_present": _winner_path(experiment, phase.name).is_file(),
            }
        )
    return {
        "experiment": experiment.experiment,
        "metric": {"name": experiment.metric.name, "goal": experiment.metric.goal},
        "phases": phases,
        "summary_present": _summary_path(experiment).is_file(),
    }
