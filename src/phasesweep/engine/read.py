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

import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml

from phasesweep.config import Experiment
from phasesweep.engine.optuna import _phase_trial_counts, _phase_trial_stats
from phasesweep.engine.state import _summary_path, _winner_path


@dataclass(frozen=True)
class PhaseWinnerView:
    """A phase winner reduced to the fields a caller may safely see.

    Notably absent: any filesystem path, the trial command, the environment,
    and the storage URL. ``effective_overrides`` is included for trusted engine
    and CLI readers that need the composed hyperparameter set; MCP filters it
    out before returning agent-visible payloads.
    """

    phase: str
    trial_number: int
    metric: float
    params: dict[str, Any]
    effective_overrides: dict[str, Any]
    gates_passed: bool | None  # None when the phase declared no gates
    incomplete: bool  # True when a wallclock timeout produced a partial winner


def _phase_status_payloads(
    experiment: Experiment,
    *,
    include_winner_path: bool,
    trial_counts: Mapping[str, dict[str, int]] | None = None,
    trial_data_available: Mapping[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Build per-phase status payloads for CLI and MCP readers.

    :param Experiment experiment: Parsed experiment whose phase study counts and winner files should be inspected.
    :param bool include_winner_path: If true, include the operator-facing winner path; otherwise return only a boolean winner flag.
    :param Mapping[str, dict[str, int]] | None trial_counts: Optional pre-read counts keyed by phase name.
    :param Mapping[str, bool] | None trial_data_available: Optional storage-read
        availability keyed by phase name. Included only in the path-free status
        view consumed by MCP.
    :return list[dict[str, Any]]: One status payload per phase in declaration order.
    """
    phases: list[dict[str, Any]] = []
    for phase in experiment.phases:
        winner_path = _winner_path(experiment, phase.name)
        counts = (
            _phase_trial_counts(experiment, phase)
            if trial_counts is None
            else trial_counts[phase.name]
        )
        payload: dict[str, Any] = {
            "trials": counts,
            "running": counts.get("RUNNING", 0),
            "n_trials": phase.n_trials,
            "completed": counts.get("COMPLETE", 0),
        }
        if include_winner_path:
            payload.update(
                {"name": phase.name, "winner": str(winner_path) if winner_path.is_file() else None}
            )
        else:
            payload.update(
                {
                    "phase": phase.name,
                    "winner_present": winner_path.is_file(),
                    "trial_data_available": (
                        trial_data_available[phase.name]
                        if trial_data_available is not None
                        else True
                    ),
                }
            )
        phases.append(payload)
    return phases


def read_winner(experiment: Experiment, phase_name: str) -> PhaseWinnerView | None:
    """Read a single phase's persisted winner, or ``None`` if not yet written.

    Args:
        experiment: Parsed experiment config; supplies the metric name used to
            pull the scalar out of the ``metric`` block of ``winner.yaml``.
        phase_name: Phase whose ``winner.yaml`` to read.

    Returns:
        A :class:`PhaseWinnerView`, or ``None`` when the phase has no usable
        winner on disk: never run, still running, selection failed, or the file
        is malformed. A malformed read is treated as "not yet written" -
        consistent with this module's permissive contract and with
        ``_phase_trial_counts`` swallowing transient backend errors. The
        strict, fingerprint-verifying read used for ``--from-phase`` resume
        lives in ``engine.state._load_winner`` and is intentionally not
        relaxed here.

    """
    path = _winner_path(experiment, phase_name)
    if not path.is_file():
        return None
    try:
        loaded = yaml.safe_load(path.read_text())
        if loaded is None:
            data: Mapping[str, Any] = {}
        elif isinstance(loaded, Mapping):
            data = loaded
        else:
            return None
        # winner.yaml stores metric as {<metric_name>: value, "goal": ...}; pull
        # the value by the configured metric name rather than positionally.
        metric_block = data.get("metric") or {}
        if not isinstance(metric_block, Mapping):
            return None
        gates = [g for g in (data.get("gates") or []) if isinstance(g, dict)]
        completion = data.get("completion") or {}
        if not isinstance(completion, Mapping):
            return None
        params = data.get("params") or {}
        if not isinstance(params, Mapping):
            return None
        effective_overrides = data.get("effective_overrides") or {}
        if not isinstance(effective_overrides, Mapping):
            return None
        return PhaseWinnerView(
            phase=phase_name,
            trial_number=int(data["trial_number"]),
            metric=float(metric_block[experiment.metric.name]),
            params=dict(params),
            effective_overrides=dict(effective_overrides),
            gates_passed=(all(bool(g.get("passed")) for g in gates) if gates else None),
            incomplete=bool(completion.get("incomplete", False)),
        )
    except (KeyError, ValueError, TypeError, OSError, yaml.YAMLError):
        # Partially-written or malformed file (or unlinked between the is_file
        # check and the read): report as no-winner-yet rather than raising.
        return None


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

    Trial counts come from ``_phase_trial_stats``, which returns ``{}`` for a
    study that does not exist yet, never creates one as a side effect, and
    swallows transient backend errors (e.g. a momentary SQLite lock while the
    runner writes) by returning ``{}`` for that phase rather than raising.
    ``trial_data_available`` distinguishes a successful empty read from missing
    or unreadable storage so callers never treat ambiguous zeros as evidence.
    ``median_trial_seconds`` follows the same permissive contract: it is the
    median wall duration of COMPLETE trials across all phases, or ``None``
    while nothing has finished — callers use it to size their poll interval.

    Args:
        experiment: Parsed experiment config whose phases are inspected.

    Returns:
        A path-free mapping with the experiment name, metric descriptor, a
        per-phase list of trial counts plus winner presence, the median
        completed-trial duration, and whether the experiment summary has been
        written.

    """
    phase_stats = {phase.name: _phase_trial_stats(experiment, phase) for phase in experiment.phases}
    durations = [seconds for stats in phase_stats.values() for seconds in stats.completed_durations]
    return {
        "experiment": experiment.experiment,
        "metric": {"name": experiment.metric.name, "goal": experiment.metric.goal},
        "phases": _phase_status_payloads(
            experiment,
            include_winner_path=False,
            trial_counts={name: stats.counts for name, stats in phase_stats.items()},
            trial_data_available={name: stats.available for name, stats in phase_stats.items()},
        ),
        "summary_present": _summary_path(experiment).is_file(),
        "median_trial_seconds": statistics.median(durations) if durations else None,
    }
