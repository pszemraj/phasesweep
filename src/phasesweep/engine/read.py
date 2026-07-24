"""Read-only views of experiment results for status and winner reporting.

This module is the single public surface for *reading* run state without
launching anything or reaching into engine-private path helpers. The MCP
layer is the primary consumer; the CLI consumes a narrow slice of it (see
``_with_generation_identity`` in ``cli.py``) for its generation-identity
split, so winner and status shapes have exactly one definition.

Reads here are intentionally permissive. They report whatever is on disk -
including partial runs - and never raise on a missing winner. They do NOT
re-verify phase fingerprints: that check belongs to the resume path in
``engine.state._load_winner``, not to a status read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from phasesweep.config import Experiment
from phasesweep.config.common import SAFE_NAME_PATTERN, _validate_safe_name
from phasesweep.engine.optuna import _phase_trial_stats
from phasesweep.engine.state import (
    WinnerSource,
    WinnerSourceKind,
    _generation_path,
    _generation_summary_path,
    _generation_winner_path,
    _last_successful_generation_id,
    _parse_winner_source,
    _published_summary_path_for,
    _published_winner_path,
    _published_winner_path_for,
)
from phasesweep.evidence.models import objective_evidence_assurance


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
    generation_id: str | None = None
    attempt_id: str | None = None
    source: WinnerSource | None = None
    promotion: dict[str, Any] | None = None


def _phase_status_payloads(
    experiment: Experiment,
    *,
    include_winner_path: bool,
    trial_counts: Mapping[str, dict[str, int]] | None = None,
    generation_trial_counts: Mapping[str, dict[str, int]] | None = None,
    trial_data_available: Mapping[str, bool] | None = None,
    winner_scope_generation_id: str | None = None,
    pinned: bool = False,
) -> list[dict[str, Any]]:
    """Build per-phase status payloads for CLI and MCP readers.

    ``winner_scope_generation_id`` must already be resolved by the caller
    exactly once (e.g. a single :func:`phasesweep.engine.state._last_successful_generation_id`
    call, or a caller-pinned id) and is reused for every phase in this one
    call -- this function never re-resolves the last-success pointer itself,
    so one status object spanning several phases can never mix identities
    from two different pointer resolutions (review v0.5.15 / blocker 3).

    :param Experiment experiment: Parsed experiment whose phase study counts and winner files should be inspected.
    :param bool include_winner_path: If true, include the operator-facing winner path; otherwise return only a boolean winner flag.
    :param Mapping[str, dict[str, int]] | None trial_counts: Optional pre-read counts keyed by phase name.
    :param Mapping[str, dict[str, int]] | None generation_trial_counts: Optional counts for the represented generation, keyed by phase name.
    :param Mapping[str, bool] | None trial_data_available: Optional storage-read
        availability keyed by phase name. Included only in the path-free status
        view consumed by MCP.
    :param str | None winner_scope_generation_id: Already-resolved generation id
        whose winner files are represented. When ``pinned`` is ``False``
        (default), this is treated as an already-captured last-success id and
        legacy-fallback semantics apply when it is ``None``. When ``pinned``
        is ``True``, this is a caller-known generation id read directly with
        no legacy fallback.
    :param bool pinned: Whether ``winner_scope_generation_id`` names an exact,
        caller-pinned generation rather than an already-captured last-success id.
    :return list[dict[str, Any]]: One status payload per phase in declaration order.
    """
    phases: list[dict[str, Any]] = []
    for phase in experiment.phases:
        winner_path: Path | None
        if pinned:
            assert winner_scope_generation_id is not None
            winner_path = _generation_winner_path(
                experiment, winner_scope_generation_id, phase.name
            )
        else:
            winner_path = _published_winner_path_for(
                experiment, winner_scope_generation_id, phase.name
            )
        winner_present = winner_path is not None and winner_path.is_file()
        counts = (
            _phase_trial_stats(experiment, phase).counts
            if trial_counts is None
            else trial_counts[phase.name]
        )
        payload: dict[str, Any] = {
            "trials": counts,
            "running": counts.get("RUNNING", 0),
            "n_trials": phase.n_trials,
            "completed": counts.get("COMPLETE", 0),
            "generation_trials": (
                generation_trial_counts[phase.name] if generation_trial_counts is not None else {}
            ),
        }
        if include_winner_path:
            payload.update(
                {"name": phase.name, "winner": str(winner_path) if winner_present else None}
            )
        else:
            payload.update(
                {
                    "phase": phase.name,
                    "winner_present": winner_present,
                    "trial_data_available": (
                        trial_data_available[phase.name]
                        if trial_data_available is not None
                        else True
                    ),
                }
            )
        phases.append(payload)
    return phases


def read_winner(
    experiment: Experiment,
    phase_name: str,
    *,
    generation_id: str | None = None,
) -> PhaseWinnerView | None:
    """Read a single phase's persisted winner, or ``None`` if not yet written.

    Args:
        experiment: Parsed experiment config; supplies the metric name used to
            pull the scalar out of the ``metric`` block of ``winner.yaml``.
        phase_name: Phase whose ``winner.yaml`` to read.
        generation_id: Optional generation whose immutable winner should be read.

    Returns:
        A :class:`PhaseWinnerView`, or ``None`` when the phase has no usable
        winner on disk: never run, still running, selection failed, or the file
        is malformed. A malformed read is treated as "not yet written" -
            consistent with this module's permissive contract and with
            ``_phase_trial_stats`` swallowing transient backend errors. The
        strict, fingerprint-verifying read used for ``--from-phase`` resume
        lives in ``engine.state._load_winner`` and is intentionally not
        relaxed here.

    """
    if generation_id is not None:
        _validate_safe_name("generation", generation_id)
    path = (
        _published_winner_path(experiment, phase_name)
        if generation_id is None
        else _generation_winner_path(experiment, generation_id, phase_name)
    )
    if path is None or not path.is_file():
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
        source_data = data.get("winner_source")
        if not isinstance(source_data, Mapping):
            return None
        source_kind = source_data.get("kind")
        if source_kind not in ("phase_trial", "promotion_baseline", "suite_baseline"):
            return None
        source = _parse_winner_source(source_data, cast(WinnerSourceKind, source_kind))
        return PhaseWinnerView(
            phase=phase_name,
            trial_number=int(data["trial_number"]),
            metric=float(metric_block[experiment.metric.name]),
            params=dict(params),
            effective_overrides=dict(effective_overrides),
            gates_passed=(all(bool(g.get("passed")) for g in gates) if gates else None),
            incomplete=bool(completion.get("incomplete", False)),
            generation_id=(
                str(data["generation_id"])
                if isinstance(data.get("generation_id"), str) and data["generation_id"]
                else None
            ),
            attempt_id=(
                str(data["attempt_id"])
                if isinstance(data.get("attempt_id"), str) and data["attempt_id"]
                else None
            ),
            source=source,
            promotion=(
                dict(data["promotion"]) if isinstance(data.get("promotion"), Mapping) else None
            ),
        )
    except (KeyError, ValueError, TypeError, OSError, yaml.YAMLError):
        # Partially-written or malformed file (or unlinked between the is_file
        # check and the read): report as no-winner-yet rather than raising.
        return None


def read_winners(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
) -> list[PhaseWinnerView]:
    """Read every persisted phase winner, in declared phase order.

    Phases without a winner yet are skipped, so the list length tells the
    caller how far the chain has progressed.

    Args:
        experiment: Parsed experiment config whose phases are read in order.
        generation_id: Optional generation whose immutable winners should be read.

    Returns:
        One :class:`PhaseWinnerView` per phase that has a winner on disk.

    """
    if generation_id is not None:
        _validate_safe_name("generation", generation_id)
    views = (
        read_winner(experiment, phase.name, generation_id=generation_id)
        for phase in experiment.phases
    )
    return [view for view in views if view is not None]


def _current_pointer_generation_id(experiment: Experiment) -> str | None:
    """Read the mutable current-generation pointer's own id, or ``None``.

    Unlike the last-success pointer, the current pointer is not validated
    against any artifact: it is progress bookkeeping, always the most recent
    invocation's own claim, and may legitimately name a failed or
    in-progress generation.

    :param Experiment experiment: Experiment whose current pointer is read.
    :return str | None: The recorded ``generation_id``, or ``None`` when the
        pointer is missing, unreadable, malformed, or unsafely named.
    """
    try:
        generation = yaml.safe_load(_generation_path(experiment).read_text())
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(generation, Mapping):
        return None
    raw_generation_id = generation.get("generation_id")
    if isinstance(raw_generation_id, str) and SAFE_NAME_PATTERN.fullmatch(raw_generation_id):
        return raw_generation_id
    return None


def read_status(experiment: Experiment, *, generation_id: str | None = None) -> dict[str, Any]:
    """Per-phase trial counts and winner presence, with no paths in the output.

    Trial counts come from ``_phase_trial_stats``, which reports empty counts
    for a study that does not exist yet, never creates one as a side effect,
    and swallows transient backend errors (e.g. a momentary SQLite lock while
    the runner writes) by reporting empty counts rather than raising.
    ``trial_data_available`` distinguishes a successful empty read from missing
    or unreadable storage so callers never treat ambiguous zeros as evidence.

    This resolves the current pointer and the last-success pointer *exactly
    once each* and reuses those two captured ids for every downstream fact in
    this call -- no helper reached from here re-resolves either pointer, so
    one status object can never mix identities from two different pointer
    reads (review v0.5.15 / blocker 3). Four identities are always reported,
    and each is truthful independent of ``generation_id``:

    - ``current_generation_id``: always the actual mutable current pointer
      (the most recent invocation's own claim, whether failed, in-progress,
      or a much older publication); *never* forced to equal a pinned
      ``generation_id``.
    - ``published_generation_id``: always the actual validated last-success
      pointer; *never* forced to equal a pinned ``generation_id`` either. A
      pinned read of a generation whose own publication failed correctly
      reports the *older* generation here, not the pinned one.
    - ``represented_generation_id``: the generation whose winner/summary
      facts this payload actually shows -- ``generation_id`` itself when
      pinned, otherwise the captured ``published_generation_id``.
    - ``is_published``: ``True`` only when ``represented_generation_id`` is
      not ``None`` and equals ``published_generation_id``. A pinned read of a
      failed-publication generation is ``is_published: False`` while still
      showing that generation's own (unpublished) winners.

    Winner/summary facts (``winner_present``, top-level ``summary_present``)
    scope to ``represented_generation_id``. ``generation_trials`` scopes to
    ``current_generation_id`` in default mode (live progress of whatever is
    currently running/most recent) but to the *pinned* id in pinned mode --
    a pinned caller wants that specific generation's own trial counts, not
    whatever else may be current by the time the read happens. ``trials``,
    ``running``, ``completed``, and ``trial_data_available`` are cumulative,
    all-time counts for the phase's study and are not generation-scoped.

    Args:
        experiment: Parsed experiment config whose phases are inspected.
        generation_id: Optional invocation identity to pin the read's
            *represented* generation: ``represented_generation_id`` and the
            ``generation_trials``/winner/summary scope all equal this id,
            while ``current_generation_id`` and ``published_generation_id``
            remain the actual (possibly different) pointers. Used by callers
            (e.g. MCP per-run reads) that already know which generation they
            mean and want its own view of itself. When omitted (the
            default), ``represented_generation_id`` is the captured
            ``published_generation_id`` and ``generation_trials`` scopes to
            the captured ``current_generation_id``.

    Returns:
        A path-free mapping with the experiment name, the four identity
        fields above, the metric descriptor, a per-phase list of trial counts
        plus winner presence, and whether the represented summary has been written.

    """
    current_generation_id = _current_pointer_generation_id(experiment)
    published_generation_id = _last_successful_generation_id(experiment)

    if generation_id is None:
        represented_generation_id = published_generation_id
        trial_scope_generation_id = current_generation_id
        winner_scope_generation_id = published_generation_id
        pinned = False
    else:
        _validate_safe_name("generation", generation_id)
        represented_generation_id = generation_id
        trial_scope_generation_id = generation_id
        winner_scope_generation_id = generation_id
        pinned = True

    is_published = (
        represented_generation_id is not None
        and represented_generation_id == published_generation_id
    )

    phase_stats = {phase.name: _phase_trial_stats(experiment, phase) for phase in experiment.phases}
    summary_path: Path | None
    if pinned:
        assert winner_scope_generation_id is not None
        summary_path = _generation_summary_path(experiment, winner_scope_generation_id)
    else:
        summary_path = _published_summary_path_for(experiment, winner_scope_generation_id)
    return {
        "experiment": experiment.experiment,
        "current_generation_id": current_generation_id,
        "published_generation_id": published_generation_id,
        "represented_generation_id": represented_generation_id,
        "is_published": is_published,
        "metric": {
            "name": experiment.metric.name,
            "goal": experiment.metric.goal,
            "objective_evidence": objective_evidence_assurance(experiment.metric.extractor),
        },
        "phases": _phase_status_payloads(
            experiment,
            include_winner_path=False,
            trial_counts={name: stats.counts for name, stats in phase_stats.items()},
            generation_trial_counts={
                name: (
                    stats.generation_counts.get(trial_scope_generation_id, {})
                    if trial_scope_generation_id
                    else {}
                )
                for name, stats in phase_stats.items()
            },
            trial_data_available={name: stats.available for name, stats in phase_stats.items()},
            winner_scope_generation_id=winner_scope_generation_id,
            pinned=pinned,
        ),
        "summary_present": summary_path is not None and summary_path.is_file(),
    }
