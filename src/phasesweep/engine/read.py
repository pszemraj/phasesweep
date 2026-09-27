"""Read-only views of experiment results for status and winner reporting.

This module is the single public surface for *reading* run state without
launching anything or reaching into engine-private path helpers. The MCP
layer is the primary consumer; the CLI consumes a narrow slice of it (see
:func:`phasesweep.engine.run.experiment_status`) for its path-bearing status
view, so winner and status shapes have exactly one definition.

Reads here are permissive about partial run state and never raise on a missing
winner. They still fail closed when the artifact root belongs to another
storage ledger: combining one database's trial counts with another database's
publication is not a partial result. They do NOT re-verify phase fingerprints:
that check belongs to the resume path in ``engine.artifacts._load_winner``, not to
a status read.

A published generation is read once, by publication validation, and its facts
come from what that validation read. Only a generation that is not the
validated publication (pinned, unfinished, or failed) is read from its files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml

from phasesweep.config import Experiment
from phasesweep.config.common import SAFE_NAME_PATTERN, _validate_safe_name
from phasesweep.config.models import _metric_semantics_payload
from phasesweep.engine.artifact_roots import _artifact_root_binding_applies
from phasesweep.engine.fingerprints import _experiment_semantic_fingerprint
from phasesweep.engine.ledger import read_trial_stats, validate_ledger
from phasesweep.engine.optuna import (
    _published_phase_trial_refs,
    _published_trial_history_available,
)
from phasesweep.engine.paths import (
    _generation_path,
    _generation_summary_path,
    _generation_winner_path,
)
from phasesweep.engine.publication import PublicationPointer, _resolve_publication_pointer
from phasesweep.engine.publication_validation import _parse_phase_plan
from phasesweep.engine.state import (
    GENERATION_SUMMARY_SCHEMA_VERSION,
    PublicationState,
    WinnerSource,
    _parse_winner_source,
)
from phasesweep.evidence.models import EXTRACTOR_KINDS, _ObjectiveEvidenceFields

ResultContext: TypeAlias = Literal["represented_generation", "current_config"]
"""Which config's semantics a result payload's labels were read under.

``"represented_generation"``: the labels come from the represented
generation's own recorded summary, so they describe the result as it was
produced. ``"current_config"``: no represented result is available, so the
currently loaded config describes the status payload.
"""


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
    # The metric semantics the winner itself recorded (review v0.5.16 /
    # blocker 4): a persisted winner is historical evidence, so its value is
    # reported under the name and goal it was optimized for, never relabeled
    # by whatever metric the current config declares.
    metric_name: str | None = None
    metric_goal: str | None = None


def _phase_status_payloads(
    experiment: Experiment,
    *,
    trial_counts: Mapping[str, dict[str, int]],
    generation_trial_counts: Mapping[str, dict[str, int]],
    trial_data_available: Mapping[str, bool],
    running_attempts: Mapping[str, list[dict[str, Any]] | None],
    unavailable_published_phases: set[str],
    winners_present: set[str],
) -> list[dict[str, Any]]:
    """Build the path-free per-phase status payloads ``read_status`` returns.

    :func:`phasesweep.engine.run.experiment_status` builds the CLI's separate,
    path-bearing phase view from this function's output instead of asking it
    for paths, so no call here can ever put a filesystem path in its result.

    ``winners_present`` must come from the caller's single resolution of the
    represented generation, so one status object spanning several phases can
    never mix identities from two different pointer resolutions (review
    v0.5.15 / blocker 3).

    :param Experiment experiment: Parsed experiment whose phase study counts are reported.
    :param Mapping[str, dict[str, int]] trial_counts: Pre-read counts keyed by phase name.
    :param Mapping[str, dict[str, int]] generation_trial_counts: Counts for the represented generation, keyed by phase name.
    :param Mapping[str, bool] trial_data_available: Storage-read
        availability keyed by phase name. True includes confirmed absent studies
        with known zero counts; false means counts could not be established.
    :param Mapping[str, list[dict[str, Any]] | None] running_attempts:
        RUNNING trial identities keyed by phase name, from the same
        storage snapshot as ``trial_counts`` -- ``None`` for a phase whose
        trial data was unreadable. Included only in the path-free status view
        consumed by MCP, whose terminal snapshot must not reread studies to
        learn which rows the RUNNING count refers to (PR #5 review /
        reviewer 2, blocker 6).
    :param set[str] unavailable_published_phases: Published phases whose local
        trial identity could not be matched in the storage snapshot.
    :param set[str] winners_present: Phases whose winner the represented
        generation holds.
    :return list[dict[str, Any]]: One status payload per phase in declaration order.
    """
    phases: list[dict[str, Any]] = []
    for phase in experiment.phases:
        counts = trial_counts[phase.name]
        payload: dict[str, Any] = {
            "trials": counts,
            "running": counts.get("RUNNING", 0),
            "n_trials": phase.n_trials,
            "completed": counts.get("COMPLETE", 0),
            "generation_trials": generation_trial_counts[phase.name],
            "trial_data_available": trial_data_available[phase.name],
            "published_study_unavailable": phase.name in unavailable_published_phases,
            "phase": phase.name,
            "winner_present": phase.name in winners_present,
            "running_attempts": running_attempts[phase.name],
        }
        phases.append(payload)
    return phases


def _winner_view(data: Mapping[str, Any], phase_name: str) -> PhaseWinnerView | None:
    """Reduce one decoded winner payload using the permissive read contract.

    :param Mapping[str, Any] data: Decoded winner payload.
    :param str phase_name: Phase name exposed by the winner.
    :return PhaseWinnerView | None: The winner view, or ``None`` when the
        payload is malformed.
    """
    try:
        # winner.yaml stores metric as {<metric_name>: value, "goal": ...}. The
        # winner is historical evidence: pull the value by the name the file
        # itself recorded, NOT the currently configured metric name — reading
        # a published x/minimize result through a config renamed to y used to
        # silently drop the winner entirely (review v0.5.16 / blocker 4).
        metric_block = data.get("metric") or {}
        if not isinstance(metric_block, Mapping):
            return None
        stored_metric_names = [key for key in metric_block if key != "goal"]
        if len(stored_metric_names) != 1:
            return None
        stored_metric_name = str(stored_metric_names[0])
        stored_goal = metric_block.get("goal")
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
        source = _parse_winner_source(data.get("winner_source"), expected_phase=phase_name)
        return PhaseWinnerView(
            phase=phase_name,
            trial_number=int(data["trial_number"]),
            metric=float(metric_block[stored_metric_name]),
            metric_name=stored_metric_name,
            metric_goal=str(stored_goal) if isinstance(stored_goal, str) else None,
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
        )
    except (KeyError, ValueError, TypeError):
        return None


def _read_winner_path(path: Path, phase_name: str) -> PhaseWinnerView | None:
    """Read one unvalidated winner file using the permissive read contract.

    Only a generation that is not the validated publication is read this way:
    a pinned, unfinished, or failed one. A published winner is never reread;
    it comes from the bytes publication validation read.

    :param Path path: Winner path in the represented generation.
    :param str phase_name: Phase name exposed by the winner.
    :return PhaseWinnerView | None: Parsed winner, or ``None`` when absent or malformed.
    """
    if not path.is_file():
        return None
    try:
        loaded = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        # Partially-written or malformed file (or unlinked between the is_file
        # check and the read): report as no-winner-yet rather than raising.
        return None
    if not isinstance(loaded, Mapping):
        return None
    return _winner_view(loaded, phase_name)


def _represented_winners(
    experiment: Experiment,
    publication: PublicationPointer,
    represented_generation_id: str | None,
    phase_names: Sequence[str],
) -> list[PhaseWinnerView]:
    """Return the represented generation's winners, in ``phase_names`` order.

    When the represented generation is the validated publication, every winner
    comes from the bytes that validation read and nothing is reread. Any other
    generation is read permissively from its files.

    :param Experiment experiment: Experiment whose artifact tree is read.
    :param PublicationPointer publication: This read's single pointer resolution.
    :param str | None represented_generation_id: Generation whose winners are
        represented, or ``None`` when there is none.
    :param Sequence[str] phase_names: Safe phase names to enumerate.
    :return list[PhaseWinnerView]: One view per named phase that has a usable winner.
    """
    if represented_generation_id is None:
        return []
    validated = publication.validated
    views: list[PhaseWinnerView | None]
    if validated is not None and validated.generation_id == represented_generation_id:
        views = [
            _winner_view(validated.winners[name].payload, name)
            for name in phase_names
            if name in validated.winners
        ]
    else:
        views = [
            _read_winner_path(
                _generation_winner_path(experiment, represented_generation_id, name), name
            )
            for name in phase_names
        ]
    return [view for view in views if view is not None]


def read_winner(
    experiment: Experiment,
    phase_name: str,
    *,
    generation_id: str | None = None,
) -> PhaseWinnerView | None:
    """Read a single phase's persisted winner, or ``None`` if not yet written.

    Args:
        experiment: Parsed experiment config; supplies the artifact roots.
            The metric value is extracted under the name the winner file
            itself recorded, so a historical winner is never reinterpreted
            through the currently configured metric (review v0.5.16 /
            blocker 4).
        phase_name: Phase whose ``winner.yaml`` to read.
        generation_id: Optional generation whose immutable winner should be read.

    Returns:
        A :class:`PhaseWinnerView`, or ``None`` when the phase has no usable
        winner on disk: never run, still running, selection failed, or the file
        is malformed. A malformed read is treated as "not yet written" -
            consistent with this module's permissive contract and with
            ``read_trial_stats`` swallowing transient backend errors. The
        strict, fingerprint-verifying read used for ``--from-phase`` resume
        lives in ``engine.artifacts._load_winner`` and is intentionally not
        relaxed here.

    Raises:
        ValueError: If ``phase_name`` or ``generation_id`` is not a safe path
            component.

    """
    views = read_winners(experiment, generation_id=generation_id, phase_names=[phase_name])
    return views[0] if views else None


def read_winners(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    phase_names: Sequence[str] | None = None,
) -> list[PhaseWinnerView]:
    """Read every persisted phase winner, in declared (or supplied) phase order.

    Phases without a winner yet are skipped, so the list length tells the
    caller how far the chain has progressed.

    Args:
        experiment: Parsed experiment config whose phases are read in order.
        generation_id: Optional generation whose immutable winners should be read.
        phase_names: Optional explicit phase plan to enumerate instead of the
            currently configured phases, in the order given. A caller reading a
            *historical* generation passes that generation's own recorded plan
            (see :func:`_summary_phase_plan`), so a phase renamed in the config
            since publication neither hides the published winner nor reports
            the new name as missing (review v0.5.16 / blocker 4). Names are
            path components, so each is validated; the default keeps the
            declared-phase behavior every existing caller relies on.

    Returns:
        One :class:`PhaseWinnerView` per named phase that has a winner on disk.

    Raises:
        ValueError: If ``generation_id`` or a name in ``phase_names`` is not a
            safe path component. Plans parsed off disk are filtered before
            they reach here, so this only fires on a programming error.

    """
    validate_ledger(experiment)
    if generation_id is not None:
        _validate_safe_name("generation", generation_id)
    names = (
        [phase.name for phase in experiment.phases]
        if phase_names is None
        else [_validate_safe_name("phase", name) for name in phase_names]
    )
    publication = _resolve_publication_pointer(experiment)
    represented_generation_id = (
        generation_id if generation_id is not None else _published_generation_id(publication)
    )
    return _represented_winners(experiment, publication, represented_generation_id, names)


def _published_generation_id(publication: PublicationPointer) -> str | None:
    """Return the generation a pointer resolution may report as published.

    :param PublicationPointer publication: One pointer resolution.
    :return str | None: The validated generation id, or ``None`` when the
        resolution is anything but ``ok``.
    """
    return publication.generation_id if publication.state == "ok" else None


def _read_summary_payload(summary_path: Path | None) -> Mapping[str, Any] | None:
    """Read a represented generation's summary for historical interpretation.

    :param Path | None summary_path: Already-resolved summary path, or ``None``.
    :return Mapping[str, Any] | None: Parsed summary mapping, or ``None`` when
        absent, unreadable, or not a mapping.
    """
    if summary_path is None:
        return None
    try:
        payload = yaml.safe_load(summary_path.read_text())
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema_version") != GENERATION_SUMMARY_SCHEMA_VERSION:
        return None
    return payload


def _summary_phase_plan(summary_payload: Mapping[str, Any] | None) -> list[str] | None:
    """Return the phase names an unvalidated represented generation recorded.

    Only the plan's names are returned. The summary's other phase records
    (``phases``) carry composed hyperparameters that agent-facing payloads
    must never see; the plan is the design-time list the engine froze beside
    the metric semantics for exactly this purpose (engine/run.py's generation
    summary). The plan is parsed by the same rule publication validation
    applies, which refuses what this permissive read reports as no plan.

    :param Mapping[str, Any] | None summary_payload: Parsed generation summary, or ``None``.
    :return list[str] | None: Recorded phase names in execution order, or
        ``None`` when the summary is absent or records no usable plan.
    """
    if summary_payload is None:
        return None
    plan = _parse_phase_plan(summary_payload.get("phase_plan"))
    return None if plan is None else [phase.name for phase in plan]


_OBJECTIVE_EVIDENCE_KEYS = frozenset(_ObjectiveEvidenceFields.model_fields)
"""The ``objective_evidence`` keys, taken from the model its writers declare."""


def _recorded_objective_evidence(candidate: object) -> dict[str, str | bool] | None:
    """Return a complete current-format assurance payload, if present.

    :param object candidate: Summary ``objective_evidence`` value.
    :return dict[str, str | bool] | None: Complete recorded flags, or ``None``.
    """
    if not isinstance(candidate, Mapping) or set(candidate) != _OBJECTIVE_EVIDENCE_KEYS:
        return None
    if candidate.get("kind") not in EXTRACTOR_KINDS:
        return None
    if any(type(candidate[key]) is not bool for key in candidate if key != "kind"):
        return None
    return dict(candidate)


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


def read_status(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    comparison_experiment: Experiment | None = None,
) -> dict[str, Any]:
    """Per-phase trial counts and winner presence for one experiment.

    The payload is always path-free: no field ever carries a filesystem path.
    The redaction layer in ``phasesweep.mcp.redaction`` takes this function's
    output as already path-free and does not strip paths itself.
    :func:`phasesweep.engine.run.experiment_status` builds the CLI's separate,
    path-bearing phase view on top of this function's path-free ``phases``
    payload instead of asking this function for paths.

    Trial counts come from ``read_trial_stats``, which reports empty counts
    for a study that does not exist yet, never creates one as a side effect,
    and swallows transient backend errors (e.g. a momentary journal-ledger lock
    while the runner writes) by reporting empty counts rather than raising.
    ``trial_data_available`` distinguishes a successful empty read from missing
    or unreadable storage so callers never treat ambiguous zeros as evidence.
    The path-free phase payload also carries ``running_attempts``: the
    identities of the RUNNING trials those same counts describe, from the same
    snapshot, or ``None`` when ``trial_data_available`` is ``False``. It exists
    so a caller that must reconcile RUNNING rows -- the MCP terminal snapshot
    -- never needs a second, intolerant storage read after this one (PR #5
    review / reviewer 2, blocker 6).

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
    - ``is_published``: ``True`` when ``represented_generation_id`` is not
      ``None`` and equals ``published_generation_id``. A pinned read of a
      failed-publication generation is ``is_published: False`` while still
      showing that generation's own (unpublished) winners. Without a
      generation identity, status is unpublished.
    - ``publication_integrity``: ``"ok"`` / ``"absent"`` / ``"failed"`` /
      ``"permission_denied"`` --
      *why* ``published_generation_id`` is what it is (review v0.5.18 /
      finding F4). ``"absent"`` means nothing was ever published, a healthy
      state for a fresh tree; ``"failed"`` means a last-success pointer exists
      but its target no longer validates; ``"permission_denied"`` means this
      user cannot complete validation without implying corruption. Both are
      accompanied by a short, path-free ``publication_error``. These states
      used to be indistinguishable,
      which invited a re-run over corrupt evidence. A ``"failed"`` payload
      reports exactly the no-publication facts it always did -- nothing is
      fabricated from an unvalidated generation -- so ``published_generation_id``
      is ``None``, ``is_published`` is ``False``, and no winner reads as
      present.

    Two views coexist in this payload and must not be confused. The
    *config-progress* view -- the ``phases`` list, with its trial counts,
    targets, and per-phase winner presence -- describes the phases the current
    config declares, because that is what a further run would execute. The
    *result* view -- the metric descriptor, ``result_phase_plan``, and the
    publication identity fields -- describes the represented generation on its
    own terms. Under a config edited since publication the two legitimately
    disagree, and ``result_context`` / ``published_config_matches_current``
    say so explicitly rather than leaving a reader to conclude the publication
    is empty (review v0.5.16 / blocker 4).

    Winner/summary facts (``winner_present``, top-level ``summary_present``)
    scope to ``represented_generation_id``. ``generation_trials`` scopes to
    ``current_generation_id`` in default mode (live progress of whatever is
    currently running/most recent) but to the *pinned* id in pinned mode --
    a pinned caller wants that specific generation's own trial counts, not
    whatever else may be current by the time the read happens. ``trials``,
    ``running``, ``completed``, and ``trial_data_available`` are cumulative,
    all-time counts for the phase's study and are not generation-scoped.
    ``published_study_unavailable`` reports a current published phase whose local
    trial identity or recorded completion boundary could not be matched in storage.
    When both it and ``trial_data_available`` are true, history is confirmed missing,
    replaced, or incomplete; executing the phase is refused, while earlier phases
    may still load saved winners via ``from_phase``. When ``trial_data_available``
    is false, the history could not be completely inspected, including an incomplete
    journal append; a run can report cleanup uncertainty and
    require operator recovery before further MCP launches. Publication
    integrity describes the artifacts separately.

    :param Experiment experiment: Parsed experiment config whose phases are inspected.
    :param str | None generation_id: Optional invocation identity to pin the
        read's *represented* generation: ``represented_generation_id`` and the
        ``generation_trials``/winner/summary scope all equal this id, while
        ``current_generation_id`` and ``published_generation_id`` remain the
        actual (possibly different) pointers. Used by callers (e.g. MCP
        per-run reads) that already know which generation they mean and want
        its own view of itself. When omitted (the default),
        ``represented_generation_id`` is the captured
        ``published_generation_id`` and ``generation_trials`` scopes to the
        captured ``current_generation_id``.
    :param Experiment | None comparison_experiment: Optional current config to
        use only for drift comparison and the no-result status semantics.
        Artifact and trial reads still use ``experiment``. This lets a frozen
        run config locate its own tree while a run-scoped MCP read compares the
        represented result with the catalog config a future run would execute.
    :raises ArtifactRootConflictError: If the artifact root cannot be read or
        is bound to a different storage ledger or experiment.
    :raises ValueError: If ``generation_id`` is not a safe generation name.
    :return dict[str, Any]: A mapping with the experiment name, the four
        identity fields above, ``publication_integrity`` (plus
        ``publication_error`` only when validation failed or was denied), the metric
        descriptor, a per-phase list of trial counts plus winner presence, and
        whether the represented summary has been written -- always path-free.
        The metric descriptor is the *represented generation's own* recorded
        metric whenever its summary declares one
        (``result_context: "represented_generation"``), falling back to the
        current config only when no summary semantics exist
        (``result_context: "current_config"``) — a published x/minimize
        result is never relabeled by a config edited to y/maximize (review
        v0.5.16 / blocker 4). ``result_phase_plan`` is likewise the represented
        generation's own recorded phase plan, falling back to the current
        config's phase names when no represented summary exists — it is the
        plan the publication's winners must be enumerated under, and equals
        the ``phases`` list's names whenever the config has not been edited
        since.
        ``published_config_matches_current`` compares the summary's recorded
        config fingerprint against the current config's semantic fingerprint;
        ``None`` when the represented summary records no fingerprint.
    """
    status, _publication = _read_status(
        experiment,
        generation_id=generation_id,
        comparison_experiment=comparison_experiment,
    )
    return status


def read_result(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    comparison_experiment: Experiment | None = None,
) -> tuple[dict[str, Any], list[PhaseWinnerView]]:
    """Read status and the represented generation's winners from one pointer resolution.

    A caller that reports both must not pair :func:`read_status` with a
    separate :func:`read_winners`: two pointer resolutions could describe two
    different publications. Winners are enumerated under the status payload's
    ``result_phase_plan``, so a phase renamed since publication neither hides
    its published winner nor reports the new name as missing (review v0.5.16 /
    blocker 4). A published generation's winners are the bytes publication
    validation read; any other represented generation is read permissively.

    :param Experiment experiment: Parsed experiment config whose phases are inspected.
    :param str | None generation_id: Optional generation to represent, as for
        :func:`read_status`.
    :param Experiment | None comparison_experiment: Optional current config
        for drift comparison, as for :func:`read_status`.
    :raises ArtifactRootConflictError: If the artifact root cannot be read or
        is bound to a different storage ledger or experiment.
    :raises ValueError: If ``generation_id`` is not a safe generation name.
    :return tuple[dict[str, Any], list[PhaseWinnerView]]: The path-free
        :func:`read_status` payload and one winner view per recorded phase
        that has a usable winner.
    """
    status, publication = _read_status(
        experiment,
        generation_id=generation_id,
        comparison_experiment=comparison_experiment,
    )
    winners = _represented_winners(
        experiment,
        publication,
        status["represented_generation_id"],
        status["result_phase_plan"],
    )
    return status, winners


def _read_status(
    experiment: Experiment,
    *,
    generation_id: str | None,
    comparison_experiment: Experiment | None,
) -> tuple[dict[str, Any], PublicationPointer]:
    """Build the :func:`read_status` payload and return the resolution behind it.

    :param Experiment experiment: Parsed experiment config whose phases are inspected.
    :param str | None generation_id: Optional generation to represent.
    :param Experiment | None comparison_experiment: Optional current config for
        drift comparison.
    :raises ArtifactRootConflictError: If the artifact root cannot be read or
        is bound to a different storage ledger or experiment.
    :raises ValueError: If ``generation_id`` is not a safe generation name.
    :return tuple[dict[str, Any], PublicationPointer]: The status payload and
        the single pointer resolution every fact in it came from.
    """
    ledger = validate_ledger(experiment)
    current_generation_id = _current_pointer_generation_id(experiment)
    publication = _resolve_publication_pointer(experiment)
    published_generation_id = _published_generation_id(publication)

    if generation_id is None:
        represented_generation_id = published_generation_id
        trial_scope_generation_id = current_generation_id
    else:
        _validate_safe_name("generation", generation_id)
        represented_generation_id = generation_id
        trial_scope_generation_id = generation_id

    is_published = (
        represented_generation_id is not None
        and represented_generation_id == published_generation_id
    )
    # A published represented generation is exactly the validated one, so its
    # facts come from what validation read and nothing below rereads it.
    validated = publication.validated if is_published else None

    published_trials = (
        _published_phase_trial_refs(
            None if publication.validated is None else publication.validated.summary
        )
        if _artifact_root_binding_applies(experiment)
        else {}
    )
    # Captured after the pointer is resolved: a publication tells its trials
    # before it moves the pointer, so this capture holds every trial the
    # pointer's generation published.
    phase_stats = read_trial_stats(ledger, published_trials)

    # The represented generation's winner/summary facts are historical
    # evidence, so the metric they are reported under must be the one that
    # generation actually optimized — never the metric the config supplied
    # today (review v0.5.16 / blocker 4). The generation's own summary is the
    # source of that interpretation; only when it records none does the
    # current config describe the result.
    comparison = comparison_experiment or experiment
    metric_payload = _metric_semantics_payload(comparison.metric)
    result_phase_plan = [phase.name for phase in experiment.phases]
    result_context: ResultContext = "current_config"
    summary_payload: Mapping[str, Any] | None
    if validated is not None:
        summary_payload = validated.summary
        summary_present = True
        winners_present = set(validated.winners)
        result_phase_plan = [phase.name for phase in validated.phase_plan]
        metric_payload = {
            "name": validated.metric["name"],
            "goal": validated.metric["goal"],
            "objective_evidence": dict(validated.metric["objective_evidence"]),
        }
        result_context = "represented_generation"
    else:
        summary_path = (
            _generation_summary_path(experiment, represented_generation_id)
            if represented_generation_id is not None
            else None
        )
        summary_payload = _read_summary_payload(summary_path)
        summary_present = summary_path is not None and summary_path.is_file()
        winners_present = (
            {
                phase.name
                for phase in experiment.phases
                if _generation_winner_path(
                    experiment, represented_generation_id, phase.name
                ).is_file()
            }
            if represented_generation_id is not None
            else set()
        )
        stored_plan = _summary_phase_plan(summary_payload)
        if stored_plan is not None:
            result_phase_plan = stored_plan
        stored_metric = None if summary_payload is None else summary_payload.get("metric")
        if (
            isinstance(stored_metric, Mapping)
            and isinstance(stored_metric.get("name"), str)
            and stored_metric.get("goal") in ("minimize", "maximize")
        ):
            stored_evidence = _recorded_objective_evidence(stored_metric.get("objective_evidence"))
            if stored_evidence is not None:
                metric_payload = {
                    "name": stored_metric["name"],
                    "goal": stored_metric["goal"],
                    "objective_evidence": stored_evidence,
                }
                result_context = "represented_generation"
    published_config_matches_current: bool | None = None
    stored_fingerprint = (
        None if summary_payload is None else summary_payload.get("config_fingerprint")
    )
    if isinstance(stored_fingerprint, str) and stored_fingerprint:
        published_config_matches_current = stored_fingerprint == _experiment_semantic_fingerprint(
            comparison
        )

    publication_state: PublicationState = publication.state

    status = {
        "experiment": experiment.experiment,
        "current_generation_id": current_generation_id,
        "published_generation_id": published_generation_id,
        "represented_generation_id": represented_generation_id,
        "is_published": is_published,
        "publication_integrity": publication_state,
        **(
            {"publication_error": publication.error}
            if publication.state in {"failed", "permission_denied"}
            else {}
        ),
        "result_context": result_context,
        "published_config_matches_current": published_config_matches_current,
        "result_phase_plan": result_phase_plan,
        "metric": metric_payload,
        "phases": _phase_status_payloads(
            experiment,
            unavailable_published_phases={
                name
                for name, stats in phase_stats.items()
                if name in published_trials
                and not _published_trial_history_available(stats, published_trials[name])
            },
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
            running_attempts={
                name: (
                    None
                    if stats.running_attempts is None
                    else [
                        {
                            "trial_number": attempt.trial_number,
                            "generation_id": attempt.generation_id,
                            "attempt_id": attempt.attempt_id,
                        }
                        for attempt in stats.running_attempts
                    ]
                )
                for name, stats in phase_stats.items()
            },
            winners_present=winners_present,
        ),
        "summary_present": summary_present,
    }
    return status, publication
