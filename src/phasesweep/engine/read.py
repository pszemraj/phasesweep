"""Read-only status, winners, and trial tables for engine and MCP callers.

The CLI and ``experiment_status`` require a started study. Partial generation
results are read permissively; latest reads fill missing phase winners from the
validated publication. Published facts come from the bytes publication validation
read. Conflicting artifact-root ownership is refused, and resume fingerprint
checks belong to ``artifacts._load_winner``.
"""

from __future__ import annotations

import logging
import math
import statistics
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

import optuna

from phasesweep.config.common import SAFE_NAME_PATTERN, _validate_safe_name
from phasesweep.config.io import load_experiment
from phasesweep.config.models import Experiment, Phase, _metric_semantics_payload
from phasesweep.engine.artifact_roots import _artifact_root_binding_applies
from phasesweep.engine.artifacts import read_yaml_or_none
from phasesweep.engine.confound import (
    _confound_summary,
    _validate_confound_block,
)
from phasesweep.engine.errors import StudyStorageUnavailableError
from phasesweep.engine.fingerprints import _experiment_semantic_fingerprint
from phasesweep.engine.ledger import (
    log_storage_read_failure,
    read_phase_trials,
    read_trial_stats,
    resolved_ledger_path,
    validate_ledger,
)
from phasesweep.engine.optuna import (
    _PhaseTrialStats,
    _published_phase_trial_refs,
    _published_trial_history_available,
    _trial_count_state,
    _TrialRef,
)
from phasesweep.engine.paths import (
    _artifact_root_binding_path,
    _experiment_dir,
    _generation_config_snapshot_path,
    _generation_dir,
    _generation_path,
    _generation_summary_path,
    _generation_winner_path,
)
from phasesweep.engine.publication import PublicationPointer, _resolve_publication_pointer
from phasesweep.engine.publication_validation import ValidatedWinner, _parse_phase_plan
from phasesweep.engine.repeats import (
    ReduceName,
    RepeatsRecord,
    RepeatsSummary,
    _repeats_summary,
    _validate_repeats_block,
    group_labels_for,
    repeat_axis,
)
from phasesweep.engine.selection import _cost_advisory, _rank_groups
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    DURATION_ATTR,
    FAILURE_REASON_ATTR,
    GENERATION_SUMMARY_SCHEMA_VERSION,
    OBSERVED_PREFIX,
    RERUN_OF_ATTR,
    TRIAL_OUTCOME_ATTR,
    PublicationState,
    WinnerSource,
    _parse_winner_source,
    _parsed_trial_outcome,
    _trial_interrupted,
    _validate_observed_block,
)
from phasesweep.errors import OperatorAction
from phasesweep.evidence.models import EXTRACTOR_KINDS, _ObjectiveEvidenceFields
from phasesweep.runtime.json import JsonObject, JsonValue, json_object

log = logging.getLogger(__name__)

ResultContext: TypeAlias = Literal["represented_generation", "current_config"]
"""Which config's semantics a result payload's labels were read under.

``"represented_generation"``: the labels come from the represented
generation's own recorded summary, so they describe the result as it was
produced. ``"current_config"``: no represented result is available, so the
currently loaded config describes the status payload.
"""


class EngineStatus(typing.TypedDict):
    """Current or frozen path-free status facts shared by engine and MCP readers."""

    experiment: typing.NotRequired[str]
    current_generation_id: str | None
    published_generation_id: str | None
    represented_generation_id: str | None
    is_published: bool
    publication_integrity: PublicationState | Literal["unknown"]
    publication_error: typing.NotRequired[str | None]
    result_context: ResultContext
    published_config_matches_current: bool | None
    result_phase_plan: list[str]
    metric: JsonObject
    phases: list[JsonObject]
    summary_present: bool
    tracking_delivery: typing.NotRequired[dict[str, int]]


@dataclass(frozen=True)
class PhaseWinnerView:
    """Trusted phase winner facts for CLI and MCP projection.

    ``effective_overrides`` and ``repeats_detail`` support local composition and
    member checkpoint paths. MCP returns only the public summary fields through
    its visibility policy.
    """

    phase: str
    trial_number: int
    metric: float
    params: dict[str, Any]
    effective_overrides: dict[str, Any]
    gates_passed: bool | None  # None when the phase declared no gates
    incomplete: bool  # True when a wallclock timeout produced a partial winner
    source: WinnerSource | None = None
    # Path-free projection of the winner's confound block: verdicts, check
    # names, counts and evaluation guidance (:func:`phasesweep.engine.confound._confound_summary`).
    confound: dict[str, Any] | None = None
    # Report-only values the winning trial recorded, by observation name.
    observed: dict[str, float] = field(default_factory=dict)
    # String-free projection of the winner's repeats block: reducer, spread
    # and member trial numbers, never the repeat key or values
    # (:func:`phasesweep.engine.repeats._repeats_summary`).
    repeats: RepeatsSummary | None = None
    repeats_detail: RepeatsRecord | None = None
    generation_id: str | None = None
    is_published: bool = False
    config_matches_current: bool | None = None
    cost_advisory: str | None = None


def _phase_status_payloads(
    experiment: Experiment,
    *,
    phase_stats: Mapping[str, _PhaseTrialStats],
    trial_scope_generation_id: str | None,
    published_trials: Mapping[str, _TrialRef | None],
    winners_present: set[str],
) -> list[dict[str, Any]]:
    """Build path-free per-phase status from one trial and publication capture.

    ``winners_present`` must refer to the caller's single represented-generation
    resolution. CLI callers add their paths separately.

    :param Experiment experiment: Parsed experiment whose phase study counts are reported.
    :param Mapping[str, _PhaseTrialStats] phase_stats: Counts, availability and
        RUNNING identities from one trial-data capture, keyed by phase name.
    :param str | None trial_scope_generation_id: Generation whose counts are reported.
    :param Mapping[str, _TrialRef | None] published_trials: Published local trial
        identities checked against that same capture.
    :param set[str] winners_present: Phases whose winner the represented
        generation holds.
    :return list[dict[str, Any]]: One status payload per phase in declaration order.
    """
    phases: list[dict[str, Any]] = []
    for phase in experiment.phases:
        stats = phase_stats[phase.name]
        counts = stats.counts
        payload: dict[str, Any] = {
            "trials": counts,
            "n_trials": phase.n_trials,
            "generation_trials": (
                stats.generation_counts.get(trial_scope_generation_id, {})
                if trial_scope_generation_id
                else {}
            ),
            "trial_data_available": stats.available,
            "published_study_unavailable": (
                phase.name in published_trials
                and not _published_trial_history_available(stats, published_trials[phase.name])
            ),
            "phase": phase.name,
            "winner_present": phase.name in winners_present,
            "running_attempts": (
                None
                if stats.running_attempts is None
                else [
                    {
                        "trial_number": attempt.trial_number,
                        "generation_id": attempt.generation_id,
                        "attempt_id": attempt.attempt_id,
                        "recovers_as": attempt.recovers_as,
                    }
                    for attempt in stats.running_attempts
                ]
            ),
        }
        phases.append(payload)
    return phases


def _winner_view(
    winner: Mapping[str, object] | ValidatedWinner, phase_name: str
) -> PhaseWinnerView | None:
    """Reduce one decoded winner payload using the permissive read contract.

    :param Mapping[str, object] | ValidatedWinner winner: Decoded winner payload or
        captured publication facts.
    :param str phase_name: Phase name exposed by the winner.
    :return PhaseWinnerView | None: The winner view, or ``None`` when the
        payload is malformed.
    """
    if isinstance(winner, ValidatedWinner):
        validated = winner
        data = winner.payload
    else:
        validated = None
        data = winner
    try:
        # Read the metric under its recorded name, even if the current config renamed it.
        metric_block = data.get("metric") or {}
        if not isinstance(metric_block, Mapping):
            return None
        stored_metric_names = [key for key in metric_block if key != "goal"]
        if len(stored_metric_names) != 1:
            return None
        stored_metric_name = str(stored_metric_names[0])
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
        trial_number = int(data["trial_number"])
        if validated is None:
            source = _parse_winner_source(data.get("winner_source"), expected_phase=phase_name)
            confound_block = _validate_confound_block(
                data.get("confound"), winner_trial=trial_number
            )
            metric = float(metric_block[stored_metric_name])
            repeats = _validate_repeats_block(
                data, metric_value=metric, ranked_groups=confound_block["ranked_groups"]
            )
            observed = _validate_observed_block(data.get("observed"))
        else:
            source = validated.source
            confound_block = dict(validated.confound)
            metric = validated.metric
            repeats = validated.repeats
            observed = dict(validated.observed)
        provenance = data.get("objective_provenance")
        extractor = provenance.get("extractor") if isinstance(provenance, Mapping) else None
        kind = extractor.get("kind") if isinstance(extractor, Mapping) else None
        confound = _confound_summary(
            confound_block,
            winner_trial=trial_number,
            extractor_kind=kind if isinstance(kind, str) else "unknown",
        )
        return PhaseWinnerView(
            phase=phase_name,
            trial_number=trial_number,
            metric=metric,
            params=dict(params),
            effective_overrides=dict(effective_overrides),
            gates_passed=(all(bool(g.get("passed")) for g in gates) if gates else None),
            incomplete=bool(completion.get("incomplete", False)),
            source=source,
            confound=confound,
            observed=observed,
            repeats=None if repeats is None else _repeats_summary(repeats),
            repeats_detail=repeats,
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
    loaded = read_yaml_or_none(path)
    if not isinstance(loaded, Mapping):
        return None
    return _winner_view(loaded, phase_name)


def _represented_winners(
    experiment: Experiment,
    publication: PublicationPointer,
    represented_generation_id: str | None,
    phase_names: Sequence[str],
    *,
    fallback_published: bool = False,
) -> list[PhaseWinnerView]:
    """Return the represented generation's winners, in ``phase_names`` order.

    Where each winner is read from follows the module docstring.

    :param Experiment experiment: Experiment whose artifact tree is read.
    :param PublicationPointer publication: This read's single pointer resolution.
    :param str | None represented_generation_id: Generation whose winners are
        represented, or ``None`` when there is none.
    :param Sequence[str] phase_names: Safe phase names to enumerate.
    :param bool fallback_published: Fill missing latest phase winners from the publication.
    :return list[PhaseWinnerView]: One view per named phase that has a usable winner.
    """
    if represented_generation_id is None:
        return []
    validated = publication.validated
    views: list[PhaseWinnerView | None]
    if validated is not None and validated.generation_id == represented_generation_id:
        views = [
            _winner_view(validated.winners[name], name) if name in validated.winners else None
            for name in phase_names
        ]
    else:
        views = [
            _read_winner_path(
                _generation_winner_path(experiment, represented_generation_id, name), name
            )
            for name in phase_names
        ]
    chosen: list[PhaseWinnerView] = []
    for name, view in zip(phase_names, views, strict=True):
        view_generation = represented_generation_id
        if (
            view is None
            and fallback_published
            and validated is not None
            and name in validated.winners
        ):
            view = _winner_view(validated.winners[name], name)
            view_generation = validated.generation_id
        if view is not None:
            is_published = validated is not None and view_generation == validated.generation_id
            if is_published and validated is not None:
                stored_fingerprint = validated.summary.get("config_fingerprint")
                prefix = experiment.model_copy(
                    update={"phases": experiment.phases[: len(validated.phase_plan)]}
                )
                config_matches = (
                    stored_fingerprint == _experiment_semantic_fingerprint(prefix)
                    if isinstance(stored_fingerprint, str)
                    else None
                )
            else:
                config_matches = _generation_config_matches(experiment, view_generation)
            chosen.append(
                replace(
                    view,
                    generation_id=view_generation,
                    is_published=is_published,
                    config_matches_current=config_matches,
                    cost_advisory=(
                        winner_cost_advisory(experiment, view) if config_matches is True else None
                    ),
                )
            )
    return chosen


def require_study(experiment: Experiment) -> None:
    """Require evidence of a started study at the configured artifact root.

    :param Experiment experiment: Config locating the study.
    :raises StudyStorageUnavailableError: No study has initialized this root.
    """
    ledger_path = resolved_ledger_path(experiment)
    if not _experiment_dir(experiment).is_dir() or not any(
        path.is_file()
        for path in (
            _artifact_root_binding_path(experiment),
            _generation_path(experiment),
            *((ledger_path,) if ledger_path is not None else ()),
        )
    ):
        raise StudyStorageUnavailableError(
            f"No study directory at {_experiment_dir(experiment)} contains a started study. "
            "Use the same config and --set overrides as run.",
            action=OperatorAction.FIX_CONFIG,
        )


def _generation_config_matches(experiment: Experiment, generation_id: str | None) -> bool | None:
    """Compare recorded semantics with the matching prefix of the current phase plan.

    :param Experiment experiment: Current config to compare.
    :param str | None generation_id: Producing generation, when known.
    :return bool | None: Semantic match, or unknown when its snapshot is absent.
    """
    if generation_id is None:
        return None
    path = _generation_config_snapshot_path(experiment, generation_id)
    if not path.is_file():
        return None
    producing = load_experiment(path)
    prefix = experiment.model_copy(update={"phases": experiment.phases[: len(producing.phases)]})
    return _experiment_semantic_fingerprint(producing) == _experiment_semantic_fingerprint(prefix)


def winner_cost_advisory(experiment: Experiment, winner: PhaseWinnerView) -> str | None:
    """Recompute cost advice from retained repeat groups under the current display config.

    :param Experiment experiment: Current selection semantics and optional cost key.
    :param PhaseWinnerView winner: Winner whose repeat group anchors the window.
    :return str | None: Advisory text when a cheaper eligible group is in the window.
    """
    phase = next((phase for phase in experiment.phases if phase.name == winner.phase), None)
    if phase is None or phase.cost_key is None or winner.repeats is None:
        return None
    axis = repeat_axis(phase)
    if axis is None:
        return None
    try:
        read = read_phase_trials(validate_ledger(experiment), phase)
    except StudyStorageUnavailableError:
        return None
    if read is None:
        return None
    ranking = _rank_groups(read.trials, experiment, axis)
    numbers = tuple(winner.repeats["member_trials"])
    group = next((group for group in ranking.groups if group.numbers == numbers), None)
    return (
        None
        if group is None
        else _cost_advisory(
            phase,
            group,
            ranking.groups,
            goal=experiment.metric.goal,
        )
    )


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
    payload = read_yaml_or_none(summary_path)
    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema_version") != GENERATION_SUMMARY_SCHEMA_VERSION:
        return None
    return payload


def _summary_phase_plan(summary_payload: Mapping[str, Any] | None) -> list[str] | None:
    """Return the phase names recorded by an unvalidated represented generation.

    Use the recorded design-time plan, excluding the summary's composed overrides.
    Malformed plans return no plan; publication validation refuses them.

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


def _recorded_objective_evidence(candidate: object) -> JsonObject | None:
    """Return a complete current-format assurance payload, if present.

    :param object candidate: Summary ``objective_evidence`` value.
    :return JsonObject | None: Complete recorded flags, or ``None``.
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
    against any artifact: it is progress bookkeeping, naming the most recently
    claimed generation, and may legitimately name a failed or in-progress one.

    :param Experiment experiment: Experiment whose current pointer is read.
    :return str | None: The recorded ``generation_id``, or ``None`` when the
        pointer is missing, unreadable, malformed, or unsafely named.
    """
    generation = read_yaml_or_none(_generation_path(experiment))
    if not isinstance(generation, Mapping):
        return None
    raw_generation_id = generation.get("generation_id")
    if isinstance(raw_generation_id, str) and SAFE_NAME_PATTERN.fullmatch(raw_generation_id):
        return raw_generation_id
    return None


def _read_generation_selection(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    latest: bool = False,
) -> tuple[str | None, str | None, PublicationPointer]:
    """Capture both pointers and choose the represented generation.

    :param Experiment experiment: Experiment whose pointers are inspected.
    :param str | None generation_id: Explicit generation pin, overriding latest selection.
    :param bool latest: Prefer the captured current generation when its namespace exists.
    :raises ValueError: An explicit generation name is unsafe.
    :return tuple[str | None, str | None, PublicationPointer]: Current id, represented id and publication.
    """
    current_id = _current_pointer_generation_id(experiment)
    publication = _resolve_publication_pointer(experiment)
    if generation_id is not None:
        _validate_safe_name("generation", generation_id)
    elif (
        latest
        and publication.state in {"ok", "absent"}
        and current_id is not None
        and _generation_dir(experiment, current_id).is_dir()
    ):
        generation_id = current_id
    else:
        generation_id = _published_generation_id(publication)
    return current_id, generation_id, publication


def read_status(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    trial_generation_id: str | None = None,
    comparison_experiment: Experiment | None = None,
) -> EngineStatus:
    """Read path-free progress and represented-result status for one experiment.

    Resolve current and last-success pointers once. Published facts use that capture;
    pinned unpublished generations use their own files. Generation fields mean:

    - ``current_generation_id``: mutable pointer to the most recently claimed generation.
    - ``published_generation_id``: validated last-success pointer.
    - ``represented_generation_id``: pinned id, or published id by default.
    - ``is_published``: non-null represented id equals the published id.
    - ``publication_integrity``: ``ok``, ``absent``, ``failed``, or ``permission_denied``.
      Failed validation exposes no published winner; denial does not imply corruption.

    Progress ``phases`` use current config. Metric labels, ``result_phase_plan``, winner
    and summary presence use the represented generation. Missing result semantics fall
    back to current config; ``result_context`` names the source. Semantic fingerprints
    determine ``published_config_matches_current``, or ``None`` if no fingerprint exists.

    ``generation_trials`` counts ``trial_generation_id`` when given, else the pinned
    generation, else the current generation; other counts are cumulative. Counts and ``running_attempts`` share one storage capture.
    Absent studies have zero counts; unreadable storage sets ``trial_data_available=False``
    and ``running_attempts=None``. ``published_study_unavailable`` marks missing/replaced
    trial identities or incomplete history when readable, otherwise incomplete inspection.
    Confirmed missing history blocks execution but permits skipped-phase winner loading;
    incomplete inspection may require MCP recovery. Artifact integrity is independent.

    :param Experiment experiment: Experiment whose artifacts and trials are inspected.
    :param str | None generation_id: Pin represented artifacts and generation counts,
        leaving pointer identities unchanged.
    :param str | None trial_generation_id: Generation whose trials ``generation_trials``
        counts instead of the represented one.
    :param Experiment | None comparison_experiment: Current config for drift comparison
        and fallback semantics; reads still use ``experiment``.
    :raises ArtifactRootConflictError: The root is unreadable or has different ownership.
    :raises ValueError: ``generation_id`` or ``trial_generation_id`` is not a safe
        generation name.
    :return EngineStatus: Identities, integrity, result semantics, progress and artifact
        presence. Path-free ``publication_error`` appears only for failed or denied validation.
    """
    status, _publication = _read_status(
        experiment,
        generation_id=generation_id,
        trial_generation_id=trial_generation_id,
        comparison_experiment=comparison_experiment,
    )
    return status


def read_result(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    comparison_experiment: Experiment | None = None,
    phase_names: Sequence[str] | None = None,
) -> tuple[EngineStatus, list[PhaseWinnerView]]:
    """Read status and represented-generation winners from one pointer resolution.

    Identity and result semantics follow :func:`read_status`. Winners default to
    ``result_phase_plan``; explicit ``phase_names`` filter winners without changing
    status. Published winners use validated bytes; other generations are read permissively.

    :param Experiment experiment: Parsed experiment config whose phases are inspected.
    :param str | None generation_id: Optional generation to represent.
    :param Experiment | None comparison_experiment: Current config for drift comparison.
    :param Sequence[str] | None phase_names: Safe phase names in output order;
        an empty sequence selects none.
    :raises ArtifactRootConflictError: If the artifact root cannot be read or
        is bound to a different storage ledger or experiment.
    :raises ValueError: If ``generation_id`` or an explicit phase name is not safe.
    :return tuple[EngineStatus, list[PhaseWinnerView]]: Status and usable winners
        in the selected phase order.
    """
    names = (
        None
        if phase_names is None
        else [_validate_safe_name("phase", name) for name in phase_names]
    )
    status, publication = _read_status(
        experiment,
        generation_id=generation_id,
        trial_generation_id=None,
        comparison_experiment=comparison_experiment,
    )
    winners = _represented_winners(
        experiment,
        publication,
        status["represented_generation_id"],
        status["result_phase_plan"] if names is None else names,
    )
    return status, winners


def _read_status(
    experiment: Experiment,
    *,
    generation_id: str | None,
    trial_generation_id: str | None,
    comparison_experiment: Experiment | None,
    latest: bool = False,
) -> tuple[EngineStatus, PublicationPointer]:
    """Build the :func:`read_status` payload and return the resolution behind it.

    :param Experiment experiment: Parsed experiment config whose phases are inspected.
    :param str | None generation_id: Optional generation to represent.
    :param str | None trial_generation_id: Optional generation whose trials
        ``generation_trials`` counts instead of the represented one.
    :param Experiment | None comparison_experiment: Optional current config for
        drift comparison.
    :param bool latest: Represent the captured current generation when its namespace exists.
    :raises ArtifactRootConflictError: If the artifact root cannot be read or
        is bound to a different storage ledger or experiment.
    :raises ValueError: If ``generation_id`` or ``trial_generation_id`` is not a safe
        generation name.
    :return tuple[EngineStatus, PublicationPointer]: The status payload and
        the single pointer resolution every fact in it came from.
    """
    ledger = validate_ledger(experiment)
    current_generation_id, represented_generation_id, publication = _read_generation_selection(
        experiment, generation_id=generation_id, latest=latest
    )
    published_generation_id = _published_generation_id(publication)

    trial_scope_generation_id = (
        represented_generation_id if generation_id is not None or latest else current_generation_id
    )
    if trial_generation_id is not None:
        trial_scope_generation_id = _validate_safe_name("generation", trial_generation_id)

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

    # Prefer the represented generation's recorded metric semantics; use config as fallback.
    comparison = comparison_experiment or experiment
    metric_payload: JsonObject = _metric_semantics_payload(comparison.metric)
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

    latest_winners = (
        {
            view.phase: view
            for view in _represented_winners(
                experiment,
                publication,
                represented_generation_id,
                result_phase_plan,
                fallback_published=True,
            )
        }
        if latest and generation_id is None and publication.state in {"ok", "absent"}
        else None
    )
    if latest_winners is not None:
        winners_present = set(latest_winners)

    status: EngineStatus = {
        "experiment": experiment.experiment,
        "current_generation_id": current_generation_id,
        "published_generation_id": published_generation_id,
        "represented_generation_id": represented_generation_id,
        "is_published": is_published,
        "publication_integrity": publication_state,
        "result_context": result_context,
        "published_config_matches_current": published_config_matches_current,
        "result_phase_plan": result_phase_plan,
        "metric": metric_payload,
        "phases": _phase_status_payloads(
            experiment,
            phase_stats=phase_stats,
            trial_scope_generation_id=trial_scope_generation_id,
            published_trials=published_trials,
            winners_present=winners_present,
        ),
        "summary_present": summary_present,
    }
    if publication.state in {"failed", "permission_denied"}:
        status["publication_error"] = publication.error
    if latest_winners is not None:
        for phase_payload in status["phases"]:
            view = latest_winners.get(str(phase_payload["phase"]))
            phase_payload["winner_generation_id"] = None if view is None else view.generation_id
            phase_payload["winner_is_published"] = False if view is None else view.is_published
            phase_payload["winner_config_matches_current"] = (
                None if view is None else view.config_matches_current
            )
    return status, publication


@dataclass(frozen=True)
class RepeatGroupView:
    """Statistics over current completed members of one repeat configuration.

    ``n`` counts finite measured values, including infeasible ones. Missing,
    failed, interrupted and replaced attempts do not contribute. ``aggregate``
    is present only when the full group is eligible under the current config;
    ``sd`` needs at least two measurements. ``winner`` names the represented phase winner.
    """

    group: int
    params: JsonObject
    n: int
    expected: int
    reduce: ReduceName
    aggregate: float | None
    mean: float | None
    sd: float | None
    min: float | None
    max: float | None
    eligible: bool
    winner: bool
    observed: dict[str, float]
    observed_n: dict[str, int]


@dataclass(frozen=True)
class TrialTable:
    """One phase's trials as table rows, in trial-number order.

    ``columns`` lists the keys every row carries, in display order: ``trial``,
    ``state``, ``value``, ``params``, ``duration_s``, then ``repeat_group`` for a
    repeats phase, ``rerun_of`` once any trial is a rerun, one ``observed:<name>``
    per report-only value any trial recorded, an optional ``reason`` when any
    trial has a recorded cause, and ``winner``. ``state`` is the Optuna state, except that a
    COMPLETE trial that violated a constraint or a metric bound reads
    ``INFEASIBLE``. ``winner`` is ``*`` on the trials the represented phase's
    winner stands on and empty elsewhere.
    """

    phase: str
    #: ``False`` when the phase's trials could not be read; the rows are then empty.
    available: bool
    columns: tuple[str, ...]
    rows: tuple[JsonObject, ...]
    groups: tuple[RepeatGroupView, ...] = ()


@dataclass(frozen=True)
class TrialTables:
    """Every configured phase's trial table, from one generation-pointer capture."""

    experiment: str
    #: Last validated publication, independent of each displayed winner's generation.
    published_generation_id: str | None
    publication_integrity: PublicationState
    #: Validation diagnostic beside a ``failed`` or ``permission_denied`` integrity.
    publication_error: str | None
    phases: tuple[TrialTable, ...]
    represented_generation_id: str | None = None
    winner_generations: dict[str, tuple[str, bool]] = field(default_factory=dict)


def _cli_phase_payloads(
    experiment: Experiment,
    phases: Sequence[JsonObject],
    represented_generation_id: str | None,
    published_generation_id: str | None,
) -> list[JsonObject]:
    """Add winner paths to the captured phase status for CLI inspection.

    :param Experiment experiment: Experiment supplying artifact paths.
    :param Sequence[JsonObject] phases: Path-free phase payloads from the shared read.
    :param str | None represented_generation_id: Generation whose winner paths appear.
    :param str | None published_generation_id: Validated publication for per-phase labels.
    :return list[JsonObject]: CLI phase payloads with name and winner path fields.
    """
    cli_phases: list[JsonObject] = []
    for phase in phases:
        name = phase["phase"]
        if not isinstance(name, str):
            raise TypeError("Phase status name must be a string.")
        winner_generation = phase.get("winner_generation_id", represented_generation_id)
        winner_path = (
            _generation_winner_path(experiment, winner_generation, name)
            if phase["winner_present"] and isinstance(winner_generation, str)
            else None
        )
        cli_phases.append(
            {
                "name": name,
                **{
                    key: value
                    for key, value in phase.items()
                    if key not in {"phase", "winner_present", "running_attempts"}
                },
                "winner": None if winner_path is None else str(winner_path),
                "winner_generation_id": winner_generation if winner_path is not None else None,
                "winner_is_published": winner_path is not None
                and winner_generation == published_generation_id,
                "winner_config_matches_current": phase.get("winner_config_matches_current"),
            }
        )
    return cli_phases


def experiment_status(experiment: Experiment, *, generation_id: str | None = None) -> JsonObject:
    """Inspect study progress and phase winners without creating or modifying state.

    By default each phase uses its latest usable winner, falling back to the validated
    publication. Per-phase generation, publication and config-match labels identify
    that choice. An explicit generation pin reads only that generation. Corrupt
    publications expose diagnostics and counts without winner paths.

    :param Experiment experiment: Parsed experiment config to inspect.
    :param str | None generation_id: Optional generation pin, disabling published fallback.
    :raises StudyStorageUnavailableError: No study has started at the configured root.
    :raises ArtifactRootConflictError: The root belongs to another ledger or experiment.
    :raises StudySchemaMismatchError: The stored study format is unsupported.
    :return JsonObject: Paths, pointer identities, integrity, phase counts and labeled winners.
    """
    require_study(experiment)
    status, _ = _read_status(
        experiment,
        generation_id=generation_id,
        trial_generation_id=None,
        comparison_experiment=None,
        latest=True,
    )
    integrity = status["publication_integrity"]
    ledger_path = resolved_ledger_path(experiment)
    payload: dict[str, object] = {
        "kind": "experiment",
        "experiment": status["experiment"],
        "workdir": str(_experiment_dir(experiment)),
        "ledger_path": None if ledger_path is None else str(ledger_path),
        "current_generation_id": status["current_generation_id"],
        "published_generation_id": status["published_generation_id"],
        "represented_generation_id": status["represented_generation_id"],
        "is_published": status["is_published"],
        "publication_integrity": integrity,
        **(
            {"publication_error": status["publication_error"]}
            if integrity in {"failed", "permission_denied"}
            else {}
        ),
        "phases": _cli_phase_payloads(
            experiment,
            status["phases"],
            status["represented_generation_id"] if integrity in {"ok", "absent"} else None,
            status["published_generation_id"],
        ),
    }
    if experiment.mlflow is not None:
        from phasesweep.tracking.publication import delivery_status

        payload["tracking_delivery"] = delivery_status(experiment)
    return json_object(payload)


def _recorded_number(value: object) -> float | None:
    """Return a recorded user-attr number, or ``None`` when it is not one.

    :param object value: A trial user-attr value.
    :return float | None: The value as a float, or ``None`` for anything but a real number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _trial_reason(trial: optuna.trial.FrozenTrial) -> str | None:
    """Read the recorded execution failure or outcome cause for a trial.

    :param optuna.trial.FrozenTrial trial: Trial whose diagnostic is displayed.
    :return str | None: Recorded cause, or None when no diagnostic was recorded.
    """
    failure = trial.user_attrs.get(FAILURE_REASON_ATTR)
    if isinstance(failure, str) and failure:
        return failure
    outcome = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    return None if outcome is None else outcome.cause


#: Columns every trial table starts with, as ``(header, getter)`` pairs.
_CORE_TRIAL_COLUMNS: tuple[tuple[str, Callable[[optuna.trial.FrozenTrial], JsonValue]], ...] = (
    ("trial", lambda trial: trial.number),
    ("state", _trial_count_state),
    ("value", lambda trial: trial.value),
    ("params", lambda trial: dict(trial.params)),
    ("duration_s", lambda trial: _recorded_number(trial.user_attrs.get(DURATION_ATTR))),
)
_REPEAT_GROUP_COLUMN = "repeat_group"
_RERUN_OF_COLUMN = "rerun_of"
_WINNER_COLUMN = "winner"


def _winning_trials(winner: PhaseWinnerView | None) -> dict[int, str | None]:
    """Map the trial numbers a published winner stands on to the attempt each must carry.

    A repeats winner stands on every member of its group; its scalar fields
    name the highest-numbered member, so only that member's attempt is known
    from the path-free view and the others are matched by number alone.

    :param PhaseWinnerView | None winner: The phase's published winner, if any.
    :return dict[int, str | None]: Attempt id to require by trial number, ``None`` where unchecked.
    """
    if winner is None:
        return {}
    standing: dict[int, str | None] = {
        winner.trial_number: winner.source.attempt_id if winner.source is not None else None
    }
    if winner.repeats is not None:
        for number in winner.repeats["member_trials"]:
            standing.setdefault(number, None)
    return standing


def _trial_table(
    experiment: Experiment,
    phase: Phase,
    trials: Sequence[optuna.trial.FrozenTrial],
    winner: PhaseWinnerView | None,
    *,
    available: bool = True,
) -> TrialTable:
    """Build one phase's table from its trials and the winner the publication holds for it.

    A ``rerun_of`` column, naming the interrupted trial or failed grid cell each rerun
    reran, appears once the phase holds a rerun.

    :param Experiment experiment: Current metric and constraint eligibility bounds.
    :param Phase phase: The configured phase the trials belong to.
    :param Sequence[optuna.trial.FrozenTrial] trials: The phase study's trials.
    :param PhaseWinnerView | None winner: The published winner for the phase, if any.
    :param bool available: ``False`` when the trials could not be read, so ``trials`` is empty.
    :return TrialTable: The rows and the columns they carry.
    """
    axis = repeat_axis(phase)
    groups = group_labels_for(trials, axis)
    observed = sorted(
        {name for t in trials for name in t.user_attrs if name.startswith(OBSERVED_PREFIX)}
    )
    reruns = any(RERUN_OF_ATTR in t.user_attrs for t in trials)
    reasons = {trial.number: _trial_reason(trial) for trial in trials}
    has_reason = any(reasons.values())
    columns = (
        *(header for header, _ in _CORE_TRIAL_COLUMNS),
        *([] if axis is None else [_REPEAT_GROUP_COLUMN]),
        *([_RERUN_OF_COLUMN] if reruns else []),
        *observed,
        *(["reason"] if has_reason else []),
        _WINNER_COLUMN,
    )
    standing = _winning_trials(winner)
    rows: list[JsonObject] = []
    for trial in trials:
        row: JsonObject = {header: getter(trial) for header, getter in _CORE_TRIAL_COLUMNS}
        if axis is not None:
            row[_REPEAT_GROUP_COLUMN] = groups.get(trial.number)
        if reruns:
            row[_RERUN_OF_COLUMN] = trial.user_attrs.get(RERUN_OF_ATTR)
        for name in observed:
            row[name] = _recorded_number(trial.user_attrs.get(name))
        if has_reason:
            row["reason"] = reasons[trial.number]
        attempt = standing.get(trial.number)
        marked = trial.number in standing and (
            attempt is None or trial.user_attrs.get(ATTEMPT_ID_ATTR) == attempt
        )
        row[_WINNER_COLUMN] = "*" if marked else ""
        rows.append(row)
    return TrialTable(
        phase=phase.name,
        available=available,
        columns=columns,
        rows=tuple(rows),
        groups=_repeat_group_views(experiment, phase, trials, rows),
    )


def _repeat_group_views(
    experiment: Experiment,
    phase: Phase,
    trials: Sequence[optuna.trial.FrozenTrial],
    rows: Sequence[JsonObject],
) -> tuple[RepeatGroupView, ...]:
    """Summarize every represented configuration using the trial table's group labels.

    :param Experiment experiment: Current selection bounds.
    :param Phase phase: Configured repeat axis and reducer.
    :param Sequence[optuna.trial.FrozenTrial] trials: The same capture as the member rows.
    :param Sequence[JsonObject] rows: Member rows carrying group and published-winner labels.
    :return tuple[RepeatGroupView, ...]: Groups in ascending label order, empty without repeats.
    """
    axis = repeat_axis(phase)
    if axis is None:
        return ()
    labels = group_labels_for(trials, axis)
    ranking = _rank_groups(trials, experiment, axis)
    eligible = {labels[group.representative]: group for group in ranking.groups}
    replaced = {trial.user_attrs.get(RERUN_OF_ATTR) for trial in trials}
    marked = {row["trial"] for row in rows if row["winner"] == "*"}
    members: dict[int, list[optuna.trial.FrozenTrial]] = {}
    params: dict[int, JsonObject] = {}
    for trial in trials:
        label = labels.get(trial.number)
        if label is None:
            continue
        members.setdefault(label, [])
        params[label] = {name: value for name, value in trial.params.items() if name != axis.key}
        if trial.number not in replaced and not _trial_interrupted(trial):
            members[label].append(trial)
    views: list[RepeatGroupView] = []
    for label, found in sorted(members.items()):
        values = [
            float(trial.value)
            for trial in found
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and math.isfinite(trial.value)
        ]
        observations: dict[str, list[float]] = {}
        for trial in found:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            for name, raw in trial.user_attrs.items():
                value = _recorded_number(raw)
                if name.startswith(OBSERVED_PREFIX) and value is not None and math.isfinite(value):
                    observations.setdefault(name[len(OBSERVED_PREFIX) :], []).append(value)
        group = eligible.get(label)
        views.append(
            RepeatGroupView(
                group=label,
                params=params[label],
                n=len(values),
                expected=len(axis.values),
                reduce=axis.reduce,
                aggregate=None if group is None else group.aggregate,
                mean=statistics.fmean(values) if values else None,
                sd=statistics.stdev(values) if len(values) > 1 else None,
                min=min(values) if values else None,
                max=max(values) if values else None,
                eligible=group is not None,
                winner=bool(found) and all(trial.number in marked for trial in found),
                observed={name: statistics.fmean(values) for name, values in observations.items()},
                observed_n={name: len(values) for name, values in observations.items()},
            )
        )
    return tuple(views)


def read_trial_tables(
    experiment: Experiment, *, generation_id: str | None = None, latest: bool = False
) -> TrialTables:
    """Read configured phases' trial tables without creating storage or writing files.

    Resolve publication once before capturing trials; winner marks identify the
    requested generation, or that publication by default. Absent studies and in-memory ledgers
    have no rows. Unreadable phases are reported unavailable.

    :param Experiment experiment: Parsed experiment config whose phases are read.
    :param str | None generation_id: Generation whose winners are marked, defaulting to publication.
    :param bool latest: Mark the captured current generation when its namespace exists.
    :raises ArtifactRootConflictError: The artifact root cannot be read or is bound to another ledger.
    :raises StudySchemaMismatchError: A phase's study is populated under an unsupported schema.
    :return TrialTables: One table per configured phase, in declaration order.
    """
    ledger = validate_ledger(experiment)
    _, represented_generation_id, publication = _read_generation_selection(
        experiment, generation_id=generation_id, latest=latest
    )
    published_generation_id = _published_generation_id(publication)
    if publication.state not in {"ok", "absent"}:
        represented_generation_id = None
    winners = {
        view.phase: view
        for view in _represented_winners(
            experiment,
            publication,
            represented_generation_id,
            [phase.name for phase in experiment.phases],
            fallback_published=latest and generation_id is None,
        )
    }
    tables: list[TrialTable] = []
    for phase in experiment.phases:
        try:
            read = read_phase_trials(ledger, phase)
        except StudyStorageUnavailableError as exc:
            log_storage_read_failure(ledger, phase, exc, logger=log)
            tables.append(_trial_table(experiment, phase, [], None, available=False))
            continue
        trials = [] if read is None else read.trials
        tables.append(_trial_table(experiment, phase, trials, winners.get(phase.name)))
    return TrialTables(
        experiment=experiment.experiment,
        published_generation_id=published_generation_id,
        publication_integrity=publication.state,
        publication_error=publication.error,
        phases=tuple(tables),
        represented_generation_id=represented_generation_id,
        winner_generations={
            name: (view.generation_id, view.is_published)
            for name, view in winners.items()
            if view.generation_id is not None
        },
    )
