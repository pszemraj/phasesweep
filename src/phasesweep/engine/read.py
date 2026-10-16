"""Read-only status, winners, and trial tables for engine and MCP callers.

Missing and partial results are reported permissively; conflicting artifact-root
ownership is refused. Resume fingerprint checks belong to ``artifacts._load_winner``.
Published facts come from the bytes publication validation read. Unpublished
represented generations are read permissively from their own files.
"""

from __future__ import annotations

import logging
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import optuna

from phasesweep.config.common import SAFE_NAME_PATTERN, _validate_safe_name
from phasesweep.config.models import Experiment, Phase, _metric_semantics_payload
from phasesweep.engine.artifact_roots import _artifact_root_binding_applies
from phasesweep.engine.artifacts import read_yaml_or_none
from phasesweep.engine.confound import _confound_summary, _validate_confound_block
from phasesweep.engine.errors import StudyStorageUnavailableError
from phasesweep.engine.fingerprints import _experiment_semantic_fingerprint
from phasesweep.engine.ledger import (
    log_storage_read_failure,
    read_phase_trials,
    read_trial_stats,
    validate_ledger,
)
from phasesweep.engine.optuna import (
    _PhaseTrialStats,
    _published_phase_trial_refs,
    _published_trial_history_available,
    _trial_state_label,
    _TrialRef,
)
from phasesweep.engine.paths import (
    _generation_path,
    _generation_summary_path,
    _generation_winner_path,
)
from phasesweep.engine.publication import PublicationPointer, _resolve_publication_pointer
from phasesweep.engine.publication_validation import _parse_phase_plan
from phasesweep.engine.repeats import (
    RepeatsSummary,
    _repeats_summary,
    _validate_repeats_block,
    group_labels_for,
    repeat_axis,
)
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    DURATION_ATTR,
    GENERATION_SUMMARY_SCHEMA_VERSION,
    OBSERVED_PREFIX,
    RERUN_OF_ATTR,
    PublicationState,
    WinnerSource,
    _parse_winner_source,
    _validate_observed_block,
)
from phasesweep.evidence.models import EXTRACTOR_KINDS, _ObjectiveEvidenceFields
from phasesweep.runtime.json import JsonObject, JsonValue

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
    source: WinnerSource | None = None
    # Path-free projection of the winner's confound block: verdicts, check
    # names, and counts only (:func:`phasesweep.engine.confound._confound_summary`).
    confound: dict[str, Any] | None = None
    # Report-only values the winning trial recorded, by observation name.
    observed: dict[str, float] = field(default_factory=dict)
    # String-free projection of the winner's repeats block: reducer, spread
    # and member trial numbers, never the repeat key or values
    # (:func:`phasesweep.engine.repeats._repeats_summary`).
    repeats: RepeatsSummary | None = None


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
            "running": counts.get("RUNNING", 0),
            "n_trials": phase.n_trials,
            "completed": counts.get("COMPLETE", 0),
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


def _winner_view(data: Mapping[str, Any], phase_name: str) -> PhaseWinnerView | None:
    """Reduce one decoded winner payload using the permissive read contract.

    :param Mapping[str, Any] data: Decoded winner payload.
    :param str phase_name: Phase name exposed by the winner.
    :return PhaseWinnerView | None: The winner view, or ``None`` when the
        payload is malformed.
    """
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
        source = _parse_winner_source(data.get("winner_source"), expected_phase=phase_name)
        trial_number = int(data["trial_number"])
        confound_block = _validate_confound_block(data.get("confound"), winner_trial=trial_number)
        confound = _confound_summary(confound_block, winner_trial=trial_number)
        metric = float(metric_block[stored_metric_name])
        repeats = _validate_repeats_block(
            data, metric_value=metric, ranked_groups=confound_block["ranked_groups"]
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
            observed=_validate_observed_block(data.get("observed")),
            repeats=None if repeats is None else _repeats_summary(repeats),
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
    against any artifact: it is progress bookkeeping, always the most recent
    invocation's own claim, and may legitimately name a failed or
    in-progress generation.

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


def read_status(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    comparison_experiment: Experiment | None = None,
) -> EngineStatus:
    """Read path-free progress and represented-result status for one experiment.

    Resolve the current and last-success pointers once each. Published facts come
    from that validation capture; pinned unpublished generations use their own files.

    - ``current_generation_id`` names the most recent invocation's mutable pointer.
    - ``published_generation_id`` names the validated last-success pointer.
    - ``represented_generation_id`` is the pinned id, or the published id by default.
    - ``is_published`` is true only when a non-null represented id equals the published id.
    - ``publication_integrity`` is ``ok``, ``absent``, ``failed``, or ``permission_denied``.
      Failed validation exposes no published winner. Failure details are path-free;
      permission denial does not imply artifact corruption.

    Progress ``phases`` follow the current config. Metric labels and ``result_phase_plan``
    use the represented summary's semantics, falling back to the current config when
    absent; ``result_context`` identifies the source. ``published_config_matches_current``
    compares semantic fingerprints, or is ``None`` without a recorded fingerprint.
    Winner and summary presence refer to the represented generation.

    ``generation_trials`` counts the pinned generation, or the current generation in
    default mode. Other trial counts are cumulative. Counts and ``running_attempts``
    come from one storage capture. Absent studies have known zero counts; unreadable
    storage has ``trial_data_available=False`` and ``running_attempts=None``.

    ``published_study_unavailable`` marks unmatched published trial identities or
    completion boundaries. Available trial data confirms missing/replaced/incomplete
    history and refuses execution, although skipped phases may load saved winners.
    Unavailable data means incomplete inspection and may require MCP recovery.
    Artifact integrity is separate; the CLI adds paths in its own view.

    :param Experiment experiment: Parsed experiment config whose phases are inspected.
    :param str | None generation_id: Pin represented artifacts and generation counts
        without changing either pointer identity.
    :param Experiment | None comparison_experiment: Current config for drift comparison
        and no-result semantics; artifact and trial reads still use ``experiment``.
    :raises ArtifactRootConflictError: If the artifact root cannot be read or
        is bound to a different storage ledger or experiment.
    :raises ValueError: If ``generation_id`` is not a safe generation name.
    :return EngineStatus: Path-free identities, integrity, metric/phase semantics,
        trial counts, running attempts, and artifact presence. ``publication_error``
        appears only for failed or permission-denied validation.
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
    comparison_experiment: Experiment | None,
) -> tuple[EngineStatus, PublicationPointer]:
    """Build the :func:`read_status` payload and return the resolution behind it.

    :param Experiment experiment: Parsed experiment config whose phases are inspected.
    :param str | None generation_id: Optional generation to represent.
    :param Experiment | None comparison_experiment: Optional current config for
        drift comparison.
    :raises ArtifactRootConflictError: If the artifact root cannot be read or
        is bound to a different storage ledger or experiment.
    :raises ValueError: If ``generation_id`` is not a safe generation name.
    :return tuple[EngineStatus, PublicationPointer]: The status payload and
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
    return status, publication


@dataclass(frozen=True)
class TrialTable:
    """One phase's trials as table rows, in trial-number order.

    ``columns`` lists the keys every row carries, in display order: ``trial``,
    ``state``, ``value``, ``params``, ``duration_s``, then ``repeat_group`` for a
    repeats phase, one ``observed:<name>`` per report-only value any trial
    recorded, and ``winner``. ``state`` is the Optuna state, except that a
    COMPLETE trial that violated a constraint or a metric bound reads
    ``INFEASIBLE``. ``winner`` is ``*`` on the trials the published generation's
    winner stands on and empty elsewhere.
    """

    phase: str
    #: ``False`` when the phase's trials could not be read; the rows are then empty.
    available: bool
    columns: tuple[str, ...]
    rows: tuple[JsonObject, ...]


@dataclass(frozen=True)
class TrialTables:
    """Every configured phase's trial table, from one publication resolution."""

    experiment: str
    #: The generation whose winners are marked, or ``None`` when nothing validates as published.
    published_generation_id: str | None
    publication_integrity: PublicationState
    #: Validation diagnostic beside a ``failed`` or ``permission_denied`` integrity.
    publication_error: str | None
    phases: tuple[TrialTable, ...]


def _recorded_number(value: object) -> float | None:
    """Return a recorded user-attr number, or ``None`` when it is not one.

    :param object value: A trial user-attr value.
    :return float | None: The value as a float, or ``None`` for anything but a real number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


#: Columns every trial table starts with, as ``(header, getter)`` pairs.
_CORE_TRIAL_COLUMNS: tuple[tuple[str, Callable[[optuna.trial.FrozenTrial], JsonValue]], ...] = (
    ("trial", lambda trial: trial.number),
    ("state", _trial_state_label),
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
    phase: Phase,
    trials: Sequence[optuna.trial.FrozenTrial],
    winner: PhaseWinnerView | None,
    *,
    available: bool = True,
) -> TrialTable:
    """Build one phase's table from its trials and the winner the publication holds for it.

    A ``rerun_of`` column, naming the interrupted trial each rerun reran, appears
    once the phase holds a rerun.

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
    columns = (
        *(header for header, _ in _CORE_TRIAL_COLUMNS),
        *([] if axis is None else [_REPEAT_GROUP_COLUMN]),
        *([_RERUN_OF_COLUMN] if reruns else []),
        *observed,
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
        attempt = standing.get(trial.number)
        marked = trial.number in standing and (
            attempt is None or trial.user_attrs.get(ATTEMPT_ID_ATTR) == attempt
        )
        row[_WINNER_COLUMN] = "*" if marked else ""
        rows.append(row)
    return TrialTable(phase=phase.name, available=available, columns=columns, rows=tuple(rows))


def read_trial_tables(experiment: Experiment) -> TrialTables:
    """Read configured phases' trial tables without creating storage or writing files.

    Resolve publication once before capturing trials; winner marks identify that
    publication, excluding uncommitted reruns. Absent studies and in-memory ledgers
    have no rows. Unreadable phases are reported unavailable.

    :param Experiment experiment: Parsed experiment config whose phases are read.
    :raises ArtifactRootConflictError: The artifact root cannot be read or is bound to another ledger.
    :raises StudySchemaMismatchError: A phase's study is populated under an unsupported schema.
    :return TrialTables: One table per configured phase, in declaration order.
    """
    ledger = validate_ledger(experiment)
    publication = _resolve_publication_pointer(experiment)
    published_generation_id = _published_generation_id(publication)
    winners = {
        view.phase: view
        for view in _represented_winners(
            experiment,
            publication,
            published_generation_id,
            [phase.name for phase in experiment.phases],
        )
    }
    tables: list[TrialTable] = []
    for phase in experiment.phases:
        try:
            read = read_phase_trials(ledger, phase)
        except StudyStorageUnavailableError as exc:
            log_storage_read_failure(ledger, phase, exc, logger=log)
            tables.append(_trial_table(phase, [], None, available=False))
            continue
        trials = [] if read is None else read.trials
        tables.append(_trial_table(phase, trials, winners.get(phase.name)))
    return TrialTables(
        experiment=experiment.experiment,
        published_generation_id=published_generation_id,
        publication_integrity=publication.state,
        publication_error=publication.error,
        phases=tuple(tables),
    )
