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
from stat import S_ISDIR, S_ISREG
from typing import Any, Literal, TypeAlias

import optuna

from phasesweep.config.common import SAFE_NAME_PATTERN, _validate_safe_name
from phasesweep.config.models import Experiment, Phase, _metric_semantics_payload
from phasesweep.engine.artifact_roots import _artifact_path_mode
from phasesweep.engine.artifacts import read_yaml_or_none
from phasesweep.engine.confound import (
    _confound_summary,
    _validate_confound_block,
)
from phasesweep.engine.errors import StudySchemaMismatchError, StudyStorageUnavailableError
from phasesweep.engine.fingerprints import (
    _experiment_semantic_fingerprint,
    _recorded_config_matches,
)
from phasesweep.engine.growth import recorded_growth
from phasesweep.engine.ledger import (
    ValidatedLedger,
    log_storage_read_failure,
    read_phase_studies,
    read_phase_trials,
    read_trial_stats,
    resolved_ledger_path,
    validate_ledger,
)
from phasesweep.engine.optuna import (
    _finished_trial_count,
    _PhaseTrialStats,
    _published_phase_trial_refs,
    _published_trial_history_available,
    _trial_count_state,
    _TrialRef,
)
from phasesweep.engine.paths import (
    _artifact_root_binding_path,
    _experiment_dir,
    _generation_dir,
    _generation_path,
    _generation_summary_path,
    _generation_winner_path,
)
from phasesweep.engine.provenance import _load_generation_experiment
from phasesweep.engine.publication import PublicationPointer, _resolve_publication_pointer
from phasesweep.engine.publication_validation import ValidatedWinner, _parse_phase_plan
from phasesweep.engine.repeats import (
    RankedGroup,
    ReduceName,
    RepeatsRecord,
    RepeatsSummary,
    _repeats_summary,
    _validate_repeats_block,
    group_labels_for,
    ranked_group,
    repeat_axis,
)
from phasesweep.engine.selection import (
    PhaseAbortRefusedError,
    TieGroupComparison,
    _cost_advisory,
    _edge_advisory,
    _eligible,
    _no_feasible_remedy,
    _phase_refusal,
    _rank_groups,
    tie_group_comparisons,
)
from phasesweep.engine.state import (
    DURATION_ATTR,
    FAILURE_REASON_ATTR,
    FINISHED_TRIAL_STATES,
    GENERATION_SUMMARY_SCHEMA_VERSION,
    OBSERVED_PREFIX,
    RERUN_OF_ATTR,
    TRIAL_OUTCOME_ATTR,
    TRIAL_STATE_NAMES,
    PublicationState,
    WinnerSource,
    _parse_winner_source,
    _parsed_trial_outcome,
    _recorded_identity,
    _recorded_trial_dir,
    _stands_under_winner,
    _unreplaced_attempts,
    _validate_observed_block,
    _winning_trials,
)
from phasesweep.engine.study_policy import (
    _at_accepted_partial_target,
    _load_accepted_partial_decision,
    _load_phase_policy_state,
)
from phasesweep.errors import (
    PERSISTENT_STUDY_KEEPS_TRIALS,
    START_AGAIN_STEP,
    OperatorAction,
    operator_step,
)
from phasesweep.evidence.models import EXTRACTOR_KINDS
from phasesweep.runtime.json import JsonObject, JsonValue, float_or_none, json_object

log = logging.getLogger(__name__)

ResultContext: TypeAlias = Literal["represented_generation", "current_config"]
"""Which config's semantics a result payload's labels were read under.

``"represented_generation"``: the labels come from the represented
generation's own recorded summary, so they describe the result as it was
produced. ``"current_config"``: no represented result is available, so the
currently loaded config describes the status payload.
"""

CostAdvisoryStatus: TypeAlias = Literal[
    "not_configured", "unavailable", "no_cheaper_configuration", "available"
]

EdgeAdvisoryStatus: TypeAlias = Literal["unavailable", "not_at_edge", "available"]
"""Whether the edge advisory could be checked, and whether it found a parameter at an edge."""


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
    cost_advisory_status: CostAdvisoryStatus = "not_configured"
    tie_groups: list[TieGroupComparison] | None = None
    edge_advisory: str | None = None
    edge_advisory_status: EdgeAdvisoryStatus = "unavailable"
    # Frozen population identity used to reconstruct display details from retained trials.
    confound_detail: Mapping[str, object] | None = None
    # Finished trials the phase held at selection (``completion.finished_trials``); later
    # trials have higher numbers, so the first this many finished ones are the ones it saw.
    finished_trials: int | None = None


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
    winner: Mapping[str, object] | ValidatedWinner,
    phase_name: str,
    *,
    persistent: bool,
    dependents: Sequence[str] = (),
) -> PhaseWinnerView | None:
    """Reduce one decoded winner payload using the permissive read contract.

    :param Mapping[str, object] | ValidatedWinner winner: Decoded winner payload or
        captured publication facts.
    :param str phase_name: Phase name exposed by the winner.
    :param bool persistent: Whether the display config keeps its trials between runs, which
        decides the survivorship remedy (:func:`phasesweep.engine.confound._confound_summary`).
    :param Sequence[str] dependents: Phases that inherit this one in the display config.
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
            confound_block = validated.confound
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
            persistent=persistent,
            dependents=dependents,
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
            confound_detail=confound_block,
            finished_trials=_recorded_count(completion.get("finished_trials")),
            observed=observed,
            repeats=None if repeats is None else _repeats_summary(repeats),
            repeats_detail=repeats,
        )
    except (KeyError, ValueError, TypeError, OverflowError):
        return None


def _read_winner_path(
    path: Path, phase_name: str, *, persistent: bool, dependents: Sequence[str]
) -> PhaseWinnerView | None:
    """Read one unvalidated winner file using the permissive read contract.

    Only a generation that is not the validated publication is read this way:
    a pinned, unfinished, or failed one. A published winner is never reread;
    it comes from the bytes publication validation read.

    :param Path path: Winner path in the represented generation.
    :param str phase_name: Phase name exposed by the winner.
    :param bool persistent: Whether the display config keeps its trials between runs.
    :param Sequence[str] dependents: Phases that inherit this one in the display config.
    :return PhaseWinnerView | None: Parsed winner, or ``None`` when absent or malformed.
    """
    if not path.is_file():
        return None
    loaded = read_yaml_or_none(path)
    if not isinstance(loaded, Mapping):
        return None
    return _winner_view(loaded, phase_name, persistent=persistent, dependents=dependents)


def _represented_winners(
    experiment: Experiment,
    publication: PublicationPointer,
    represented_generation_id: str | None,
    phase_names: Sequence[str],
    *,
    ledger: ValidatedLedger,
    fallback_published: bool = False,
    comparison_experiment: Experiment | None = None,
) -> list[PhaseWinnerView]:
    """Return the represented generation's winners, in ``phase_names`` order.

    Where each winner is read from follows the module docstring.

    :param Experiment experiment: Experiment whose artifact tree is read.
    :param PublicationPointer publication: This read's single pointer resolution.
    :param str | None represented_generation_id: Generation whose winners are
        represented, or ``None`` when there is none.
    :param Sequence[str] phase_names: Safe phase names to enumerate.
    :param ValidatedLedger ledger: Handle the command validated for ``experiment``, which
        the read-time advice reads the phases' trials through (:func:`winner_advisories`).
    :param bool fallback_published: Fill missing latest phase winners from the publication.
    :param Experiment | None comparison_experiment: Current config for drift comparisons.
    :return list[PhaseWinnerView]: One view per named phase that has a usable winner.
    """
    if represented_generation_id is None:
        return []
    validated = publication.validated
    comparison = comparison_experiment or experiment
    # The next run uses the current config's storage and inheritance graph.
    persistent = comparison.persistent
    views: list[PhaseWinnerView | None]
    if validated is not None and validated.generation_id == represented_generation_id:
        views = [
            _winner_view(
                validated.winners[name],
                name,
                persistent=persistent,
                dependents=comparison.phases_inheriting(name),
            )
            if name in validated.winners
            else None
            for name in phase_names
        ]
    else:
        views = [
            _read_winner_path(
                _generation_winner_path(experiment, represented_generation_id, name),
                name,
                persistent=persistent,
                dependents=comparison.phases_inheriting(name),
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
            view = _winner_view(
                validated.winners[name],
                name,
                persistent=persistent,
                dependents=comparison.phases_inheriting(name),
            )
            view_generation = validated.generation_id
        if view is not None:
            is_published = validated is not None and view_generation == validated.generation_id
            if is_published and validated is not None:
                config_matches = _recorded_config_matches(
                    comparison,
                    validated.summary.get("config_fingerprint"),
                    phase_count=len(validated.phase_plan),
                )
            else:
                producing = _load_generation_experiment(
                    _experiment_dir(experiment), view_generation
                )
                config_matches = _recorded_config_matches(
                    comparison,
                    _experiment_semantic_fingerprint(producing),
                    phase_count=len(producing.phases),
                )
            advisories = winner_advisories(
                experiment, view, config_matches=config_matches is True, ledger=ledger
            )
            cost_status = advisories.cost_advisory_status
            if config_matches is not True and cost_status == "not_configured":
                producing = _load_generation_experiment(
                    _experiment_dir(experiment), view_generation
                )
                cost_status = cost_advisory_status(producing, name, None, available=False)
            chosen.append(
                replace(
                    view,
                    generation_id=view_generation,
                    is_published=is_published,
                    config_matches_current=config_matches,
                    cost_advisory=advisories.cost_advisory,
                    cost_advisory_status=cost_status,
                    tie_groups=advisories.tie_groups,
                    edge_advisory=advisories.edge_advisory,
                    edge_advisory_status=advisories.edge_advisory_status,
                )
            )
    return chosen


def require_study(experiment: Experiment) -> None:
    """Require evidence of a started study at the configured artifact root.

    :param Experiment experiment: Config locating the study.
    :raises ArtifactRootConflictError: Access to the study path is denied.
    :raises StudyStorageUnavailableError: No study has initialized this root.
    """
    ledger_path = resolved_ledger_path(experiment)
    artifact_root = _experiment_dir(experiment)
    if not S_ISDIR(_artifact_path_mode(artifact_root)) or not any(
        S_ISREG(_artifact_path_mode(path))
        for path in (
            _artifact_root_binding_path(experiment),
            _generation_path(experiment),
            *((ledger_path,) if ledger_path is not None else ()),
        )
    ):
        raise StudyStorageUnavailableError(
            f"No study directory at {artifact_root} contains a started study.",
            action=OperatorAction.FIX_CONFIG,
            remediation=(
                "Use the same config and --set overrides as the run that created the study. "
                "If no run has started, run `phasesweep run CONFIG` first."
            ),
        )


def cost_advisory_status(
    experiment: Experiment, phase_name: str, advisory: str | None, *, available: bool
) -> CostAdvisoryStatus:
    """Distinguish a disabled cost comparison from its checked and unavailable outcomes.

    :param Experiment experiment: Display config carrying the optional cost key.
    :param str phase_name: Phase whose cost comparison is described.
    :param str | None advisory: A cheaper configuration, if one was found.
    :param bool available: Whether the winner agrees with the display config and its retained groups are readable.
    :return CostAdvisoryStatus: Whether cost advice was configured, available, or found.
    """
    phase = next((phase for phase in experiment.phases if phase.name == phase_name), None)
    if phase is None:
        return "unavailable"
    if phase.cost_key is None:
        return "not_configured"
    if not available:
        return "unavailable"
    return "no_cheaper_configuration" if advisory is None else "available"


class WinnerAdvisories(typing.NamedTuple):
    """The advice recomputed at read from a winner's retained trials, never stored."""

    cost_advisory: str | None
    cost_advisory_status: CostAdvisoryStatus
    tie_groups: list[TieGroupComparison] | None
    edge_advisory: str | None
    edge_advisory_status: EdgeAdvisoryStatus


def winner_advisories(
    experiment: Experiment,
    winner: PhaseWinnerView,
    *,
    config_matches: bool,
    ledger: ValidatedLedger,
) -> WinnerAdvisories:
    """Recompute a winner's tie groups, cost advisory and edge advisory from its phase's trials.

    The phase's trials are read once for all three, and each is judged over the population
    that selected the winner, never over trials a later run added: the recorded repeat groups,
    and the first ``finished_trials`` finished trials. Tie groups need only that population.
    Cost and edge advice also read the phase's config (its ``cost_key``, its search space), so
    they are computed only when ``experiment`` is the config that produced the winner, and are
    otherwise ``None`` with status ``unavailable``, as they are when the trials cannot be read.

    :param Experiment experiment: Display config locating the retained phase ledger.
    :param PhaseWinnerView winner: Winner whose recorded population anchors the comparisons.
    :param bool config_matches: Whether ``experiment`` agrees with the producing config.
    :param ValidatedLedger ledger: Handle the command validated for ``experiment``, so a read
        of every winner's advice validates the ledger once.
    :return WinnerAdvisories: The advice and whether each could be checked.
    """
    phase = next((phase for phase in experiment.phases if phase.name == winner.phase), None)
    try:
        read = read_phase_trials(ledger, winner.phase)
    except StudyStorageUnavailableError:
        read = None
    trials = None if read is None else read.trials
    ranking = _winner_ranked_groups(winner, trials)
    tie_groups = (
        []
        if winner.repeats is None
        else None
        if ranking is None
        else tie_group_comparisons(*ranking)
    )
    cost = (
        _cost_advisory(phase, *ranking, goal=experiment.metric.goal)
        if config_matches
        and phase is not None
        and phase.cost_key is not None
        and ranking is not None
        else None
    )
    finished = sorted(
        (trial for trial in trials or () if _trial_count_state(trial) in FINISHED_TRIAL_STATES),
        key=lambda trial: trial.number,
    )
    edge_status: EdgeAdvisoryStatus = "unavailable"
    edge = None
    if (
        config_matches
        and phase is not None
        and trials is not None
        and winner.finished_trials is not None
        and winner.finished_trials <= len(finished)
    ):
        edge = _edge_advisory(
            phase,
            finished[: winner.finished_trials],
            winner.params,
            persistent=experiment.persistent,
        )
        edge_status = "not_at_edge" if edge is None else "available"
    return WinnerAdvisories(
        cost_advisory=cost,
        cost_advisory_status=cost_advisory_status(
            experiment, winner.phase, cost, available=config_matches and tie_groups is not None
        ),
        tie_groups=tie_groups,
        edge_advisory=edge,
        edge_advisory_status=edge_status,
    )


def _winner_ranked_groups(
    winner: PhaseWinnerView, retained: Sequence[optuna.trial.FrozenTrial] | None
) -> tuple[RankedGroup, list[RankedGroup]] | None:
    """Reconstruct the exact repeat-group population that selected one winner.

    :param PhaseWinnerView winner: Winner carrying its frozen confound population and reducer.
    :param Sequence[optuna.trial.FrozenTrial] | None retained: The phase's trials in the
        ledger, or ``None`` when they could not be read.
    :return tuple[RankedGroup, list[RankedGroup]] | None: Winner and recorded groups, or null when unavailable.
    """
    if winner.repeats is None or winner.confound_detail is None or retained is None:
        return None
    raw_groups = winner.confound_detail["ranked_groups"]
    if not isinstance(raw_groups, list):
        return None
    if winner.repeats_detail is None:
        return None
    trials = {trial.number: trial for trial in retained}
    groups = []
    for numbers in raw_groups:
        if not isinstance(numbers, list) or not all(isinstance(n, int) for n in numbers):
            raise TypeError("validated repeat-group trial numbers are not integer lists")
        if any(n not in trials or trials[n].value is None for n in numbers):
            return None
        members: list[tuple[int, float]] = []
        for number in numbers:
            value = trials[number].value
            if value is None:
                return None
            members.append((number, value))
        groups.append(
            ranked_group(
                members,
                {
                    k: v
                    for k, v in trials[numbers[-1]].params.items()
                    if k != winner.repeats_detail["key"]
                },
                winner.repeats["reduce"],
            )
        )
    winning = next((group for group in groups if group.representative == winner.trial_number), None)
    return None if winning is None else (winning, groups)


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
        and _generation_dir(_experiment_dir(experiment), current_id).is_dir()
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
    ledger: ValidatedLedger | None = None,
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
    determine ``published_config_matches_current`` against the producing phase prefix,
    or ``None`` if no fingerprint exists. Appended phases do not change that agreement.

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
    :param ValidatedLedger | None ledger: Handle the caller already validated for ``experiment``,
        so one command validates once; validated here when omitted.
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
        ledger=ledger,
    )
    return status


def read_result(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    comparison_experiment: Experiment | None = None,
    phase_names: Sequence[str] | None = None,
    ledger: ValidatedLedger | None = None,
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
    :param ValidatedLedger | None ledger: Handle the caller already validated for ``experiment``,
        so one command validates once; validated here when omitted.
    :raises ArtifactRootConflictError: If the artifact root cannot be read or
        is bound to a different storage ledger or experiment.
    :raises WinnerIntegrityError: An unpublished winner's producing snapshot is unavailable or invalid.
    :raises ValueError: If ``generation_id`` or an explicit phase name is not safe.
    :return tuple[EngineStatus, list[PhaseWinnerView]]: Status and usable winners
        in the selected phase order.
    """
    names = (
        None
        if phase_names is None
        else [_validate_safe_name("phase", name) for name in phase_names]
    )
    ledger = validate_ledger(experiment) if ledger is None else ledger
    status, publication = _read_status(
        experiment,
        generation_id=generation_id,
        trial_generation_id=None,
        comparison_experiment=comparison_experiment,
        ledger=ledger,
    )
    winners = _represented_winners(
        experiment,
        publication,
        status["represented_generation_id"],
        status["result_phase_plan"] if names is None else names,
        ledger=ledger,
        comparison_experiment=comparison_experiment,
    )
    return status, winners


def _read_status(
    experiment: Experiment,
    *,
    generation_id: str | None,
    trial_generation_id: str | None,
    comparison_experiment: Experiment | None,
    latest: bool = False,
    ledger: ValidatedLedger | None = None,
) -> tuple[EngineStatus, PublicationPointer]:
    """Build the :func:`read_status` payload and return the resolution behind it.

    :param Experiment experiment: Parsed experiment config whose phases are inspected.
    :param str | None generation_id: Optional generation to represent.
    :param str | None trial_generation_id: Optional generation whose trials
        ``generation_trials`` counts instead of the represented one.
    :param Experiment | None comparison_experiment: Optional current config for
        drift comparison.
    :param bool latest: Represent the captured current generation when its namespace exists.
    :param ValidatedLedger | None ledger: Handle the caller already validated for ``experiment``,
        so one command validates once; validated here when omitted.
    :raises ArtifactRootConflictError: If the artifact root cannot be read or
        is bound to a different storage ledger or experiment.
    :raises ValueError: If ``generation_id`` or ``trial_generation_id`` is not a safe
        generation name.
    :return tuple[EngineStatus, PublicationPointer]: The status payload and
        the single pointer resolution every fact in it came from.
    """
    ledger = validate_ledger(experiment) if ledger is None else ledger
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
        if experiment.persistent
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
        stored_evidence = (
            stored_metric.get("objective_evidence") if isinstance(stored_metric, Mapping) else None
        )
        if (
            isinstance(stored_metric, Mapping)
            and isinstance(stored_metric.get("name"), str)
            and stored_metric.get("goal") in ("minimize", "maximize")
            and isinstance(stored_evidence, Mapping)
            and stored_evidence.get("kind") in EXTRACTOR_KINDS
        ):
            metric_payload = {
                "name": stored_metric["name"],
                "goal": stored_metric["goal"],
                "objective_evidence": {"kind": stored_evidence["kind"]},
            }
            result_context = "represented_generation"
    stored_fingerprint = (
        None if summary_payload is None else summary_payload.get("config_fingerprint")
    )
    published_config_matches_current = _recorded_config_matches(
        comparison, stored_fingerprint, phase_count=len(result_phase_plan)
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
                ledger=ledger,
                fallback_published=True,
                comparison_experiment=comparison_experiment,
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
    ``state``, ``value``, ``params``, ``duration_s``, ``trial_dir`` (the trial directory the
    trial recorded, ``None`` when it recorded none), then ``repeat_group`` for a
    repeats phase, ``rerun_of`` once any trial is a rerun, one ``observed:<name>``
    per report-only value any trial recorded, an optional ``reason`` when any
    trial has a recorded cause, and ``winner``. ``state`` is the Optuna state, except that a
    COMPLETE trial that violated a constraint or a metric bound reads
    ``INFEASIBLE``. ``winner`` is ``*`` on the trials the represented phase's
    winner stands on and empty elsewhere. ``finished`` is how many rows count toward the
    phase's ``n_trials``, and ``uncounted`` pairs each state of the rows that do not with its
    row count, in :data:`~phasesweep.engine.state.TRIAL_STATE_NAMES` order.
    """

    phase: str
    #: ``False`` when the phase's trials could not be read; the rows are then empty.
    available: bool
    columns: tuple[str, ...]
    rows: tuple[JsonObject, ...]
    finished: int
    uncounted: tuple[tuple[str, int], ...]
    groups: tuple[RepeatGroupView, ...] = ()
    #: The diagnosis and step a run of the same config would be refused with, from the phase's
    #: recorded abort or failure streak (:func:`phasesweep.engine.selection._phase_refusal`).
    refusal: tuple[str, str] | None = None
    #: The metric name and goal the study recorded, which its trial values were measured under;
    #: ``None`` when the phase has no study or the study kept no identity payload.
    recorded_metric: tuple[str, str] | None = None
    #: Each search-space growth the study recorded: the draw position it took effect at and the
    #: values it added (:func:`phasesweep.engine.growth.recorded_growth`).
    growth: tuple[tuple[int, str], ...] = ()


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


@dataclass(frozen=True)
class UnpublishedWinnerStep:
    """How a retained winner gets published: the recorded cause when one blocks it, then the fix."""

    cause: str | None
    fix: str


def _status_phase_refusal(
    experiment: Experiment, phase: Phase, study: optuna.Study
) -> PhaseAbortRefusedError | None:
    """Give status the failure-policy refusal when a partial decision does not take precedence.

    :param Experiment experiment: Current configuration, which identifies dependent phases.
    :param Phase phase: Phase whose target and failure limit apply.
    :param optuna.Study study: Read-only phase study containing the recorded decisions.
    :return PhaseAbortRefusedError | None: Failure-policy refusal to report, if any.
    """
    policy = _load_phase_policy_state(study)
    if _at_accepted_partial_target(_load_accepted_partial_decision(study), phase):
        return None
    return _phase_refusal(
        phase,
        study,
        policy,
        dependents=experiment.phases_inheriting(phase.name),
    )


def phase_refusals(
    experiment: Experiment, ledger: ValidatedLedger
) -> dict[str, PhaseAbortRefusedError | StudySchemaMismatchError]:
    """Give the refusal each phase's recorded failure policy would meet on a run.

    Invalid policy state is reported for its phase alongside other refusals. It does not
    prevent status from reporting trial counts and publication facts.

    :param Experiment experiment: Current config, which decides whether a recorded abort is
        retired.
    :param ValidatedLedger ledger: Handle the caller validated for ``experiment``.
    :return dict[str, PhaseAbortRefusedError | StudySchemaMismatchError]: Refusals or invalid
        policy diagnostics by phase name; empty without a readable ledger.
    """
    if not experiment.persistent:
        return {}
    try:
        studies = read_phase_studies(ledger)
    except StudyStorageUnavailableError:
        return {}
    refusals: dict[str, PhaseAbortRefusedError | StudySchemaMismatchError] = {}
    for phase in experiment.phases:
        study = studies[phase.name]
        if study is None:
            continue
        try:
            refusal = _status_phase_refusal(experiment, phase, study)
        except StudySchemaMismatchError as exc:
            refusals[phase.name] = exc
            continue
        if refusal is not None:
            refusals[phase.name] = refusal
    return refusals


def unpublished_winner_step(
    experiment: Experiment, ledger: ValidatedLedger
) -> UnpublishedWinnerStep | None:
    """Give one publication remedy from a phase without a feasible winner, for an idle experiment.

    A recorded abort or failure streak is :func:`phase_refusals`' to report.

    :param Experiment experiment: Current config used to inspect durable phase policy.
    :param ValidatedLedger ledger: Handle the caller validated for ``experiment``.
    :return UnpublishedWinnerStep | None: The recorded cause and remedy, the rerun instruction,
        or ``None`` if storage is unreadable.
    """
    if not experiment.persistent:
        return UnpublishedWinnerStep(None, START_AGAIN_STEP)
    try:
        studies = read_phase_studies(ledger)
    except StudyStorageUnavailableError:
        return None
    for phase in experiment.phases:
        study = studies[phase.name]
        if study is None:
            continue
        trials = study.get_trials(deepcopy=False)
        if _finished_trial_count(trials) < phase.n_trials:
            continue
        axis = repeat_axis(phase)
        eligible = (
            any(_eligible(trial, experiment) for trial in trials)
            if axis is None
            else bool(_rank_groups(trials, experiment, axis).groups)
        )
        if not eligible:
            return UnpublishedWinnerStep(
                f"Phase {phase.name!r} has no feasible winner.",
                _no_feasible_remedy(experiment, trials, phase.name),
            )
    return UnpublishedWinnerStep(
        None, f"Rerun the same config to resume and publish; {PERSISTENT_STUDY_KEEPS_TRIALS}."
    )


def experiment_status(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    ledger: ValidatedLedger | None = None,
) -> JsonObject:
    """Inspect study progress and phase winners without creating or modifying state.

    By default each phase uses its latest usable winner, falling back to the validated
    publication. Per-phase generation, publication and config-match labels identify
    that choice. An explicit generation pin reads only that generation. Corrupt
    publications expose diagnostics and counts without winner paths.

    :param Experiment experiment: Parsed experiment config to inspect.
    :param str | None generation_id: Optional generation pin, disabling published fallback.
    :param ValidatedLedger | None ledger: Handle the caller already validated for ``experiment``,
        so one command validates once; validated here when omitted.
    :raises StudyStorageUnavailableError: No study has started at the configured root.
    :raises ArtifactRootConflictError: The root belongs to another ledger or experiment.
    :raises StudySchemaMismatchError: The stored study format is unsupported.
    :raises WinnerIntegrityError: An unpublished winner's producing snapshot is unavailable or invalid.
    :return JsonObject: Paths, pointer identities, integrity, phase counts and labeled winners.
    """
    require_study(experiment)
    status, _ = _read_status(
        experiment,
        generation_id=generation_id,
        trial_generation_id=None,
        comparison_experiment=None,
        latest=True,
        ledger=ledger,
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


def _recorded_count(value: object) -> int | None:
    """Read a count a winner file recorded, or ``None`` when it is not a non-negative integer.

    :param object value: The recorded value.
    :return int | None: The count, or ``None``.
    """
    return value if type(value) is int and value >= 0 else None


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


def _trial_dir_text(trial: optuna.trial.FrozenTrial) -> str | None:
    """Read the trial directory a trial recorded, as text.

    :param optuna.trial.FrozenTrial trial: Trial whose directory is displayed.
    :return str | None: The directory, or ``None`` when none was recorded.
    """
    directory = _recorded_trial_dir(trial)
    return None if directory is None else str(directory)


#: Columns every trial table starts with, as ``(header, getter)`` pairs.
_CORE_TRIAL_COLUMNS: tuple[tuple[str, Callable[[optuna.trial.FrozenTrial], JsonValue]], ...] = (
    ("trial", lambda trial: trial.number),
    ("state", _trial_count_state),
    ("value", lambda trial: trial.value),
    ("params", lambda trial: dict(trial.params)),
    ("duration_s", lambda trial: float_or_none(trial.user_attrs.get(DURATION_ATTR))),
    ("trial_dir", _trial_dir_text),
)
_REPEAT_GROUP_COLUMN = "repeat_group"
_RERUN_OF_COLUMN = "rerun_of"
_WINNER_COLUMN = "winner"


def _trial_table(
    experiment: Experiment,
    phase: Phase,
    trials: Sequence[optuna.trial.FrozenTrial],
    winner: PhaseWinnerView | None,
    *,
    available: bool = True,
    refusal: tuple[str, str] | None = None,
    recorded_metric: tuple[str, str] | None = None,
    growth: tuple[tuple[int, str], ...] = (),
) -> TrialTable:
    """Build one phase's table from its trials and the winner the publication holds for it.

    A ``rerun_of`` column, naming the interrupted trial or failed grid cell each rerun
    reran, appears once the phase holds a rerun.

    :param Experiment experiment: Current metric and constraint eligibility bounds.
    :param Phase phase: The configured phase the trials belong to.
    :param Sequence[optuna.trial.FrozenTrial] trials: The phase study's trials.
    :param PhaseWinnerView | None winner: The published winner for the phase, if any.
    :param bool available: ``False`` when the trials could not be read, so ``trials`` is empty.
    :param tuple[str, str] | None refusal: The diagnosis and step a run would be refused with,
        defaults to none.
    :param tuple[str, str] | None recorded_metric: The name and goal the study recorded,
        defaults to none.
    :param tuple[tuple[int, str], ...] growth: The search-space growths the study recorded,
        defaults to none.
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
    standing = (
        {}
        if winner is None
        else _winning_trials(
            winner.trial_number,
            None if winner.source is None else winner.source.attempt_id,
            winner.repeats_detail,
        )
    )
    rows: list[JsonObject] = []
    for trial in trials:
        row: JsonObject = {header: getter(trial) for header, getter in _CORE_TRIAL_COLUMNS}
        if axis is not None:
            row[_REPEAT_GROUP_COLUMN] = groups.get(trial.number)
        if reruns:
            row[_RERUN_OF_COLUMN] = trial.user_attrs.get(RERUN_OF_ATTR)
        for name in observed:
            row[name] = float_or_none(trial.user_attrs.get(name))
        if has_reason:
            row["reason"] = reasons[trial.number]
        row[_WINNER_COLUMN] = "*" if _stands_under_winner(trial, standing) else ""
        rows.append(row)
    states = [_trial_count_state(trial) for trial in trials]
    return TrialTable(
        phase=phase.name,
        available=available,
        columns=columns,
        rows=tuple(rows),
        finished=_finished_trial_count(trials),
        uncounted=tuple(
            (state, states.count(state))
            for state in TRIAL_STATE_NAMES
            if state not in FINISHED_TRIAL_STATES and state in states
        ),
        groups=_repeat_group_views(experiment, phase, trials, rows),
        refusal=refusal,
        recorded_metric=recorded_metric,
        growth=growth,
    )


def _recorded_metric(study: optuna.Study) -> tuple[str, str] | None:
    """Read the metric name and goal a study recorded in its identity payload.

    The recorded values are the ones its trials were measured and ranked under, so a table
    labels them with these, whatever the config names now.

    :param optuna.Study study: The phase's study.
    :return tuple[str, str] | None: ``(name, goal)``, or ``None`` when the study kept no identity.
    """
    record = _recorded_identity(study)
    metric = None if record is None else record["payload"].get("metric")
    name = metric.get("name") if isinstance(metric, dict) else None
    goal = metric.get("goal") if isinstance(metric, dict) else None
    return (name, goal) if isinstance(name, str) and isinstance(goal, str) else None


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
    standing = {trial.number for trial in _unreplaced_attempts(trials, interrupted=False)}
    marked = {row["trial"] for row in rows if row["winner"] == "*"}
    members: dict[int, list[optuna.trial.FrozenTrial]] = {}
    params: dict[int, JsonObject] = {}
    # Every labelled trial opens its group's entry, so a group whose members were all
    # replaced or interrupted still gets a view; only the standing members are measured.
    for trial in trials:
        label = labels.get(trial.number)
        if label is None:
            continue
        members.setdefault(label, [])
        params[label] = {name: value for name, value in trial.params.items() if name != axis.key}
        if trial.number in standing:
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
                value = float_or_none(raw)
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
    have no rows. Unreadable phases are reported unavailable. A phase whose recorded abort or
    failure streak would refuse a run carries that refusal, read from the same study snapshot as
    its trials.

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
            ledger=ledger,
            fallback_published=latest and generation_id is None,
        )
    }
    tables: list[TrialTable] = []
    unreadable: StudyStorageUnavailableError | None = None
    studies: dict[str, optuna.Study | None] = {}
    try:
        studies = read_phase_studies(ledger)
    except StudyStorageUnavailableError as exc:
        unreadable = exc
    for phase in experiment.phases:
        if unreadable is not None:
            log_storage_read_failure(ledger, phase, unreadable, logger=log)
            tables.append(_trial_table(experiment, phase, [], None, available=False))
            continue
        study = studies[phase.name]
        if study is None:
            tables.append(_trial_table(experiment, phase, [], winners.get(phase.name)))
            continue
        refusal = _status_phase_refusal(experiment, phase, study)
        identity = _recorded_identity(study)
        tables.append(
            _trial_table(
                experiment,
                phase,
                study.get_trials(deepcopy=False),
                winners.get(phase.name),
                refusal=(str(refusal), operator_step(refusal)) if refusal is not None else None,
                recorded_metric=_recorded_metric(study),
                growth=() if identity is None else recorded_growth(identity),
            )
        )
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
