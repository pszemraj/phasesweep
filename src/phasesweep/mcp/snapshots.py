"""Validated, path-free terminal result snapshots for MCP run handles."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import asdict
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from phasesweep.config.models import Experiment, _metric_semantics_payload
from phasesweep.engine.artifacts import read_yaml_or_none
from phasesweep.engine.confound import ConfoundCheck, ConfoundVerdict, _confound_summary
from phasesweep.engine.fingerprints import _experiment_semantic_fingerprint
from phasesweep.engine.ledger import ValidatedLedger, validate_ledger
from phasesweep.engine.optuna import _published_phase_trial_refs
from phasesweep.engine.paths import _generation_record_path
from phasesweep.engine.read import (
    CostAdvisoryStatus,
    EdgeAdvisoryStatus,
    EngineStatus,
    PhaseWinnerView,
    ResultContext,
    WinnerAdvisories,
    _recorded_count,
    read_result,
    read_status,
    winner_advisories,
)
from phasesweep.engine.repeats import _repeats_summary
from phasesweep.engine.state import (
    TRIAL_STATE_NAMES,
    PublicationState,
    RecoveredTrialState,
    TrialStateName,
    Winner,
    WinnerSourceKind,
)
from phasesweep.evidence.models import ExtractorKind
from phasesweep.runtime.json import JsonObject, dump_json

log = logging.getLogger("phasesweep.mcp.snapshots")

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
McpPublicationState = PublicationState | Literal["unknown"]
"""Engine publication verdict plus the MCP snapshot-unavailable state."""

RESULT_SNAPSHOT_SCHEMA_VERSION: Literal[6] = 6


class _SnapshotModel(BaseModel):
    """Strict base for persisted result snapshot records."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ObjectiveEvidenceSnapshot(_SnapshotModel):
    """The objective extractor kind a result was produced with."""

    kind: ExtractorKind


class MetricSnapshot(_SnapshotModel):
    """Optimization metric and the extractor kind that produced it."""

    name: str = Field(description="Metric key extracted from trial output.")
    goal: Literal["minimize", "maximize"] = Field(description="Optimization direction.")
    objective_evidence: ObjectiveEvidenceSnapshot = Field(
        description=(
            "kind names the extractor that produced the objective. Only json_envelope, wandb "
            "and mlflow bind an evaluation position."
        )
    )


class RunningAttemptSnapshot(_SnapshotModel):
    """Concrete RUNNING row identity captured with a terminal snapshot."""

    trial_number: NonNegativeInt
    generation_id: str | None = None
    attempt_id: str | None = None
    recovers_as: RecoveredTrialState
    """Captured recovery prediction; successful trainer evidence stays pending until extraction."""


class PhaseStatusSnapshot(_SnapshotModel):
    """One phase's trial counts by state at capture, and winner presence."""

    phase: str
    trials: dict[str, NonNegativeInt]
    n_trials: NonNegativeInt
    generation_trials: dict[str, NonNegativeInt]
    winner_present: bool
    trial_data_available: bool
    published_study_unavailable: bool | None = None
    """Published study unavailable at capture; ``None`` means unchecked."""
    running_attempts: list[RunningAttemptSnapshot] | None = None
    """RUNNING rows behind the frozen counts; ``None`` means no trial data was read.

    Known counts carry a list, empty when no row is RUNNING or the study is absent.
    """


class StatusSnapshot(_SnapshotModel):
    """Terminal status with generation identity and publication frozen at capture."""

    current_generation_id: str | None = None
    published_generation_id: str | None = None
    represented_generation_id: str | None = None
    is_published: bool = False
    publication_integrity: McpPublicationState
    """Publication verdict recorded at capture time."""

    metric: MetricSnapshot
    phases: list[PhaseStatusSnapshot]
    summary_present: bool

    result_context: ResultContext
    """Source of ``metric`` and ``result_phase_plan``: represented generation or config."""

    published_config_matches_current: bool | None = None
    """Config agreement at capture, recomputed against the catalog on run-scoped reads.

    ``None`` means the comparison config or represented fingerprint was unavailable.
    """

    result_phase_plan: list[str]
    """Phase plan the represented generation published under."""


def status_snapshot(status: EngineStatus) -> StatusSnapshot:
    """Project an engine status read onto the model frozen reads and redaction share.

    Engine-only keys (the experiment label, the operator-only publication error and the
    mutable tracking counts) have no snapshot field and are dropped.

    :param EngineStatus status: Path-free status from :func:`phasesweep.engine.read.read_status`
        or :func:`phasesweep.engine.read.read_result`.
    :raises ValidationError: The read does not fit the strict snapshot model.
    :return StatusSnapshot: Validated status.
    """
    return StatusSnapshot.model_validate(
        {key: value for key, value in status.items() if key in StatusSnapshot.model_fields}
    )


class WinnerSourceSnapshot(_SnapshotModel):
    """Concrete source trial for an exposed phase winner."""

    kind: WinnerSourceKind
    phase: str
    trial_number: int
    generation_id: str
    attempt_id: str


_VERDICT_DESCRIPTION = (
    "ok: checked, nothing found. heterogeneous: a potential confound in the population "
    "this winner was chosen from. n/a: not checked."
)


class EvaluationPointSummary(_SnapshotModel):
    """Whether every ranked objective was measured at one training position.

    Positions are on the experiment's configured evaluation axis -- the
    progress axis its sweep holds fixed, such as optimizer step or tokens.
    """

    verdict: ConfoundVerdict = Field(
        description=(
            f"{_VERDICT_DESCRIPTION} Always n/a for json and log_regex objectives, which "
            "record no evaluation position, and for mlflow select: min or max, whose step is "
            "where the best value fell; json_envelope, wandb and other mlflow selectors bind one."
        )
    )
    distinct_values: NonNegativeInt | None = Field(
        description=(
            "Distinct positions on the configured evaluation axis the ranked objectives "
            "were measured at; null when not checked."
        )
    )
    axis: str | None = Field(
        description="Declared axis compared across ranked objectives; null when not checked."
    )
    remedy: str | None = Field(
        description="Reporting remedy for a training-length comparison; null when no mismatch was found."
    )
    winner_value: NonNegativeInt | NonNegativeFloat | None = Field(
        description=(
            "Position on the configured evaluation axis the winning objective was "
            "measured at; null when not checked."
        )
    )


class SurvivorshipSummary(_SnapshotModel):
    """How the configurations the phase tried ended: ranked, infeasible, failed or never tried.

    A configuration is a trial's params, or a repeat group in a repeats phase.
    """

    verdict: ConfoundVerdict = Field(
        description=(
            f"{_VERDICT_DESCRIPTION} Heterogeneous when any configuration failed or a grid "
            "cell was never tried, so the winner was chosen without them; infeasible "
            "configurations never flag."
        )
    )
    remedy: str | None = Field(
        description=(
            "How to give the configurations without a result one when the check flagged: a "
            "raised n_trials with persistent storage, a rerun with storage: null; otherwise null."
        )
    )
    ranked: NonNegativeInt = Field(description="Configurations selection ranked.")
    infeasible: NonNegativeInt = Field(
        description="Configurations measured outside a metric bound or constraint."
    )
    failed: NonNegativeInt = Field(
        description="Configurations whose every attempt failed, with no rerun pending."
    )
    untried: NonNegativeInt = Field(
        description=(
            "Grid cells' configurations no trial reached although the grid finished a "
            "target covering every cell; 0 for any other phase."
        )
    )
    failed_trials: list[NonNegativeInt] = Field(
        description="The latest failed attempt of each failed configuration, by trial number."
    )


class TieSummary(_SnapshotModel):
    """Whether draw order or seed noise could have decided the winner.

    Without repeats, ok only means no other configuration matched the winner's value
    exactly; it says nothing about the margin. With mean repeats, it also flags groups whose
    mean is within the noise band pooled over every ranked group. A unit is a trial, or a
    complete repeat group in a repeats phase.
    """

    verdict: ConfoundVerdict = Field(
        description=(
            f"{_VERDICT_DESCRIPTION} With reduce: median, n/a unless a group matched the "
            "winner's median exactly, since medians have no noise band."
        )
    )
    remedy: str | None = Field(
        description=(
            "For a flagged exact tie between single trials, what draw order decided and how to "
            "separate the configurations, distinct values first; otherwise null."
        )
    )
    tied_trials: NonNegativeInt | None = Field(
        description=(
            "Units tying the winner's metric exactly with other params; null when not checked."
        )
    )
    within_noise: NonNegativeInt | None = Field(
        description=(
            "Repeat groups whose mean differs from the winning group's by at most the "
            "two-sided 5% Student-t value at df times pooled_sd * sqrt(1/n_winner + 1/n_other), "
            "exact ties included. Null without repeats, with reduce: median, or when not checked."
        )
    )
    pooled_sd: NonNegativeFloat | None = Field(
        description=(
            "Run-to-run standard deviation pooled over every ranked repeat group, assuming the "
            "noise is about the same size at each configuration. Null when within_noise is."
        )
    )
    df: PositiveInt | None = Field(
        description="Degrees of freedom of pooled_sd: the sum of n - 1 over the ranked groups."
    )


class CodeRevisionSummary(_SnapshotModel):
    """Whether every ranked trial ran under one recorded trainer code state.

    A generation is one recorded state: the trainer working directory's git commit, and
    whether tracked files differed from it. Edits made during a generation, and differences
    between two dirty trees, are not recorded. A changed commit is allowed on resume; this
    only reports what the winner was chosen among.
    """

    verdict: ConfoundVerdict = Field(
        description=(
            f"{_VERDICT_DESCRIPTION} Heterogeneous when ranked trials span generations that "
            "recorded more than one commit, a dirty tree, or no readable revision; a single "
            "generation is never flagged."
        )
    )
    remedy: str | None = Field(
        description="What the ranking compared and what to weigh; null unless flagged."
    )
    generations: NonNegativeInt | None = Field(
        description="Generations the ranked trials ran in; null when not checked."
    )
    commits: NonNegativeInt | None = Field(
        description="Distinct commits those generations recorded; null when not checked."
    )
    dirty: NonNegativeInt | None = Field(
        description="Generations that recorded a dirty tree; null when not checked."
    )
    unknown: NonNegativeInt | None = Field(
        description=(
            "Generations with no readable revision: not a git work tree, or an unreadable "
            "record; null when not checked."
        )
    )


class ConfoundSummary(_SnapshotModel):
    """Advisory verdicts on the population the winner was selected from.

    Frozen when the winner was selected. The evidence behind each verdict
    stays in the operator-facing winner file; this summary carries verdicts,
    counts, the evaluation axis and reporting guidance.
    """

    flagged: list[ConfoundCheck] = Field(
        description="Checks whose verdict is heterogeneous, in report order."
    )
    evaluation_point: EvaluationPointSummary
    survivorship: SurvivorshipSummary
    tie: TieSummary
    code_revision: CodeRevisionSummary


class RepeatsSpread(_SnapshotModel):
    """How a repeats winner's members spread around the value it ranked by.

    The repeat key and its values are config-authored and never appear here.
    """

    reduce: Literal["mean", "median"] = Field(
        description="Reducer that turned the member metrics into the winner's metric."
    )
    n: NonNegativeInt = Field(description="Members in the winning group, one per repeat value.")
    mean: float
    sd: float = Field(description="Sample standard deviation of the member metrics.")
    min: float
    max: float
    member_trials: list[NonNegativeInt] = Field(
        description="Trial numbers of the members, ascending; the last one is trial_number."
    )


class TieGroupSnapshot(_SnapshotModel):
    """Configuration whose mean is within the phase's pooled noise band of the winner's.

    ``combined_se`` is the pooled standard error of the difference between the two means and
    ``threshold`` the Student-t critical value times it. Groups ranked by median have no band:
    one ties only when its median equals the winner's, at ``gap`` 0 with both fields null.
    """

    trial_number: NonNegativeInt
    params: dict[str, bool | int | float | str | None]
    gap: NonNegativeFloat
    combined_se: NonNegativeFloat | None
    threshold: NonNegativeFloat | None


class WinnerSnapshot(_SnapshotModel):
    """One sampled phase winner captured without effective overrides."""

    phase: str
    trial_number: NonNegativeInt
    metric: float
    params: dict[str, Any]
    gates_passed: bool | None
    incomplete: bool
    generation_id: str | None = None
    attempt_id: str | None = None
    source: WinnerSourceSnapshot
    confound: ConfoundSummary
    observed: dict[str, float]
    repeats: RepeatsSpread | None
    cost_advisory: str | None = None
    cost_advisory_status: CostAdvisoryStatus
    tie_groups: list[TieGroupSnapshot] | None
    edge_advisory: str | None
    edge_advisory_status: EdgeAdvisoryStatus


def _winner_snapshot(
    phase: str,
    winner: PhaseWinnerView | Winner,
    *,
    experiment: Experiment,
    ledger: ValidatedLedger,
) -> WinnerSnapshot:
    """Freeze either engine winner representation into one validated snapshot.

    A read view already carries its read-time advice; an engine winner gets it here.

    :param str phase: Phase under which the winner is exposed.
    :param PhaseWinnerView | Winner winner: Winner representation to serialize.
    :param Experiment experiment: The config the runner executed, which produced an engine
        winner: it locates the trials the advice is recomputed from
        (:func:`phasesweep.engine.read.winner_advisories`) and decides the survivorship remedy.
    :param ValidatedLedger ledger: Handle the snapshot validated once for ``experiment``.
    :raises RuntimeError: Either representation carries no confound verdicts or no source
        provenance.
    :return WinnerSnapshot: Validated path-free winner snapshot.
    """
    advisories: WinnerAdvisories
    if isinstance(winner, Winner):
        gates_passed = (
            all(bool(gate.get("passed")) for gate in winner.gates) if winner.gates else None
        )
        incomplete = bool(winner.completion.get("incomplete", False))
        provenance = winner.objective_provenance
        extractor = provenance.get("extractor") if isinstance(provenance, Mapping) else None
        kind = extractor.get("kind") if isinstance(extractor, Mapping) else None
        if winner.confound is None:
            raise RuntimeError("Current-format winner is missing confound verdicts.")
        confound = _confound_summary(
            winner.confound,
            winner_trial=winner.trial_number,
            extractor_kind=kind if isinstance(kind, str) else "unknown",
            persistent=experiment.persistent,
            dependents=experiment.phases_inheriting(phase),
        )
        repeats = None if winner.repeats is None else _repeats_summary(winner.repeats)
        view = PhaseWinnerView(
            phase=phase,
            trial_number=winner.trial_number,
            metric=winner.metric,
            params=dict(winner.params),
            effective_overrides={},
            gates_passed=gates_passed,
            incomplete=incomplete,
            repeats=repeats,
            repeats_detail=winner.repeats,
            confound_detail=winner.confound,
            finished_trials=_recorded_count(winner.completion.get("finished_trials")),
        )
        advisories = winner_advisories(experiment, view, config_matches=True, ledger=ledger)
    else:
        gates_passed = winner.gates_passed
        incomplete = winner.incomplete
        if winner.confound is None:
            raise RuntimeError("Current-format winner is missing confound verdicts.")
        confound = winner.confound
        repeats = winner.repeats
        advisories = WinnerAdvisories(
            winner.cost_advisory,
            winner.cost_advisory_status,
            winner.tie_groups,
            winner.edge_advisory,
            winner.edge_advisory_status,
        )
    tie_groups = advisories.tie_groups
    source = winner.source
    if source is None:
        raise RuntimeError("Current-format winner is missing source provenance.")
    return WinnerSnapshot(
        phase=phase,
        trial_number=winner.trial_number,
        metric=winner.metric,
        params=dict(winner.params),
        gates_passed=gates_passed,
        incomplete=incomplete,
        generation_id=winner.generation_id if isinstance(winner, Winner) else source.generation_id,
        attempt_id=winner.attempt_id if isinstance(winner, Winner) else source.attempt_id,
        source=WinnerSourceSnapshot.model_validate(asdict(source)),
        confound=ConfoundSummary.model_validate(confound),
        observed=dict(winner.observed),
        repeats=None if repeats is None else RepeatsSpread.model_validate(repeats),
        cost_advisory=advisories.cost_advisory,
        cost_advisory_status=advisories.cost_advisory_status,
        tie_groups=(
            None if tie_groups is None else [TieGroupSnapshot.model_validate(g) for g in tie_groups]
        ),
        edge_advisory=advisories.edge_advisory,
        edge_advisory_status=advisories.edge_advisory_status,
    )


class RunResultSnapshot(_SnapshotModel):
    """Terminal status and winners frozen for one MCP run id."""

    schema_version: Literal[6]
    status: StatusSnapshot
    winners: list[WinnerSnapshot]
    represented_config_fingerprint: str | None = None
    """Represented config's semantic fingerprint, captured using the run's working directory."""


def capture_pre_generation_result_snapshot(experiment: Experiment) -> RunResultSnapshot:
    """Freeze config shape with zero trial counts, no winners and no generation ID.

    Reads no mutable study or winner state; publication and trial availability are unknown.

    :param Experiment experiment: Exact config snapshot the detached runner
        attempted to execute.
    :return RunResultSnapshot: Validated placeholder result snapshot.
    """
    zero_counts: dict[str, int] = {state: 0 for state in TRIAL_STATE_NAMES}
    return RunResultSnapshot(
        schema_version=RESULT_SNAPSHOT_SCHEMA_VERSION,
        status=StatusSnapshot(
            publication_integrity="unknown",
            result_context="current_config",
            result_phase_plan=[phase.name for phase in experiment.phases],
            metric=MetricSnapshot.model_validate(_metric_semantics_payload(experiment.metric)),
            phases=[
                PhaseStatusSnapshot(
                    phase=phase.name,
                    trials=dict(zero_counts),
                    n_trials=phase.n_trials,
                    generation_trials={},
                    winner_present=False,
                    trial_data_available=False,
                )
                for phase in experiment.phases
            ],
            summary_present=False,
        ),
        winners=[],
    )


def capture_result_snapshot(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    trial_generation_id: str | None = None,
    generation_claimed: bool = True,
    engine_winners: Mapping[str, Winner] | None = None,
) -> dict[str, Any]:
    """Capture path-free status and winners for the requested generation.

    A pinned generation retains its own partial winners even if unpublished. One tolerant
    status read supplies all counts and RUNNING identities; unreadable phases retain
    ``trial_data_available: false``. Engine-supplied winners are authoritative, and each
    phase's ``winner_present`` says whether they hold its winner; otherwise read the
    generation's validated winner bytes and recorded phase plan.

    An absent post-commit lifecycle record is tolerated; mismatches are logged.

    :param Experiment experiment: Exact config snapshot the detached runner executed.
    :param str | None generation_id: Generation to represent: the engine generation that
        owns the experiment lock, or the published generation a replay matched.
    :param str | None trial_generation_id: Generation whose trials ``generation_trials``
        counts when it differs from ``generation_id``; only with ``engine_winners``.
    :param bool generation_claimed: ``False`` when ``generation_id`` is the id of an
        invocation that never claimed its generation; that id names no files, so the
        snapshot represents no generation while still counting its trials.
    :param Mapping[str, Winner] | None engine_winners: The engine's own
        winner mapping from its terminal report, when the run succeeded.
    :return dict[str, Any]: JSON-serializable terminal result snapshot.
    """
    if generation_id is not None:
        lifecycle = read_yaml_or_none(_generation_record_path(experiment, generation_id))
        if lifecycle is not None and (
            not isinstance(lifecycle, Mapping) or lifecycle.get("generation_id") != generation_id
        ):
            log.warning(
                "generation %s lifecycle record does not match its identity; "
                "capturing the terminal snapshot from the authoritative artifacts anyway",
                generation_id,
            )
    ledger = validate_ledger(experiment)
    if engine_winners is not None:
        status = read_status(
            experiment,
            generation_id=generation_id,
            trial_generation_id=trial_generation_id,
            ledger=ledger,
        )
        # The engine's winners count even where the generation holds no winner file.
        for phase in status["phases"]:
            phase["winner_present"] = phase["phase"] in engine_winners
        winner_snapshots = [
            _winner_snapshot(phase_name, winner, experiment=experiment, ledger=ledger)
            for phase_name, winner in engine_winners.items()
        ]
    else:
        status, winner_views = read_result(experiment, generation_id=generation_id, ledger=ledger)
        winner_snapshots = [
            _winner_snapshot(winner.phase, winner, experiment=experiment, ledger=ledger)
            for winner in winner_views
        ]
    if not generation_claimed:
        status["represented_generation_id"] = None
    snapshot = RunResultSnapshot(
        schema_version=RESULT_SNAPSHOT_SCHEMA_VERSION,
        status=status_snapshot(status),
        winners=winner_snapshots,
        represented_config_fingerprint=(
            _experiment_semantic_fingerprint(
                experiment, phase_count=len(status["result_phase_plan"])
            )
            if status["published_config_matches_current"] is True
            else None
        ),
    )
    return snapshot.model_dump(mode="json")


def mark_result_snapshot_published(
    snapshot: Mapping[str, object],
    *,
    generation_id: str,
    published_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a prepared snapshot to its committed publication and summary availability flags.

    Preserve captured winner facts without rereading shared study state.

    :param Mapping[str, object] snapshot: Snapshot prepared under the experiment lock.
    :param str generation_id: Generation whose last-success pointer just committed.
    :param Mapping[str, Any] published_summary: Already-validated summary committed by
        that pointer.
    :return dict[str, Any]: Validated snapshot recording the committed publication.
    :raises RuntimeError: If the snapshot represents another generation or was
        captured without that generation's summary.
    :raises ValidationError: If ``snapshot`` is not a valid :class:`RunResultSnapshot`.
    """
    parsed = RunResultSnapshot.model_validate(snapshot)
    status = parsed.status
    if status.represented_generation_id != generation_id:
        raise RuntimeError("prepared result snapshot represents a different generation")
    if not status.summary_present:
        raise RuntimeError("prepared result snapshot has no generation summary")
    published_phases = set(_published_phase_trial_refs(published_summary))
    for phase in status.phases:
        # Rebind selected phases to the committed summary; retain carried-phase availability.
        if phase.phase not in published_phases:
            phase.published_study_unavailable = False
        elif not phase.trial_data_available:
            phase.published_study_unavailable = True
    status.published_generation_id = generation_id
    status.is_published = True
    status.publication_integrity = "ok"
    return parsed.model_dump(mode="json")


def finalize_result_snapshot(
    snapshot: Mapping[str, object],
    *,
    confirmed_attempt_ids: Collection[str] = (),
    confirmed_attempt_locations: Mapping[str, tuple[str, int, str]] | None = None,
    confirmed_attempt_states: Mapping[str, TrialStateName] | None = None,
) -> JsonObject:
    """Reconcile captured RUNNING identities without rereading shared state.

    Use the exact recovered terminal state for completed evidence awaiting extraction,
    and the captured prediction for other attempts. Preserve all unrelated frozen counts.

    :param Mapping[str, object] snapshot: Raw snapshot captured under the experiment lock.
    :param Collection[str] confirmed_attempt_ids: Exact RUNNING attempts reconciled by recovery.
    :param Mapping[str, tuple[str, int, str]] | None confirmed_attempt_locations:
        Reconciled attempt ids mapped to phase, trial number, and generation;
        used when the frozen row was captured before its Optuna identity attrs
        were set.
    :param Mapping[str, TrialStateName] | None confirmed_attempt_states: Actual terminal states of reconciled attempts, required for captured pending evidence.
    :return JsonObject: Validated terminal snapshot with truthful trial states.
    :raises RuntimeError: If the cleanup report names more recovered attempts
        than the snapshot counts as RUNNING, for a phase or for the represented
        generation.
    :raises ValidationError: If ``snapshot`` is not a valid ``RunResultSnapshot``.
    """
    parsed = RunResultSnapshot.model_validate(snapshot)
    confirmed = set(confirmed_attempt_ids)
    confirmed_by_location = {
        (phase_name, trial_number): (attempt_id, generation_id)
        for attempt_id, (phase_name, trial_number, generation_id) in (
            (confirmed_attempt_locations or {}).items()
        )
        if attempt_id in confirmed
    }
    for phase in parsed.status.phases:
        if phase.running_attempts is None:
            continue
        recovered: list[tuple[RunningAttemptSnapshot, str | None, TrialStateName]] = []
        for attempt in phase.running_attempts:
            location = confirmed_by_location.get((phase.phase, attempt.trial_number))
            attempt_id = attempt.attempt_id
            if attempt_id is None and location is not None:
                attempt_id = location[0]
            if attempt_id is None or attempt_id not in confirmed:
                continue
            state: TrialStateName
            if attempt.recovers_as == "EVIDENCE_PENDING":
                actual = (confirmed_attempt_states or {}).get(attempt_id)
                if actual is None or actual not in {
                    "COMPLETE",
                    "INFEASIBLE",
                    "FAIL",
                    "INTERRUPTED",
                }:
                    raise RuntimeError(
                        f"cleanup report lacks the terminal state of pending attempt {attempt_id}"
                    )
                state = actual
            else:
                state = attempt.recovers_as
            recovered.append((attempt, None if location is None else location[1], state))
        if not recovered:
            continue
        running = phase.trials.get("RUNNING", 0)
        if len(recovered) > running:
            raise RuntimeError("cleanup report identifies more attempts than the snapshot records")
        phase.trials["RUNNING"] = running - len(recovered)
        for _attempt, _generation_id, state in recovered:
            phase.trials[state] = phase.trials.get(state, 0) + 1
        # Generation counts belong to represented_generation_id: a run that launched a
        # trial, and so can leave one RUNNING, represents its own generation.
        generation_id = parsed.status.represented_generation_id
        generation_recovered = [
            (attempt, state)
            for attempt, recovered_generation_id, state in recovered
            if generation_id is not None
            and (
                attempt.generation_id == generation_id
                or (attempt.generation_id is None and recovered_generation_id == generation_id)
            )
        ]
        if generation_recovered:
            generation_running = phase.generation_trials.get("RUNNING", 0)
            if len(generation_recovered) > generation_running:
                raise RuntimeError(
                    "cleanup report identifies more generation attempts than the snapshot records"
                )
            phase.generation_trials["RUNNING"] = generation_running - len(generation_recovered)
            for _attempt, state in generation_recovered:
                phase.generation_trials[state] = phase.generation_trials.get(state, 0) + 1
        recovered_trial_numbers = {
            attempt.trial_number for attempt, _generation_id, _state in recovered
        }
        phase.running_attempts = [
            attempt
            for attempt in phase.running_attempts
            if attempt.trial_number not in recovered_trial_numbers
        ]
    return dump_json(parsed)


def parse_result_snapshot(status: Mapping[str, object]) -> RunResultSnapshot | None:
    """Parse a terminal status's result snapshot, returning None when absent or malformed.

    :param Mapping[str, object] status: Validated runner terminal status payload.
    :return RunResultSnapshot | None: Strict snapshot model when usable.
    """
    raw = status.get("result_snapshot")
    if raw is None:
        return None
    try:
        return RunResultSnapshot.model_validate(raw)
    except ValidationError:
        return None
