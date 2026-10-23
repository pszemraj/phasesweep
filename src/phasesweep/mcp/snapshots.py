"""Validated, path-free terminal result snapshots for MCP run handles."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import asdict
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from phasesweep.config.models import Experiment, _metric_semantics_payload
from phasesweep.engine.artifacts import read_yaml_or_none
from phasesweep.engine.confound import (
    ConfoundCheck,
    ConfoundVerdict,
    _confound_summary,
    _validate_confound_block,
)
from phasesweep.engine.fingerprints import _experiment_semantic_fingerprint
from phasesweep.engine.optuna import _published_phase_trial_refs
from phasesweep.engine.paths import _generation_record_path
from phasesweep.engine.read import (
    EngineStatus,
    PhaseWinnerView,
    ResultContext,
    read_result,
    read_status,
)
from phasesweep.engine.repeats import _repeats_summary
from phasesweep.engine.state import (
    TRIAL_STATE_NAMES,
    PublicationState,
    RecoveredTrialState,
    Winner,
    WinnerSourceKind,
)
from phasesweep.evidence.models import _ObjectiveEvidenceFields

log = logging.getLogger("phasesweep.mcp.snapshots")

NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
McpPublicationState = PublicationState | Literal["unknown"]
"""Engine publication verdict plus the MCP snapshot-unavailable state."""


class _SnapshotModel(BaseModel):
    """Strict base for persisted result snapshot records."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ObjectiveEvidenceSnapshot(_SnapshotModel, _ObjectiveEvidenceFields):
    """Identity checks enforced by the configured objective extractor.

    False means unenforced. ``*_declared`` marks configured expectations;
    ``*_value_bound`` marks enforced equality with those expectations.
    """


class MetricSnapshot(_SnapshotModel):
    """Optimization metric and objective-evidence assurance."""

    name: str = Field(description="Metric key extracted from trial output.")
    goal: Literal["minimize", "maximize"] = Field(description="Optimization direction.")
    objective_evidence: ObjectiveEvidenceSnapshot = Field(
        description=(
            "False flags mean the identity is not enforced, not that it failed. kind names the "
            "extractor. Only json_envelope, wandb and mlflow bind an evaluation position."
        )
    )


class RunningAttemptSnapshot(_SnapshotModel):
    """Concrete RUNNING row identity captured with a terminal snapshot."""

    trial_number: NonNegativeInt
    generation_id: str | None = None
    attempt_id: str | None = None
    recovers_as: RecoveredTrialState
    """The state stale recovery records for this row, read from its outcome record at capture."""


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


class WinnerSourceSnapshot(_SnapshotModel):
    """Concrete source trial for an exposed phase winner."""

    kind: WinnerSourceKind
    phase: str
    trial_number: int
    generation_id: str
    attempt_id: str


_VERDICT_DESCRIPTION = (
    "ok: checked, nothing found. heterogeneous: a potential confound in the population "
    "this winner was chosen from; report it before relying on the winner. n/a: not "
    "checked; never read it as clean."
)


class EvaluationPointSummary(_SnapshotModel):
    """Whether every ranked objective was measured at one training position.

    Positions are on the experiment's configured evaluation axis -- the
    progress axis its sweep holds fixed, such as optimizer step or tokens.
    """

    verdict: ConfoundVerdict = Field(
        description=(
            f"{_VERDICT_DESCRIPTION} Always n/a for json and log_regex objectives, which "
            "record no evaluation position; only json_envelope, wandb and mlflow bind one."
        )
    )
    distinct_values: NonNegativeInt | None = Field(
        description=(
            "Distinct positions on the configured evaluation axis the ranked objectives "
            "were measured at; null when not checked."
        )
    )
    winner_value: NonNegativeInt | NonNegativeFloat | None = Field(
        description=(
            "Position on the configured evaluation axis the winning objective was "
            "measured at; null when not checked."
        )
    )


class SurvivorshipSummary(_SnapshotModel):
    """How many finished trials were ranked versus excluded from ranking."""

    verdict: ConfoundVerdict = Field(
        description=(
            f"{_VERDICT_DESCRIPTION} Heterogeneous when failed trials are at least as "
            "numerous as ranked ones; infeasible and incomplete_group trials never flag."
        )
    )
    ranked: NonNegativeInt
    infeasible: NonNegativeInt
    incomplete_group: NonNegativeInt = Field(
        description=(
            "Feasible trials of a repeats phase left unranked because another member of "
            "their repeat group failed, was infeasible, or never ran; 0 without repeats."
        )
    )
    failed: NonNegativeInt


class TieSummary(_SnapshotModel):
    """Whether the winner stands clear of ranked units that used other params.

    A unit is a trial, or a complete repeat group in a repeats phase.
    """

    verdict: ConfoundVerdict = Field(description=_VERDICT_DESCRIPTION)
    tied_trials: NonNegativeInt | None = Field(
        description=(
            "Units tying the winner's metric exactly with other params; null when not checked."
        )
    )
    within_noise: NonNegativeInt | None = Field(
        description=(
            "Repeat groups whose aggregate is within two standard errors of the winning "
            "group's, exact ties included: an advisory heuristic, not a significance test. "
            "Null without repeats or when not checked."
        )
    )


class ConfoundSummary(_SnapshotModel):
    """Advisory verdicts on the population the winner was selected from.

    Frozen when the winner was selected. The evidence behind each verdict
    stays in the operator-facing winner file; this summary carries only
    verdicts and counts.
    """

    flagged: list[ConfoundCheck] = Field(
        description="Checks whose verdict is heterogeneous, in report order."
    )
    evaluation_point: EvaluationPointSummary
    survivorship: SurvivorshipSummary
    tie: TieSummary


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


def _winner_snapshot(
    phase: str,
    winner: PhaseWinnerView | Winner,
) -> WinnerSnapshot:
    """Freeze either engine winner representation into one validated snapshot.

    :param str phase: Phase under which the winner is exposed.
    :param PhaseWinnerView | Winner winner: Winner representation to serialize.
    :raises RuntimeError: A ``PhaseWinnerView`` carries no confound verdicts, or either
        representation carries no source provenance.
    :raises ValueError: A stored ``Winner``'s confound block fails validation.
    :return WinnerSnapshot: Validated path-free winner snapshot.
    """
    if isinstance(winner, Winner):
        gates_passed = (
            all(bool(gate.get("passed")) for gate in winner.gates) if winner.gates else None
        )
        incomplete = bool(winner.completion.get("incomplete", False))
        confound = _confound_summary(
            _validate_confound_block(winner.confound, winner_trial=winner.trial_number),
            winner_trial=winner.trial_number,
        )
        repeats = None if winner.repeats is None else _repeats_summary(winner.repeats)
    else:
        gates_passed = winner.gates_passed
        incomplete = winner.incomplete
        if winner.confound is None:
            raise RuntimeError("Current-format winner is missing confound verdicts.")
        confound = winner.confound
        repeats = winner.repeats
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
    )


class RunResultSnapshot(_SnapshotModel):
    """Terminal status and winners frozen for one MCP run id."""

    status: StatusSnapshot
    winners: list[WinnerSnapshot]
    represented_config_fingerprint: str | None = None
    """Represented config's semantic fingerprint, captured using the run's working directory."""

    def status_payload(self) -> EngineStatus:
        """Return the stored status in the engine reader's path-free shape.

        :return EngineStatus: Status mapping accepted by the MCP payload builder.
        """
        status = self.status
        return {
            "current_generation_id": status.current_generation_id,
            "published_generation_id": status.published_generation_id,
            "represented_generation_id": status.represented_generation_id,
            "is_published": status.is_published,
            "publication_integrity": status.publication_integrity,
            "metric": status.metric.model_dump(mode="json"),
            "phases": [
                phase.model_dump(mode="json", exclude={"running_attempts"})
                for phase in status.phases
            ],
            "summary_present": status.summary_present,
            "result_context": status.result_context,
            "published_config_matches_current": status.published_config_matches_current,
            "result_phase_plan": list(status.result_phase_plan),
        }


def capture_pre_generation_result_snapshot(experiment: Experiment) -> RunResultSnapshot:
    """Freeze config shape with zero trial counts, no winners and no generation ID.

    Reads no mutable study or winner state; publication and trial availability are unknown.

    :param Experiment experiment: Exact config snapshot the detached runner
        attempted to execute.
    :return RunResultSnapshot: Validated placeholder result snapshot.
    """
    zero_counts: dict[str, int] = {state: 0 for state in TRIAL_STATE_NAMES}
    return RunResultSnapshot(
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
    if engine_winners is not None:
        status = read_status(
            experiment, generation_id=generation_id, trial_generation_id=trial_generation_id
        )
        # The engine's winners count even where the generation holds no winner file.
        for phase in status["phases"]:
            phase["winner_present"] = phase["phase"] in engine_winners
        winner_snapshots = [
            _winner_snapshot(phase_name, winner) for phase_name, winner in engine_winners.items()
        ]
    else:
        status, winner_views = read_result(experiment, generation_id=generation_id)
        winner_snapshots = [_winner_snapshot(winner.phase, winner) for winner in winner_views]
    if not generation_claimed:
        status["represented_generation_id"] = None
    snapshot = RunResultSnapshot(
        status=StatusSnapshot.model_validate(
            {key: value for key, value in status.items() if key in StatusSnapshot.model_fields}
        ),
        winners=winner_snapshots,
        represented_config_fingerprint=(
            _experiment_semantic_fingerprint(experiment)
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
) -> dict[str, Any]:
    """Reconcile captured RUNNING identities without rereading shared state.

    Recovered attempts become their captured ``INTERRUPTED`` or ``FAIL`` state. Phases
    with unavailable trial data and no RUNNING identities remain unchanged.

    :param Mapping[str, object] snapshot: Raw snapshot captured under the experiment lock.
    :param Collection[str] confirmed_attempt_ids: Exact RUNNING attempts reconciled by recovery.
    :param Mapping[str, tuple[str, int, str]] | None confirmed_attempt_locations:
        Reconciled attempt ids mapped to phase, trial number, and generation;
        used when the frozen row was captured before its Optuna identity attrs
        were set.
    :return dict[str, Any]: Validated terminal snapshot with truthful trial states.
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
        recovered: list[tuple[RunningAttemptSnapshot, str | None]] = []
        for attempt in phase.running_attempts:
            location = confirmed_by_location.get((phase.phase, attempt.trial_number))
            if attempt.attempt_id is not None and attempt.attempt_id in confirmed:
                recovered.append((attempt, None if location is None else location[1]))
            elif attempt.attempt_id is None and location is not None:
                recovered.append((attempt, location[1]))
        if not recovered:
            continue
        running = phase.trials.get("RUNNING", 0)
        if len(recovered) > running:
            raise RuntimeError("cleanup report identifies more attempts than the snapshot records")
        phase.trials["RUNNING"] = running - len(recovered)
        for attempt, _generation_id in recovered:
            phase.trials[attempt.recovers_as] = phase.trials.get(attempt.recovers_as, 0) + 1
        # Generation counts belong to represented_generation_id: a run that launched a
        # trial, and so can leave one RUNNING, represents its own generation.
        generation_id = parsed.status.represented_generation_id
        generation_recovered = [
            attempt
            for attempt, recovered_generation_id in recovered
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
            for attempt in generation_recovered:
                phase.generation_trials[attempt.recovers_as] = (
                    phase.generation_trials.get(attempt.recovers_as, 0) + 1
                )
        recovered_trial_numbers = {attempt.trial_number for attempt, _generation_id in recovered}
        phase.running_attempts = [
            attempt
            for attempt in phase.running_attempts
            if attempt.trial_number not in recovered_trial_numbers
        ]
    return parsed.model_dump(mode="json")


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
