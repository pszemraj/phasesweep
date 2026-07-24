"""Validated, path-free terminal result snapshots for MCP run handles."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from phasesweep.config import Experiment
from phasesweep.engine import PhaseWinnerView, read_status, read_winners
from phasesweep.engine.optuna import _load_existing_phase_study
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GENERATION_ID_ATTR,
    WinnerSource,
    _generation_record_path,
)
from phasesweep.evidence.models import _ObjectiveEvidenceFields, objective_evidence_assurance

NonNegativeInt = Annotated[int, Field(ge=0)]


class _SnapshotModel(BaseModel):
    """Strict base for persisted result snapshot records."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ObjectiveEvidenceSnapshot(_SnapshotModel, _ObjectiveEvidenceFields):
    """Assurance properties of the configured objective extractor.

    See :func:`phasesweep.evidence.models.objective_evidence_assurance` for
    exactly what each flag means and which runtime checks back it.
    """


class MetricSnapshot(_SnapshotModel):
    """Optimization metric stored with a terminal run snapshot."""

    name: str
    goal: Literal["minimize", "maximize"]
    objective_evidence: ObjectiveEvidenceSnapshot


class RunningAttemptSnapshot(_SnapshotModel):
    """Concrete RUNNING row identity captured with a terminal snapshot."""

    trial_number: NonNegativeInt
    generation_id: str | None = None
    attempt_id: str | None = None


class PhaseStatusSnapshot(_SnapshotModel):
    """One phase's terminal trial counts and winner presence."""

    phase: str
    trials: dict[str, NonNegativeInt]
    running: NonNegativeInt
    n_trials: NonNegativeInt
    completed: NonNegativeInt
    generation_trials: dict[str, NonNegativeInt]
    winner_present: bool
    trial_data_available: bool
    running_attempts: list[RunningAttemptSnapshot] = Field(default_factory=list)


class StatusSnapshot(_SnapshotModel):
    """Path-free terminal status view captured by the detached runner.

    ``current_generation_id`` and ``published_generation_id`` are always the
    actual mutable/last-success pointers and may differ from each other and
    from ``represented_generation_id`` (e.g. a failed rerun, or a pinned
    snapshot of a generation whose own publication failed).
    ``represented_generation_id`` is the generation whose winner/summary facts
    this snapshot shows, and ``is_published`` says whether that generation is
    the actual published one -- a failed-publication generation's snapshot
    correctly reports ``is_published: False`` while remaining fully readable.
    See :func:`phasesweep.engine.read.read_status`.
    """

    current_generation_id: str | None = None
    published_generation_id: str | None = None
    represented_generation_id: str | None = None
    is_published: bool = False
    metric: MetricSnapshot
    phases: list[PhaseStatusSnapshot]
    summary_present: bool


class WinnerSourceSnapshot(_SnapshotModel):
    """Concrete source trial for an exposed phase winner."""

    kind: Literal["phase_trial", "promotion_baseline", "suite_baseline"]
    phase: str
    trial_number: int
    generation_id: str | None = None
    attempt_id: str | None = None
    study: str | None = None


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
    promotion: dict[str, Any] | None = None


def _winner_source_snapshot(winner: PhaseWinnerView) -> WinnerSourceSnapshot:
    """Return the concrete source model for a winner view.

    Falls back to a ``"phase_trial"`` source built from the winner's own
    phase/trial/generation/attempt identity when ``winner.source`` is unset
    (winners persisted before source tracking was added have no ``source``).

    :param PhaseWinnerView winner: Winner view whose source should be captured.
    :return WinnerSourceSnapshot: Concrete, validated source snapshot.
    """
    source = winner.source or WinnerSource(
        kind="phase_trial",
        phase=winner.phase,
        trial_number=winner.trial_number,
        generation_id=winner.generation_id,
        attempt_id=winner.attempt_id,
    )
    return WinnerSourceSnapshot(
        kind=source.kind,
        phase=source.phase,
        trial_number=source.trial_number,
        generation_id=source.generation_id,
        attempt_id=source.attempt_id,
        study=source.study,
    )


class RunResultSnapshot(_SnapshotModel):
    """Terminal status and winners frozen for one MCP run id."""

    status: StatusSnapshot
    winners: list[WinnerSnapshot]

    def status_payload(self) -> dict[str, Any]:
        """Return the stored status in the engine reader's path-free shape.

        :return dict[str, Any]: Status mapping accepted by the MCP payload builder.
        """
        payload = self.status.model_dump(mode="json")
        for phase in payload["phases"]:
            phase.pop("running_attempts", None)
        return payload

    def winner_views(self) -> list[PhaseWinnerView]:
        """Return stored winners as the engine view consumed by MCP redaction.

        :return list[PhaseWinnerView]: Winner views with intentionally empty effective overrides.
        """
        return [
            PhaseWinnerView(
                phase=winner.phase,
                trial_number=winner.trial_number,
                metric=winner.metric,
                params=winner.params,
                effective_overrides={},
                gates_passed=winner.gates_passed,
                incomplete=winner.incomplete,
                generation_id=winner.generation_id,
                attempt_id=winner.attempt_id,
                source=WinnerSource(
                    kind=winner.source.kind,
                    phase=winner.source.phase,
                    trial_number=winner.source.trial_number,
                    generation_id=winner.source.generation_id,
                    attempt_id=winner.source.attempt_id,
                    study=winner.source.study,
                ),
                promotion=winner.promotion,
            )
            for winner in self.winners
        ]


def capture_pre_generation_result_snapshot(experiment: Experiment) -> dict[str, Any]:
    """Freeze known config shape without reading mutable study or winner state.

    Used when the engine crashed before ever claiming a generation, so no
    lock-protected read of shared Optuna study/winner state is safe. Builds a
    snapshot purely from the static experiment config: every phase's declared
    ``n_trials`` with all trial-state counts at zero, no winners, and no
    generation id.

    :param Experiment experiment: Exact config snapshot the detached runner
        attempted to execute.
    :return dict[str, Any]: JSON-serializable placeholder result snapshot.
    """
    zero_counts = {state: 0 for state in ("WAITING", "RUNNING", "COMPLETE", "PRUNED", "FAIL")}
    snapshot = RunResultSnapshot(
        status=StatusSnapshot(
            current_generation_id=None,
            published_generation_id=None,
            represented_generation_id=None,
            is_published=False,
            metric=MetricSnapshot(
                name=experiment.metric.name,
                goal=experiment.metric.goal,
                objective_evidence=ObjectiveEvidenceSnapshot.model_validate(
                    objective_evidence_assurance(experiment.metric.extractor)
                ),
            ),
            phases=[
                PhaseStatusSnapshot(
                    phase=phase.name,
                    trials=dict(zero_counts),
                    running=0,
                    n_trials=phase.n_trials,
                    completed=0,
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
    return snapshot.model_dump(mode="json")


def capture_result_snapshot(
    experiment: Experiment,
    *,
    generation_id: str | None = None,
    require_trial_data: bool = False,
) -> dict[str, Any]:
    """Capture one experiment's current path-free status and sampled winners.

    When ``generation_id`` is given (the detached runner always pins its own
    generation), the captured status's ``represented_generation_id`` equals
    it and the frozen winners are read from that exact generation, regardless
    of whether it ever became the actual last-success pointer -- so a
    failed-publication generation's terminal snapshot still reports its own
    (unpublished) partial results, correctly flagged ``is_published: False``.

    :param Experiment experiment: Exact config snapshot the detached runner executed.
    :param str | None generation_id: Engine generation known to own the experiment lock.
    :param bool require_trial_data: Refuse ambiguous storage reads when the engine succeeded.
    :return dict[str, Any]: JSON-serializable terminal result snapshot.
    """
    if generation_id is not None:
        try:
            lifecycle = yaml.safe_load(
                _generation_record_path(experiment, generation_id).read_text()
            )
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(
                "requested generation has no readable immutable lifecycle record"
            ) from exc
        if not isinstance(lifecycle, Mapping) or lifecycle.get("generation_id") != generation_id:
            raise RuntimeError("requested generation lifecycle record does not match its identity")
    status = read_status(experiment, generation_id=generation_id)
    if require_trial_data:
        unavailable = [
            phase["phase"] for phase in status["phases"] if not phase["trial_data_available"]
        ]
        if unavailable:
            raise RuntimeError(
                "terminal trial data is unavailable for phase(s) "
                f"{', '.join(unavailable)}; refusing to freeze ambiguous counts"
            )
    phases_by_name = {phase.name: phase for phase in experiment.phases}
    for phase_status in status["phases"]:
        running_attempts: list[dict[str, Any]] = []
        if phase_status["trial_data_available"]:
            phase = phases_by_name[phase_status["phase"]]
            study = _load_existing_phase_study(experiment, phase)
            if study is not None:
                for trial in study.get_trials(deepcopy=False):
                    if trial.state.name != "RUNNING":
                        continue
                    generation = trial.user_attrs.get(GENERATION_ID_ATTR)
                    attempt = trial.user_attrs.get(ATTEMPT_ID_ATTR)
                    running_attempts.append(
                        {
                            "trial_number": trial.number,
                            "generation_id": generation if isinstance(generation, str) else None,
                            "attempt_id": attempt if isinstance(attempt, str) else None,
                        }
                    )
        phase_status["running_attempts"] = running_attempts
    # Winners must be scoped to the generation this snapshot *represents*, not
    # the (possibly different) true current pointer: a pinned capture wants
    # exactly its own generation's winners even when a newer generation has
    # since become current (review v0.5.15 / blocker 3).
    winners = read_winners(experiment, generation_id=status["represented_generation_id"])
    snapshot = RunResultSnapshot(
        status=StatusSnapshot(
            current_generation_id=status["current_generation_id"],
            published_generation_id=status["published_generation_id"],
            represented_generation_id=status["represented_generation_id"],
            is_published=status["is_published"],
            metric=status["metric"],
            phases=status["phases"],
            summary_present=status["summary_present"],
        ),
        winners=[
            WinnerSnapshot(
                phase=winner.phase,
                trial_number=winner.trial_number,
                metric=winner.metric,
                params=winner.params,
                gates_passed=winner.gates_passed,
                incomplete=winner.incomplete,
                generation_id=winner.generation_id,
                attempt_id=winner.attempt_id,
                source=_winner_source_snapshot(winner),
                promotion=winner.promotion,
            )
            for winner in winners
        ],
    )
    return snapshot.model_dump(mode="json")


def finalize_result_snapshot(
    snapshot: Mapping[str, object],
    *,
    confirmed_attempt_ids: Collection[str] = (),
) -> dict[str, Any]:
    """Finalize a previously captured snapshot without rereading shared state.

    :param Mapping[str, object] snapshot: Raw snapshot captured under the experiment lock.
    :param Collection[str] confirmed_attempt_ids: Exact RUNNING attempts reconciled to FAIL.
    :return dict[str, Any]: Validated terminal snapshot with truthful trial states.
    """
    parsed = RunResultSnapshot.model_validate(snapshot)
    confirmed = set(confirmed_attempt_ids)
    for phase in parsed.status.phases:
        recovered = [
            attempt
            for attempt in phase.running_attempts
            if attempt.attempt_id is not None and attempt.attempt_id in confirmed
        ]
        if not recovered:
            continue
        running = phase.trials.get("RUNNING", 0)
        if len(recovered) > running:
            raise RuntimeError("cleanup report identifies more attempts than the snapshot records")
        phase.trials["RUNNING"] = running - len(recovered)
        phase.trials["FAIL"] = phase.trials.get("FAIL", 0) + len(recovered)
        phase.running = phase.trials["RUNNING"]
        # generation_trials was captured scoped to represented_generation_id
        # (the pinned id for a run snapshot), not current_generation_id, which
        # is now always the actual mutable pointer and may be unrelated to --
        # or absent for -- this snapshot's own generation (review v0.5.15 /
        # blocker 3).
        generation_id = parsed.status.represented_generation_id
        generation_recovered = [
            attempt
            for attempt in recovered
            if generation_id is not None and attempt.generation_id == generation_id
        ]
        if generation_recovered:
            generation_running = phase.generation_trials.get("RUNNING", 0)
            if len(generation_recovered) > generation_running:
                raise RuntimeError(
                    "cleanup report identifies more generation attempts than the snapshot records"
                )
            phase.generation_trials["RUNNING"] = generation_running - len(generation_recovered)
            phase.generation_trials["FAIL"] = phase.generation_trials.get("FAIL", 0) + len(
                generation_recovered
            )
        recovered_ids = {attempt.attempt_id for attempt in recovered}
        phase.running_attempts = [
            attempt for attempt in phase.running_attempts if attempt.attempt_id not in recovered_ids
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
