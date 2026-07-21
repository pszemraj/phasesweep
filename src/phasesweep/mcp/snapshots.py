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
from phasesweep.evidence.models import objective_evidence_assurance

NonNegativeInt = Annotated[int, Field(ge=0)]


class _SnapshotModel(BaseModel):
    """Strict base for persisted result snapshot records."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ObjectiveEvidenceSnapshot(_SnapshotModel):
    """Assurance properties of the configured objective extractor."""

    kind: Literal["json_envelope", "log_regex", "wandb"]
    attempt_bound: bool
    checkpoint_bound: bool
    evaluation_policy_bound: bool


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
    """Path-free terminal status view captured by the detached runner."""

    generation_id: str | None = None
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
    """Return the concrete source model for a winner view."""
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
    """Freeze known config shape without reading mutable study or winner state."""
    zero_counts = {state: 0 for state in ("WAITING", "RUNNING", "COMPLETE", "PRUNED", "FAIL")}
    snapshot = RunResultSnapshot(
        status=StatusSnapshot(
            generation_id=None,
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
    cleanup_confirmed: bool,
    generation_id: str | None = None,
    require_trial_data: bool = False,
) -> dict[str, Any]:
    """Capture one experiment's current path-free status and sampled winners.

    :param Experiment experiment: Exact config snapshot the detached runner executed.
    :param bool cleanup_confirmed: Whether all trainer process groups are confirmed gone.
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
    del cleanup_confirmed
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
    winners = read_winners(experiment, generation_id=status["generation_id"])
    snapshot = RunResultSnapshot(
        status=StatusSnapshot(
            generation_id=status["generation_id"],
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
    cleanup_confirmed: bool,
    confirmed_attempt_ids: Collection[str] = (),
) -> dict[str, Any]:
    """Finalize a previously captured snapshot without rereading shared state.

    :param Mapping[str, object] snapshot: Raw snapshot captured under the experiment lock.
    :param bool cleanup_confirmed: Whether remaining trainer process groups are confirmed gone.
    :param Collection[str] confirmed_attempt_ids: Exact RUNNING attempts reconciled to FAIL.
    :return dict[str, Any]: Validated terminal snapshot with truthful trial states.
    """
    del cleanup_confirmed
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
        generation_id = parsed.status.generation_id
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
