"""Validated, path-free terminal result snapshots for MCP run handles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from phasesweep.config import Experiment
from phasesweep.engine import PhaseWinnerView, read_status, read_winners

NonNegativeInt = Annotated[int, Field(ge=0)]


class _SnapshotModel(BaseModel):
    """Strict base for persisted result snapshot records."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MetricSnapshot(_SnapshotModel):
    """Optimization metric stored with a terminal run snapshot."""

    name: str
    goal: Literal["minimize", "maximize"]


class PhaseStatusSnapshot(_SnapshotModel):
    """One phase's terminal trial counts and winner presence."""

    phase: str
    trials: dict[str, NonNegativeInt]
    running: NonNegativeInt
    n_trials: NonNegativeInt
    completed: NonNegativeInt
    winner_present: bool
    trial_data_available: bool


class StatusSnapshot(_SnapshotModel):
    """Path-free terminal status view captured by the detached runner."""

    generation_id: str | None = None
    metric: MetricSnapshot
    phases: list[PhaseStatusSnapshot]
    summary_present: bool


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


class RunResultSnapshot(_SnapshotModel):
    """Terminal status and winners frozen for one MCP run id."""

    status: StatusSnapshot
    winners: list[WinnerSnapshot]

    def status_payload(self) -> dict[str, Any]:
        """Return the stored status in the engine reader's path-free shape.

        :return dict[str, Any]: Status mapping accepted by the MCP payload builder.
        """
        return self.status.model_dump(mode="json")

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
            )
            for winner in self.winners
        ]


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
    status = read_status(experiment)
    if generation_id is not None and status["generation_id"] != generation_id:
        raise RuntimeError(
            "current generation marker does not match the locked engine generation; "
            "refusing to freeze stale result artifacts"
        )
    if require_trial_data:
        unavailable = [
            phase["phase"] for phase in status["phases"] if not phase["trial_data_available"]
        ]
        if unavailable:
            raise RuntimeError(
                "terminal trial data is unavailable for phase(s) "
                f"{', '.join(unavailable)}; refusing to freeze ambiguous counts"
            )
    if cleanup_confirmed:
        # A signal can escape Optuna before it changes its RUNNING row to FAIL.
        # Confirmed process cleanup means those trials are terminal in reality;
        # the normal stale reaper will persist the same transition before a
        # later resume. Freeze that truthful terminal view for this run now.
        for phase in status["phases"]:
            running = phase["trials"].pop("RUNNING", 0)
            if running:
                phase["trials"]["FAIL"] = phase["trials"].get("FAIL", 0) + running
                phase["running"] = 0
    winners = read_winners(experiment)
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
            )
            for winner in winners
        ],
    )
    return snapshot.model_dump(mode="json")


def finalize_result_snapshot(
    snapshot: Mapping[str, object],
    *,
    cleanup_confirmed: bool,
) -> dict[str, Any]:
    """Finalize a previously captured snapshot without rereading shared state.

    :param Mapping[str, object] snapshot: Raw snapshot captured under the experiment lock.
    :param bool cleanup_confirmed: Whether remaining trainer process groups are confirmed gone.
    :return dict[str, Any]: Validated terminal snapshot with truthful trial states.
    """
    parsed = RunResultSnapshot.model_validate(snapshot)
    if not cleanup_confirmed:
        return parsed.model_dump(mode="json")
    for phase in parsed.status.phases:
        running = phase.trials.pop("RUNNING", 0)
        if running:
            phase.trials["FAIL"] = phase.trials.get("FAIL", 0) + running
            phase.running = 0
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
