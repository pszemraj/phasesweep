"""Winner selection."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

import optuna

from phasesweep.config import Experiment, check_bounds
from phasesweep.engine.artifacts import _winner_common_payload
from phasesweep.engine.errors import OperatorAction, PhaseSweepError, TrialEvidenceMissingError
from phasesweep.engine.evidence import _selection_candidate_identity, _trial_objective_provenance
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GATES_ATTR,
    GENERATION_ID_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
    Winner,
    constraint_attr,
)

log = logging.getLogger("phasesweep.engine.selection")


@dataclass
class SelectedTrial:
    """The best feasible trial from a completed phase.

    Does not include effective_overrides — the orchestrator adds those.
    """

    trial_number: int
    params: dict[str, Any]
    metric: float
    generation_id: str
    attempt_id: str
    # Frozen evidence provenance recorded when the trial's objective was
    # extracted (review v0.5.17 / finding F).
    objective_provenance: dict[str, Any]
    # Digest of the trainer environment this trial ran under (review v0.5.18 /
    # finding F3).
    trainer_env_digest: str
    # Versioned identity of the exact generated input consumed by the trainer.
    trainer_input: dict[str, Any]
    constraints: dict[str, float] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)


class NoFeasibleTrialError(PhaseSweepError):
    """Raised when no trial in a completed phase satisfies all constraints."""

    default_action: ClassVar[OperatorAction] = OperatorAction.INSPECT_LOGS


def select_winner(
    study: optuna.Study,
    experiment: Experiment,
    *,
    phase_name: str | None = None,
) -> SelectedTrial:
    """Pick the best feasible completed trial from a phase study.

    Rules:
      1. Trial must be COMPLETE, with a finite value, a truthy feasibility
         attr, and nonempty generation/attempt ids
         (:func:`phasesweep.engine.evidence._selection_candidate_identity`).
      2. All constraint values (read from user_attrs) must satisfy bounds.
      3. Among survivors, argmin/argmax on metric.
      4. Ties — exact float equality only — broken by lower trial_number.

    Ordering is exact, never approximate. An earlier absolute epsilon (1e-12)
    folded everything within that distance of the optimum into one "tie" band,
    which silently reordered any objective whose natural scale sits at or below
    1e-12: two genuinely different results at 1e-13 and 3e-13 ranked by trial
    number rather than by value (review v0.5.17 / finding H). PhaseSweep has no
    way to know a config's meaningful resolution, so it does not guess one.

    Args:
        study: Optuna study for the phase whose winner we want.
        experiment: Parsed experiment config. Provides the optimization goal
            (minimize/maximize) and the constraint definitions used to filter
            trials.
        phase_name: Phase label used in log messages; defaults to the study
            name when the caller has no phase context.

    Returns:
        The winning trial as :class:`SelectedTrial` (number, params, metric,
        constraint readings, and persisted evidence-gate results).

    Raises:
        NoFeasibleTrialError: If no trial in the study is both COMPLETE and
            satisfies every constraint.

    """
    minimize = experiment.metric.goal == "minimize"
    constraints_by_name = {c.name: c for c in experiment.constraints}

    survivors: list[optuna.trial.FrozenTrial] = []
    for t in study.get_trials(deepcopy=False):
        if _selection_candidate_identity(t) is None:
            continue
        # Re-verify constraints from user_attrs in case rules changed or stored
        # values are non-finite (defense in depth — review v0.5.2 / item 3).
        ok = True
        for name, c in constraints_by_name.items():
            v = t.user_attrs.get(constraint_attr(name))
            if v is None:
                ok = False
                break
            try:
                v_f = float(v)
            except (TypeError, ValueError):
                ok = False
                break
            if not check_bounds(v_f, min_value=c.min, max_value=c.max):
                ok = False
                break
        if not ok:
            continue
        survivors.append(t)

    if not survivors:
        raise NoFeasibleTrialError(
            "No feasible completed trials in phase. "
            "Check stdout/stderr logs in the phase's trial_* directories."
        )

    best_value = (
        min(_trial_value(t) for t in survivors)
        if minimize
        else max(_trial_value(t) for t in survivors)
    )
    # Exact equality, not a tolerance band (review v0.5.17 / finding H).
    tied = [t for t in survivors if _trial_value(t) == best_value]
    best = min(tied, key=lambda t: t.number)

    constraint_vals = {
        name: float(best.user_attrs[constraint_attr(name)]) for name in constraints_by_name
    }
    selected_value = best.value
    assert selected_value is not None  # same invariant
    raw_gates = best.user_attrs.get(GATES_ATTR)
    gates: list[dict[str, Any]] = []
    if raw_gates is not None:
        if not isinstance(raw_gates, str) or not raw_gates:
            raise TrialEvidenceMissingError(
                f"Winning trial {best.number} has malformed {GATES_ATTR!r} evidence."
            )
        try:
            parsed_gates = json.loads(raw_gates)
        except json.JSONDecodeError as exc:
            raise TrialEvidenceMissingError(
                f"Winning trial {best.number} has corrupt {GATES_ATTR!r} JSON evidence."
            ) from exc
        if not isinstance(parsed_gates, list) or any(
            not isinstance(item, dict) or type(item.get("passed")) is not bool
            for item in parsed_gates
        ):
            raise TrialEvidenceMissingError(
                f"Winning trial {best.number} has malformed {GATES_ATTR!r} evidence."
            )
        gates = parsed_gates

    return SelectedTrial(
        trial_number=best.number,
        params=dict(best.params),
        metric=float(selected_value),
        constraints=constraint_vals,
        gates=gates,
        generation_id=str(best.user_attrs[GENERATION_ID_ATTR]),
        attempt_id=str(best.user_attrs[ATTEMPT_ID_ATTR]),
        objective_provenance=_trial_objective_provenance(best),
        # Allocation writes both before the trainer can run, so every
        # completed trial carries them.
        trainer_env_digest=str(best.user_attrs[TRAINER_ENV_DIGEST_ATTR]),
        trainer_input=dict(best.user_attrs[TRAINER_INPUT_ATTR]),
    )


def _trial_value(trial: optuna.trial.FrozenTrial) -> float:
    """Return the non-None metric value for a known survivor trial.

    :param optuna.trial.FrozenTrial trial: Completed feasible trial already filtered by
        :func:`select_winner`.
    :return float: Scalar objective value for the trial.
    """
    value = trial.value
    assert value is not None  # survivor invariant from select_winner
    return value


def _winner_summary_item(name: str, winner: Winner) -> dict[str, Any]:
    """Return the compact winner payload used in run summaries.

    :param str name: Phase label for the winner.
    :param Winner winner: Winner to serialize into a summary item.
    :return dict[str, Any]: Compact summary payload for the winner.
    """
    payload = {
        "name": name,
        "metric": winner.metric,
        **_winner_common_payload(winner),
    }
    return payload
