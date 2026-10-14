"""Winner selection."""

from __future__ import annotations

import contextlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import optuna

from phasesweep.config import Experiment, check_bounds
from phasesweep.engine.artifacts import _winner_common_payload
from phasesweep.engine.confound import _assess_population, _describe_flagged
from phasesweep.engine.errors import OperatorAction, PhaseSweepError, TrialEvidenceMissingError
from phasesweep.engine.evidence import _selection_candidate_identity, _trial_objective_provenance
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GATES_ATTR,
    GENERATION_ID_ATTR,
    OBSERVED_PREFIX,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
    Winner,
    _validate_observed_block,
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
    # Advisory verdicts on the population this selection ranked
    # (:mod:`phasesweep.engine.confound`).
    confound: dict[str, Any]
    constraints: dict[str, float] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    # Report-only values the selected trial recorded, by observation name.
    observed: dict[str, float] = field(default_factory=dict)


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
      2. The metric value must lie inside the metric's ``min``/``max`` bounds,
         and all constraint values (read from user_attrs) must satisfy theirs.
      3. Among survivors, argmin/argmax on metric.
      4. Ties — exact float equality only — broken by lower trial_number.

    Ordering is exact, never approximate. An earlier absolute epsilon (1e-12)
    folded everything within that distance of the optimum into one "tie" band,
    which silently reordered any objective whose natural scale sits at or below
    1e-12: two genuinely different results at 1e-13 and 3e-13 ranked by trial
    number rather than by value (review v0.5.17 / finding H). PhaseSweep has no
    way to know a config's meaningful resolution, so it does not guess one.

    The ranked population is also classified for confounds
    (:func:`phasesweep.engine.confound._assess_population`) and any flagged
    check is logged once per selection. The verdicts are advisory: they never
    change which trial wins or whether selection succeeds.

    Args:
        study: Optuna study for the phase whose winner we want.
        experiment: Parsed experiment config. Provides the optimization goal
            (minimize/maximize) and the constraint definitions used to filter
            trials.
        phase_name: Phase label used in log messages; defaults to the study
            name when the caller has no phase context.

    Returns:
        The winning trial as :class:`SelectedTrial` (number, params, metric,
        constraint readings, persisted evidence-gate results, and the
        population's confound block).

    Raises:
        NoFeasibleTrialError: If no trial in the study is both COMPLETE and
            satisfies every constraint and the metric's bounds. When the metric
            has bounds, the message names them and the best COMPLETE value seen.

    """
    minimize = experiment.metric.goal == "minimize"
    constraints_by_name = {c.name: c for c in experiment.constraints}

    trials = study.get_trials(deepcopy=False)
    survivors = [t for t in trials if _eligible(t, experiment)]

    if not survivors:
        raise NoFeasibleTrialError(
            "No feasible completed trials in phase. The phase's trials.csv records each "
            "failed trial's reason; trainer output is in its trial_* directory."
            + _metric_bounds_note(trials, experiment)
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

    objective_provenance = _trial_objective_provenance(best)
    provenance = {best.number: objective_provenance}
    # Only the winner's record backs the published result. The others feed the
    # advisory evaluation_point check, which reports one it cannot read
    # instead of failing selection over a trial that lost.
    for t in survivors:
        if t is not best:
            with contextlib.suppress(TrialEvidenceMissingError):
                provenance[t.number] = _trial_objective_provenance(t)
    confound = _assess_population(trials, survivors, best, provenance=provenance)
    flagged = _describe_flagged(confound)
    if flagged:
        log.warning(
            "[%s] potential confounds in the %d ranked trials (advisory; recorded in the "
            "winner's confound block): %s.",
            phase_name or study.study_name,
            len(survivors),
            "; ".join(flagged),
        )

    return SelectedTrial(
        trial_number=best.number,
        params=dict(best.params),
        metric=float(selected_value),
        constraints=constraint_vals,
        gates=gates,
        generation_id=str(best.user_attrs[GENERATION_ID_ATTR]),
        attempt_id=str(best.user_attrs[ATTEMPT_ID_ATTR]),
        objective_provenance=objective_provenance,
        # Allocation writes both before the trainer can run, so every
        # completed trial carries them.
        trainer_env_digest=str(best.user_attrs[TRAINER_ENV_DIGEST_ATTR]),
        trainer_input=dict(best.user_attrs[TRAINER_INPUT_ATTR]),
        confound=confound,
        observed=_observed_values(best),
    )


def _observed_values(trial: optuna.trial.FrozenTrial) -> dict[str, float]:
    """Read the report-only values one trial recorded, from its own attributes.

    The values come from the trial, not the current config, so a later edit to
    ``observe`` cannot change what an already-run trial reports.

    :param optuna.trial.FrozenTrial trial: The selected trial.
    :raises TrialEvidenceMissingError: A recorded value is not a finite number.
    :return dict[str, float]: Observed value by observation name, sorted by name.
    """
    recorded = {
        key.removeprefix(OBSERVED_PREFIX): value
        for key, value in sorted(trial.user_attrs.items())
        if key.startswith(OBSERVED_PREFIX)
    }
    try:
        return _validate_observed_block(recorded)
    except ValueError as exc:
        raise TrialEvidenceMissingError(
            f"Winning trial {trial.number} has a malformed {OBSERVED_PREFIX!r} value: {exc}."
        ) from exc


def _eligible(trial: optuna.trial.FrozenTrial, experiment: Experiment) -> bool:
    """Return whether a trial may be ranked under the current bounds.

    The trial must pass :func:`_selection_candidate_identity`, hold a metric
    value inside the metric's viability bounds, and hold every constraint
    value inside its bounds. The bounds are re-checked here, from the stored
    values, in case the rules changed or a stored value is non-finite (defense
    in depth).

    :param optuna.trial.FrozenTrial trial: Persisted trial to classify.
    :param Experiment experiment: Parsed experiment; supplies the metric and constraint bounds.
    :return bool: ``True`` when the trial is a selection candidate.
    """
    if _selection_candidate_identity(trial) is None:
        return False
    metric = experiment.metric
    if not check_bounds(_trial_value(trial), min_value=metric.min, max_value=metric.max):
        return False
    for constraint in experiment.constraints:
        raw = trial.user_attrs.get(constraint_attr(constraint.name))
        if raw is None:
            return False
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return False
        if not check_bounds(value, min_value=constraint.min, max_value=constraint.max):
            return False
    return True


def _metric_bounds_note(trials: Sequence[optuna.trial.FrozenTrial], experiment: Experiment) -> str:
    """Name the metric's viability bounds and the best value seen, for an empty ranking.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial in the phase study.
    :param Experiment experiment: Parsed experiment; supplies the bounds and the goal.
    :return str: A sentence naming each set bound and the best COMPLETE finite value with
        its trial number, or an empty string when the metric has no bounds.
    """
    metric = experiment.metric
    named = [
        f"metric.{label}={value!r}"
        for label, value in (("min", metric.min), ("max", metric.max))
        if value is not None
    ]
    if not named:
        return ""
    seen = [
        t
        for t in trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and math.isfinite(t.value)
    ]
    note = f" Metric bounds: {', '.join(named)}."
    if not seen:
        return note + " No COMPLETE trial recorded a finite metric value."
    sign = 1.0 if metric.goal == "minimize" else -1.0
    best = min(seen, key=lambda t: (sign * _trial_value(t), t.number))
    return note + (
        f" The best value among COMPLETE trials with a finite value is "
        f"{_trial_value(best)!r} (trial {best.number})."
    )


def _trial_value(trial: optuna.trial.FrozenTrial) -> float:
    """Return the non-None metric value for a completed trial known to hold one.

    :param optuna.trial.FrozenTrial trial: COMPLETE trial already screened for a value, as
        :func:`select_winner` does before ranking.
    :return float: Scalar objective value for the trial.
    """
    value = trial.value
    assert value is not None  # callers screen for a recorded value first
    return value
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
