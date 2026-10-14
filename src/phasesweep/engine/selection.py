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
from phasesweep.engine.optuna import _draw_positions
from phasesweep.engine.repeats import (
    RankedGroup,
    RepeatAxis,
    RepeatsRecord,
    group_key,
    observed_means,
    ranked_group,
)
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
    # extracted.
    objective_provenance: dict[str, Any]
    # Digest of the trainer environment this trial ran under.
    trainer_env_digest: str
    # Versioned identity of the exact generated input consumed by the trainer.
    trainer_input: dict[str, Any]
    # Advisory verdicts on the population this selection ranked
    # (:mod:`phasesweep.engine.confound`).
    confound: dict[str, Any]
    constraints: dict[str, float] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    # Report-only values the selected trial recorded, by observation name; for
    # a repeats phase, the mean over the members that recorded each name.
    observed: dict[str, float] = field(default_factory=dict)
    # The winning repeat group, or ``None`` for a phase without repeats. With
    # repeats, the scalar fields describe the group's highest-numbered member,
    # ``params`` omits the repeat key, and ``metric`` is the reduced value.
    repeats: RepeatsRecord | None = None


class NoFeasibleTrialError(PhaseSweepError):
    """Raised when no trial in a completed phase satisfies all constraints."""

    default_action: ClassVar[OperatorAction] = OperatorAction.INSPECT_LOGS


@dataclass(frozen=True)
class _GroupRanking:
    """The complete repeat groups of one phase study and what selection left out."""

    groups: list[RankedGroup]
    members: dict[int, list[optuna.trial.FrozenTrial]]
    incomplete_group: list[int]
    incomplete_groups: int


def select_winner(
    study: optuna.Study,
    experiment: Experiment,
    *,
    phase_name: str | None = None,
    repeats: RepeatAxis | None = None,
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

    With ``repeats``, the ranked units are complete repeat groups: a
    configuration ranks only when it has one trial per repeat value and every
    one of them passes rules 1 and 2. Groups rank by the reducer over their
    member metrics, exact ties go to the group with the lowest member trial
    number, and the group's highest-numbered member names the winner.

    Ordering is exact, never approximate. An absolute epsilon (say 1e-12)
    would fold everything within that distance of the optimum into one "tie"
    band, which silently reorders any objective whose natural scale sits at or
    below it: two genuinely different results at 1e-13 and 3e-13 would rank by
    trial number rather than by value. PhaseSweep has no way to know a
    config's meaningful resolution, so it does not guess one.

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
        repeats: The phase's repeat axis
            (:func:`phasesweep.engine.repeats.repeat_axis`), or ``None`` for a
            phase without repeats.

    Returns:
        The winning trial as :class:`SelectedTrial` (number, params, metric,
        constraint readings, persisted evidence-gate results, and the
        population's confound block).

    Raises:
        NoFeasibleTrialError: If no trial in the study is both COMPLETE and
            satisfies every constraint and the metric's bounds, or with
            ``repeats`` if no group is complete. When the metric has bounds,
            the message names them and the best COMPLETE value seen.

    """
    minimize = experiment.metric.goal == "minimize"
    constraints_by_name = {c.name: c for c in experiment.constraints}

    trials = study.get_trials(deepcopy=False)
    ranking = None if repeats is None else _rank_groups(trials, experiment, repeats)
    survivors = (
        [t for t in trials if _eligible(t, experiment)]
        if ranking is None
        else [t for group in ranking.groups for t in ranking.members[group.representative]]
    )

    if repeats is not None and ranking is not None and not ranking.groups:
        raise NoFeasibleTrialError(
            f"No complete repeat group: a configuration ranks only when all "
            f"{len(repeats.values)} trials (one per repeats.values of {repeats.key!r}) are "
            f"COMPLETE, feasible and within bounds; {ranking.incomplete_groups} group(s) were "
            "incomplete. The phase's trials.csv labels each trial's repeat_group and records "
            "each failed trial's reason." + _metric_bounds_note(trials, experiment)
        )
    if not survivors:
        raise NoFeasibleTrialError(
            "No feasible completed trials in phase. The phase's trials.csv records each "
            "failed trial's reason; trainer output is in its trial_* directory."
            + _metric_bounds_note(trials, experiment)
        )

    winning: RankedGroup | None = None
    if ranking is None:
        best_value = (
            min(_trial_value(t) for t in survivors)
            if minimize
            else max(_trial_value(t) for t in survivors)
        )
        # Exact equality, not a tolerance band.
        tied = [t for t in survivors if _trial_value(t) == best_value]
        best = min(tied, key=lambda t: t.number)
    else:
        sign = 1.0 if minimize else -1.0
        winning = min(ranking.groups, key=lambda group: (sign * group.aggregate, group.numbers[0]))
        best = ranking.members[winning.representative][-1]

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
    confound = _assess_population(
        trials,
        survivors,
        best,
        provenance=provenance,
        groups=None if ranking is None else ranking.groups,
        incomplete_group=() if ranking is None else ranking.incomplete_group,
    )
    flagged = _describe_flagged(confound)
    if flagged:
        log.warning(
            "[%s] potential confounds in the %d ranked trials (advisory; recorded in the "
            "winner's confound block): %s.",
            phase_name or study.study_name,
            len(survivors),
            "; ".join(flagged),
        )

    record = (
        None
        if repeats is None or ranking is None or winning is None
        else _repeats_record(ranking.members[winning.representative], winning, repeats, experiment)
    )
    return SelectedTrial(
        trial_number=best.number,
        params={
            name: value
            for name, value in best.params.items()
            if record is None or name != record["key"]
        },
        metric=float(selected_value) if winning is None else winning.aggregate,
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
        observed=(
            _observed_values(best)
            if record is None
            else observed_means([member["observed"] for member in record["members"]])
        ),
        repeats=record,
    )


def _rank_groups(
    trials: Sequence[optuna.trial.FrozenTrial],
    experiment: Experiment,
    axis: RepeatAxis,
) -> _GroupRanking:
    """Group a repeats phase's trials by configuration and keep the complete groups.

    A group is complete when it holds one trial per repeat value and every one
    of them is eligible (:func:`_eligible`). Trials that never recorded the
    repeat param were pruned before their first suggestion and belong to no group.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial in the phase study.
    :param Experiment experiment: Parsed experiment; supplies the eligibility bounds.
    :param RepeatAxis axis: The phase's repeat axis.
    :return _GroupRanking: The complete groups with their member trials, the
        eligible trials left in incomplete groups, and how many groups were incomplete.
    """
    by_key: dict[str, list[optuna.trial.FrozenTrial]] = {}
    draw_positions = _draw_positions(trials)
    for trial in trials:
        if axis.key in trial.params:
            key = group_key(draw_positions[trial.number], trial.params, axis)
            by_key.setdefault(key, []).append(trial)
    groups: list[RankedGroup] = []
    members: dict[int, list[optuna.trial.FrozenTrial]] = {}
    for found in by_key.values():
        ordered = sorted(found, key=lambda t: t.number)
        positions = {axis.index_of(t.params[axis.key]) for t in ordered}
        if (
            len(ordered) == len(axis.values)
            and positions == set(range(len(axis.values)))
            and all(_eligible(t, experiment) for t in ordered)
        ):
            group = ranked_group(
                [(t.number, _trial_value(t)) for t in ordered],
                {name: value for name, value in ordered[-1].params.items() if name != axis.key},
                axis.reduce,
            )
            groups.append(group)
            members[group.representative] = ordered
    ranked = {number for group in groups for number in group.numbers}
    return _GroupRanking(
        groups=groups,
        members=members,
        incomplete_group=[
            t.number for t in trials if t.number not in ranked and _eligible(t, experiment)
        ],
        incomplete_groups=len(by_key) - len(groups),
    )


def _repeats_record(
    members: Sequence[optuna.trial.FrozenTrial],
    group: RankedGroup,
    axis: RepeatAxis,
    experiment: Experiment,
) -> RepeatsRecord:
    """Build the winner's ``repeats`` block from the winning group's members.

    :param Sequence[optuna.trial.FrozenTrial] members: The group's trials, ascending.
    :param RankedGroup group: The winning group.
    :param RepeatAxis axis: The phase's repeat axis.
    :param Experiment experiment: Parsed experiment; names the constraints each member recorded.
    :raises TrialEvidenceMissingError: A member's objective provenance or observed values
        cannot be read.
    :return RepeatsRecord: Key, reducer, spread, and every member's evidence identity.
    """
    return {
        "key": axis.key,
        "reduce": axis.reduce,
        "spread": group.spread,
        "members": [
            {
                "trial_number": t.number,
                "repeat": t.params[axis.key],
                "metric": _trial_value(t),
                "generation_id": str(t.user_attrs[GENERATION_ID_ATTR]),
                "attempt_id": str(t.user_attrs[ATTEMPT_ID_ATTR]),
                "objective_provenance": _trial_objective_provenance(t),
                "trainer_input": dict(t.user_attrs[TRAINER_INPUT_ATTR]),
                "constraints": {
                    c.name: float(t.user_attrs[constraint_attr(c.name)])
                    for c in experiment.constraints
                },
                "observed": _observed_values(t),
            }
            for t in members
        ],
    }


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
