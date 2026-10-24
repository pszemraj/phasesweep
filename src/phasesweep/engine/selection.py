"""Winner selection."""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import math
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import optuna

from phasesweep.config.common import check_bounds
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import grid_search_space, phase_draws, sampler_search_space
from phasesweep.engine.artifacts import _winner_common_payload
from phasesweep.engine.confound import (
    _assess_population,
    _describe_flagged,
    _evaluation_point_remedy,
)
from phasesweep.engine.errors import TrialEvidenceMissingError
from phasesweep.engine.evidence import _selection_candidate_identity, _trial_objective_provenance
from phasesweep.engine.optuna import _draw_positions, _trial_count_state
from phasesweep.engine.repeats import (
    RankedGroup,
    RepeatAxis,
    RepeatsRecord,
    combined_standard_error,
    group_key,
    observed_means,
    ranked_group,
    within_noise,
)
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GATES_ATTR,
    GENERATION_ID_ATTR,
    OBSERVED_PREFIX,
    RERUN_OF_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
    Winner,
    _trial_interrupted,
    _validate_observed_block,
    constraint_attr,
)
from phasesweep.errors import OperatorAction, PhaseSweepError

log = logging.getLogger("phasesweep.engine.selection")

# Grid cells without a successful trial that one advisory spells out before it counts the rest.
_MAX_LISTED_CELLS = 10


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
    """Raised when no trial in a completed phase satisfies all constraints.

    Its subclasses name why the phase holds no winner.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.INSPECT_LOGS


class PhaseTrialsFailedError(NoFeasibleTrialError):
    """A counted trial failed, so the phase holds no winner."""

    default_action: ClassVar[OperatorAction] = OperatorAction.INSPECT_LOGS


class PhaseTrialsInfeasibleError(NoFeasibleTrialError):
    """Only infeasible trials kept the phase from a winner."""

    default_action: ClassVar[OperatorAction] = OperatorAction.INSPECT_LOGS


class ReplayWithoutFeasibleTrialError(NoFeasibleTrialError):
    """The run launched no trial and the recorded trials it replayed hold no feasible one."""

    default_action: ClassVar[OperatorAction] = OperatorAction.INSPECT_LOGS


def _no_feasible_error_type(
    trials: Sequence[optuna.trial.FrozenTrial],
) -> type[NoFeasibleTrialError]:
    """Return the error type that names why a phase's finished trials hold no winner.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial of the phase's study.
    :return type[NoFeasibleTrialError]: :class:`PhaseTrialsFailedError` when any trial
        counts as ``FAIL``, otherwise :class:`PhaseTrialsInfeasibleError`.
    """
    if any(_trial_count_state(trial) == "FAIL" for trial in trials):
        return PhaseTrialsFailedError
    return PhaseTrialsInfeasibleError


class PhaseAbortRefusedError(PhaseSweepError):
    """Raised when a phase's recorded abort or failure streak stops a run before it launches a trial."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


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
    emit_advisories: bool = True,
) -> SelectedTrial:
    """Select the best feasible COMPLETE trial, or complete repeat group.

    Require finite metric values, execution ids, feasibility, and metric/constraint
    bounds. Rank by exact argmin or argmax, breaking exact ties by lower draw position.

    With repeats, require one eligible trial per repeat value, rank the reduced member
    metrics, and break ties by the group's lowest draw position. Its highest-numbered
    member supplies the scalar winner identity.

    Assess and log confounds over the ranked population. These verdicts are advisory
    and never alter ranking or selection success.

    :param optuna.Study study: Study of the phase whose winner is selected.
    :param Experiment experiment: Parsed experiment. Supplies the optimization goal
        (minimize/maximize), the metric bounds, and the constraint definitions used to
        filter trials.
    :param str | None phase_name: Phase label used in log messages and to find the phase for
        the grid advisories. Log messages fall back to the study name when ``None``, and
        the grid advisories are skipped.
    :param RepeatAxis | None repeats: The phase's repeat axis
        (:func:`phasesweep.engine.repeats.repeat_axis`), or ``None`` for a phase without repeats.
    :param bool emit_advisories: Whether this fresh selection should log advisories; replay retains verdicts silently.
    :raises NoFeasibleTrialError: No trial in the study is both COMPLETE and within every
        constraint and the metric's bounds, or with ``repeats`` no group is complete. When
        the metric has bounds, the message names them and the best COMPLETE value seen. The
        raised subclass comes from :func:`_no_feasible_error_type`.
    :raises TrialEvidenceMissingError: The winning trial's (with ``repeats``, any winning
        group member's) persisted gate results, objective provenance, or observed values are
        missing, corrupt, or malformed.
    :return SelectedTrial: The winning trial: number, params, metric, constraint readings,
        persisted evidence-gate results, objective provenance, trainer identity, observed
        values, the population's confound block, and the repeat group when ``repeats`` is set.
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
        raise _no_feasible_error_type(trials)(
            f"No complete repeat group: a configuration ranks only when all "
            f"{len(repeats.values)} trials (one per repeats.values of {repeats.key!r}) are "
            f"COMPLETE, feasible and within bounds; {ranking.incomplete_groups} group(s) were "
            "incomplete. The phase's trials.csv labels each trial's repeat_group and records "
            "why each failed or infeasible trial finished that way."
            + _metric_bounds_note(trials, experiment)
        )
    if not survivors:
        raise _no_feasible_error_type(trials)(
            "No completed trials in phase. The phase's trials.csv records why each "
            "failed or infeasible trial finished that way; trainer output is in its "
            "trial_* directory." + _metric_bounds_note(trials, experiment)
        )

    winning: RankedGroup | None = None
    positions = _draw_positions(trials)
    if ranking is None:
        best_value = (
            min(_trial_value(t) for t in survivors)
            if minimize
            else max(_trial_value(t) for t in survivors)
        )
        # Exact equality, not a tolerance band.
        tied = [t for t in survivors if _trial_value(t) == best_value]
        best = min(tied, key=lambda t: positions[t.number])
    else:
        sign = 1.0 if minimize else -1.0
        winning = min(
            ranking.groups,
            key=lambda group: (sign * group.aggregate, min(positions[n] for n in group.numbers)),
        )
        best = ranking.members[winning.representative][-1]

    constraint_vals = {
        name: float(best.user_attrs[constraint_attr(name)]) for name in constraints_by_name
    }
    selected_value = best.value
    assert selected_value is not None
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
    if confound["checks"]["evaluation_point"]["verdict"] == "heterogeneous":
        flagged.append(
            _evaluation_point_remedy(
                experiment.metric.extractor.type,
                axis=confound["checks"]["evaluation_point"]["detail"]["axis"],
            )
        )
    if winning is not None and ranking is not None:
        for group in ranking.groups:
            if dict(group.params) != dict(winning.params) and within_noise(winning, group):
                flagged.append(
                    f"group {group.representative} params={dict(group.params)} versus "
                    f"winner params={dict(winning.params)}, "
                    f"gap={abs(group.aggregate - winning.aggregate):.6g}, "
                    f"combined_se={combined_standard_error(winning, group):.6g}"
                )
    if flagged and emit_advisories:
        log.warning(
            "[%s] potential confounds in the %d ranked trials (advisory; verdicts recorded in the "
            "winner's confound block): %s.",
            phase_name or study.study_name,
            len(survivors),
            "; ".join(fragment.rstrip(".") for fragment in flagged),
        )
    phase = next((phase for phase in experiment.phases if phase.name == phase_name), None)
    if phase is not None and emit_advisories:
        _warn_grid_edges(phase, best.params)
        _warn_unsuccessful_grid_cells(phase, trials)
        if winning is not None and ranking is not None and phase.cost_key is not None:
            advisory = _cost_advisory(phase, winning, ranking.groups, goal=experiment.metric.goal)
            if advisory is not None:
                log.warning("[%s] %s", phase.name, advisory)

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
            _observed_values(best, subject=f"Winning trial {best.number}")
            if record is None
            else observed_means([member["observed"] for member in record["members"]])
        ),
        repeats=record,
    )


def _cost_advisory(
    phase: Phase,
    winner: RankedGroup,
    groups: Sequence[RankedGroup],
    *,
    goal: typing.Literal["minimize", "maximize"],
) -> str | None:
    """Describe a cheaper configuration within one standard error of the winner's mean.

    :param Phase phase: Phase declaring a numeric sampled cost key and mean repeats.
    :param RankedGroup winner: Configuration selected by the objective.
    :param Sequence[RankedGroup] groups: Complete feasible configuration groups.
    :param typing.Literal["minimize", "maximize"] goal: Objective direction for breaking cost ties.
    :return str | None: Advisory text, or ``None`` without a cheaper eligible group or cost key.
    """
    key = phase.cost_key
    if key is None:
        return None
    winner_se = winner.spread["sd"] / math.sqrt(winner.spread["n"])
    eligible = [group for group in groups if abs(group.aggregate - winner.aggregate) <= winner_se]

    def cost(group: RankedGroup) -> float:
        """Read the numeric parameter guaranteed by the phase's cost-key validator.

        :param RankedGroup group: Configuration group.
        :return float: Its declared cost.
        """
        value = group.params[key]
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"cost_key {key!r} is not numeric in group {group.representative}.")
        return float(value)

    sign = 1.0 if goal == "minimize" else -1.0
    cheapest = min(eligible, key=lambda group: (cost(group), sign * group.aggregate, group.numbers))
    if cost(cheapest) < cost(winner):
        return (
            f"one-standard-error cost advisory: params={dict(cheapest.params)} has "
            f"{key}={cost(cheapest):g} versus winner {key}={cost(winner):g}, "
            f"gap={abs(cheapest.aggregate - winner.aggregate):g} <= winner_se={winner_se:g}. "
            "The selected winner is unchanged."
        )
    return None


def _warn_grid_edges(phase: Phase, params: Mapping[str, object]) -> None:
    """Warn when selected params touch an endpoint of a numeric grid axis.

    Numeric categorical choices are ordered by value. Mixed, boolean and
    axes with fewer than three values have no reported boundary; the repeat axis is not searched.

    :param Phase phase: Configured search space and sampler.
    :param Mapping[str, object] params: Selected configuration's sampled values.
    """
    if phase.sampler.type != "grid":
        return
    for name, choices in grid_search_space(phase.search_space, phase_name=phase.name).items():
        values = [
            value
            for value in choices
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        if len(values) != len(choices) or len(set(values)) < 3:
            continue
        value = params.get(name)
        lower, upper = min(values), max(values)
        edge = "lower" if value == lower else "upper" if value == upper else None
        if edge is not None:
            log.warning(
                "[%s] winner is at the %s numeric grid boundary for %s=%s (grid %s..%s); "
                "consider expanding this range if feasible (advisory).",
                phase.name,
                edge,
                name,
                value,
                lower,
                upper,
            )


def _warn_unsuccessful_grid_cells(phase: Phase, trials: Sequence[optuna.trial.FrozenTrial]) -> None:
    """Warn when a grid winner was chosen while some cell has no successful trial.

    A cell is successful when one of its trials is COMPLETE, feasible or not, which is
    when the retry rule closes it. A cell whose attempts all failed, were interrupted or
    never ran has none. The repeat axis counts as a grid axis. The advisory never changes
    selection.

    :param Phase phase: Configured search space and sampler.
    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial of the phase's study.
    """
    if phase.sampler.type != "grid" or not phase_draws(phase):
        return
    axes = grid_search_space(sampler_search_space(phase), phase_name=phase.name)
    succeeded = {
        tuple(trial.params.get(name) for name in axes)
        for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    }
    cells = [dict(zip(axes, values, strict=True)) for values in itertools.product(*axes.values())]
    missing = [cell for cell in cells if tuple(cell.values()) not in succeeded]
    if not missing:
        return
    listed = "; ".join(str(cell) for cell in missing[:_MAX_LISTED_CELLS])
    if len(missing) > _MAX_LISTED_CELLS:
        listed += f"; and {len(missing) - _MAX_LISTED_CELLS} more"
    log.warning(
        "[%s] only %d of %d grid cells succeeded (advisory); the winner was chosen without a "
        "successful trial for: %s.",
        phase.name,
        len(cells) - len(missing),
        len(cells),
        listed,
    )


def _rank_groups(
    trials: Sequence[optuna.trial.FrozenTrial],
    experiment: Experiment,
    axis: RepeatAxis,
) -> _GroupRanking:
    """Group a repeats phase's trials by configuration and keep the complete groups.

    A group is complete when it holds one trial per repeat value and every one
    of them is eligible (:func:`_eligible`). Trials that never recorded the
    repeat param belong to no group.
    An interrupted trial belongs to none either: its rerun takes its place, in
    the same group because it has the same draw position. A failed grid member
    replaced by a retry likewise leaves its group to that replacement.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial in the phase study.
    :param Experiment experiment: Parsed experiment; supplies the eligibility bounds.
    :param RepeatAxis axis: The phase's repeat axis.
    :return _GroupRanking: The complete groups with their member trials, the
        eligible trials left in incomplete groups, and how many groups were incomplete.
    """
    by_key: dict[str, list[optuna.trial.FrozenTrial]] = {}
    draw_positions = _draw_positions(trials)
    replaced = {trial.user_attrs.get(RERUN_OF_ATTR) for trial in trials}
    for trial in trials:
        if (
            axis.key in trial.params
            and not _trial_interrupted(trial)
            and trial.number not in replaced
        ):
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
                "observed": _observed_values(
                    t, subject=f"Trial {t.number} of the winning repeat group"
                ),
            }
            for t in members
        ],
    }


def _observed_values(trial: optuna.trial.FrozenTrial, *, subject: str) -> dict[str, float]:
    """Read the report-only values one trial recorded, from its own attributes.

    The values come from the trial, not the current config, so a later edit to
    ``observe`` cannot change what an already-run trial reports.

    :param optuna.trial.FrozenTrial trial: The selected trial, or a member of the winning group.
    :param str subject: How an error names ``trial``, such as ``"Winning trial 3"``.
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
            f"{subject} has a malformed {OBSERVED_PREFIX!r} value: {exc}."
        ) from exc


def _eligible(trial: optuna.trial.FrozenTrial, experiment: Experiment) -> bool:
    """Require selection identity and stored metric/constraint values within current bounds.

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
        except (TypeError, ValueError, OverflowError):
            return False
        if not check_bounds(value, min_value=constraint.min, max_value=constraint.max):
            return False
    return True


def _metric_bounds_note(trials: Sequence[optuna.trial.FrozenTrial], experiment: Experiment) -> str:
    """Name the metric's viability bounds and the best value seen, for an empty ranking.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial in the phase study.
    :param Experiment experiment: Parsed experiment; supplies the bounds and the goal.
    :return str: Text naming each set bound, then either the best finite value among the
        trials Optuna holds COMPLETE, infeasible ones included, with its trial number or that
        none has one; empty when the metric has no bounds.
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
        return note + " No COMPLETE or INFEASIBLE trial recorded a finite metric value."
    sign = 1.0 if metric.goal == "minimize" else -1.0
    best = min(seen, key=lambda t: (sign * _trial_value(t), t.number))
    return note + (
        f" The best value among COMPLETE and INFEASIBLE trials with a finite value is "
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
