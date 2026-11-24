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
from pathlib import Path
from typing import Any, ClassVar

import optuna

from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import (
    CategoricalParam,
    SearchParam,
    grid_search_space,
    phase_draws,
    sampler_search_space,
)
from phasesweep.engine.artifacts import _winner_common_payload
from phasesweep.engine.bounds import trial_bound_violations
from phasesweep.engine.confound import (
    ConfoundBlock,
    RevisionRead,
    _assess_population,
    _describe_flagged,
    _params_text,
)
from phasesweep.engine.errors import TrialEvidenceMissingError, WinnerIntegrityError
from phasesweep.engine.evidence import _selection_candidate_identity, _trial_objective_provenance
from phasesweep.engine.growth import STANDALONE_PHASE, _grid_kind
from phasesweep.engine.optuna import (
    _any_objective_unreadable,
    _any_trial_timed_out,
    _draw_positions,
    _finished_trial_count,
    _trial_count_state,
    _trial_run_state,
)
from phasesweep.engine.paths import (
    TRIALS_CSV_FILENAME,
    _experiment_dir,
    _trials_csv_path,
)
from phasesweep.engine.provenance import _load_generation_code_revision
from phasesweep.engine.repeats import (
    RankedGroup,
    RepeatAxis,
    RepeatsRecord,
    group_key,
    noise_band,
    observed_means,
    ranked_group,
)
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FINISHED_TRIAL_STATES,
    GATES_ATTR,
    GENERATION_ID_ATTR,
    OBSERVED_PREFIX,
    TRAINER_ENV_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
    TRIAL_DIR_ATTR,
    Winner,
    _unreplaced_attempts,
    _validate_observed_block,
    constraint_attr,
)
from phasesweep.engine.study_policy import (
    _accepted_trial_target,
    _awaits_cleanup_recovery,
    _config_retires_abort,
    _PhasePolicyState,
    _raised_target_step,
    _recovery_action,
    _recovery_advice,
    _replayed_streak_abort,
)
from phasesweep.engine.trial_dirs import _rerun_directory_sources
from phasesweep.errors import (
    REJUDGE_BOUNDS_STEP,
    STORAGE_NULL_KEEPS_NOTHING,
    OperatorAction,
    PhaseSweepError,
    ProcessCleanupUncertainError,
    as_sentence,
    timeout_step,
    unreadable_objective_step,
)
from phasesweep.evidence.models import MlflowExtractor
from phasesweep.runtime.json import JsonObject, float_or_none, json_object, json_text_map
from phasesweep.runtime.terminal import (
    display_path,
    objective_text,
    param_text,
    plural,
)

log = logging.getLogger("phasesweep.engine.selection")


class TieGroupComparison(typing.TypedDict):
    """One configuration compared with the selected repeat group.

    ``combined_se`` and ``threshold`` are ``None`` for groups ranked by median, which have no
    noise band and tie only exactly, at ``gap`` 0.
    """

    trial_number: int
    params: JsonObject
    gap: float
    combined_se: float | None
    threshold: float | None


def tie_group_comparisons(
    winner: RankedGroup, groups: Sequence[RankedGroup]
) -> list[TieGroupComparison]:
    """Describe the other configurations that tie the winner: within its noise band, or exactly.

    :param RankedGroup winner: Selected repeat group.
    :param Sequence[RankedGroup] groups: Every group of the same ranked population, which the
        band pools (:func:`phasesweep.engine.repeats.noise_band`).
    :return list[TieGroupComparison]: Parameters, objective gap, standard error of the difference
        and threshold for each overlap; for groups ranked by median, which have no band, each
        group whose median equals the winner's, without the two band fields.
    """
    band = noise_band(groups)
    if band is None:
        return [
            {
                "trial_number": group.representative,
                "params": json_object(group.params),
                "gap": 0.0,
                "combined_se": None,
                "threshold": None,
            }
            for group in groups
            if dict(group.params) != dict(winner.params) and group.aggregate == winner.aggregate
        ]
    return [
        {
            "trial_number": group.representative,
            "params": json_object(group.params),
            "gap": abs(group.aggregate - winner.aggregate),
            "combined_se": band.standard_error(winner, group),
            "threshold": band.threshold(winner, group),
        }
        for group in groups
        if dict(group.params) != dict(winner.params) and band.within(winner, group)
    ]


def _inspect_step(experiment: Experiment, phase_name: str | None) -> str:
    """Point at a phase's record of its trials, which gives each failed trial's reason.

    :param Experiment experiment: Config locating the artifact root.
    :param str | None phase_name: Phase whose ``trials.csv`` is named, if known.
    :return str: The first sentence of a no-winner remedy.
    """
    if phase_name is None:
        return f"Inspect the phase's {TRIALS_CSV_FILENAME}."
    return f"Inspect {display_path(_trials_csv_path(experiment, phase_name))}."


def _no_feasible_remedy(
    experiment: Experiment, trials: Sequence[optuna.trial.FrozenTrial], phase_name: str | None
) -> str:
    """Name the failed-trial retry or infeasible-bound remedy for the recorded population.

    A fix that changes the study's identity, such as its ``metric`` or ``trial_command``, meets
    the fingerprint refusal on the next run, which gives that change's own step.

    :param Experiment experiment: Config whose storage controls trial retention.
    :param Sequence[optuna.trial.FrozenTrial] trials: Recorded phase attempts.
    :param str | None phase_name: Phase supplying its accepted trial budget, if known.
    :return str: Operator steps appropriate to failures, bounds, and storage.
    """
    evidence = _inspect_step(experiment, phase_name)
    unreadable = _any_objective_unreadable(trials)
    if not experiment.persistent:
        if unreadable:
            return f"{evidence} {unreadable_objective_step(persistent=False)}"
        return f"{evidence} Fix the failure, or the metric bounds or constraints, then rerun."
    infeasible = any(_trial_count_state(trial, experiment) == "INFEASIBLE" for trial in trials)
    if infeasible:
        return f"{evidence} {REJUDGE_BOUNDS_STEP}"
    fix = "Fix the failure"
    step: str | None = "raise the phase's n_trials and rerun."
    if phase_name is not None:
        phase = experiment.phase(phase_name)
        step = _raised_target_step(phase, trials)
        if _any_trial_timed_out(trials):
            fix = f"{timeout_step(phase.name)} and fix any other failure"
    objective_lead = f"{unreadable_objective_step(persistent=True)} " if unreadable else ""
    then = (
        f"{fix}, then {step}"
        if step is not None
        else f"{fix}; every configuration was measured, so no larger n_trials retries one."
    )
    return f"{evidence} {objective_lead}{then}"


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
    # Name of the trial directory inside its phase that the attempt ran in.
    trial_dir: str
    # Frozen evidence provenance recorded when the trial's objective was
    # extracted.
    objective_provenance: dict[str, Any]
    # Digest of the trainer environment this trial ran under.
    trainer_env_digest: str
    # Versioned identity of the exact generated input consumed by the trainer.
    trainer_input: dict[str, Any]
    # Advisory verdicts on the population this selection ranked.
    confound: ConfoundBlock
    # The tracked environment mapping behind ``trainer_env_digest``, or ``None`` when the
    # trial recorded none. Only the ledger trial holds it; winner.yaml keeps the digest.
    trainer_env: dict[str, str] | None = None
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
    if any(_trial_run_state(trial) == "FAIL" for trial in trials):
        return PhaseTrialsFailedError
    return PhaseTrialsInfeasibleError


class PhaseAbortRefusedError(PhaseSweepError):
    """Raised when a phase's recorded abort or failure streak stops a run before it launches a trial."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


def _prior_phase_abort_error(
    experiment: Experiment, phase: Phase, abort_record: JsonObject, study: optuna.Study
) -> PhaseAbortRefusedError:
    """Build the refusal of a phase whose recorded abort the current config does not retire.

    The advice comes from :func:`~phasesweep.engine.study_policy._recovery_advice` and
    the action from :func:`~phasesweep.engine.study_policy._recovery_action`.

    :param Experiment experiment: Config whose bounds judge the recorded trials.
    :param Phase phase: Phase whose accepted trial target is quoted in the
        message.
    :param JsonObject abort_record: Recorded abort; supplies
        ``trial_target`` and ``cause``.
    :param optuna.Study study: The phase's study.
    :return PhaseAbortRefusedError: The refusal, for the caller to raise or report.
    """
    advice = _recovery_advice(experiment, phase, study, abort_record)
    return PhaseAbortRefusedError(
        f"Phase {phase.name!r} previously aborted at its accepted "
        f"n_trials={abort_record['trial_target']}: {as_sentence(abort_record['cause'])}",
        action=_recovery_action(study, abort_record),
        remediation=advice,
    )


def _streak_at_limit_error(
    experiment: Experiment, phase: Phase, abort_record: JsonObject, study: optuna.Study
) -> PhaseAbortRefusedError:
    """Build the refusal of new work while the resumed failure streak meets the current limit.

    :param Experiment experiment: Config whose bounds judge the recorded trials.
    :param Phase phase: Phase whose current ``max_consecutive_failures`` the
        streak meets.
    :param JsonObject abort_record: The abort the streak would record, from
        :func:`~phasesweep.engine.study_policy._failure_policy_abort_record`; supplies the
        replayed count and the accepted target.
    :param optuna.Study study: The phase's study.
    :return PhaseAbortRefusedError: The refusal, for the caller to raise or report.
    """
    advice = _recovery_advice(experiment, phase, study, abort_record)
    return PhaseAbortRefusedError(
        f"Phase {phase.name!r} has {abort_record['consecutive_failures']} consecutive "
        f"failed/infeasible trials, which meets "
        f"max_consecutive_failures={phase.max_consecutive_failures}.",
        action=OperatorAction.FIX_CONFIG,
        remediation=advice,
    )


def _phase_refusal(
    experiment: Experiment,
    phase: Phase,
    study: optuna.Study,
    policy_state: _PhasePolicyState,
) -> PhaseAbortRefusedError | None:
    """Return the refusal a run would raise for the phase's recorded abort or failure streak.

    A recorded abort refuses until the current config retires it or its cleanup recovery runs,
    and a replayed streak at the current limit refuses until ``n_trials`` exceeds the accepted
    target. ``run`` raises it and ``status`` prints it, so both give one diagnosis and step.

    :param Experiment experiment: Config whose bounds judge the recorded trials.
    :param Phase phase: The phase being judged.
    :param optuna.Study study: The phase's study.
    :param _PhasePolicyState policy_state: The study's replayed failure-policy state.
    :return PhaseAbortRefusedError | None: The refusal, or ``None`` when the phase may run.
    """
    abort = policy_state.abort
    if abort is not None and (
        _awaits_cleanup_recovery(study, abort)
        or not _config_retires_abort(phase, study.get_trials(deepcopy=False), abort)
    ):
        return _prior_phase_abort_error(experiment, phase, abort, study)
    streak = _replayed_streak_abort(phase, study, policy_state)
    if streak is not None and phase.n_trials <= _accepted_trial_target(study):
        return _streak_at_limit_error(experiment, phase, streak, study)
    return None


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
        the edge advisory, a grid's untried cells and the no-winner remedy's retry step. Log
        messages fall back to the study name when ``None``, and the others are skipped.
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

    # The remedy points at trials.csv, which gives each failed trial's reason and, with
    # repeats, its repeat_group.
    if repeats is not None and ranking is not None and not ranking.groups:
        incomplete = ranking.incomplete_groups
        raise _no_feasible_error_type(trials)(
            f"No complete repeat group: a configuration ranks only when all "
            f"{len(repeats.values)} trials (one per repeats.values of {repeats.key!r}) are "
            f"COMPLETE, feasible and within bounds; {plural(incomplete, 'group')} "
            f"{'was' if incomplete == 1 else 'were'} incomplete."
            + _metric_bounds_note(trials, experiment),
            remediation=_no_feasible_remedy(experiment, trials, phase_name),
        )
    if not survivors:
        raise _no_feasible_error_type(trials)(
            "No completed trials in phase." + _metric_bounds_note(trials, experiment),
            remediation=_no_feasible_remedy(experiment, trials, phase_name),
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
    phase = next((phase for phase in experiment.phases if phase.name == phase_name), None)
    untried_cells = _untried_grid_cells(phase, trials)
    extractor = experiment.metric.extractor
    revision_sources = _ranked_revision_sources(trials, survivors)
    confound = _assess_population(
        trials,
        survivors,
        best,
        provenance=provenance,
        revisions=_ranked_code_revisions(experiment, survivors, sources=revision_sources),
        revision_sources=revision_sources,
        axis=repeats,
        groups=None if ranking is None else ranking.groups,
        incomplete_group=() if ranking is None else ranking.incomplete_group,
        untried_cells=untried_cells,
        best_value_select=(
            extractor.select
            if isinstance(extractor, MlflowExtractor) and extractor.select in ("min", "max")
            else None
        ),
    )
    flagged = _describe_flagged(
        confound,
        winner_trial=best.number,
        trials=trials,
        untried_cells=untried_cells,
        noise_gaps=(
            None
            if winning is None or ranking is None
            else {
                comparison["trial_number"]: (comparison["gap"], threshold)
                for comparison in tie_group_comparisons(winning, ranking.groups)
                if (threshold := comparison["threshold"]) is not None
            }
        ),
    )
    if flagged and emit_advisories:
        # The warning states what each check found; the run's closing summary gives the
        # show-winners command as typed, which prints each flagged check's counts and remedy.
        log.warning("[%s] %s", phase_name or study.study_name, "; ".join(flagged))
    if phase is not None and emit_advisories:
        edge = _edge_advisory(
            phase,
            trials,
            best.params if winning is None else winning.params,
            persistent=experiment.persistent,
        )
        if edge is not None:
            log.warning("[%s] %s", phase.name, edge)
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
        trial_dir=Path(str(best.user_attrs[TRIAL_DIR_ATTR])).name,
        objective_provenance=objective_provenance,
        # Allocation writes both before the trainer can run, so every
        # completed trial carries them.
        trainer_env_digest=str(best.user_attrs[TRAINER_ENV_DIGEST_ATTR]),
        trainer_env=json_text_map(best.user_attrs.get(TRAINER_ENV_ATTR)),
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

    The standard error is the phase's pooled run-to-run sd over the square root of the
    winner's group size, from the same noise band the tie check uses
    (:func:`phasesweep.engine.repeats.noise_band`), so a phase has one noise estimate.

    :param Phase phase: Phase declaring a numeric sampled cost key and mean repeats.
    :param RankedGroup winner: Configuration selected by the objective.
    :param Sequence[RankedGroup] groups: Complete feasible configuration groups.
    :param typing.Literal["minimize", "maximize"] goal: Objective direction for breaking cost ties.
    :raises ValueError: The groups rank by median, which the cost-key validator refuses.
    :return str | None: Advisory text, or ``None`` without a cheaper eligible group or cost key.
    """
    key = phase.cost_key
    if key is None:
        return None
    band = noise_band(groups)
    if band is None:
        raise ValueError(f"cost_key {key!r} needs repeats with reduce: mean.")
    winner_se = band.pooled_sd / math.sqrt(winner.spread["n"])
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
        # The params carry the cheaper cost, so the winner's one value says by how much.
        return (
            f"cost advisory: {_params_text(cheapest.params).lstrip()} is cheaper than the "
            f"winner's {key}={param_text(winner.params[key])}, with a mean within one standard "
            f"error of the winner's (gap {objective_text(abs(cheapest.aggregate - winner.aggregate))}"
            f", standard error {objective_text(winner_se)}); the selected winner is unchanged"
        )
    return None


def _configured_end(param: SearchParam, *, upper: bool) -> float | None:
    """Return the most extreme value a numeric parameter's configuration allows on one side.

    :param SearchParam param: A searched parameter.
    :param bool upper: Whether to return the highest value rather than the lowest.
    :return float | None: ``high`` or ``low`` of a range, the largest or smallest choice of an
        all-numeric categorical, or ``None`` for choices that are not all numbers.
    """
    if isinstance(param, CategoricalParam):
        choices = [float_or_none(choice) for choice in param.choices]
        numbers = [choice for choice in choices if choice is not None]
        if len(numbers) != len(choices):
            return None
        return max(numbers) if upper else min(numbers)
    return float(param.high if upper else param.low)


def _at_configured_end(param: SearchParam, value: float, end: float) -> bool:
    """Say whether no configured value lies beyond ``value`` on the side of ``end``.

    A stepped range allows values one step apart, so within half a step of the end is at it;
    a continuous float reaches its end only by equalling it.

    :param SearchParam param: The parameter ``end`` came from (:func:`_configured_end`).
    :param float value: The winner's value.
    :param float end: The configured end on the side the winner is extreme on.
    :return bool: Whether the winner sits on the configured end.
    """
    if isinstance(param, CategoricalParam):
        return value == end
    step = param.step or 0
    return abs(end - value) <= step / 2


def _edge_advisory(
    phase: Phase,
    trials: Sequence[optuna.trial.FrozenTrial],
    params: Mapping[str, object],
    *,
    persistent: bool,
) -> str | None:
    """Say which searched numeric parameters no finished trial tried beyond the winner's value.

    A best value is located only once worse values lie on both sides of it. Every sampler gets
    the one rule, stated in terms of what was tried: a numeric parameter (an int or float range,
    or all-numeric categorical choices) with at least three distinct values across the trials
    that finished, COMPLETE, INFEASIBLE or FAIL, whose winning value is the lowest or highest of
    them. A failed or infeasible trial beyond the winner counts as tried, and an interrupted one
    tried nothing. Choices containing booleans or strings have no numeric order. The repeat
    axis is not searched.

    :param Phase phase: Configured search space.
    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial of the phase's study.
    :param Mapping[str, object] params: The winning configuration's params; for a repeats
        phase, the winning group's.
    :param bool persistent: Whether subsequent runs can keep the finished trials.
    :return str | None: One clause per such parameter, naming the values tried and whether the
        winner sits on the configured end, then how to search past each configured end
        reached; ``None`` when no parameter qualifies.
    """
    finished = [trial for trial in trials if _trial_run_state(trial) in FINISHED_TRIAL_STATES]
    clauses: list[str] = []
    remedies: dict[str, None] = {}
    for name, param in phase.search_space.items():
        value = float_or_none(params.get(name))
        tried = sorted(
            {
                number
                for trial in finished
                if (number := float_or_none(trial.params.get(name))) is not None
            }
        )
        if value is None or len(tried) < 3 or value not in (tried[0], tried[-1]):
            continue
        upper = value == tried[-1]
        end = _configured_end(param, upper=upper)
        if end is None:
            continue
        at_end = _at_configured_end(param, value, end)
        place = (
            f"at the configured {'upper' if upper else 'lower'} bound"
            if at_end
            else f"configured range extends to {param_text(end)}"
        )
        clauses.append(
            f"{'above' if upper else 'below'} the winner's {name}={param_text(params.get(name))} "
            f"(tried {param_text(tried[0])}..{param_text(tried[-1])}, {place})"
        )
        if at_end:
            remedies[_edge_remedy(phase, param, persistent=persistent)] = None
    if not clauses:
        return None
    return "; ".join(["no trial ran " + " or ".join(clauses), *remedies])


def _edge_remedy(phase: Phase, param: SearchParam, *, persistent: bool) -> str:
    """Say how to search beyond the configured end the winner sits on.

    A persistent grid's int or float axis and a range under random or TPE can grow in place.
    An axis Optuna records as categorical or a CMA-ES range needs a separate phase to keep
    the earlier trials. An in-memory run can change its original phase and starts over.

    :param Phase phase: The phase whose winner sits on the end.
    :param SearchParam param: The parameter at its configured end.
    :param bool persistent: Whether subsequent runs can keep the finished trials.
    :return str: The remedy clause.
    """
    sampler = phase.sampler.type
    if not persistent:
        target = " and raise n_trials to the new grid size" if sampler == "grid" else ""
        return (
            f"extend search_space beyond the bound{target} and rerun; {STORAGE_NULL_KEEPS_NOTHING}"
        )
    kept = "; the study keeps its finished trials"
    if sampler == "grid" and _grid_kind(param) != "categorical":
        return f"add values beyond the bound to search_space and raise n_trials to cover the enlarged grid{kept}"
    if sampler == "cmaes":
        return "add a seeded random or TPE phase over the wider range, " + STANDALONE_PHASE.format(
            phase=phase.name
        )
    if isinstance(param, CategoricalParam):
        return "add a phase over values beyond the bound, " + STANDALONE_PHASE.format(
            phase=phase.name
        )
    return f"widen the range in search_space and raise n_trials{kept}"


def _untried_grid_cells(
    phase: Phase | None, trials: Sequence[optuna.trial.FrozenTrial]
) -> list[dict[str, object]]:
    """Return the cells of a grid that reached its target covering every cell, yet no trial ran.

    Raising a grid's target after failures schedules the failed cells' retries ahead of the
    cells it never reached, so a finished target can leave cells nobody tried, and nothing
    else reports them. A target below the grid's size leaves cells out by design, and a
    phase cut short by an accepted timeout reports its shortfall in the completion record,
    so neither counts. The repeat axis counts as a grid axis.

    :param Phase | None phase: The phase, or ``None`` when selection has none.
    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial of the phase's study.
    :return list[dict[str, object]]: The untried cells in grid order, empty for any other phase.
    """
    if phase is None or phase.sampler.type != "grid" or not phase_draws(phase):
        return []
    axes = grid_search_space(sampler_search_space(phase), phase_name=phase.name)
    cells: list[dict[str, object]] = [
        dict(zip(axes, values, strict=True)) for values in itertools.product(*axes.values())
    ]
    if phase.n_trials < len(cells) or _finished_trial_count(trials) < phase.n_trials:
        return []
    tried = {tuple(trial.params.get(name) for name in axes) for trial in trials}
    return [cell for cell in cells if tuple(cell.values()) not in tried]


def _rank_groups(
    trials: Sequence[optuna.trial.FrozenTrial],
    experiment: Experiment,
    axis: RepeatAxis,
) -> _GroupRanking:
    """Group a repeats phase's trials by configuration and keep the complete groups.

    A group is complete when it holds one trial per repeat value and every one
    of them is eligible (:func:`_eligible`). Only the attempts that stand for
    their cell join a group (:func:`~phasesweep.engine.state._unreplaced_attempts`):
    an interrupted trial's rerun takes its place, in the same group because it has
    the same draw position, and a failed grid member replaced by a retry leaves its
    group to that replacement. Trials that never recorded the repeat param belong
    to no group.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial in the phase study.
    :param Experiment experiment: Parsed experiment; supplies the eligibility bounds.
    :param RepeatAxis axis: The phase's repeat axis.
    :return _GroupRanking: The complete groups with their member trials, the
        eligible trials left in incomplete groups, and how many groups were incomplete.
    """
    by_key: dict[str, list[optuna.trial.FrozenTrial]] = {}
    draw_positions = _draw_positions(trials)
    for trial in _unreplaced_attempts(trials, interrupted=False):
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
                "trial_dir": Path(str(t.user_attrs[TRIAL_DIR_ATTR])).name,
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
        if key.startswith(OBSERVED_PREFIX) and value is not None
    }
    try:
        return _validate_observed_block(recorded)
    except ValueError as exc:
        raise TrialEvidenceMissingError(
            f"{subject} has a malformed {OBSERVED_PREFIX!r} value: {exc}."
        ) from exc


def _ranked_revision_sources(
    trials: Sequence[optuna.trial.FrozenTrial], ranked: Sequence[optuna.trial.FrozenTrial]
) -> dict[int, list[str]]:
    """Find the earlier generations that may have trained in each ranked trial's directory.

    Every ancestor back to the attempt that opened the same directory
    (:func:`~phasesweep.engine.trial_dirs._rerun_directory_sources`) and recorded a generation counts.
    One that never launched its trainer may add a revision that trained nothing, which the
    check's "may have trained" wording covers.

    :param Sequence[optuna.trial.FrozenTrial] trials: The phase's full trial history.
    :param Sequence[optuna.trial.FrozenTrial] ranked: The trials eligible for selection.
    :return dict[int, list[str]]: Possible contributing generations per ranked trial, oldest first.
    """
    sources: dict[int, list[str]] = {}
    if len(ranked) < 2:
        return sources
    by_number = {trial.number: trial for trial in trials}
    for trial in ranked:
        generations: list[str] = []
        # A malformed ancestor must not turn a code revision advisory into a refusal.
        with contextlib.suppress(ProcessCleanupUncertainError):
            for source in _rerun_directory_sources(by_number, trial):
                generation = source.user_attrs.get(GENERATION_ID_ATTR)
                if isinstance(generation, str) and generation:
                    generations.append(generation)
        if generations:
            sources[trial.number] = generations[::-1]
    return sources


def _ranked_code_revisions(
    experiment: Experiment,
    ranked: Sequence[optuna.trial.FrozenTrial],
    *,
    sources: Mapping[int, Sequence[str]],
) -> dict[str, RevisionRead]:
    """Read each recorded trainer revision that a ranked trial may share, once per generation.

    A missing or invalid record is ``unreadable`` rather than an error: the check that uses
    these is advisory, and a trial that lost must not fail selection.

    :param Experiment experiment: Parsed experiment; locates the generation records.
    :param Sequence[optuna.trial.FrozenTrial] ranked: Ranked trials; selection requires each
        to carry a generation id (:func:`_eligible`).
    :param Mapping[int, Sequence[str]] sources: Possible contributing generations per
        ranked trial, before the generation that finished it.
    :return dict[str, RevisionRead]: Each ranked generation's revision, ``unknown`` when it
        recorded none; empty for fewer than two ranked trials, which the check does not run on.
    """
    revisions: dict[str, RevisionRead] = {}
    if len(ranked) < 2:
        return revisions
    artifact_root = _experiment_dir(experiment)
    generation_ids = dict.fromkeys(str(trial.user_attrs[GENERATION_ID_ATTR]) for trial in ranked)
    generation_ids.update(dict.fromkeys(g for earlier in sources.values() for g in earlier))
    for generation_id in generation_ids:
        try:
            revision = _load_generation_code_revision(artifact_root, generation_id)
        except WinnerIntegrityError:
            revisions[generation_id] = "unreadable"
        else:
            revisions[generation_id] = "unknown" if revision is None else revision
    return revisions


def _eligible(trial: optuna.trial.FrozenTrial, experiment: Experiment) -> bool:
    """Require selection identity and stored metric/constraint values within current bounds.

    :param optuna.trial.FrozenTrial trial: Persisted trial to classify.
    :param Experiment experiment: Parsed experiment; supplies the metric and constraint bounds.
    :return bool: ``True`` when the trial is a selection candidate.
    """
    if _selection_candidate_identity(trial) is None:
        return False
    return not trial_bound_violations(trial, experiment)


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
