"""Advisory confound verdicts over a phase's ranked population.

Without repeats, ranked units are individual trials. With repeats, they are complete groups,
whose aggregates are also compared using combined standard errors as a noise heuristic.
Survivorship counts configurations instead: a trial's params, or a repeat group.

Verdicts are frozen at selection, carried unchanged with the winner, and excluded from
fingerprints. Readers validate the block's shape and never re-judge a verdict. They never
block selection, publication, or top-ups.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Literal, TypeAlias, TypedDict, get_args

import optuna

from phasesweep.engine.optuna import _trial_count_state
from phasesweep.engine.repeats import (
    RankedGroup,
    RepeatAxis,
    configuration_slot,
    configuration_slots,
    within_noise,
)
from phasesweep.engine.state import _unreplaced_attempts
from phasesweep.errors import FRESH_STUDY
from phasesweep.evidence.models import POSITIONED_EXTRACTOR_KINDS
from phasesweep.runtime.json import progress_position
from phasesweep.runtime.terminal import param_text

ConfoundVerdict = Literal["ok", "heterogeneous", "n/a"]
"""Outcome of one check. ``ok``: the check ran and found nothing.
``heterogeneous``: the check ran and flagged the population, with evidence.
``n/a``: the check could not run; it means *not checked*, never *clean*."""

CONFOUND_VERDICTS: tuple[ConfoundVerdict, ...] = get_args(ConfoundVerdict)

CheckedVerdict = Literal["ok", "heterogeneous"]
"""The verdict of a check that ran."""

ConfoundCheck = Literal["evaluation_point", "survivorship", "tie"]
"""One comparability dimension, in the order every surface reports them."""

CONFOUND_CHECKS: tuple[ConfoundCheck, ...] = get_args(ConfoundCheck)

CONFOUND_SCHEMA_VERSION = 4


class PositionGroup(TypedDict):
    """Ranked trials whose objectives were measured at one evaluation position."""

    value: int | float
    trials: list[int]


class EvaluationDetail(TypedDict):
    """Every ranked objective's position on the one axis they declared, ascending."""

    axis: str
    values: list[PositionGroup]


class SurvivorshipDetail(TypedDict):
    """How the configurations the phase tried, and a full grid's untried cells, ended.

    ``failed_trials`` holds each failed configuration's latest failed attempt.
    """

    ranked: int
    infeasible: int
    failed: int
    untried: int
    failed_trials: list[int]


class TieDetail(TypedDict):
    """Units that tied the winner exactly, and repeat groups within noise of it."""

    tied_trials: list[int]
    within_noise: list[int] | None


class NotChecked(TypedDict):
    """A check that could not run, and why."""

    verdict: Literal["n/a"]
    reason: str


class EvaluationChecked(TypedDict):
    """A checked ``evaluation_point`` record."""

    verdict: CheckedVerdict
    detail: EvaluationDetail


class SurvivorshipCheck(TypedDict):
    """The ``survivorship`` record, which always runs."""

    verdict: CheckedVerdict
    detail: SurvivorshipDetail


class TieChecked(TypedDict):
    """A checked ``tie`` record."""

    verdict: CheckedVerdict
    detail: TieDetail


EvaluationCheck: TypeAlias = EvaluationChecked | NotChecked
TieCheck: TypeAlias = TieChecked | NotChecked


class ConfoundChecks(TypedDict):
    """The three check records, keyed by check name."""

    evaluation_point: EvaluationCheck
    survivorship: SurvivorshipCheck
    tie: TieCheck


class ConfoundBlock(TypedDict):
    """The ``confound`` block a phase winner carries."""

    schema_version: int
    ranked_trials: list[int]
    ranked_groups: list[list[int]] | None
    checks: ConfoundChecks


SlotState: TypeAlias = Literal["measured", "infeasible", "pending", "untried"] | int
"""How one slot of a configuration ended: measured inside the bounds, measured outside them,
still waiting for an attempt, a grid cell no trial reached, or failed, as the number of its
latest failed attempt."""

_MEASURED_STATES = frozenset({"COMPLETE", "INFEASIBLE"})
_PENDING_STATES = frozenset({"WAITING", "RUNNING"})
# Evaluation positions within this fraction of the furthest one count as one
# point: a fixed-token batch-size sweep stops each trial within a batch of its
# budget, while a truncated trial falls short by far more.
_POSITION_TOLERANCE = 0.01
# Trials or configurations a log line spells out before counting the rest.
_LOGGED_VALUES = 5


def _assess_population(
    trials: Sequence[optuna.trial.FrozenTrial],
    ranked: Sequence[optuna.trial.FrozenTrial],
    winner: optuna.trial.FrozenTrial,
    *,
    provenance: Mapping[int, Mapping[str, object]],
    axis: RepeatAxis | None = None,
    groups: Sequence[RankedGroup] | None = None,
    incomplete_group: Collection[int] = (),
    untried_cells: Sequence[Mapping[str, object]] = (),
    best_value_select: str | None = None,
) -> ConfoundBlock:
    """Classify the population one winner selection compared.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial in the phase study, the
        population survivorship accounts for.
    :param Sequence[optuna.trial.FrozenTrial] ranked: Trials selection ranked:
        candidates that satisfy every constraint, and for a repeats phase every
        member of a complete group.
    :param optuna.trial.FrozenTrial winner: The selected trial, one of
        ``ranked``; for a repeats phase, the last member of the winning group.
    :param Mapping[int, Mapping[str, object]] provenance: Validated objective
        provenance keyed by trial number, for every ranked trial whose record
        could be read.
    :param RepeatAxis | None axis: The phase's repeat axis, which keys configurations by
        repeat group, or ``None`` for a phase without repeats.
    :param Sequence[RankedGroup] | None groups: The complete repeat groups
        selection ranked, or ``None`` for a phase without repeats.
    :param Collection[int] incomplete_group: Numbers of feasible trials whose
        repeat group was incomplete, so selection could not rank them.
    :param Sequence[Mapping[str, object]] untried_cells: The cells, repeat key included,
        that no trial reached in a grid whose finished target covers every cell
        (:func:`phasesweep.engine.selection._untried_grid_cells`), defaults to none.
    :param str | None best_value_select: The MLflow selector, ``min`` or ``max``, when the
        objective is the best value of its history, defaults to None.
    :return ConfoundBlock: The ``confound`` block persisted on the winner.
    """
    ordered = sorted(ranked, key=lambda trial: trial.number)
    ordered_groups = None if groups is None else sorted(groups, key=lambda group: group.numbers)
    return {
        "schema_version": CONFOUND_SCHEMA_VERSION,
        "ranked_trials": [trial.number for trial in ordered],
        "ranked_groups": (
            None if ordered_groups is None else [list(group.numbers) for group in ordered_groups]
        ),
        "checks": {
            "evaluation_point": _evaluation_point(
                ordered, provenance, best_value_select=best_value_select
            ),
            "survivorship": _survivorship(
                trials,
                {trial.number for trial in ordered},
                feasible={trial.number for trial in ordered} | set(incomplete_group),
                axis=axis,
                untried_cells=untried_cells,
            ),
            "tie": _tie(ordered, winner)
            if ordered_groups is None
            else _group_tie(ordered_groups, winner),
        },
    }


def _evaluation_point(
    ranked: Sequence[optuna.trial.FrozenTrial],
    provenance: Mapping[int, Mapping[str, object]],
    *,
    best_value_select: str | None,
) -> EvaluationCheck:
    """Compare ranked objectives on their declared training-progress axis.

    JSON envelopes record the position; W&B uses its selected history position and
    MLflow the selected observation's step. Record all positions and flag those
    more than ``_POSITION_TOLERANCE`` short of the furthest. Other axes, including
    checkpoint labels, are excluded from this comparison. An MLflow ``select: min`` or
    ``max`` objective is not checked: its step is where the best value fell, which says
    nothing about how far the trial trained.

    :param Sequence[optuna.trial.FrozenTrial] ranked: Ranked trials by number.
    :param Mapping[int, Mapping[str, object]] provenance: Their readable objective
        provenance.
    :param str | None best_value_select: The MLflow selector, ``min`` or ``max``, when the
        objective is the best value of its history, otherwise ``None``.
    :return EvaluationCheck: The check record.
    """
    if len(ranked) < 2:
        return _not_checked("only one trial was ranked")
    if best_value_select is not None:
        return _not_checked(
            f"mlflow select: {best_value_select} records the step where the best value fell, "
            "not how far each trial trained"
        )
    unreadable = [str(trial.number) for trial in ranked if trial.number not in provenance]
    if unreadable:
        return _not_checked(
            f"trial(s) {', '.join(unreadable)} have unreadable objective provenance"
        )
    records = [(trial.number, provenance[trial.number]) for trial in ranked]
    kinds = sorted({_extractor_kind(record) for _, record in records})
    if len(kinds) != 1 or kinds[0] not in POSITIONED_EXTRACTOR_KINDS:
        return _not_checked(
            f"{', '.join(kinds)} objectives report no evaluation position; json_envelope "
            "(report_objective), wandb (evaluation_axis) or mlflow (select: last or step) "
            "would bind one"
        )
    positions = [(number, _recorded_position(record)) for number, record in records]
    axes = sorted({axis for _, (axis, _) in positions})
    if len(axes) != 1:
        return _not_checked(f"ranked objectives declare different axes ({', '.join(axes)})")
    located: list[tuple[int, int | float]] = []
    unknown: list[str] = []
    for number, (_, value) in positions:
        if value is None:
            unknown.append(str(number))
        else:
            located.append((number, value))
    if unknown:
        return _not_checked(
            f"trial(s) {', '.join(unknown)} recorded no {axes[0]} position for their objective"
        )
    values = _position_groups(located)
    flagged = values[0]["value"] < (1 - _POSITION_TOLERANCE) * values[-1]["value"]
    return {
        "verdict": "heterogeneous" if flagged else "ok",
        "detail": {"axis": axes[0], "values": values},
    }


def _extractor_kind(record: Mapping[str, object]) -> str:
    """Return the extractor kind a validated objective provenance record names.

    :param Mapping[str, object] record: Validated objective provenance.
    :raises ValueError: The record names no extractor kind, which validation rules out.
    :return str: The kind.
    """
    extractor = record.get("extractor")
    kind = extractor.get("kind") if isinstance(extractor, Mapping) else None
    if not isinstance(kind, str):
        raise ValueError("objective provenance names no extractor kind")
    return kind


def _recorded_position(record: Mapping[str, object]) -> tuple[str, int | float | None]:
    """Return the evaluation axis and position a positioned objective recorded.

    :param Mapping[str, object] record: Validated provenance of a positioned extractor kind.
    :raises ValueError: The record has no evaluation axis, which validation rules out.
    :return tuple[str, int | float | None]: The axis, and the position or ``None`` when the
        reader could not locate one.
    """
    source = record.get("source")
    evaluation = source.get("evaluation") if isinstance(source, Mapping) else None
    progress = evaluation.get("progress") if isinstance(evaluation, Mapping) else None
    axis = progress.get("axis") if isinstance(progress, Mapping) else None
    if not isinstance(progress, Mapping) or not isinstance(axis, str):
        raise ValueError("positioned objective provenance records no evaluation axis")
    return axis, progress_position(progress.get("value"))


def _position_groups(pairs: Iterable[tuple[int, int | float]]) -> list[PositionGroup]:
    """Group trial numbers by the position each recorded, ordered by position.

    :param Iterable[tuple[int, int | float]] pairs: ``(trial_number, position)`` pairs in
        trial-number order.
    :return list[PositionGroup]: ``[{"value": ..., "trials": [...]}, ...]``.
    """
    by_value: dict[int | float, list[int]] = {}
    for number, value in pairs:
        by_value.setdefault(value, []).append(number)
    return [{"value": value, "trials": by_value[value]} for value in sorted(by_value)]


def _survivorship(
    trials: Sequence[optuna.trial.FrozenTrial],
    ranked: Collection[int],
    *,
    feasible: Collection[int],
    axis: RepeatAxis | None,
    untried_cells: Sequence[Mapping[str, object]],
) -> SurvivorshipCheck:
    """Account for every configuration the phase tried, and flag any left without a result.

    A configuration is a trial's params, or with repeats its repeat group, each repeat value
    a slot (:func:`phasesweep.engine.repeats.configuration_slot`). It is ranked when selection
    ranked one of its trials; infeasible when a slot was measured outside a bound or
    constraint; failed when a slot's attempts all failed, with nothing measured and no rerun
    pending for it; untried when a slot is a grid cell no trial reached. Any other
    configuration is still waiting for an attempt, which only a phase cut short by an
    accepted timeout leaves, and the winner's completion record reports that shortfall
    instead. Infeasible configurations never flag: the experiment's own bounds and
    constraints define what may win. A failed or untried configuration flags the check.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial in the study.
    :param Collection[int] ranked: Numbers of the ranked trials.
    :param Collection[int] feasible: Numbers of the trials measured inside every bound,
        ranked or left in an incomplete repeat group.
    :param RepeatAxis | None axis: The phase's repeat axis, or ``None`` without repeats.
    :param Sequence[Mapping[str, object]] untried_cells: Grid cells no trial reached.
    :return SurvivorshipCheck: The check record.
    """
    waiting = {trial.number for trial in _unreplaced_attempts(trials, interrupted=True)}
    by_number = {trial.number: trial for trial in trials}
    slots: dict[str, dict[str, list[optuna.trial.FrozenTrial]]] = {}
    for number, (configuration, slot) in configuration_slots(trials, axis).items():
        slots.setdefault(configuration, {}).setdefault(slot, []).append(by_number[number])
    for cell in untried_cells:
        # Only seeded random keys a group by draw position, and a grid cell has none.
        keys = configuration_slot(0, cell, axis)
        if keys is not None:
            slots.setdefault(keys[0], {}).setdefault(keys[1], [])
    counts = {"ranked": 0, "infeasible": 0, "untried": 0}
    failed: list[int] = []
    for by_slot in slots.values():
        states = [
            _slot_state(attempts, feasible=feasible, waiting=waiting)
            for attempts in by_slot.values()
        ]
        failures = [state for state in states if isinstance(state, int)]
        if any(trial.number in ranked for attempts in by_slot.values() for trial in attempts):
            counts["ranked"] += 1
        elif "infeasible" in states:
            counts["infeasible"] += 1
        elif failures:
            failed.append(max(failures))
        elif "untried" in states:
            counts["untried"] += 1
    return {
        "verdict": "heterogeneous" if failed or counts["untried"] else "ok",
        "detail": {
            "ranked": counts["ranked"],
            "infeasible": counts["infeasible"],
            "failed": len(failed),
            "untried": counts["untried"],
            "failed_trials": sorted(failed),
        },
    }


def _slot_state(
    attempts: Sequence[optuna.trial.FrozenTrial],
    *,
    feasible: Collection[int],
    waiting: Collection[int],
) -> SlotState:
    """Say how the attempts at one slot of a configuration ended.

    :param Sequence[optuna.trial.FrozenTrial] attempts: Every attempt at the slot, none for
        an untried grid cell.
    :param Collection[int] feasible: Numbers of the trials measured inside every bound.
    :param Collection[int] waiting: Numbers of the interrupted trials no rerun names yet.
    :return SlotState: ``untried`` without attempts, ``measured`` or ``infeasible`` once an
        attempt was measured, ``pending`` while one is queued, running or waiting for its
        rerun, otherwise the number of the latest failed attempt.
    """
    if not attempts:
        return "untried"
    states = {trial.number: _trial_count_state(trial) for trial in attempts}
    measured = [number for number, state in states.items() if state in _MEASURED_STATES]
    if measured:
        return "measured" if any(number in feasible for number in measured) else "infeasible"
    if any(state in _PENDING_STATES or number in waiting for number, state in states.items()):
        return "pending"
    failures = [number for number, state in states.items() if state == "FAIL"]
    return max(failures) if failures else "pending"


def _tie(
    ranked: Sequence[optuna.trial.FrozenTrial],
    winner: optuna.trial.FrozenTrial,
) -> TieCheck:
    """Check whether the draw-position tiebreak chose between distinct configs.

    Selection breaks exact ties by lower draw position. Tied trials that share
    the winner's params are repeats of one configuration and change nothing;
    tied trials with other params mean the winner was chosen arbitrarily.
    Single trials carry no noise estimate, so ``within_noise`` is ``None``.

    :param Sequence[optuna.trial.FrozenTrial] ranked: Ranked trials by number.
    :param optuna.trial.FrozenTrial winner: The selected trial.
    :return TieCheck: The check record.
    """
    if len(ranked) < 2:
        return _not_checked("only one trial was ranked")
    if all(trial.params == winner.params for trial in ranked):
        return _not_checked("only one configuration was ranked")
    tied = [
        trial.number
        for trial in ranked
        if trial.number != winner.number
        and trial.value == winner.value
        and trial.params != winner.params
    ]
    return {
        "verdict": "heterogeneous" if tied else "ok",
        "detail": {"tied_trials": tied, "within_noise": None},
    }


def _group_tie(
    groups: Sequence[RankedGroup],
    winner: optuna.trial.FrozenTrial,
) -> TieCheck:
    """Check whether the winning repeat group stands clear of the other groups.

    ``tied_trials`` names the groups whose aggregate equals the winner's
    exactly; the lowest member draw position broke that tie. ``within_noise``
    names the groups whose aggregate lies within two combined standard errors
    of the winner's, exact ties included. Each group is named by its last
    member. A group with the winner's params is another slot of the same
    configuration and is skipped.

    :param Sequence[RankedGroup] groups: Ranked groups ordered by their members.
    :param optuna.trial.FrozenTrial winner: The last member of the winning group.
    :return TieCheck: The check record.
    """
    if len(groups) < 2:
        return _not_checked("only one repeat group was ranked")
    best = next(group for group in groups if group.representative == winner.number)
    others = [
        group for group in groups if group is not best and dict(group.params) != dict(best.params)
    ]
    tied = sorted(group.representative for group in others if group.aggregate == best.aggregate)
    noisy = sorted(group.representative for group in others if within_noise(best, group))
    return {
        "verdict": "heterogeneous" if tied or noisy else "ok",
        "detail": {"tied_trials": tied, "within_noise": noisy},
    }


def _not_checked(reason: str) -> NotChecked:
    """Return the record of a check that could not run.

    :param str reason: Why the population could not be checked.
    :return NotChecked: The check record.
    """
    return {"verdict": "n/a", "reason": reason}


def _validate_confound_block(data: object, *, winner_trial: int) -> ConfoundBlock:
    """Validate the shape of a persisted ``confound`` block and return a plain copy.

    The verdicts were judged once, at selection, and are read as recorded: the winner file's
    manifest hash already catches an edited block, so nothing here re-judges a verdict against
    its evidence or cross-checks counts against the ranked trials. What is checked is what the
    readers rely on: the keys and schema version, the winner among ``ranked_trials``, the shape
    of ``ranked_groups`` that the ``repeats`` validation and the group readers index, each
    verdict's value, a reason on every ``n/a``, and the types of the evidence fields that
    :func:`_confound_summary` and the MCP snapshot read.

    :param object data: Parsed ``confound`` value from a ``winner.yaml``.
    :param int winner_trial: Trial number of the winner the block sits on.
    :raises ValueError: The block is absent, has another schema version or other keys, does
        not rank the winner, or a check record or its evidence is malformed.
    :return ConfoundBlock: The validated block, rebuilt from plain values.
    """
    keys = {"schema_version", "ranked_trials", "ranked_groups", "checks"}
    if not isinstance(data, Mapping) or set(data) != keys:
        raise ValueError("confound block is missing or has unexpected keys")
    if type(data["schema_version"]) is not int or data["schema_version"] != CONFOUND_SCHEMA_VERSION:
        raise ValueError(f"confound schema_version is not {CONFOUND_SCHEMA_VERSION}")
    ranked = _trial_numbers(data["ranked_trials"], label="ranked_trials")
    if winner_trial not in ranked:
        raise ValueError(
            f"confound ranked_trials does not include the winner, trial {winner_trial}"
        )
    checks = data["checks"]
    if not isinstance(checks, Mapping) or set(checks) != set(CONFOUND_CHECKS):
        raise ValueError(f"confound checks must be exactly {', '.join(CONFOUND_CHECKS)}")
    survivorship = _checked_record("survivorship", checks["survivorship"])
    if survivorship is None:
        raise ValueError("confound check 'survivorship' always runs and cannot be n/a")
    evaluation = _checked_record("evaluation_point", checks["evaluation_point"])
    tie = _checked_record("tie", checks["tie"])
    return {
        "schema_version": CONFOUND_SCHEMA_VERSION,
        "ranked_trials": ranked,
        "ranked_groups": _validate_ranked_groups(data["ranked_groups"]),
        "checks": {
            "evaluation_point": (
                _not_checked(_reason(checks["evaluation_point"]))
                if evaluation is None
                else {
                    "verdict": evaluation[0],
                    "detail": _validate_evaluation_detail(evaluation[1]),
                }
            ),
            "survivorship": {
                "verdict": survivorship[0],
                "detail": _validate_survivorship_detail(survivorship[1]),
            },
            "tie": (
                _not_checked(_reason(checks["tie"]))
                if tie is None
                else {"verdict": tie[0], "detail": _validate_tie_detail(tie[1])}
            ),
        },
    }


def _validate_ranked_groups(value: object) -> list[list[int]] | None:
    """Validate the shape of the ranked repeat groups.

    :param object value: Parsed ``ranked_groups``: ``None``, or a list of member lists.
    :raises ValueError: The value is neither ``None`` nor a nonempty list of lists of at
        least two ascending trial numbers.
    :return list[list[int]] | None: The groups, or ``None`` for a phase without repeats.
    """
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("confound ranked_groups is neither null nor a list of groups")
    groups = [_trial_numbers(group, label="ranked_groups member list") for group in value]
    if any(len(group) < 2 for group in groups):
        raise ValueError("confound ranked_groups has a group of fewer than two trials")
    return groups


def _checked_record(name: ConfoundCheck, check: object) -> tuple[CheckedVerdict, object] | None:
    """Validate one check record's verdict and split a checked one into verdict and evidence.

    :param ConfoundCheck name: Check the record belongs to.
    :param object check: Parsed record.
    :raises ValueError: The verdict is unknown, an ``n/a`` record has no reason, or a
        checked record carries no evidence.
    :return tuple[CheckedVerdict, object] | None: The verdict and the unvalidated evidence,
        or ``None`` for an ``n/a`` record with its reason.
    """
    verdict = check.get("verdict") if isinstance(check, Mapping) else None
    if not isinstance(check, Mapping) or verdict not in CONFOUND_VERDICTS:
        raise ValueError(f"confound check {name!r} has no valid verdict")
    if verdict == "n/a":
        _reason(check, name=name)
        return None
    if set(check) != {"verdict", "detail"}:
        raise ValueError(f"confound check {name!r} has no evidence")
    return ("ok" if verdict == "ok" else "heterogeneous"), check["detail"]


def _reason(check: object, *, name: str = "") -> str:
    """Return an ``n/a`` record's reason.

    :param object check: An ``n/a`` record whose verdict was validated.
    :param str name: Check name for the diagnostic, defaults to empty.
    :raises ValueError: The record has keys besides its verdict and a nonempty reason.
    :return str: The reason.
    """
    reason = check.get("reason") if isinstance(check, Mapping) else None
    if not isinstance(check, Mapping) or set(check) != {"verdict", "reason"}:
        raise ValueError(f"confound check {name!r} is n/a without a reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError(f"confound check {name!r} is n/a without a reason")
    return reason


def _validate_evaluation_detail(detail: object) -> EvaluationDetail:
    """Validate evaluation-point evidence: the axis and its position groups.

    :param object detail: Parsed evidence.
    :raises ValueError: The axis is missing, or a group is malformed.
    :return EvaluationDetail: The validated evidence.
    """
    if not isinstance(detail, Mapping) or set(detail) != {"axis", "values"}:
        raise ValueError("confound evaluation_point evidence must name its axis and values")
    axis, groups = detail["axis"], detail["values"]
    if not isinstance(axis, str) or not axis:
        raise ValueError("confound evaluation_point axis is not a nonempty string")
    if not isinstance(groups, list) or not groups:
        raise ValueError("confound evaluation_point has no value groups")
    parsed: list[PositionGroup] = []
    for group in groups:
        if not isinstance(group, Mapping) or set(group) != {"value", "trials"}:
            raise ValueError("confound evaluation_point value group is malformed")
        value = progress_position(group["value"])
        trials = _trial_numbers(group["trials"], label="evaluation_point trials")
        if value is None or not trials:
            raise ValueError("confound evaluation_point value group is malformed")
        if parsed and not parsed[-1]["value"] < value:
            raise ValueError("confound evaluation_point values are not ascending and distinct")
        parsed.append({"value": value, "trials": trials})
    return {"axis": axis, "values": parsed}


def _validate_survivorship_detail(detail: object) -> SurvivorshipDetail:
    """Validate survivorship evidence: the counts and the failed configurations' trials.

    :param object detail: Parsed evidence.
    :raises ValueError: A field is missing or extra, a count is not a non-negative int, or
        ``failed_trials`` is not a list of trial numbers.
    :return SurvivorshipDetail: The validated evidence.
    """
    fields = set(SurvivorshipDetail.__annotations__)
    if not isinstance(detail, Mapping) or set(detail) != fields:
        raise ValueError(f"confound survivorship evidence must hold {', '.join(sorted(fields))}")
    return {
        "ranked": _count(detail["ranked"], label="ranked"),
        "infeasible": _count(detail["infeasible"], label="infeasible"),
        "failed": _count(detail["failed"], label="failed"),
        "untried": _count(detail["untried"], label="untried"),
        "failed_trials": _trial_numbers(
            detail["failed_trials"], label="survivorship failed_trials"
        ),
    }


def _count(value: object, *, label: str) -> int:
    """Validate one survivorship count.

    :param object value: Parsed count.
    :param str label: Count name for the diagnostic.
    :raises ValueError: The value is not a non-negative int.
    :return int: The count.
    """
    if type(value) is not int or value < 0:
        raise ValueError(f"confound survivorship {label} is not a non-negative count")
    return value


def _validate_tie_detail(detail: object) -> TieDetail:
    """Validate tie evidence: exact ties with other params, and groups within noise.

    :param object detail: Parsed evidence.
    :raises ValueError: A list is missing or malformed.
    :return TieDetail: The validated evidence.
    """
    if not isinstance(detail, Mapping) or set(detail) != {"tied_trials", "within_noise"}:
        raise ValueError("confound tie evidence must list tied_trials and within_noise")
    noisy = detail["within_noise"]
    return {
        "tied_trials": _trial_numbers(detail["tied_trials"], label="tie tied_trials"),
        "within_noise": (
            None if noisy is None else _trial_numbers(noisy, label="tie within_noise")
        ),
    }


def _trial_numbers(value: object, *, label: str) -> list[int]:
    """Validate a list of distinct trial numbers in ascending order.

    :param object value: Parsed list.
    :param str label: Field name for the diagnostic.
    :raises ValueError: The value is not such a list.
    :return list[int]: The trial numbers.
    """
    if not isinstance(value, list):
        raise ValueError(f"confound {label} is not a list of trial numbers")
    numbers: list[int] = []
    for number in value:
        if type(number) is not int or number < 0:
            raise ValueError(f"confound {label} is not a list of trial numbers")
        numbers.append(number)
    if numbers != sorted(set(numbers)):
        raise ValueError(f"confound {label} is not ascending and distinct")
    return numbers


def _flagged_checks(block: ConfoundBlock) -> list[ConfoundCheck]:
    """Return the checks whose verdict is ``heterogeneous``, in report order.

    :param ConfoundBlock block: A validated or freshly assessed block.
    :return list[ConfoundCheck]: Flagged check names.
    """
    checks = block["checks"]
    verdicts: dict[ConfoundCheck, ConfoundVerdict] = {
        "evaluation_point": checks["evaluation_point"]["verdict"],
        "survivorship": checks["survivorship"]["verdict"],
        "tie": checks["tie"]["verdict"],
    }
    return [name for name in CONFOUND_CHECKS if verdicts[name] == "heterogeneous"]


def _describe_flagged(
    block: ConfoundBlock,
    *,
    winner_trial: int,
    trials: Sequence[optuna.trial.FrozenTrial] = (),
    untried_cells: Sequence[Mapping[str, object]] = (),
    retry_step: str | None = None,
) -> list[str]:
    """Describe each flagged check with the numbers behind it, for operator logs.

    :param ConfoundBlock block: A freshly assessed block.
    :param int winner_trial: The selected trial, one of the block's ranked trials.
    :param Sequence[optuna.trial.FrozenTrial] trials: The assessed study's trials, whose
        params name the failed configurations, defaults to none.
    :param Sequence[Mapping[str, object]] untried_cells: The untried grid cells passed to
        :func:`_assess_population`, defaults to none.
    :param str | None retry_step: The raised ``n_trials`` that retries the failed
        configurations (:func:`phasesweep.engine.study_policy._raised_target_step`), defaults
        to None.
    :return list[str]: One sentence per flagged check; empty when none flagged.
    """
    checks = block["checks"]
    lines: list[str] = []
    evaluation = checks["evaluation_point"]
    if evaluation["verdict"] == "heterogeneous":
        lines.append(_describe_evaluation_point(evaluation["detail"], winner_trial=winner_trial))
    survivorship = checks["survivorship"]
    if survivorship["verdict"] == "heterogeneous":
        lines.append(
            _describe_survivorship(
                survivorship["detail"],
                trial_params={trial.number: trial.params for trial in trials},
                untried_cells=untried_cells,
                retry_step=retry_step,
            )
        )
    tie = checks["tie"]
    if tie["verdict"] == "heterogeneous":
        detail = tie["detail"]
        if detail["within_noise"] is None:
            tied = ", ".join(str(number) for number in detail["tied_trials"])
            lines.append(
                f"tie: trial(s) {tied} matched the winner's value exactly with different "
                "params. " + _noise_remedy()
            )
        else:
            lines.extend(_describe_group_tie(detail["tied_trials"], detail["within_noise"]))
    return lines


def _describe_evaluation_point(detail: EvaluationDetail, *, winner_trial: int) -> str:
    """Say where the winner was measured and which ranked trials fell short of the furthest.

    :param EvaluationDetail detail: Flagged evaluation-point evidence.
    :param int winner_trial: The selected trial, one of the ranked trials the evidence places.
    :return str: The warning sentence, naming the ordinary causes without picking one.
    """
    axis, groups = detail["axis"], detail["values"]
    furthest = groups[-1]["value"]
    position = {number: group["value"] for group in groups for number in group["trials"]}
    short = [
        f"trial {number} at {param_text(value)}"
        for number, value in sorted(position.items())
        if value < (1 - _POSITION_TOLERANCE) * furthest
    ]
    listed = ", ".join(short[:_LOGGED_VALUES])
    if len(short) > _LOGGED_VALUES:
        listed += f" and {len(short) - _LOGGED_VALUES} more"
    return (
        f"evaluation_point: the winner, trial {winner_trial}, was measured at {axis} "
        f"{param_text(position[winner_trial])}; {len(short)} ranked "
        f"{'trial' if len(short) == 1 else 'trials'} fell more than {_POSITION_TOLERANCE:.0%} "
        f"short of the furthest, {param_text(furthest)}: {listed}. Ordinary causes are a "
        "search space that sets training length (for example epochs or max_steps), a trainer "
        "that stops early, and a trainer that reports its best checkpoint rather than how far "
        "it trained"
    )


def _describe_survivorship(
    detail: SurvivorshipDetail,
    *,
    trial_params: Mapping[int, Mapping[str, object]],
    untried_cells: Sequence[Mapping[str, object]],
    retry_step: str | None,
) -> str:
    """Say which configurations have no result, so the winner was chosen among the rest.

    :param SurvivorshipDetail detail: Flagged survivorship evidence.
    :param Mapping[int, Mapping[str, object]] trial_params: Recorded params by trial number.
    :param Sequence[Mapping[str, object]] untried_cells: The untried grid cells.
    :param str | None retry_step: The raised ``n_trials`` that retries them, if any.
    :return str: The warning sentence.
    """
    missing = detail["failed"] + detail["untried"]
    total = detail["ranked"] + detail["infeasible"] + missing
    entries = [
        f"trial {number}{_params_text(trial_params.get(number, {}))} failed"
        for number in detail["failed_trials"]
    ] + [f"{_params_text(cell).lstrip()} was never tried" for cell in untried_cells]
    listed = ", ".join(entries[:_LOGGED_VALUES])
    if len(entries) > _LOGGED_VALUES:
        listed += f" and {len(entries) - _LOGGED_VALUES} more"
    step = "" if retry_step is None else f" {retry_step[0].upper()}{retry_step[1:]}"
    return (
        f"survivorship: {missing} of {total} configurations "
        f"{'has' if missing == 1 else 'have'} no result, so the winner was chosen among the "
        f"rest: {listed}.{step}"
    )


def _params_text(params: Mapping[str, object]) -> str:
    """Write params as the run report writes them, after a leading space.

    :param Mapping[str, object] params: Recorded params, empty for a fixed configuration.
    :return str: `` key=value`` items, or empty without params.
    """
    return "".join(f" {name}={param_text(value)}" for name, value in params.items())


def _evaluation_point_remedy(kind: str, *, axis: str = "step") -> str:
    """Say how a phase that searches training length compares relative completion instead.

    Every branch is for that phase only: when the positions differ for another reason, the
    trainer is what to fix. The axis and the metric key are extractor config, so each branch
    also says that changing them starts a new study.

    :param str kind: Objective extractor kind.
    :param str axis: Recorded evaluation axis, defaults to ``step``.
    :return str: Trainer reporting and evaluation-axis instructions.
    """
    new_study = f"Changing evaluation_axis changes the study's identity, so {FRESH_STUDY}."
    if axis == "schedule_fraction":
        return (
            "In a phase that searches training length, schedule_fraction already measures "
            "actual_steps / planned_steps, so a fraction below 1 is a trial that stopped before "
            "its own planned steps, and its trial log says why. Keep the lower fractions "
            f"rather than rescaling them. {new_study}"
        )
    if kind == "wandb":
        return (
            "For a phase that searches training length, log schedule_fraction=actual_steps / "
            "planned_steps alongside the objective in the same wandb.log() call and set "
            f"metric.extractor.evaluation_axis to schedule_fraction. {new_study}"
        )
    if kind == "mlflow":
        return (
            "For a phase that searches training length, MLflow compares the selected metric's "
            "integer step, so log the objective under a separate metric key (for example "
            "objective_schedule_fraction) at step=int(10000 * actual_steps / planned_steps), "
            "keeping native-step history under its original key; a trial that stopped early "
            "logs a lower step. Set metric.extractor.metric_key to that key and select: last. "
            f"Changing the metric key changes the study's identity, so {FRESH_STUDY}."
        )
    return (
        "For a phase that searches training length, report "
        'progress={"schedule_fraction": actual_steps / planned_steps} through report_objective '
        "and set metric.extractor.evaluation_axis to schedule_fraction on the json_envelope "
        f"metric. {new_study}"
    )


def _noise_remedy() -> str:
    """Return guidance for an exact tie between configurations measured once each.

    :return str: Configuration-comparison guidance shared by winner output and MCP results.
    """
    return (
        "The tied configurations scored identically, so draw order chose the winner that "
        "later phases inherit. First compare more distinct values and inspect trends across "
        "neighboring configurations. Use phases[].repeats for a few discrete finalists or an "
        "explicit noise check."
    )


def _describe_group_tie(tied: Sequence[int], noisy: Sequence[int]) -> list[str]:
    """Describe a flagged tie between repeat groups, each named by its last member.

    :param Sequence[int] tied: Groups that matched the winning group's aggregate exactly.
    :param Sequence[int] noisy: Groups within noise of it, exact ties included.
    :return list[str]: A sentence for exact ties and one for the other groups within noise.
    """
    lines: list[str] = []
    if tied:
        lines.append(
            f"tie: the repeat group(s) ending at trial(s) {', '.join(map(str, tied))} matched "
            "the winning group's aggregate exactly with different params, and the group "
            "with the lowest draw position won"
        )
    untied = [number for number in noisy if number not in tied]
    if untied:
        lines.append(
            ("tie: " if not lines else "")
            + f"the repeat group(s) ending at trial(s) {', '.join(map(str, untied))} are "
            "within two standard errors of the winning group's aggregate, so their order "
            "may be seed noise"
        )
    return lines


def _confound_summary(
    block: ConfoundBlock, *, winner_trial: int, extractor_kind: str
) -> dict[str, object]:
    """Project a validated block onto the enums and numbers agents may see.

    Include the declared evaluation axis and actionable guidance alongside counts.
    The full trial evidence stays in ``winner.yaml`` for operators.

    :param ConfoundBlock block: A validated confound block.
    :param int winner_trial: Trial number of the winner the block belongs to.
    :param str extractor_kind: Objective reader that recorded the winner's provenance.
    :return dict[str, object]: The summary, one entry per check plus the ``flagged`` check names.
    """
    checks = block["checks"]
    evaluation = checks["evaluation_point"]
    if evaluation["verdict"] == "n/a":
        groups: list[PositionGroup] | None = None
        axis: str | None = None
    else:
        groups, axis = evaluation["detail"]["values"], evaluation["detail"]["axis"]
    tie = checks["tie"]
    tie_detail = None if tie["verdict"] == "n/a" else tie["detail"]
    survivorship = checks["survivorship"]
    return {
        "flagged": list(_flagged_checks(block)),
        "evaluation_point": {
            "verdict": evaluation["verdict"],
            "axis": axis,
            "remedy": (
                _evaluation_point_remedy(extractor_kind, axis=axis or "step")
                if evaluation["verdict"] == "heterogeneous"
                else None
            ),
            "distinct_values": None if groups is None else len(groups),
            "winner_value": (
                None
                if groups is None
                else next(
                    (group["value"] for group in groups if winner_trial in group["trials"]),
                    None,
                )
            ),
        },
        "survivorship": {"verdict": survivorship["verdict"], **survivorship["detail"]},
        "tie": {
            "verdict": tie["verdict"],
            "remedy": (
                _noise_remedy()
                if block["ranked_groups"] is None and tie["verdict"] == "heterogeneous"
                else None
            ),
            "tied_trials": None if tie_detail is None else len(tie_detail["tied_trials"]),
            "within_noise": (
                None
                if tie_detail is None or tie_detail["within_noise"] is None
                else len(tie_detail["within_noise"])
            ),
        },
    }
