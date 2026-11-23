"""Advisory confound verdicts over a phase's ranked population.

Without repeats, ranked units are individual trials. With repeats, they are complete groups,
whose means are also compared against a noise band pooled over every ranked group
(:func:`phasesweep.engine.repeats.noise_band`). Survivorship counts configurations instead: a
trial's params, or a repeat group.

Verdicts are frozen at selection, carried unchanged with the winner, and excluded from
fingerprints. Readers validate the block's shape and never re-judge a verdict. They never
block selection, publication, or top-ups.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Literal, TypeAlias, TypedDict, get_args

import optuna

from phasesweep.engine.optuna import _trial_count_state
from phasesweep.engine.provenance import CodeRevision
from phasesweep.engine.repeats import (
    RankedGroup,
    RepeatAxis,
    configuration_slot,
    configuration_slots,
    noise_band,
)
from phasesweep.engine.state import GENERATION_ID_ATTR, _unreplaced_attempts
from phasesweep.errors import STORAGE_NULL_KEEPS_NOTHING, rename_step
from phasesweep.evidence.models import (
    POSITIONED_EXTRACTOR_KINDS,
    best_value_position_reason,
    no_position_reason,
)
from phasesweep.runtime.json import finite_nonnegative
from phasesweep.runtime.terminal import objective_text, param_text, plural

ConfoundVerdict = Literal["ok", "heterogeneous", "n/a"]
"""Outcome of one check. ``ok``: the check ran and found nothing.
``heterogeneous``: the check ran and flagged the population, with evidence.
``n/a``: the check could not run; it means *not checked*, never *clean*."""

CONFOUND_VERDICTS: tuple[ConfoundVerdict, ...] = get_args(ConfoundVerdict)

CheckedVerdict = Literal["ok", "heterogeneous"]
"""The verdict of a check that ran."""

ConfoundCheck = Literal["evaluation_point", "survivorship", "tie", "code_revision"]
"""One comparability dimension, in the order every surface reports them."""

CONFOUND_CHECKS: tuple[ConfoundCheck, ...] = get_args(ConfoundCheck)

CONFOUND_SCHEMA_VERSION = 6


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
    """Units that tied the winner exactly, and repeat groups within noise of it.

    ``pooled_sd`` and ``df`` are the noise band the groups' means were compared with
    (:class:`phasesweep.engine.repeats.NoiseBand`). All three noise fields are ``None`` when
    no band was judged: for single trials, and for groups ranked by median.
    """

    tied_trials: list[int]
    within_noise: list[int] | None
    pooled_sd: float | None
    df: int | None


class GenerationRevision(TypedDict):
    """The trainer revision one generation of ranked trials recorded.

    ``record`` is ``recorded`` with the revision in ``commit`` and ``dirty``, ``unknown`` when
    the generation found no git work tree to read, or ``unreadable`` when its record could
    not be read; ``commit`` and ``dirty`` are ``None`` for the last two.
    """

    generation_id: str
    record: Literal["recorded", "unknown", "unreadable"]
    commit: str | None
    dirty: bool | None
    trials: list[int]


class CodeRevisionDetail(TypedDict):
    """Every generation the ranked trials ran in, ordered by its first ranked trial."""

    generations: list[GenerationRevision]


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


class CodeRevisionChecked(TypedDict):
    """A checked ``code_revision`` record."""

    verdict: CheckedVerdict
    detail: CodeRevisionDetail


EvaluationCheck: TypeAlias = EvaluationChecked | NotChecked
TieCheck: TypeAlias = TieChecked | NotChecked
CodeRevisionCheck: TypeAlias = CodeRevisionChecked | NotChecked

RevisionRead: TypeAlias = CodeRevision | Literal["unknown", "unreadable"]
"""What selection read of one generation's recorded trainer revision."""


class ConfoundChecks(TypedDict):
    """The four check records, keyed by check name."""

    evaluation_point: EvaluationCheck
    survivorship: SurvivorshipCheck
    tie: TieCheck
    code_revision: CodeRevisionCheck


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
# Trials, configurations or generations a log line spells out before counting the rest.
_LOGGED_VALUES = 5


def _assess_population(
    trials: Sequence[optuna.trial.FrozenTrial],
    ranked: Sequence[optuna.trial.FrozenTrial],
    winner: optuna.trial.FrozenTrial,
    *,
    provenance: Mapping[int, Mapping[str, object]],
    revisions: Mapping[str, RevisionRead],
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
    :param Mapping[str, RevisionRead] revisions: What selection read of the trainer revision each
        ranked generation recorded, keyed by generation id; a generation without an entry
        counts as unreadable.
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
            "code_revision": _code_revision_check(ordered, revisions),
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
        return _not_checked(best_value_position_reason(best_value_select))
    unreadable = [str(trial.number) for trial in ranked if trial.number not in provenance]
    if unreadable:
        return _not_checked(
            f"trial(s) {', '.join(unreadable)} have unreadable objective provenance"
        )
    records = [(trial.number, provenance[trial.number]) for trial in ranked]
    kinds = sorted({_extractor_kind(record) for _, record in records})
    if len(kinds) != 1 or kinds[0] not in POSITIONED_EXTRACTOR_KINDS:
        # docs/runtime.md ("Confound verdicts") says which extractors record a position.
        return _not_checked(no_position_reason(kinds))
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
    return axis, finite_nonnegative(progress.get("value"))


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
        "detail": {"tied_trials": tied, "within_noise": None, "pooled_sd": None, "df": None},
    }


def _group_tie(
    groups: Sequence[RankedGroup],
    winner: optuna.trial.FrozenTrial,
) -> TieCheck:
    """Check whether the winning repeat group stands clear of the other groups.

    ``tied_trials`` names the groups whose aggregate equals the winner's
    exactly; the lowest member draw position broke that tie. ``within_noise``
    names the groups whose mean lies within the noise band pooled over every
    ranked group (:func:`phasesweep.engine.repeats.noise_band`), exact ties
    included. Each group is named by its last member. A group with the winner's
    params is another slot of the same configuration and is skipped, so with no other
    configuration ranked the check is ``n/a``.

    Groups ranked by median have no band. An exact tie still flags the check; without one
    the check is ``n/a``, because seed noise was not judged and ``ok`` would claim it was.

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
    if not others:
        # Every other group is another slot of the winner's configuration: nothing to compare.
        return _not_checked("only one configuration was ranked")
    tied = sorted(group.representative for group in others if group.aggregate == best.aggregate)
    band = noise_band(groups)
    if band is None:
        if not tied:
            return _not_checked(
                "reduce: median has no noise band, and no group matched the winner's median exactly"
            )
        return {
            "verdict": "heterogeneous",
            "detail": {"tied_trials": tied, "within_noise": None, "pooled_sd": None, "df": None},
        }
    noisy = sorted(group.representative for group in others if band.within(best, group))
    return {
        "verdict": "heterogeneous" if tied or noisy else "ok",
        "detail": {
            "tied_trials": tied,
            "within_noise": noisy,
            "pooled_sd": band.pooled_sd,
            "df": band.df,
        },
    }


def _code_revision_check(
    ranked: Sequence[optuna.trial.FrozenTrial], revisions: Mapping[str, RevisionRead]
) -> CodeRevisionCheck:
    """Compare the trainer revisions of the generations the ranked trials ran in.

    One generation is one recorded code state, so a population from one generation is never
    flagged. Across generations the check flags more than one commit, any dirty tree (two
    dirty trees at one commit may differ and the record cannot show them equal), and any
    generation whose revision is unknown or unreadable. A changed commit stays allowed on
    resume; this only says what the winner was chosen among.

    :param Sequence[optuna.trial.FrozenTrial] ranked: Ranked trials by number; each carries
        its generation id, which selection requires.
    :param Mapping[str, RevisionRead] revisions: The revision read for each generation.
    :return CodeRevisionCheck: The check record.
    """
    if len(ranked) < 2:
        return _not_checked("only one trial was ranked")
    members: dict[str, list[int]] = {}
    for trial in ranked:
        members.setdefault(str(trial.user_attrs[GENERATION_ID_ATTR]), []).append(trial.number)
    generations = [
        _generation_revision(generation_id, revisions.get(generation_id, "unreadable"), numbers)
        for generation_id, numbers in members.items()
    ]
    commits = {entry["commit"] for entry in generations if entry["commit"] is not None}
    # A dirty flag of None is a generation with no revision to compare.
    flagged = len(generations) > 1 and (
        len(commits) > 1 or any(entry["dirty"] is not False for entry in generations)
    )
    return {
        "verdict": "heterogeneous" if flagged else "ok",
        "detail": {"generations": generations},
    }


def _generation_revision(
    generation_id: str, read: RevisionRead, trials: list[int]
) -> GenerationRevision:
    """Record one generation's revision beside the ranked trials that ran in it.

    :param str generation_id: The generation.
    :param RevisionRead read: What selection read of its recorded revision.
    :param list[int] trials: Ranked trials of the generation, ascending.
    :return GenerationRevision: The generation's entry.
    """
    if isinstance(read, str):
        return {
            "generation_id": generation_id,
            "record": read,
            "commit": None,
            "dirty": None,
            "trials": trials,
        }
    return {
        "generation_id": generation_id,
        "record": "recorded",
        "commit": read["commit"],
        "dirty": read["dirty"],
        "trials": trials,
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
    revision = _checked_record("code_revision", checks["code_revision"])
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
            "code_revision": (
                _not_checked(_reason(checks["code_revision"]))
                if revision is None
                else {
                    "verdict": revision[0],
                    "detail": _validate_code_revision_detail(revision[1]),
                }
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
        value = finite_nonnegative(group["value"])
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
    """Validate tie evidence: exact ties with other params, groups within noise, and the band.

    :param object detail: Parsed evidence.
    :raises ValueError: A field is missing or extra, a list is malformed, ``pooled_sd`` is
        not null or a finite non-negative number, or ``df`` is not null or a positive int.
    :return TieDetail: The validated evidence.
    """
    fields = set(TieDetail.__annotations__)
    if not isinstance(detail, Mapping) or set(detail) != fields:
        raise ValueError(f"confound tie evidence must hold {', '.join(sorted(fields))}")
    noisy, pooled_sd, df = detail["within_noise"], detail["pooled_sd"], detail["df"]
    if pooled_sd is not None and (
        isinstance(pooled_sd, bool)
        or not isinstance(pooled_sd, int | float)
        or not math.isfinite(pooled_sd)
        or pooled_sd < 0
    ):
        raise ValueError("confound tie pooled_sd is not a finite non-negative number")
    if df is not None and (type(df) is not int or df < 1):
        raise ValueError("confound tie df is not a positive count")
    return {
        "tied_trials": _trial_numbers(detail["tied_trials"], label="tie tied_trials"),
        "within_noise": (
            None if noisy is None else _trial_numbers(noisy, label="tie within_noise")
        ),
        "pooled_sd": None if pooled_sd is None else float(pooled_sd),
        "df": df,
    }


def _validate_code_revision_detail(detail: object) -> CodeRevisionDetail:
    """Validate code-revision evidence: each generation's revision and its ranked trials.

    :param object detail: Parsed evidence.
    :raises ValueError: The generations are missing or malformed, or an entry's revision
        fields disagree with its record state.
    :return CodeRevisionDetail: The validated evidence.
    """
    if not isinstance(detail, Mapping) or set(detail) != {"generations"}:
        raise ValueError("confound code_revision evidence must hold generations")
    entries = detail["generations"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("confound code_revision has no generations")
    fields = set(GenerationRevision.__annotations__)
    generations: list[GenerationRevision] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != fields:
            raise ValueError("confound code_revision generation is malformed")
        generation_id, record = entry["generation_id"], entry["record"]
        commit, dirty = entry["commit"], entry["dirty"]
        trials = _trial_numbers(entry["trials"], label="code_revision trials")
        if not isinstance(generation_id, str) or not generation_id or not trials:
            raise ValueError("confound code_revision generation is malformed")
        if record == "recorded" and isinstance(commit, str) and commit and isinstance(dirty, bool):
            generations.append(
                {
                    "generation_id": generation_id,
                    "record": "recorded",
                    "commit": commit,
                    "dirty": dirty,
                    "trials": trials,
                }
            )
        elif record in ("unknown", "unreadable") and commit is None and dirty is None:
            generations.append(
                {
                    "generation_id": generation_id,
                    "record": record,
                    "commit": None,
                    "dirty": None,
                    "trials": trials,
                }
            )
        else:
            raise ValueError("confound code_revision generation record is inconsistent")
    return {"generations": generations}


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
        "code_revision": checks["code_revision"]["verdict"],
    }
    return [name for name in CONFOUND_CHECKS if verdicts[name] == "heterogeneous"]


def _describe_flagged(
    block: ConfoundBlock,
    *,
    winner_trial: int,
    trials: Sequence[optuna.trial.FrozenTrial] = (),
    untried_cells: Sequence[Mapping[str, object]] = (),
    noise_gaps: Mapping[int, tuple[float, float]] | None = None,
) -> list[str]:
    """Describe each flagged check with the numbers behind it, for operator logs.

    Each sentence states what the check found; ``show-winners`` and the MCP results carry each
    flagged check's remedy (:func:`_confound_summary`).

    :param ConfoundBlock block: A freshly assessed block.
    :param int winner_trial: The selected trial, one of the block's ranked trials.
    :param Sequence[optuna.trial.FrozenTrial] trials: The assessed study's trials, whose
        params name the failed configurations, defaults to none.
    :param Sequence[Mapping[str, object]] untried_cells: The untried grid cells passed to
        :func:`_assess_population`, defaults to none.
    :param Mapping[int, tuple[float, float]] | None noise_gaps: The gap and threshold of each
        group within noise of the winning group, by the group's last member
        (:func:`phasesweep.engine.selection.tie_group_comparisons`), which a repeats phase's
        flagged tie needs; defaults to None.
    :raises ValueError: A repeats phase's tie flags groups within noise without ``noise_gaps``
        for them.
    :return list[str]: One sentence per flagged check, led by the check's name and without a
        final period; empty when none flagged.
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
            )
        )
    tie = checks["tie"]
    if tie["verdict"] == "heterogeneous":
        detail = tie["detail"]
        groups = block["ranked_groups"]
        if groups is None:
            lines.append(
                f"tie: {_numbered('trial', detail['tied_trials'])} matched the winner's value "
                "exactly with different params"
            )
        else:
            lines.append(_describe_group_tie(detail, groups=len(groups), noise_gaps=noise_gaps))
    revision = checks["code_revision"]
    if revision["verdict"] == "heterogeneous":
        lines.append(_describe_code_revision(revision["detail"]))
    return lines


def _describe_evaluation_point(detail: EvaluationDetail, *, winner_trial: int) -> str:
    """Say where the winner was measured and which ranked trials fell short of the furthest.

    The ordinary causes are in docs/runtime.md ("Confound verdicts").

    :param EvaluationDetail detail: Flagged evaluation-point evidence.
    :param int winner_trial: The selected trial, one of the ranked trials the evidence places.
    :return str: The warning sentence.
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
        f"{param_text(position[winner_trial])}; {plural(len(short), 'ranked trial')} fell more "
        f"than {_POSITION_TOLERANCE:.0%} short of the furthest, {param_text(furthest)}: {listed}"
    )


def _describe_code_revision(detail: CodeRevisionDetail) -> str:
    """Say which trainer revisions the ranked trials ran under, generation by generation.

    :param CodeRevisionDetail detail: Flagged code-revision evidence.
    :return str: The warning sentence.
    """
    generations = detail["generations"]
    listed = ", ".join(_generation_text(entry) for entry in generations[:_LOGGED_VALUES])
    if len(generations) > _LOGGED_VALUES:
        listed += f" and {len(generations) - _LOGGED_VALUES} more"
    return (
        f"code_revision: ranked trials span {plural(len(generations), 'generation')} whose "
        f"trainer code may differ: {listed}"
    )


def _generation_text(entry: GenerationRevision) -> str:
    """Write one generation's revision and its ranked trials, as the warning lists it.

    :param GenerationRevision entry: The generation's evidence.
    :return str: Such as ``1a2b3c4d5e6f dirty (trials 3, 4)``, with at most five trials named.
    """
    commit = entry["commit"]
    if commit is None:
        label = entry["record"]
    else:
        label = f"{commit[:12]}{' dirty' if entry['dirty'] else ''}"
    trials = entry["trials"]
    named = _numbered("trial", trials[:_LOGGED_VALUES])
    if len(trials) > _LOGGED_VALUES:
        named += f" and {len(trials) - _LOGGED_VALUES} more"
    return f"{label} ({named})"


def _describe_survivorship(
    detail: SurvivorshipDetail,
    *,
    trial_params: Mapping[int, Mapping[str, object]],
    untried_cells: Sequence[Mapping[str, object]],
) -> str:
    """Say which configurations have no result, so the winner was chosen among the rest.

    :param SurvivorshipDetail detail: Flagged survivorship evidence.
    :param Mapping[int, Mapping[str, object]] trial_params: Recorded params by trial number.
    :param Sequence[Mapping[str, object]] untried_cells: The untried grid cells.
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
    return (
        f"survivorship: {missing} of {plural(total, 'configuration')} "
        f"{'has' if missing == 1 else 'have'} no result ({listed})"
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
    trainer is what to fix. The axis and the metric key are extractor config, part of the
    study's identity, and the fingerprint refusal a changed one meets gives its own step;
    docs/config.md ("Training-length comparison") has the rest.

    :param str kind: Objective extractor kind.
    :param str axis: Recorded evaluation axis, defaults to ``step``.
    :return str: Trainer reporting and evaluation-axis instructions.
    """
    if axis == "schedule_fraction":
        return (
            "A schedule_fraction below 1 is a trial that stopped before its planned steps; its "
            "trial log says why."
        )
    if kind == "wandb":
        return (
            "For a phase that searches training length, log schedule_fraction=actual_steps / "
            "planned_steps with the objective in one wandb.log() call and set "
            "metric.extractor.evaluation_axis to schedule_fraction."
        )
    if kind == "mlflow":
        return (
            "For a phase that searches training length, log the objective under a separate "
            "metric key at step=int(10000 * actual_steps / planned_steps) and set "
            "metric.extractor.metric_key to it with select: last."
        )
    return (
        "For a phase that searches training length, report "
        'progress={"schedule_fraction": actual_steps / planned_steps} through report_objective '
        "and set metric.extractor.evaluation_axis to schedule_fraction."
    )


def _survivorship_remedy(
    detail: SurvivorshipDetail, *, persistent: bool, dependents: Sequence[str]
) -> str:
    """Say how to give the configurations without a result one, for every sampler.

    The summary holds no phase config or trials, so it cannot name the exact target
    :func:`phasesweep.engine.study_policy._raised_target_step` names; a raised ``n_trials`` is
    the one step that is true for every sampler, and docs/runtime.md ("Retry fixed
    configurations and grids") gives the ceiling. A dependent study with trials must get a new
    name before an upstream top-up. Without storage a rerun tries every configuration again.

    :param SurvivorshipDetail detail: Flagged survivorship evidence.
    :param bool persistent: Whether the display config keeps its trials between runs.
    :param Sequence[str] dependents: Phases that inherit the phase being topped up.
    :return str: The step.
    """
    if not persistent:
        return f"Fix the failure and rerun; {STORAGE_NULL_KEEPS_NOTHING}."
    if detail["failed"]:
        step = (
            "Fix the failure, then raise the phase's n_trials and rerun: a grid or fixed "
            "configuration retries its failed cells, and another sampler draws new configurations."
        )
    else:
        step = "Raise the phase's n_trials and rerun to run its untried cells."
    if dependents:
        step += (
            " Before rerunning, if any phase that inherits this one already has trials, "
            f"{rename_step(dependents)}."
        )
    return step


def _noise_remedy() -> str:
    """Return guidance for an exact tie between configurations measured once each.

    It leads with coverage, not seeds: docs/config.md ("Noisy objectives and small budgets")
    says when repeats are worth their budget.

    :return str: Configuration-comparison guidance shared by winner output and MCP results.
    """
    return (
        "The tied configurations scored identically, so draw order chose the winner; compare "
        "more distinct values around them."
    )


def _code_revision_remedy() -> str:
    """Say what the ranking compared when ranked trials may have run under different code.

    A changed trainer commit stays allowed on resume, so this states the fact and leaves the
    judgment to the reader: only they know whether the difference can move the objective.

    :return str: The fact and what to weigh, shared by winner output and MCP results.
    """
    return (
        "Ranked trials may have run under different trainer code; judge whether a difference "
        "can move the objective."
    )


def _numbered(noun: str, numbers: Sequence[int]) -> str:
    """Name trials or groups by number, the noun in the plural for more than one.

    :param str noun: ``trial`` or ``group``.
    :param Sequence[int] numbers: Their trial numbers, at least one.
    :return str: Such as ``group 5`` or ``groups 5, 8``.
    """
    return f"{noun}{'' if len(numbers) == 1 else 's'} {', '.join(map(str, numbers))}"


def _describe_group_tie(
    detail: TieDetail, *, groups: int, noise_gaps: Mapping[int, tuple[float, float]] | None
) -> str:
    """Describe a flagged tie between repeat groups, each named by its last member.

    Exact ties are within noise too, so with a band they are listed among those groups at a
    gap of 0. Groups ranked by median have no band and flag only on an exact tie.

    :param TieDetail detail: Flagged tie evidence of a repeats phase.
    :param int groups: How many groups were ranked, which the band pools.
    :param Mapping[int, tuple[float, float]] | None noise_gaps: Gap and threshold by group.
    :raises ValueError: A group within noise has no entry in ``noise_gaps``.
    :return str: The warning fragment.
    """
    noisy, pooled_sd, df = detail["within_noise"], detail["pooled_sd"], detail["df"]
    if noisy is None or pooled_sd is None or df is None:
        return (
            f"tie: {_numbered('group', detail['tied_trials'])} matched the winner's median exactly"
        )
    if noise_gaps is None or any(number not in noise_gaps for number in noisy):
        raise ValueError("every repeat group within noise needs its gap and threshold")
    # Complete groups all hold one member per repeat value, so every pair shares a threshold.
    threshold = noise_gaps[noisy[0]][1]
    listed = ", ".join(
        f"group {number} at gap {objective_text(noise_gaps[number][0])}"
        for number in noisy[:_LOGGED_VALUES]
    )
    if len(noisy) > _LOGGED_VALUES:
        listed += f" and {len(noisy) - _LOGGED_VALUES} more"
    return (
        f"tie: {len(noisy)} {'group is' if len(noisy) == 1 else 'groups are'} within noise of "
        f"the winner, threshold {objective_text(threshold)} (sd {objective_text(pooled_sd)} "
        f"pooled over {groups} groups, df {df}): {listed}"
    )


def _confound_summary(
    block: ConfoundBlock,
    *,
    winner_trial: int,
    extractor_kind: str,
    persistent: bool,
    dependents: Sequence[str] = (),
) -> dict[str, object]:
    """Project a validated block onto the enums and numbers agents may see.

    Include the declared evaluation axis and actionable guidance alongside counts.
    The full trial evidence stays in ``winner.yaml`` for operators.

    :param ConfoundBlock block: A validated confound block.
    :param int winner_trial: Trial number of the winner the block belongs to.
    :param str extractor_kind: Objective reader that recorded the winner's provenance.
    :param bool persistent: Whether the display config keeps its trials between runs, which
        decides the survivorship remedy.
    :param Sequence[str] dependents: Phases that inherit this one in the display config.
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
    revision = checks["code_revision"]
    generations = None if revision["verdict"] == "n/a" else revision["detail"]["generations"]
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
        "survivorship": {
            "verdict": survivorship["verdict"],
            "remedy": (
                _survivorship_remedy(
                    survivorship["detail"], persistent=persistent, dependents=dependents
                )
                if survivorship["verdict"] == "heterogeneous"
                else None
            ),
            **survivorship["detail"],
        },
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
            "pooled_sd": None if tie_detail is None else tie_detail["pooled_sd"],
            "df": None if tie_detail is None else tie_detail["df"],
        },
        "code_revision": {
            "verdict": revision["verdict"],
            "remedy": _code_revision_remedy() if revision["verdict"] == "heterogeneous" else None,
            "generations": None if generations is None else len(generations),
            "commits": (
                None
                if generations is None
                else len({entry["commit"] for entry in generations if entry["commit"]})
            ),
            "dirty": (
                None
                if generations is None
                else sum(entry["dirty"] is True for entry in generations)
            ),
            "unknown": (
                None
                if generations is None
                else sum(entry["record"] != "recorded" for entry in generations)
            ),
        },
    }
