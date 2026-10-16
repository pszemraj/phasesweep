"""Repeat groups: how a repeats phase groups, reduces, and records its trials.

A phase with ``repeats`` runs every configuration once per repeat value, and
selection ranks only complete groups by a reducer over their members. This
module holds the pure parts of that contract: the group key, the statistics,
the winner's ``repeats`` record and its validator, and the evidence members a
winner stands on.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, TypedDict, get_args

from phasesweep.engine.state import _validate_observed_block
from phasesweep.runtime.json import JsonObject, json_object, nonempty_text

if TYPE_CHECKING:
    from optuna.trial import FrozenTrial

    from phasesweep.config.models import Phase
    from phasesweep.config.search import Repeats

ReduceName = Literal["mean", "median"]
"""How a group's member metrics combine into the value that ranks it."""

REDUCERS: tuple[ReduceName, ...] = get_args(ReduceName)

RepeatValue: TypeAlias = str | int | float | bool | None
"""One value of the repeat axis, such as a seed."""

# Groups whose aggregates differ by at most this many combined standard errors
# are flagged as within noise of each other.
_NOISE_STANDARD_ERRORS = 2.0

_REPEATS_KEYS = frozenset({"key", "reduce", "spread", "members"})
_SPREAD_KEYS = ("n", "mean", "sd", "min", "max")
_MEMBER_KEYS = (
    "trial_number",
    "repeat",
    "metric",
    "generation_id",
    "attempt_id",
    "objective_provenance",
    "trainer_input",
    "constraints",
    "observed",
)


class RepeatSpread(TypedDict):
    """Sample statistics of one group's member metrics."""

    n: int
    mean: float
    sd: float
    min: float
    max: float


class RepeatMember(TypedDict):
    """One trial of a winning group, with the evidence identity it was ranked on."""

    trial_number: int
    repeat: RepeatValue
    metric: float
    generation_id: str
    attempt_id: str
    objective_provenance: JsonObject
    trainer_input: JsonObject
    constraints: dict[str, float]
    observed: dict[str, float]


class RepeatsRecord(TypedDict):
    """The ``repeats`` block of a repeats-phase winner."""

    key: str
    reduce: ReduceName
    spread: RepeatSpread
    members: list[RepeatMember]


class RepeatsSummary(TypedDict):
    """String-free projection of a ``repeats`` block for agent-visible payloads."""

    reduce: ReduceName
    n: int
    mean: float
    sd: float
    min: float
    max: float
    member_trials: list[int]


@dataclass(frozen=True)
class RepeatAxis:
    """The repeat axis of one phase, as selection and the trial CSV read it.

    ``slot_size`` is the repeat count for seeded random, whose draws are keyed
    by configuration slot, and ``None`` for grid, whose cells are unique.
    """

    key: str
    values: tuple[RepeatValue, ...]
    reduce: ReduceName
    slot_size: int | None

    def index_of(self, value: object) -> int | None:
        """Return the position of a recorded repeat value on the axis.

        :param object value: The repeat param a trial recorded.
        :return int | None: Its index in ``values``, or ``None`` when it is not one of them.
        """
        return next((index for index, item in enumerate(self.values) if item == value), None)


@dataclass(frozen=True)
class RankedGroup:
    """One complete repeat group as selection ranked it."""

    numbers: tuple[int, ...]
    params: Mapping[str, object]
    aggregate: float
    spread: RepeatSpread

    @property
    def representative(self) -> int:
        """Return the highest member trial number, which names the group.

        :return int: The last member's trial number.
        """
        return self.numbers[-1]


@dataclass(frozen=True)
class EvidenceMember:
    """One trial whose evidence backs a winner's published result."""

    trial_number: int
    generation_id: str
    attempt_id: str
    metric: float
    objective_provenance: Mapping[str, object]
    trainer_input: Mapping[str, object]
    constraints: Mapping[str, float]


class EvidenceSource(Protocol):
    """A selected trial or winner whose evidence members can be enumerated."""

    @property
    def trial_number(self) -> int:
        """Trial number of the scalar source trial."""
        ...

    @property
    def generation_id(self) -> str | None:
        """Generation that ran the scalar source trial."""
        ...

    @property
    def attempt_id(self) -> str | None:
        """Attempt that ran the scalar source trial."""
        ...

    @property
    def metric(self) -> float:
        """Published metric value."""
        ...

    @property
    def objective_provenance(self) -> Mapping[str, object] | None:
        """Frozen objective provenance of the scalar source trial."""
        ...

    @property
    def trainer_input(self) -> Mapping[str, object] | None:
        """Generated-input identity of the scalar source trial."""
        ...

    @property
    def constraints(self) -> Mapping[str, float]:
        """Constraint readings of the scalar source trial."""
        ...

    @property
    def repeats(self) -> RepeatsRecord | None:
        """The ``repeats`` block, or ``None`` for a phase without repeats."""
        ...


def repeat_axis(phase: Phase) -> RepeatAxis | None:
    """Return a phase's repeat axis, or ``None`` for a phase without repeats.

    :return RepeatAxis | None: The axis, slotted by repeat count under seeded random.
    """
    repeats = phase.repeats
    if repeats is None:
        return None
    return RepeatAxis(
        key=repeats.key,
        values=tuple(repeats.values),
        reduce=repeats.reduce,
        slot_size=len(repeats.values) if phase.sampler.type == "random" else None,
    )


def group_key(position: int, params: Mapping[str, object], axis: RepeatAxis) -> str:
    """Return the key that groups a trial with the other repeats of its configuration.

    The key is the params without the repeat key, plus the configuration slot
    for seeded random, so two slots that drew the same params stay separate
    groups and a complete group never gains a member. The slot comes from the
    trial's draw position (:func:`phasesweep.engine.optuna._draw_positions`),
    so a rerun lands in its source's slot.

    :param int position: The trial's draw position.
    :param Mapping[str, object] params: The trial's recorded params.
    :param RepeatAxis axis: The phase's repeat axis.
    :return str: Canonical JSON of the slot and the remaining params.
    """
    slot = None if axis.slot_size is None else position // axis.slot_size
    rest = {name: value for name, value in params.items() if name != axis.key}
    return json.dumps([slot, rest], sort_keys=True, separators=(",", ":"))


def group_labels(
    trials: Iterable[tuple[int, int, Mapping[str, object]]], axis: RepeatAxis
) -> dict[int, int]:
    """Label each trial of a repeats phase with the highest trial number of its group so far.

    The groups are the ones selection ranks, by :func:`group_key`. An incomplete
    group's label moves as more members are allocated. A trial without the
    repeat key, which was stopped before sampling it, belongs to no group and
    gets no label.

    :param Iterable[tuple[int, int, Mapping[str, object]]] trials: ``(trial_number,
        draw_position, params)`` of every trial of the phase.
    :param RepeatAxis axis: The phase's repeat axis.
    :return dict[int, int]: Group label by trial number, for the trials that have one.
    """
    keyed = {
        number: group_key(position, params, axis)
        for number, position, params in trials
        if axis.key in params
    }
    last: dict[str, int] = {}
    for number, key in keyed.items():
        last[key] = max(last.get(key, number), number)
    return {number: last[key] for number, key in keyed.items()}


def group_labels_for(trials: Sequence[FrozenTrial], axis: RepeatAxis | None) -> dict[int, int]:
    """Label recorded trials by their repeat groups, keeping reruns in their source slots.

    :param Sequence[FrozenTrial] trials: Trials from one phase study.
    :param RepeatAxis | None axis: Configured repeat axis, or ``None`` without repeats.
    :return dict[int, int]: Group label by trial number; empty without a repeat axis.
    """
    from phasesweep.engine.optuna import _draw_positions

    positions = _draw_positions(trials)
    return (
        {}
        if axis is None
        else group_labels(((t.number, positions[t.number], t.params) for t in trials), axis)
    )


def reduce_values(values: Sequence[float], reduce: ReduceName) -> float:
    """Combine member metrics into the value a group ranks by.

    :param Sequence[float] values: Member metrics in ascending trial-number order.
    :param ReduceName reduce: The configured reducer.
    :return float: Their mean or median.
    """
    if reduce == "mean":
        return statistics.fmean(values)
    return float(statistics.median(values))


def spread(values: Sequence[float]) -> RepeatSpread:
    """Summarize member metrics with their count, mean, sample SD, and range.

    :param Sequence[float] values: At least two member metrics, in ascending trial-number order.
    :return RepeatSpread: The statistics.
    """
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def ranked_group(
    members: Sequence[tuple[int, float]],
    params: Mapping[str, object],
    reduce: ReduceName,
) -> RankedGroup:
    """Build a ranked group from its members' trial numbers and metrics.

    :param Sequence[tuple[int, float]] members: ``(trial_number, metric)`` pairs.
    :param Mapping[str, object] params: The configuration's params without the repeat key.
    :param ReduceName reduce: The configured reducer.
    :return RankedGroup: The group with its aggregate and spread.
    """
    ordered = sorted(members)
    values = [value for _, value in ordered]
    return RankedGroup(
        numbers=tuple(number for number, _ in ordered),
        params=dict(params),
        aggregate=reduce_values(values, reduce),
        spread=spread(values),
    )


def within_noise(winner: RankedGroup, other: RankedGroup) -> bool:
    """Say whether two groups' aggregates differ by no more than their noise.

    Two combined standard errors is an advisory heuristic, not a significance
    test, and it is applied to the median reducer as well.

    :param RankedGroup winner: The winning group.
    :param RankedGroup other: Another ranked group.
    :return bool: Whether the aggregates differ by at most two combined standard errors.
    """
    first, second = winner.spread, other.spread
    error = math.sqrt(first["sd"] ** 2 / first["n"] + second["sd"] ** 2 / second["n"])
    return abs(winner.aggregate - other.aggregate) <= _NOISE_STANDARD_ERRORS * error


def observed_means(members: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average each report-only value over the members that recorded it.

    :param Sequence[Mapping[str, float]] members: Each member's observed values.
    :return dict[str, float]: Mean value by observation name, sorted by name.
    """
    names = sorted({name for observed in members for name in observed})
    return {
        name: statistics.fmean(observed[name] for observed in members if name in observed)
        for name in names
    }


def _finite_number(value: object, *, label: str) -> float:
    """Validate a finite JSON number.

    :param object value: Parsed value.
    :param str label: Field name for the diagnostic.
    :raises ValueError: The value is not a finite number.
    :return float: The value as a float.
    """
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"repeats {label} is not a finite number")
    return float(value)


def _repeat_value(value: object) -> RepeatValue:
    """Validate one recorded repeat value.

    :param object value: Parsed value.
    :raises ValueError: The value is not a JSON scalar, or is a non-finite float.
    :return RepeatValue: The value.
    """
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("repeats member repeat value is not a finite JSON scalar")


def _validate_member(data: object) -> RepeatMember:
    """Validate one member record of a ``repeats`` block.

    :param object data: Parsed member record.
    :raises ValueError: A key is missing or extra, or a field is malformed.
    :return RepeatMember: The member, rebuilt from plain values.
    """
    if not isinstance(data, Mapping) or set(data) != set(_MEMBER_KEYS):
        raise ValueError(f"repeats member must have exactly {', '.join(_MEMBER_KEYS)}")
    number = data["trial_number"]
    if type(number) is not int or number < 0:
        raise ValueError("repeats member trial_number is not a trial number")
    constraints = data["constraints"]
    if not isinstance(constraints, Mapping):
        raise ValueError("repeats member constraints is not a mapping")
    try:
        provenance = json_object(data["objective_provenance"])
        trainer_input = json_object(data["trainer_input"])
        observed = _validate_observed_block(data["observed"])
    except ValueError as exc:
        raise ValueError(f"repeats member {number} is malformed: {exc}") from exc
    return {
        "trial_number": number,
        "repeat": _repeat_value(data["repeat"]),
        "metric": _finite_number(data["metric"], label="member metric"),
        "generation_id": nonempty_text(data["generation_id"], label="repeats member generation_id"),
        "attempt_id": nonempty_text(data["attempt_id"], label="repeats member attempt_id"),
        "objective_provenance": provenance,
        "trainer_input": trainer_input,
        "constraints": {
            nonempty_text(name, label="repeats member constraint name"): _finite_number(
                value, label="member constraint value"
            )
            for name, value in constraints.items()
        },
        "observed": observed,
    }


def _representative_fields(member: RepeatMember) -> dict[str, object]:
    """Return the scalar winner fields that describe a group's highest-numbered member.

    :param RepeatMember member: The group's last member.
    :return dict[str, object]: Winner field name to the value the member requires.
    """
    return {
        "trial_number": member["trial_number"],
        "generation_id": member["generation_id"],
        "attempt_id": member["attempt_id"],
        "objective_provenance": member["objective_provenance"],
        "trainer_input": member["trainer_input"],
        "constraints": member["constraints"],
    }


def _validate_spread(data: object) -> RepeatSpread:
    """Validate a recorded ``spread`` mapping.

    :param object data: Parsed spread.
    :raises ValueError: A key is missing or extra, or a value is malformed.
    :return RepeatSpread: The spread, rebuilt from plain values.
    """
    if not isinstance(data, Mapping) or set(data) != set(_SPREAD_KEYS):
        raise ValueError(f"repeats spread must have exactly {', '.join(_SPREAD_KEYS)}")
    count = data["n"]
    if type(count) is not int:
        raise ValueError("repeats spread n is not an integer")
    return {
        "n": count,
        "mean": _finite_number(data["mean"], label="spread mean"),
        "sd": _finite_number(data["sd"], label="spread sd"),
        "min": _finite_number(data["min"], label="spread min"),
        "max": _finite_number(data["max"], label="spread max"),
    }


def _validate_repeats_block(
    payload: Mapping[str, object],
    *,
    metric_value: float,
    ranked_groups: Sequence[Sequence[int]] | None,
) -> RepeatsRecord | None:
    """Validate a repeat-group record against its winner and ranked confound population.

    Scalar identity fields must match the highest-numbered member. Recompute the
    aggregate metric, spread, and observations, and require membership in the
    confound block's ranked groups.

    :param Mapping[str, object] payload: Decoded winner payload holding ``repeats``.
    :param float metric_value: The winner's published metric value.
    :param Sequence[Sequence[int]] | None ranked_groups: ``ranked_groups`` of the
        winner's validated confound block.
    :raises ValueError: The block is absent or malformed, its members are not
        ascending with distinct repeat values, the last member disagrees with
        the scalar winner fields, a recomputed value differs from the recorded
        one, the repeat key appears in the winner's params or overrides, or the
        block and the confound block disagree on whether and which group ranked.
    :return RepeatsRecord | None: The validated block, or ``None`` for a phase
        without repeats.
    """
    if "repeats" not in payload:
        raise ValueError("repeats block is missing")
    data = payload["repeats"]
    if data is None:
        if ranked_groups is not None:
            raise ValueError("repeats block is null though the confound block ranked repeat groups")
        return None
    if not isinstance(data, Mapping) or set(data) != _REPEATS_KEYS:
        raise ValueError("repeats block must have exactly key, reduce, spread and members")
    key = nonempty_text(data["key"], label="repeats key")
    reduce = data["reduce"]
    if reduce not in REDUCERS:
        raise ValueError(f"repeats reduce is not one of {', '.join(REDUCERS)}")
    raw_members = data["members"]
    if not isinstance(raw_members, list) or len(raw_members) < 2:
        raise ValueError("repeats members is not a list of at least two trials")
    members = [_validate_member(member) for member in raw_members]
    numbers = [member["trial_number"] for member in members]
    if numbers != sorted(set(numbers)):
        raise ValueError("repeats members are not ascending and distinct")
    if ranked_groups is None or numbers not in [list(group) for group in ranked_groups]:
        raise ValueError("repeats members are not one of the confound block's ranked groups")
    for index, member in enumerate(members):
        if any(member["repeat"] == earlier["repeat"] for earlier in members[:index]):
            raise ValueError("repeats members repeat a repeat value")
    for field, value in _representative_fields(members[-1]).items():
        if payload.get(field) != value:
            raise ValueError(f"repeats last member disagrees with the winner's {field}")
    values = [member["metric"] for member in members]
    if reduce_values(values, reduce) != metric_value:
        raise ValueError(f"repeats {reduce} of the member metrics differs from the winner's metric")
    recomputed = spread(values)
    if _validate_spread(data["spread"]) != recomputed:
        raise ValueError("repeats spread differs from what its members recompute to")
    if payload.get("observed") != observed_means([member["observed"] for member in members]):
        raise ValueError("repeats observed means differ from the winner's observed values")
    for field in ("params", "effective_overrides"):
        block = payload.get(field)
        if isinstance(block, Mapping) and key in block:
            raise ValueError(f"repeats key {key!r} appears in the winner's {field}")
    return {"key": key, "reduce": reduce, "spread": recomputed, "members": members}


def _scalar_identity(value: RepeatValue) -> tuple[type, RepeatValue]:
    """Return a repeat value paired with its type, so ``1``, ``1.0`` and ``True`` stay distinct.

    :param RepeatValue value: One repeat value.
    :return tuple[type, RepeatValue]: The value's type and the value.
    """
    return (type(value), value)


def _check_repeats_axis(record: RepeatsRecord | None, configured: Repeats | None) -> None:
    """Match a validated group to its configured repeat key, reducer, and values.

    Require every configured repeat value exactly once.

    :param RepeatsRecord | None record: The block :func:`_validate_repeats_block` returned.
    :param Repeats | None configured: The phase's configured ``repeats``, or ``None``.
    :raises ValueError: The winner has a block and the phase configures none, or the
        reverse, or the block's key, reducer, member count or repeat values differ
        from the configured ones.
    """
    if record is None and configured is None:
        return
    if configured is None:
        raise ValueError("the winner records repeats but the phase configures none")
    if record is None:
        raise ValueError("the winner records no repeats but the phase configures them")
    if record["key"] != configured.key:
        raise ValueError(
            f"repeats key {record['key']!r} is not the configured repeats.key {configured.key!r}"
        )
    if record["reduce"] != configured.reduce:
        raise ValueError(
            f"repeats reduce {record['reduce']!r} is not the configured "
            f"repeats.reduce {configured.reduce!r}"
        )
    recorded = [_scalar_identity(member["repeat"]) for member in record["members"]]
    if len(recorded) != len(configured.values):
        raise ValueError(
            f"repeats has {len(recorded)} members but repeats.values configures "
            f"{len(configured.values)}"
        )
    if Counter(recorded) != Counter(_scalar_identity(value) for value in configured.values):
        raise ValueError("repeats member values are not the configured repeats.values")


def _repeats_summary(record: RepeatsRecord) -> RepeatsSummary:
    """Project a validated block onto the enums and numbers agents may see.

    The repeat key and its values are config-authored, so the projection keeps
    only the reducer, the spread, and the member trial numbers.

    :param RepeatsRecord record: A validated ``repeats`` block.
    :return RepeatsSummary: The string-free summary.
    """
    stats = record["spread"]
    return {
        "reduce": record["reduce"],
        "n": stats["n"],
        "mean": stats["mean"],
        "sd": stats["sd"],
        "min": stats["min"],
        "max": stats["max"],
        "member_trials": [member["trial_number"] for member in record["members"]],
    }


def evidence_members(source: EvidenceSource) -> list[EvidenceMember]:
    """List the trials whose evidence backs a winner's published result.

    One path for every winner: a phase without repeats yields its single
    scalar source trial, and a repeats phase yields every member of the group.

    :param EvidenceSource source: A selected trial or a winner.
    :raises ValueError: A winner without repeats carries no execution identity
        or provenance, which only the never-persisted dry-run placeholder lacks.
    :return list[EvidenceMember]: The members in ascending trial-number order.
    """
    record = source.repeats
    if record is not None:
        return [
            EvidenceMember(
                trial_number=member["trial_number"],
                generation_id=member["generation_id"],
                attempt_id=member["attempt_id"],
                metric=member["metric"],
                objective_provenance=member["objective_provenance"],
                trainer_input=member["trainer_input"],
                constraints=member["constraints"],
            )
            for member in record["members"]
        ]
    generation_id, attempt_id = source.generation_id, source.attempt_id
    provenance, trainer_input = source.objective_provenance, source.trainer_input
    if generation_id is None or attempt_id is None or provenance is None or trainer_input is None:
        raise ValueError("a winner without execution identity has no evidence to verify")
    return [
        EvidenceMember(
            trial_number=source.trial_number,
            generation_id=generation_id,
            attempt_id=attempt_id,
            metric=source.metric,
            objective_provenance=provenance,
            trainer_input=trainer_input,
            constraints=source.constraints,
        )
    ]
