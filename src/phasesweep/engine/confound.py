"""Advisory confound verdicts over a phase's ranked population.

Without repeats, ranked units are individual trials. With repeats, they are
complete groups; feasible members of unfinished groups count as ``incomplete_group``.
Group aggregates are also compared using combined standard errors as a noise
heuristic, not a significance test.

Verdicts are frozen at selection, carried unchanged with the winner, and excluded
from fingerprints. They never block selection, publication, or top-ups.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, Literal, get_args

import optuna

from phasesweep.engine.repeats import RankedGroup, within_noise
from phasesweep.engine.state import RERUN_OF_ATTR, _trial_interrupted
from phasesweep.evidence.models import POSITIONED_EXTRACTOR_KINDS
from phasesweep.runtime.json import progress_position

ConfoundVerdict = Literal["ok", "heterogeneous", "n/a"]
"""Outcome of one check. ``ok``: the check ran and found nothing.
``heterogeneous``: the check ran and flagged the population, with evidence.
``n/a``: the check could not run; it means *not checked*, never *clean*."""

CONFOUND_VERDICTS: tuple[ConfoundVerdict, ...] = get_args(ConfoundVerdict)

ConfoundCheck = Literal["evaluation_point", "survivorship", "tie"]
"""One comparability dimension, in the order every surface reports them."""

CONFOUND_CHECKS: tuple[ConfoundCheck, ...] = get_args(ConfoundCheck)

CONFOUND_SCHEMA_VERSION = 2

_SURVIVORSHIP_COUNTS = ("ranked", "infeasible", "incomplete_group", "failed")
# Where a finished trial that selection did not rank is counted. A COMPLETE
# trial outside the ranked set violated a constraint or the metric's bounds,
# unless it was feasible and only its repeat group was incomplete; a trial
# that failed an evidence gate or extraction is FAIL.
_EXCLUDED_BY_STATE = {
    optuna.trial.TrialState.COMPLETE: "infeasible",
    optuna.trial.TrialState.FAIL: "failed",
}
# Evaluation positions within this fraction of the furthest one count as one
# point: a fixed-token batch-size sweep stops each trial within a batch of its
# budget, while a truncated trial falls short by far more.
_POSITION_TOLERANCE = 0.01
# Distinct values a log line spells out before eliding the rest.
_LOGGED_VALUES = 5


def _assess_population(
    trials: Sequence[optuna.trial.FrozenTrial],
    ranked: Sequence[optuna.trial.FrozenTrial],
    winner: optuna.trial.FrozenTrial,
    *,
    provenance: Mapping[int, Mapping[str, Any]],
    groups: Sequence[RankedGroup] | None = None,
    incomplete_group: Collection[int] = (),
) -> dict[str, Any]:
    """Classify the population one winner selection compared.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial in the phase
        study, the population survivorship counts.
    :param Sequence[optuna.trial.FrozenTrial] ranked: Trials selection ranked:
        candidates that satisfy every constraint, and for a repeats phase every
        member of a complete group.
    :param optuna.trial.FrozenTrial winner: The selected trial, one of
        ``ranked``; for a repeats phase, the last member of the winning group.
    :param Mapping[int, Mapping[str, Any]] provenance: Validated objective
        provenance keyed by trial number, for every ranked trial whose record
        could be read.
    :param Sequence[RankedGroup] | None groups: The complete repeat groups
        selection ranked, or ``None`` for a phase without repeats.
    :param Collection[int] incomplete_group: Numbers of feasible trials whose
        repeat group was incomplete, so selection could not rank them.
    :return dict[str, Any]: The ``confound`` block persisted on the winner.
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
            "evaluation_point": _evaluation_point(ordered, provenance),
            "survivorship": _survivorship(trials, ordered, incomplete_group),
            "tie": _tie(ordered, winner)
            if ordered_groups is None
            else _group_tie(ordered_groups, winner),
        },
    }


def _evaluation_point(
    ranked: Sequence[optuna.trial.FrozenTrial],
    provenance: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare ranked objectives on their declared training-progress axis.

    JSON envelopes record the position; W&B uses its selected history position and
    MLflow the selected observation's step. Record all positions and flag those
    more than ``_POSITION_TOLERANCE`` short of the furthest. Other axes, including
    checkpoint labels, are excluded from this comparison.

    :param Sequence[optuna.trial.FrozenTrial] ranked: Ranked trials by number.
    :param Mapping[int, Mapping[str, Any]] provenance: Their readable objective
        provenance.
    :return dict[str, Any]: The check record.
    """
    if len(ranked) < 2:
        return _not_checked("only one trial was ranked")
    unreadable = [str(trial.number) for trial in ranked if trial.number not in provenance]
    if unreadable:
        return _not_checked(
            f"trial(s) {', '.join(unreadable)} have unreadable objective provenance"
        )
    records = [(trial.number, provenance[trial.number]) for trial in ranked]
    kinds = sorted({record["extractor"]["kind"] for _, record in records})
    if len(kinds) != 1 or kinds[0] not in POSITIONED_EXTRACTOR_KINDS:
        return _not_checked(
            f"{', '.join(kinds)} objectives report no evaluation position; json_envelope "
            "(report_objective), wandb (evaluation_axis) or mlflow (selected step) would bind one"
        )
    positions = [(number, record["source"]["evaluation"]["progress"]) for number, record in records]
    axes = sorted({progress["axis"] for _, progress in positions})
    if len(axes) != 1:
        return _not_checked(f"ranked objectives declare different axes ({', '.join(axes)})")
    unknown = [str(number) for number, progress in positions if progress["value"] is None]
    if unknown:
        return _not_checked(
            f"trial(s) {', '.join(unknown)} recorded no {axes[0]} position for their objective"
        )
    groups = _groups((number, progress["value"]) for number, progress in positions)
    return _checked("evaluation_point", {"axis": axes[0], "values": groups})


def _survivorship(
    trials: Sequence[optuna.trial.FrozenTrial],
    ranked: Sequence[optuna.trial.FrozenTrial],
    incomplete_group: Collection[int],
) -> dict[str, Any]:
    """Flag when failed trials are at least as numerous as ranked units.

    Failures include evidence-gate and extraction errors. Infeasible trials and
    feasible members of incomplete repeat groups are counted separately and do not
    flag; interrupted trials are excluded, and so is a failed attempt that a
    COMPLETE or INFEASIBLE retry replaced (:func:`_replaced_by_measured_retry`), since
    its cell has a measured trial. Counts are recorded for every verdict.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial in the study.
    :param Sequence[optuna.trial.FrozenTrial] ranked: Ranked trials.
    :param Collection[int] incomplete_group: Feasible trials in incomplete repeat groups.
    :return dict[str, Any]: The check record.
    """
    ranked_numbers = {trial.number for trial in ranked}
    replaced = _replaced_by_measured_retry(trials)
    counts = dict.fromkeys(_SURVIVORSHIP_COUNTS, 0)
    counts["ranked"] = len(ranked_numbers)
    for trial in trials:
        bucket = _EXCLUDED_BY_STATE.get(trial.state)
        if bucket is None or trial.number in ranked_numbers or _trial_interrupted(trial):
            continue
        if bucket == "failed" and trial.number in replaced:
            continue
        if bucket == "infeasible" and trial.number in incomplete_group:
            bucket = "incomplete_group"
        counts[bucket] += 1
    return _checked("survivorship", counts)


def _replaced_by_measured_retry(trials: Sequence[optuna.trial.FrozenTrial]) -> set[int]:
    """Return the numbers of attempts that a COMPLETE retry replaced, however many retries ago.

    A retry names the attempt it reran under :data:`~phasesweep.engine.state.RERUN_OF_ATTR`
    and runs that attempt's configuration, repeat value included, so the links trace a
    cell's attempts. Every attempt on the chain behind a COMPLETE trial is replaced; a chain
    that ends in a failure replaces none, and the failures in it stay counted.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial in the study.
    :return set[int]: Trial numbers of the replaced attempts.
    """
    by_number = {trial.number: trial for trial in trials}
    replaced: set[int] = set()
    for trial in trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        source = trial.user_attrs.get(RERUN_OF_ATTR)
        while source in by_number and source not in replaced:
            replaced.add(source)
            source = by_number[source].user_attrs.get(RERUN_OF_ATTR)
    return replaced


def _tie(
    ranked: Sequence[optuna.trial.FrozenTrial],
    winner: optuna.trial.FrozenTrial,
) -> dict[str, Any]:
    """Check whether the draw-position tiebreak chose between distinct configs.

    Selection breaks exact ties by lower draw position. Tied trials that share
    the winner's params are repeats of one configuration and change nothing;
    tied trials with other params mean the winner was chosen arbitrarily.
    Single trials carry no noise estimate, so ``within_noise`` is ``None``.

    :param Sequence[optuna.trial.FrozenTrial] ranked: Ranked trials by number.
    :param optuna.trial.FrozenTrial winner: The selected trial.
    :return dict[str, Any]: The check record.
    """
    if len(ranked) < 2:
        return _not_checked("only one trial was ranked")
    tied = [
        trial.number
        for trial in ranked
        if trial.number != winner.number
        and trial.value == winner.value
        and trial.params != winner.params
    ]
    return _checked("tie", {"tied_trials": tied, "within_noise": None})


def _group_tie(
    groups: Sequence[RankedGroup],
    winner: optuna.trial.FrozenTrial,
) -> dict[str, object]:
    """Check whether the winning repeat group stands clear of the other groups.

    ``tied_trials`` names the groups whose aggregate equals the winner's
    exactly; the lowest member draw position broke that tie. ``within_noise``
    names the groups whose aggregate lies within two combined standard errors
    of the winner's, exact ties included. Each group is named by its last
    member. A group with the winner's params is another slot of the same
    configuration and is skipped.

    :param Sequence[RankedGroup] groups: Ranked groups ordered by their members.
    :param optuna.trial.FrozenTrial winner: The last member of the winning group.
    :return dict[str, object]: The check record.
    """
    if len(groups) < 2:
        return _not_checked("only one repeat group was ranked")
    best = next(group for group in groups if group.representative == winner.number)
    others = [
        group for group in groups if group is not best and dict(group.params) != dict(best.params)
    ]
    tied = sorted(group.representative for group in others if group.aggregate == best.aggregate)
    noisy = sorted(group.representative for group in others if within_noise(best, group))
    return _checked("tie", {"tied_trials": tied, "within_noise": noisy})


def _groups(pairs: Iterable[tuple[int, Any]]) -> list[dict[str, Any]]:
    """Group trial numbers by the value each reported, ordered by value.

    :param Iterable[tuple[int, Any]] pairs: ``(trial_number, value)`` pairs in
        trial-number order; every value is of one sortable type.
    :return list[dict[str, Any]]: ``[{"value": ..., "trials": [...]}, ...]``.
    """
    by_value: dict[Any, list[int]] = {}
    for number, value in pairs:
        by_value.setdefault(value, []).append(number)
    return [{"value": value, "trials": by_value[value]} for value in sorted(by_value)]


def _checked(name: ConfoundCheck, detail: dict[str, Any]) -> dict[str, Any]:
    """Build a check record with its verdict derived from the recorded evidence.

    :param ConfoundCheck name: The check.
    :param dict[str, Any] detail: Evidence recorded whatever the verdict.
    :return dict[str, Any]: The check record.
    """
    return {"verdict": "heterogeneous" if _FLAGGED[name](detail) else "ok", "detail": detail}


def _survivorship_flagged(counts: Mapping[str, int]) -> bool:
    """Whether at least one trial failed and failures are at least as numerous as ranked ones.

    :param Mapping[str, int] counts: Survivorship counts.
    :return bool: Whether the failures could have decided what was ranked.
    """
    return counts["failed"] >= max(1, counts["ranked"])


def _evaluation_point_flagged(detail: Mapping[str, Any]) -> bool:
    """Whether a ranked position fell more than the tolerance short of the furthest.

    :param Mapping[str, Any] detail: Evaluation-point evidence, groups ascending.
    :return bool: Whether the positions span more than ``_POSITION_TOLERANCE``.
    """
    groups = detail["values"]
    lowest: float = groups[0]["value"]
    highest: float = groups[-1]["value"]
    return lowest < (1 - _POSITION_TOLERANCE) * highest


_FLAGGED: dict[ConfoundCheck, Callable[[Mapping[str, Any]], bool]] = {
    "evaluation_point": _evaluation_point_flagged,
    "survivorship": _survivorship_flagged,
    "tie": lambda detail: bool(detail["tied_trials"]) or bool(detail["within_noise"]),
}


def _not_checked(reason: str) -> dict[str, Any]:
    """Return the record of a check that could not run.

    :param str reason: Why the population could not be checked.
    :return dict[str, Any]: The check record.
    """
    return {"verdict": "n/a", "reason": reason}


def _validate_confound_block(data: object, *, winner_trial: int) -> dict[str, Any]:
    """Validate a persisted ``confound`` block and return a plain copy.

    :param object data: Parsed ``confound`` value from a ``winner.yaml``.
    :param int winner_trial: Trial number of the winner the block sits on.
    :raises ValueError: The block is absent, has another schema version, any
        check record is malformed or contradicts its evidence, the evidence
        disagrees with the ranked trials or groups, or the winner is not one of
        them or, with repeat groups, not the last member of its own group.
    :return dict[str, Any]: The validated block, rebuilt from plain values.
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
    groups = _validate_ranked_groups(data["ranked_groups"], ranked)
    if groups is not None and not any(group[-1] == winner_trial for group in groups):
        raise ValueError(
            f"confound ranked_groups has no group whose last member is the winner, "
            f"trial {winner_trial}"
        )
    checks = data["checks"]
    if not isinstance(checks, Mapping) or set(checks) != set(CONFOUND_CHECKS):
        raise ValueError(f"confound checks must be exactly {', '.join(CONFOUND_CHECKS)}")
    validated = {name: _validate_check(name, checks[name]) for name in CONFOUND_CHECKS}
    _validate_against_ranked(validated, ranked)
    _validate_against_units(validated["tie"], validated["survivorship"], ranked, groups)
    tie = validated["tie"]
    if "detail" in tie and winner_trial in [
        *tie["detail"]["tied_trials"],
        *(tie["detail"]["within_noise"] or []),
    ]:
        raise ValueError("confound tie lists the winner as tying with itself")
    return {
        "schema_version": CONFOUND_SCHEMA_VERSION,
        "ranked_trials": ranked,
        "ranked_groups": groups,
        "checks": validated,
    }


def _validate_ranked_groups(value: object, ranked: list[int]) -> list[list[int]] | None:
    """Validate the ranked repeat groups against the ranked trials.

    :param object value: Parsed ``ranked_groups``: ``None``, or a list of member lists.
    :param list[int] ranked: The block's ranked trial numbers.
    :raises ValueError: The groups are not lists of at least two ascending
        trial numbers of one size, are not ordered by their members, or do not
        cover each ranked trial exactly once.
    :return list[list[int]] | None: The groups, or ``None`` for a phase without repeats.
    """
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("confound ranked_groups is neither null nor a list of groups")
    groups = [_trial_numbers(group, label="ranked_groups member list") for group in value]
    if any(len(group) < 2 or len(group) != len(groups[0]) for group in groups):
        raise ValueError("confound ranked_groups are not groups of one size of at least two")
    if groups != sorted(groups):
        raise ValueError("confound ranked_groups are not ordered by their members")
    if sorted(number for group in groups for number in group) != ranked:
        raise ValueError("confound ranked_groups do not cover each ranked trial once")
    return groups


def _validate_against_ranked(
    checks: Mapping[ConfoundCheck, Mapping[str, Any]], ranked: list[int]
) -> None:
    """Validate that each check's evidence describes the block's ranked trials.

    :param Mapping[ConfoundCheck, Mapping[str, Any]] checks: Individually validated records.
    :param list[int] ranked: The block's ranked trial numbers.
    :raises ValueError: A single ranked trial was compared, the evaluation
        groups do not cover each ranked trial once, the survivorship count
        differs, or a tied trial was not ranked.
    """
    evaluation, survivorship, tie = (checks[name] for name in CONFOUND_CHECKS)
    if len(ranked) < 2 and evaluation["verdict"] != "n/a":
        raise ValueError("confound compared trials though only one was ranked")
    if "detail" in evaluation:
        grouped = sorted(
            number for group in evaluation["detail"]["values"] for number in group["trials"]
        )
        if grouped != ranked:
            raise ValueError("confound evaluation_point groups do not cover each ranked trial once")
    if survivorship["detail"]["ranked"] != len(ranked):
        raise ValueError("confound survivorship ranked count differs from ranked_trials")
    if "detail" in tie and not set(tie["detail"]["tied_trials"]) <= set(ranked):
        raise ValueError("confound tie lists a trial that was not ranked")


def _validate_against_units(
    tie: Mapping[str, object],
    survivorship: Mapping[str, object],
    ranked: list[int],
    groups: list[list[int]] | None,
) -> None:
    """Validate the tie and survivorship evidence against the block's ranked units.

    A unit is a repeat group when the block has groups, and a trial otherwise.

    :param Mapping[str, object] tie: The validated tie record.
    :param Mapping[str, object] survivorship: The validated survivorship record.
    :param list[int] ranked: The block's ranked trial numbers.
    :param list[list[int]] | None groups: The block's ranked repeat groups, if any.
    :raises ValueError: A single ranked unit was compared, several were not,
        ``incomplete_group`` or ``within_noise`` is recorded without groups,
        ``within_noise`` is absent with them, or a listed trial does not name
        another ranked group.
    """
    units = len(ranked) if groups is None else len(groups)
    unit_name = "trials" if groups is None else "repeat groups"
    if units < 2 and tie["verdict"] != "n/a":
        raise ValueError(f"confound compared {unit_name} though only one was ranked")
    if units >= 2 and tie["verdict"] == "n/a":
        raise ValueError(f"confound tie is n/a though several {unit_name} were ranked")
    counts = survivorship["detail"]
    if groups is None and isinstance(counts, Mapping) and counts["incomplete_group"]:
        raise ValueError("confound survivorship counts incomplete_group trials without repeats")
    detail = tie.get("detail")
    if not isinstance(detail, Mapping):
        return
    tied, noisy = detail["tied_trials"], detail["within_noise"]
    if groups is None:
        if noisy is not None:
            raise ValueError("confound tie lists within_noise groups without repeat groups")
        return
    if noisy is None:
        raise ValueError("confound tie has no within_noise list though repeat groups were ranked")
    representatives = {group[-1] for group in groups}
    if not set(tied) <= representatives or not set(noisy) <= representatives:
        raise ValueError("confound tie lists a trial that is not the last member of a ranked group")
    if not set(tied) <= set(noisy):
        raise ValueError("confound tie lists an exact tie that within_noise omits")


def _validate_check(name: ConfoundCheck, check: object) -> dict[str, Any]:
    """Validate one check record.

    :param ConfoundCheck name: Check the record belongs to.
    :param object check: Parsed record.
    :raises ValueError: The verdict is unknown, an ``n/a`` record has no
        reason, or a checked record's evidence is malformed or yields another
        verdict.
    :return dict[str, Any]: The validated record.
    """
    if not isinstance(check, Mapping) or check.get("verdict") not in CONFOUND_VERDICTS:
        raise ValueError(f"confound check {name!r} has no valid verdict")
    if check["verdict"] == "n/a":
        if name == "survivorship":
            raise ValueError("confound check 'survivorship' always runs and cannot be n/a")
        reason = check.get("reason")
        if set(check) != {"verdict", "reason"} or not isinstance(reason, str) or not reason:
            raise ValueError(f"confound check {name!r} is n/a without a reason")
        return {"verdict": "n/a", "reason": reason}
    if set(check) != {"verdict", "detail"}:
        raise ValueError(f"confound check {name!r} has no evidence")
    record = _checked(name, _DETAIL_VALIDATORS[name](check["detail"]))
    if record["verdict"] != check["verdict"]:
        raise ValueError(f"confound check {name!r} verdict contradicts its evidence")
    return record


def _validate_evaluation_detail(detail: object) -> dict[str, Any]:
    """Validate evaluation-point evidence: the axis and its position groups.

    :param object detail: Parsed evidence.
    :raises ValueError: The axis is missing, or a group is malformed.
    :return dict[str, Any]: The validated evidence.
    """
    if not isinstance(detail, Mapping) or set(detail) != {"axis", "values"}:
        raise ValueError("confound evaluation_point evidence must name its axis and values")
    axis, groups = detail["axis"], detail["values"]
    if not isinstance(axis, str) or not axis:
        raise ValueError("confound evaluation_point axis is not a nonempty string")
    if not isinstance(groups, list) or not groups:
        raise ValueError("confound evaluation_point has no value groups")
    parsed: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, Mapping) or set(group) != {"value", "trials"}:
            raise ValueError("confound evaluation_point value group is malformed")
        trials = _trial_numbers(group["trials"], label="evaluation_point trials")
        if progress_position(group["value"]) is None or not trials:
            raise ValueError("confound evaluation_point value group is malformed")
        if parsed and not parsed[-1]["value"] < group["value"]:
            raise ValueError("confound evaluation_point values are not ascending and distinct")
        parsed.append({"value": group["value"], "trials": trials})
    return {"axis": axis, "values": parsed}


def _validate_survivorship_detail(detail: object) -> dict[str, Any]:
    """Validate survivorship evidence: the outcome counts.

    :param object detail: Parsed evidence.
    :raises ValueError: A count is missing, extra, or not a non-negative int.
    :return dict[str, Any]: The validated evidence.
    """
    if not isinstance(detail, Mapping) or set(detail) != set(_SURVIVORSHIP_COUNTS):
        raise ValueError(
            f"confound survivorship evidence must count {', '.join(_SURVIVORSHIP_COUNTS)}"
        )
    for key in _SURVIVORSHIP_COUNTS:
        if type(detail[key]) is not int or detail[key] < 0:
            raise ValueError(f"confound survivorship {key} is not a non-negative count")
    return {key: detail[key] for key in _SURVIVORSHIP_COUNTS}


def _validate_tie_detail(detail: object) -> dict[str, Any]:
    """Validate tie evidence: exact ties with other params, and groups within noise.

    :param object detail: Parsed evidence.
    :raises ValueError: A list is missing or malformed.
    :return dict[str, Any]: The validated evidence.
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


_DETAIL_VALIDATORS: dict[ConfoundCheck, Callable[[object], dict[str, Any]]] = {
    "evaluation_point": _validate_evaluation_detail,
    "survivorship": _validate_survivorship_detail,
    "tie": _validate_tie_detail,
}


def _trial_numbers(value: object, *, label: str) -> list[int]:
    """Validate a list of distinct trial numbers in ascending order.

    :param object value: Parsed list.
    :param str label: Field name for the diagnostic.
    :raises ValueError: The value is not such a list.
    :return list[int]: The trial numbers.
    """
    if not isinstance(value, list) or any(
        type(number) is not int or number < 0 for number in value
    ):
        raise ValueError(f"confound {label} is not a list of trial numbers")
    if value != sorted(set(value)):
        raise ValueError(f"confound {label} is not ascending and distinct")
    return list(value)


def _flagged_checks(block: Mapping[str, Any]) -> list[ConfoundCheck]:
    """Return the checks whose verdict is ``heterogeneous``, in report order.

    :param Mapping[str, Any] block: A validated or freshly assessed block.
    :return list[ConfoundCheck]: Flagged check names.
    """
    return [name for name in CONFOUND_CHECKS if block["checks"][name]["verdict"] == "heterogeneous"]


def _describe_flagged(block: Mapping[str, Any]) -> list[str]:
    """Describe each flagged check with the numbers behind it, for operator logs.

    :param Mapping[str, Any] block: A validated or freshly assessed block.
    :return list[str]: One sentence per flagged check; empty when none flagged.
    """
    lines: list[str] = []
    for name in _flagged_checks(block):
        detail = block["checks"][name]["detail"]
        if name == "evaluation_point":
            groups = detail["values"]
            lines.append(
                f"evaluation_point: ranked objectives were measured at {len(groups)} distinct "
                f"{detail['axis']} values ({_logged_values(groups)}), the lowest more than "
                f"{_POSITION_TOLERANCE:.0%} short of the highest. Expected when this phase's "
                "search space changes training length (for example epochs or max_steps); "
                "otherwise the trials were not trained to one budget. A training-length phase "
                "checks relative completion on a schedule_fraction axis, which still flags "
                "a run that stopped early"
            )
        elif name == "survivorship":
            lines.append(
                f"survivorship: {detail['failed']} failed trials against {detail['ranked']} ranked"
            )
        elif detail["within_noise"] is None:
            tied = ", ".join(str(number) for number in detail["tied_trials"])
            lines.append(
                f"tie: trial(s) {tied} matched the winner's value exactly with different "
                "params, and the lower draw position won"
            )
        else:
            lines.extend(_describe_group_tie(detail["tied_trials"], detail["within_noise"]))
    return lines


def _evaluation_point_remedy(kind: str) -> str:
    """Describe relative completion using the objective reader in use.

    :param str kind: Objective extractor kind.
    :return str: Trainer reporting and evaluation-axis instructions.
    """
    if kind == "wandb":
        return (
            "Log schedule_fraction=actual_steps / planned_steps alongside the objective in "
            "the same wandb.log() call and set metric.extractor.evaluation_axis to schedule_fraction."
        )
    if kind == "mlflow":
        return (
            "MLflow compares the selected metric's integer step. For a training-length "
            "comparison, log the objective at a common scaled relative-completion step "
            "(for example int(10000 * actual_steps / planned_steps)), retaining lower "
            "steps for early stops."
        )
    return (
        'Report progress={"schedule_fraction": actual_steps / planned_steps} through '
        "report_objective and set metric.extractor.evaluation_axis to schedule_fraction "
        "on the json_envelope metric."
    )


def _noise_remedy() -> str:
    """Return guidance when the selected configurations carry no repeat noise estimate.

    :return str: Repeats guidance shared by winner output and MCP results.
    """
    return (
        "No noise estimate: single trials cannot establish whether the gap is seed noise. "
        "Use phases[].repeats to compare configurations across seeds and rank group means."
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
            "may be seed noise (an advisory heuristic, not a significance test)"
        )
    return lines


def _confound_summary(block: Mapping[str, Any], *, winner_trial: int) -> dict[str, Any]:
    """Project a validated block onto the enums and numbers agents may see.

    Include the declared evaluation axis and actionable guidance alongside counts.
    The full trial evidence stays in ``winner.yaml`` for operators.

    :param Mapping[str, Any] block: A validated confound block.
    :param int winner_trial: Trial number of the winner the block belongs to.
    :return dict: The summary, one entry per check plus
        the ``flagged`` check names.
    """
    checks = block["checks"]
    evaluation = checks["evaluation_point"]
    groups = evaluation["detail"]["values"] if "detail" in evaluation else None
    tie = checks["tie"]
    return {
        "flagged": list(_flagged_checks(block)),
        "evaluation_point": {
            "verdict": evaluation["verdict"],
            "axis": evaluation["detail"]["axis"] if "detail" in evaluation else None,
            "remedy": (
                "For training-length comparisons, report schedule_fraction=actual_steps / "
                "planned_steps as the evaluation axis; retain lower fractions for early stops."
                if evaluation["verdict"] == "heterogeneous"
                else None
            ),
            "distinct_values": len(groups) if groups is not None else None,
            "winner_value": (
                next(
                    (group["value"] for group in groups if winner_trial in group["trials"]),
                    None,
                )
                if groups is not None
                else None
            ),
        },
        "survivorship": {
            "verdict": checks["survivorship"]["verdict"],
            **checks["survivorship"]["detail"],
        },
        "tie": {
            "verdict": tie["verdict"],
            "remedy": _noise_remedy() if block["ranked_groups"] is None else None,
            "tied_trials": len(tie["detail"]["tied_trials"]) if "detail" in tie else None,
            "within_noise": (
                len(tie["detail"]["within_noise"])
                if "detail" in tie and tie["detail"]["within_noise"] is not None
                else None
            ),
        },
    }


def _logged_values(groups: Sequence[Mapping[str, Any]]) -> str:
    """Render group values for a log line, eliding past a few.

    :param Sequence[Mapping[str, Any]] groups: Value groups from the evidence.
    :return str: Comma-separated values.
    """
    shown = ", ".join(str(group["value"]) for group in groups[:_LOGGED_VALUES])
    return shown + (", ..." if len(groups) > _LOGGED_VALUES else "")
