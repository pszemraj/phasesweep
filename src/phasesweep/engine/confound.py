"""Selection-time confound verdicts over a phase's ranked population.

PhaseSweep has no control trial: every trial in a phase is a peer, and
selection picks the best ranked value. A confound verdict therefore answers
whether the population selection just compared was homogeneous enough for
"best" to mean what it says. Each check is a countable fact about that
population, never a significance test -- the ledger holds no replicate trials
to estimate noise from, so a p-value would be invented. Every verdict is
advisory: it never blocks selection or publication.

The block is computed once per selection and frozen onto the winner. A
carried winner keeps it verbatim, and it never joins a fingerprint, so an
observation can never invalidate a study or block a top-up.

This module stays free of the extraction stack: read-only publication
validation imports it, so the selection caller decodes each ranked trial's
objective provenance and passes the records in.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, get_args

import optuna

ConfoundVerdict = Literal["ok", "heterogeneous", "n/a"]
"""Outcome of one check. ``ok``: the check ran and found nothing.
``heterogeneous``: the check ran and flagged the population, with evidence.
``n/a``: the check could not run; it means *not checked*, never *clean*."""

CONFOUND_VERDICTS: tuple[ConfoundVerdict, ...] = get_args(ConfoundVerdict)

ConfoundCheck = Literal["evaluation_point", "survivorship", "tie"]
"""One comparability dimension, in the order every surface reports them."""

CONFOUND_CHECKS: tuple[ConfoundCheck, ...] = get_args(ConfoundCheck)

CONFOUND_SCHEMA_VERSION = 1

# Objective kinds whose provenance records a position on the evaluation axis.
_POSITIONED_KINDS = frozenset({"json_envelope", "wandb"})
_SURVIVORSHIP_COUNTS = ("ranked", "infeasible", "failed", "pruned")
# Where a terminal trial that selection did not rank is counted. A COMPLETE
# trial outside the ranked set violated a constraint; a trial that failed an
# evidence gate or extraction is FAIL.
_EXCLUDED_BY_STATE = {
    optuna.trial.TrialState.COMPLETE: "infeasible",
    optuna.trial.TrialState.FAIL: "failed",
    optuna.trial.TrialState.PRUNED: "pruned",
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
) -> dict[str, Any]:
    """Classify the population one winner selection compared.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial in the phase
        study, the population survivorship counts.
    :param Sequence[optuna.trial.FrozenTrial] ranked: Trials selection ranked:
        candidates that satisfy every constraint.
    :param optuna.trial.FrozenTrial winner: The selected trial, one of ``ranked``.
    :param Mapping[int, Mapping[str, Any]] provenance: Validated objective
        provenance keyed by trial number, for every ranked trial whose record
        could be read.
    :return dict[str, Any]: The ``confound`` block persisted on the winner.
    """
    ordered = sorted(ranked, key=lambda trial: trial.number)
    return {
        "schema_version": CONFOUND_SCHEMA_VERSION,
        "ranked_trials": [trial.number for trial in ordered],
        "checks": {
            "evaluation_point": _evaluation_point(ordered, provenance),
            "survivorship": _survivorship(trials, ordered),
            "tie": _tie(ordered, winner),
        },
    }


def _evaluation_point(
    ranked: Sequence[optuna.trial.FrozenTrial],
    provenance: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Check that every ranked objective was measured at one training position.

    The position is on the extractor's declared ``evaluation_axis`` -- the
    progress axis the sweep holds fixed: optimizer step, tokens for a
    fixed-token sweep, or a W&B step metric. ``json_envelope`` objectives
    report it; ``wandb`` objectives locate it in run history. Comparing any
    other axis would flag designed differences: a fixed-token batch-size sweep
    ends trials at different steps, and checkpoint labels usually embed the
    step. The axis is part of the fingerprinted extractor config, so every
    ranked trial declares the same one. The exact positions are recorded, and
    the check flags only positions more than ``_POSITION_TOLERANCE`` short of
    the furthest.

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
    if len(kinds) != 1 or kinds[0] not in _POSITIONED_KINDS:
        return _not_checked(f"{', '.join(kinds)} objectives report no evaluation position")
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
) -> dict[str, Any]:
    """Check whether failed and pruned trials at least match the ranked ones.

    When they do, "best" is a claim about the survivors: anything that made a
    configuration fail, diverge, or run out of time is now correlated with
    which configurations could win. Failed trials include evidence-gate and
    extraction failures. An infeasible trial violated one of the operator's
    constraints, which define what may win, so it is counted but never flags.
    The counts are recorded whatever the verdict, and the check always runs.

    :param Sequence[optuna.trial.FrozenTrial] trials: Every trial in the study.
    :param Sequence[optuna.trial.FrozenTrial] ranked: Ranked trials.
    :return dict[str, Any]: The check record.
    """
    ranked_numbers = {trial.number for trial in ranked}
    counts = dict.fromkeys(_SURVIVORSHIP_COUNTS, 0)
    counts["ranked"] = len(ranked_numbers)
    for trial in trials:
        bucket = _EXCLUDED_BY_STATE.get(trial.state)
        if bucket is not None and trial.number not in ranked_numbers:
            counts[bucket] += 1
    return _checked("survivorship", counts)


def _tie(
    ranked: Sequence[optuna.trial.FrozenTrial],
    winner: optuna.trial.FrozenTrial,
) -> dict[str, Any]:
    """Check whether the trial-number tiebreak chose between distinct configs.

    Selection breaks exact ties by lower trial number. Tied trials that share
    the winner's params are repeats of one configuration and change nothing;
    tied trials with other params mean the winner was chosen arbitrarily.

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
    return _checked("tie", {"tied_trials": tied})


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
    """Return the record of a check that ran, its verdict derived from its evidence.

    Assessment and validation share this derivation, so a persisted verdict
    can never disagree with the evidence recorded beside it.

    :param ConfoundCheck name: The check.
    :param dict[str, Any] detail: Evidence recorded whatever the verdict.
    :return dict[str, Any]: The check record.
    """
    return {"verdict": "heterogeneous" if _FLAGGED[name](detail) else "ok", "detail": detail}


def _survivorship_flagged(counts: Mapping[str, int]) -> bool:
    """Whether failed and pruned trials were at least as numerous as ranked ones.

    :param Mapping[str, int] counts: Survivorship counts.
    :return bool: Whether the failures could have decided what was ranked.
    """
    return counts["failed"] + counts["pruned"] >= max(1, counts["ranked"])


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
    "tie": lambda detail: bool(detail["tied_trials"]),
}


def _not_checked(reason: str) -> dict[str, Any]:
    """Return the record of a check that could not run.

    :param str reason: Why the population could not be checked.
    :return dict[str, Any]: The check record.
    """
    return {"verdict": "n/a", "reason": reason}


def _validate_confound_block(data: object) -> dict[str, Any]:
    """Validate a persisted ``confound`` block and return a plain copy.

    Shared by :func:`phasesweep.engine.publication_validation._validate_generation_manifest`,
    :func:`phasesweep.engine.artifacts._load_winner`, and the winner read
    path, which differ only in how they wrap the ``ValueError``.

    :param object data: Parsed ``confound`` value from a ``winner.yaml``.
    :raises ValueError: The block is absent, has another schema version, any
        check record is malformed or contradicts its evidence, or the evidence
        disagrees with the ranked trials.
    :return dict[str, Any]: The validated block, rebuilt from plain values.
    """
    if not isinstance(data, Mapping) or set(data) != {"schema_version", "ranked_trials", "checks"}:
        raise ValueError("confound block is missing or has unexpected keys")
    if type(data["schema_version"]) is not int or data["schema_version"] != CONFOUND_SCHEMA_VERSION:
        raise ValueError(f"confound schema_version is not {CONFOUND_SCHEMA_VERSION}")
    ranked = _trial_numbers(data["ranked_trials"], label="ranked_trials")
    if not ranked:
        raise ValueError("confound ranked_trials is empty")
    checks = data["checks"]
    if not isinstance(checks, Mapping) or set(checks) != set(CONFOUND_CHECKS):
        raise ValueError(f"confound checks must be exactly {', '.join(CONFOUND_CHECKS)}")
    validated = {name: _validate_check(name, checks[name]) for name in CONFOUND_CHECKS}
    _validate_against_ranked(validated, ranked)
    return {"schema_version": CONFOUND_SCHEMA_VERSION, "ranked_trials": ranked, "checks": validated}


def _validate_against_ranked(
    checks: Mapping[ConfoundCheck, Mapping[str, Any]], ranked: list[int]
) -> None:
    """Validate that each check's evidence describes the block's ranked trials.

    :param Mapping[ConfoundCheck, Mapping[str, Any]] checks: Individually validated records.
    :param list[int] ranked: The block's ranked trial numbers.
    :raises ValueError: A single ranked trial was compared, several were not,
        the evaluation groups do not cover each ranked trial once, the
        survivorship count differs, or a tied trial was not ranked.
    """
    evaluation, survivorship, tie = (checks[name] for name in CONFOUND_CHECKS)
    if len(ranked) < 2 and (evaluation["verdict"] != "n/a" or tie["verdict"] != "n/a"):
        raise ValueError("confound compared trials though only one was ranked")
    if len(ranked) >= 2 and tie["verdict"] == "n/a":
        raise ValueError("confound tie is n/a though several trials were ranked")
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
        if not _is_position(group["value"]) or not trials:
            raise ValueError("confound evaluation_point value group is malformed")
        if parsed and not parsed[-1]["value"] < group["value"]:
            raise ValueError("confound evaluation_point values are not ascending and distinct")
        parsed.append({"value": group["value"], "trials": trials})
    return {"axis": axis, "values": parsed}


def _is_position(value: object) -> bool:
    """Whether ``value`` is a finite, non-negative JSON number (not a bool).

    :param object value: Parsed group value.
    :return bool: Whether it is a valid evaluation-axis position.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _validate_survivorship_detail(detail: object) -> dict[str, Any]:
    """Validate survivorship evidence: the four outcome counts.

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
    """Validate tie evidence: the trials that tied the winner with other params.

    :param object detail: Parsed evidence.
    :raises ValueError: The trial list is missing or malformed.
    :return dict[str, Any]: The validated evidence.
    """
    if not isinstance(detail, Mapping) or set(detail) != {"tied_trials"}:
        raise ValueError("confound tie evidence must list tied_trials")
    return {"tied_trials": _trial_numbers(detail["tied_trials"], label="tie tied_trials")}


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
                f"{_POSITION_TOLERANCE:.0%} short of the highest"
            )
        elif name == "survivorship":
            lines.append(
                f"survivorship: {detail['failed']} failed and {detail['pruned']} pruned "
                f"trials against {detail['ranked']} ranked"
            )
        else:
            tied = ", ".join(str(number) for number in detail["tied_trials"])
            lines.append(
                f"tie: trial(s) {tied} matched the winner's value exactly with different "
                "params, and the lower trial number won"
            )
    return lines


def _confound_summary(block: Mapping[str, Any], *, winner_trial: int) -> dict[str, Any]:
    """Project a validated block onto the enums and numbers agents may see.

    Evidence holds operator- and trainer-authored strings -- the axis is a
    config key, like the metric keys MCP never returns -- so the projection
    keeps only verdicts, check names, counts, and the winner's position. The
    full evidence stays in ``winner.yaml`` for operators.

    :param Mapping[str, Any] block: A validated confound block.
    :param int winner_trial: Trial number of the winner the block belongs to.
    :return dict[str, Any]: The string-free summary, one entry per check plus
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
            "tied_trials": len(tie["detail"]["tied_trials"]) if "detail" in tie else None,
        },
    }


def _logged_values(groups: Sequence[Mapping[str, Any]]) -> str:
    """Render group values for a log line, eliding past a few.

    :param Sequence[Mapping[str, Any]] groups: Value groups from the evidence.
    :return str: Comma-separated values.
    """
    shown = ", ".join(str(group["value"]) for group in groups[:_LOGGED_VALUES])
    return shown + (", ..." if len(groups) > _LOGGED_VALUES else "")
