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

_EVALUATION_FIELDS = ("checkpoint", "step")
_SURVIVORSHIP_COUNTS = ("ranked", "infeasible", "failed", "pruned")
# Where a terminal trial that selection did not rank is counted. A COMPLETE
# trial outside the ranked set failed feasibility, a gate, or a constraint.
_EXCLUDED_BY_STATE = {
    optuna.trial.TrialState.COMPLETE: "infeasible",
    optuna.trial.TrialState.FAIL: "failed",
    optuna.trial.TrialState.PRUNED: "pruned",
}
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
        provenance for every ranked trial, keyed by trial number.
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
    """Check that every ranked objective was measured at one checkpoint and step.

    Only a ``json_envelope`` objective reports where it was measured. The
    extractor kind and config are fingerprinted, and the envelope's objective
    name, split, and policy are enforced at extraction, so the reported
    checkpoint and step are the only evaluation identity that can differ
    between trials -- and only when the config leaves them unpinned.

    :param Sequence[optuna.trial.FrozenTrial] ranked: Ranked trials by number.
    :param Mapping[int, Mapping[str, Any]] provenance: Their objective provenance.
    :return dict[str, Any]: The check record.
    """
    if len(ranked) < 2:
        return _not_checked("only one trial was ranked")
    records = [(trial.number, provenance[trial.number]) for trial in ranked]
    kinds = sorted({record["extractor"]["kind"] for _, record in records})
    if kinds != ["json_envelope"]:
        return _not_checked(
            f"{', '.join(kinds)} objectives report no evaluation checkpoint or step"
        )
    detail = {
        field: _groups(
            (number, record["source"]["evaluation"][field]) for number, record in records
        )
        for field in _EVALUATION_FIELDS
    }
    return _checked(any(len(groups) > 1 for groups in detail.values()), detail)


def _survivorship(
    trials: Sequence[optuna.trial.FrozenTrial],
    ranked: Sequence[optuna.trial.FrozenTrial],
) -> dict[str, Any]:
    """Check whether the winner was ranked among at most half of what ran.

    When failures or exclusions are at least as numerous as the ranked trials,
    "best" is a claim about the survivors: anything that made a configuration
    fail is now correlated with which configurations could win. The counts are
    recorded whatever the verdict, and the check always runs.

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
    excluded = counts["infeasible"] + counts["failed"] + counts["pruned"]
    return _checked(excluded >= max(1, counts["ranked"]), counts)


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
    return _checked(bool(tied), {"tied_trials": tied})


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


def _checked(flagged: bool, detail: dict[str, Any]) -> dict[str, Any]:
    """Return the record of a check that ran.

    :param bool flagged: Whether the check found a confound.
    :param dict[str, Any] detail: Evidence recorded whatever the verdict.
    :return dict[str, Any]: The check record.
    """
    return {"verdict": "heterogeneous" if flagged else "ok", "detail": detail}


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
    :raises ValueError: The block is absent, has another schema version, or
        any check record is malformed.
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
    return {
        "schema_version": CONFOUND_SCHEMA_VERSION,
        "ranked_trials": ranked,
        "checks": {name: _validate_check(name, checks[name]) for name in CONFOUND_CHECKS},
    }


def _validate_check(name: ConfoundCheck, check: object) -> dict[str, Any]:
    """Validate one check record.

    :param ConfoundCheck name: Check the record belongs to.
    :param object check: Parsed record.
    :raises ValueError: The verdict is unknown, an ``n/a`` record has no
        reason, or a checked record's evidence is malformed.
    :return dict[str, Any]: The validated record.
    """
    if not isinstance(check, Mapping) or check.get("verdict") not in CONFOUND_VERDICTS:
        raise ValueError(f"confound check {name!r} has no valid verdict")
    if check["verdict"] == "n/a":
        reason = check.get("reason")
        if set(check) != {"verdict", "reason"} or not isinstance(reason, str) or not reason:
            raise ValueError(f"confound check {name!r} is n/a without a reason")
        return {"verdict": "n/a", "reason": reason}
    if set(check) != {"verdict", "detail"}:
        raise ValueError(f"confound check {name!r} has no evidence")
    return {"verdict": check["verdict"], "detail": _DETAIL_VALIDATORS[name](check["detail"])}


def _validate_evaluation_detail(detail: object) -> dict[str, Any]:
    """Validate evaluation-point evidence: value groups per evaluation field.

    :param object detail: Parsed evidence.
    :raises ValueError: A field is missing or a group is malformed.
    :return dict[str, Any]: The validated evidence.
    """
    if not isinstance(detail, Mapping) or set(detail) != set(_EVALUATION_FIELDS):
        raise ValueError("confound evaluation_point evidence must group checkpoint and step")
    parsed: dict[str, Any] = {}
    for field in _EVALUATION_FIELDS:
        groups = detail[field]
        if not isinstance(groups, list) or not groups:
            raise ValueError(f"confound evaluation_point {field} has no groups")
        parsed[field] = []
        for group in groups:
            if not isinstance(group, Mapping) or set(group) != {"value", "trials"}:
                raise ValueError(f"confound evaluation_point {field} group is malformed")
            value = group["value"]
            valid = (
                isinstance(value, str) and bool(value)
                if field == "checkpoint"
                else type(value) is int and value >= 0
            )
            trials = _trial_numbers(group["trials"], label=f"evaluation_point {field} trials")
            if not valid or not trials:
                raise ValueError(f"confound evaluation_point {field} group is malformed")
            parsed[field].append({"value": value, "trials": trials})
    return parsed


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
            spread = [
                f"{len(groups)} distinct {field}s ({_logged_values(groups)})"
                for field, groups in detail.items()
                if len(groups) > 1
            ]
            lines.append(
                f"evaluation_point: ranked objectives were measured at {' and '.join(spread)}"
            )
        elif name == "survivorship":
            total = sum(detail[key] for key in _SURVIVORSHIP_COUNTS)
            lines.append(
                f"survivorship: the winner was ranked among {detail['ranked']} of {total} "
                f"terminal trials ({detail['infeasible']} infeasible, {detail['failed']} "
                f"failed, {detail['pruned']} pruned)"
            )
        else:
            tied = ", ".join(str(number) for number in detail["tied_trials"])
            lines.append(
                f"tie: trial(s) {tied} matched the winner's value exactly with different "
                "params, and the lower trial number won"
            )
    return lines


def _logged_values(groups: Sequence[Mapping[str, Any]]) -> str:
    """Render group values for a log line, eliding past a few.

    :param Sequence[Mapping[str, Any]] groups: Value groups from the evidence.
    :return str: Comma-separated values.
    """
    shown = ", ".join(str(group["value"]) for group in groups[:_LOGGED_VALUES])
    return shown + (", ..." if len(groups) > _LOGGED_VALUES else "")
