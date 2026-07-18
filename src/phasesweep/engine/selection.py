"""Winner selection and promotion decisions."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import optuna

from phasesweep.config import Experiment, Phase, Promotion, Suite, check_bounds
from phasesweep.engine.state import FEASIBLE_ATTR, GATES_ATTR, Winner, constraint_attr

WINNER_TIE_EPS = 1e-12


@dataclass
class SelectedTrial:
    """The best feasible trial from a completed phase.

    Does not include effective_overrides — the orchestrator adds those.
    """

    trial_number: int
    params: dict[str, Any]
    metric: float
    constraints: dict[str, float] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)


class NoFeasibleTrialError(RuntimeError):
    """Raised when no trial in a completed phase satisfies all constraints."""


def select_winner(study: optuna.Study, experiment: Experiment) -> SelectedTrial:
    """Pick the best feasible completed trial from a phase study.

    Rules:
      1. Trial must be COMPLETE (not pruned, not failed).
      2. Trial's metric must be finite.
      3. All constraint values (read from user_attrs) must satisfy bounds.
      4. Among survivors, argmin/argmax on metric.
      5. Ties (within absolute eps 1e-12) broken by lower trial_number.

    Args:
        study: Optuna study for the phase whose winner we want.
        experiment: Parsed experiment config. Provides the optimization goal
            (minimize/maximize) and the constraint definitions used to filter
            trials.

    Returns:
        The winning trial as :class:`SelectedTrial` (number, params, metric,
        constraint readings, and persisted evidence-gate results).

    Raises:
        NoFeasibleTrialError: If no trial in the study is both COMPLETE and
            satisfies every constraint.

    """
    minimize = experiment.metric.goal == "minimize"
    constraints_by_name = {c.name: c for c in experiment.constraints}

    survivors: list[optuna.trial.FrozenTrial] = []
    for t in study.get_trials(deepcopy=False):
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.value is None or not math.isfinite(t.value):
            continue
        if not t.user_attrs.get(FEASIBLE_ATTR, False):
            continue
        # Re-verify constraints from user_attrs in case rules changed or stored
        # values are non-finite (defense in depth — review item #3).
        ok = True
        for name, c in constraints_by_name.items():
            v = t.user_attrs.get(constraint_attr(name))
            if v is None:
                ok = False
                break
            try:
                v_f = float(v)
            except (TypeError, ValueError):
                ok = False
                break
            if not check_bounds(v_f, min_value=c.min, max_value=c.max):
                ok = False
                break
        if not ok:
            continue
        survivors.append(t)

    if not survivors:
        raise NoFeasibleTrialError(
            "No feasible completed trials in phase. "
            "Check stdout/stderr logs in the phase's trial_* directories."
        )

    best_value = (
        min(_trial_value(t) for t in survivors)
        if minimize
        else max(_trial_value(t) for t in survivors)
    )
    near_best = [t for t in survivors if abs(_trial_value(t) - best_value) <= WINNER_TIE_EPS]
    best = min(near_best, key=lambda t: t.number)

    constraint_vals = {
        name: float(best.user_attrs[constraint_attr(name)]) for name in constraints_by_name
    }
    selected_value = best.value
    assert selected_value is not None  # same invariant
    raw_gates = best.user_attrs.get(GATES_ATTR)
    gates: list[dict[str, Any]] = []
    if isinstance(raw_gates, str) and raw_gates:
        try:
            parsed_gates = json.loads(raw_gates)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed_gates, list):
                gates = [item for item in parsed_gates if isinstance(item, dict)]

    return SelectedTrial(
        trial_number=best.number,
        params=dict(best.params),
        metric=float(selected_value),
        constraints=constraint_vals,
        gates=gates,
    )


def _trial_value(trial: optuna.trial.FrozenTrial) -> float:
    """Return the non-None metric value for a known survivor trial.

    :param optuna.trial.FrozenTrial trial: Completed feasible trial already filtered by
        :func:`select_winner`.
    :return float: Scalar objective value for the trial.
    """
    value = trial.value
    assert value is not None  # survivor invariant from select_winner
    return value


log = logging.getLogger("phasesweep.engine.selection")


def _gates_pass(gates: list[dict[str, Any]]) -> bool:
    """Return whether every recorded gate result passed.

    :param list[dict[str, Any]] gates: Recorded evidence gate payloads.
    :return bool: ``True`` when every gate has a truthy ``passed`` value.
    """
    return all(bool(gate.get("passed")) for gate in gates)


def _clone_winner_from_baseline(
    baseline: Winner,
    *,
    phase_fingerprint: str | None,
    completion: dict[str, Any] | None = None,
    promotion: dict[str, Any] | None = None,
) -> Winner:
    """Clone a baseline winner for exposure under another phase/study.

    :param Winner baseline: Winner to copy into the exposed result slot.
    :param str | None phase_fingerprint: Fingerprint to assign to the clone.
    :param dict[str, Any] | None completion: Optional completion payload to
        store instead of the baseline completion.
    :param dict[str, Any] | None promotion: Optional promotion audit payload.
    :return Winner: Cloned winner with copied mutable payloads.
    """
    return Winner(
        trial_number=baseline.trial_number,
        params=dict(baseline.params),
        effective_overrides=dict(baseline.effective_overrides),
        metric=baseline.metric,
        constraints=dict(baseline.constraints),
        gates=list(baseline.gates),
        completion=dict(completion or baseline.completion),
        promotion=promotion,
        phase_fingerprint=phase_fingerprint,
    )


def _metric_improvement(goal: str, candidate: Winner, baseline: Winner) -> float:
    """Return candidate improvement over baseline for a metric goal.

    :param str goal: Optimization direction, either ``"minimize"`` or
        ``"maximize"``.
    :param Winner candidate: Candidate winner being evaluated.
    :param Winner baseline: Baseline winner to compare against.
    :return float: Signed improvement where larger values are better.
    """
    if goal == "minimize":
        return baseline.metric - candidate.metric
    return candidate.metric - baseline.metric


def _evaluate_promotion_rule(
    *,
    goal: str,
    promotion: Promotion,
    candidate: Winner,
    baseline: Winner,
) -> tuple[bool, float | None, bool, str]:
    """Evaluate shared gate and metric-delta promotion semantics.

    :param str goal: Optimization direction for the metric comparison.
    :param Promotion promotion: Promotion rule to apply.
    :param Winner candidate: Candidate winner being considered for promotion.
    :param Winner baseline: Baseline winner used for the delta comparison.
    :return tuple[bool, float | None, bool, str]: Promotion flag, improvement
        value, gate pass flag, and decision reason.
    """
    gates_passed = _gates_pass(candidate.gates)
    if promotion.requires_gates and not gates_passed:
        return False, None, gates_passed, "gates_failed"
    improvement = _metric_improvement(goal, candidate, baseline)
    promoted = improvement >= promotion.min_delta
    reason = "promoted" if promoted else "insufficient_delta"
    return promoted, improvement, gates_passed, reason


def _winner_summary_item(name: str, winner: Winner) -> dict[str, Any]:
    """Return the compact winner payload used in run summaries.

    :param str name: Phase or study label for the winner.
    :param Winner winner: Winner to serialize into a summary item.
    :return dict[str, Any]: Compact summary payload for the winner.
    """
    payload = {
        "name": name,
        "trial_number": winner.trial_number,
        "metric": winner.metric,
        "params": winner.params,
        "effective_overrides": winner.effective_overrides,
        "constraints": winner.constraints,
        "gates": winner.gates,
        "completion": winner.completion,
    }
    if winner.promotion is not None:
        payload["promotion"] = winner.promotion
    return payload


def _apply_promotion(
    experiment: Experiment,
    phase: Phase,
    candidate: Winner,
    winners: dict[str, Winner],
) -> tuple[Winner | None, dict[str, Any] | None]:
    """Apply a phase promotion rule and return the exposed winner plus audit payload.

    :param Experiment experiment: Experiment config that supplies metric goal.
    :param Phase phase: Phase whose promotion rule is being applied.
    :param Winner candidate: Candidate winner from the current phase.
    :param dict[str, Winner] winners: Winners from previous phases.
    :return tuple[Winner | None, dict[str, Any] | None]: Exposed winner and
        promotion audit payload, or ``None`` values when no rule applies.
    """
    promotion = phase.promotion
    if promotion is None:
        return candidate, None

    baseline = winners[promotion.min_delta_vs]
    promoted, improvement, gates_passed, reason = _evaluate_promotion_rule(
        goal=experiment.metric.goal,
        promotion=promotion,
        candidate=candidate,
        baseline=baseline,
    )
    if reason == "gates_failed":
        message = f"Phase {phase.name!r} failed promotion: required evidence gates did not pass."
    else:
        assert improvement is not None
        message = (
            ""
            if promoted
            else f"Phase {phase.name!r} failed promotion: improvement {improvement:g} "
            f"vs {promotion.min_delta_vs!r} is below min_delta {promotion.min_delta:g}."
        )

    action = (
        "promote"
        if promoted
        else "continue_baseline"
        if promotion.on_fail == "continue_baseline"
        else promotion.on_fail
    )
    exposed_trial_number = (
        candidate.trial_number
        if action == "promote"
        else baseline.trial_number
        if action == "continue_baseline"
        else None
    )
    exposed_source = (
        "candidate"
        if action == "promote"
        else "baseline"
        if action == "continue_baseline"
        else None
    )
    decision: dict[str, Any] = {
        "phase": phase.name,
        "baseline": promotion.min_delta_vs,
        "candidate_trial_number": candidate.trial_number,
        "exposed_trial_number": exposed_trial_number,
        "exposed_source": exposed_source,
        "candidate_metric": candidate.metric,
        "baseline_metric": baseline.metric,
        "min_delta": promotion.min_delta,
        "improvement": improvement,
        "requires_gates": promotion.requires_gates,
        "gates_passed": gates_passed,
        "promoted": promoted,
        "on_fail": promotion.on_fail,
        "action": action,
        "reason": reason,
    }
    if message:
        decision["message"] = message

    if promoted:
        assert improvement is not None
        log.info(
            "phase=%s PROMOTED improvement=%g baseline=%s min_delta=%g",
            phase.name,
            improvement,
            promotion.min_delta_vs,
            promotion.min_delta,
        )
        candidate.promotion = decision
        return candidate, decision

    if promotion.on_fail == "stop":
        return None, decision
    if promotion.on_fail == "skip":
        log.warning("%s Skipping remaining dependent phases.", message)
        return None, decision

    log.warning("%s Continuing with baseline winner.", message)
    return (
        _clone_winner_from_baseline(
            baseline,
            phase_fingerprint=candidate.phase_fingerprint,
            completion=candidate.completion,
            promotion=decision,
        ),
        decision,
    )


def _study_phase_winner(
    study_name: str,
    results: dict[str, dict[str, Winner]],
    selector: str,
) -> tuple[str, Winner]:
    """Resolve a suite promotion selector to a prior study winner.

    :param str study_name: Name of the study whose rule references the selector.
    :param dict[str, dict[str, Winner]] results: Prior study winners keyed by
        study and phase name.
    :param str selector: Baseline selector, either a study name or
        ``"study.phase"``.
    :return tuple[str, Winner]: Resolved baseline label and winner.
    """
    if "." in selector:
        baseline_study, phase_name = selector.split(".", 1)
    else:
        baseline_study, phase_name = selector, ""
    if baseline_study not in results:
        raise RuntimeError(
            f"Study {study_name!r} promotion references unknown baseline study {baseline_study!r}."
        )
    study_winners = results[baseline_study]
    if not study_winners:
        raise RuntimeError(f"Baseline study {baseline_study!r} has no winners.")
    if phase_name:
        if phase_name not in study_winners:
            raise RuntimeError(
                f"Study {study_name!r} promotion references missing baseline phase {selector!r}."
            )
        return baseline_study, study_winners[phase_name]
    final_phase = next(reversed(study_winners))
    return f"{baseline_study}.{final_phase}", study_winners[final_phase]


def _apply_study_promotion(
    *,
    suite: Suite,
    study_name: str,
    experiment: Experiment,
    study_winners: dict[str, Winner],
    prior_results: dict[str, dict[str, Winner]],
) -> tuple[dict[str, Winner] | None, dict[str, Any] | None]:
    """Apply a suite study promotion rule against a prior study winner.

    :param Suite suite: Suite config containing study promotion definitions.
    :param str study_name: Study whose promotion rule is being evaluated.
    :param Experiment experiment: Experiment config that supplies metric goal.
    :param dict[str, Winner] study_winners: Winners produced by the current
        study.
    :param dict[str, dict[str, Winner]] prior_results: Winners from earlier
        studies in the suite.
    :return tuple[dict[str, Winner] | None, dict[str, Any] | None]: Exposed
        study winners and promotion decision payload.
    """
    study_spec = next(study for study in suite.studies if study.name == study_name)
    promotion = study_spec.promotion
    if promotion is None:
        return study_winners, None
    if not study_winners:
        raise RuntimeError(f"Study {study_name!r} has no winner to promote.")

    baseline_label, baseline = _study_phase_winner(
        study_name,
        prior_results,
        promotion.min_delta_vs,
    )
    final_phase = next(reversed(study_winners))
    candidate = study_winners[final_phase]

    promoted, improvement, gates_passed, _reason = _evaluate_promotion_rule(
        goal=experiment.metric.goal,
        promotion=promotion,
        candidate=candidate,
        baseline=baseline,
    )

    decision: dict[str, Any] = {
        "study": study_name,
        "phase": final_phase,
        "baseline": baseline_label,
        "candidate_metric": candidate.metric,
        "baseline_metric": baseline.metric,
        "min_delta": promotion.min_delta,
        "improvement": improvement,
        "requires_gates": promotion.requires_gates,
        "gates_passed": gates_passed,
        "promoted": promoted,
        "on_fail": promotion.on_fail,
    }
    if promoted:
        log.info(
            "suite=%s study=%s PROMOTED improvement=%s baseline=%s min_delta=%g",
            suite.suite,
            study_name,
            improvement,
            baseline_label,
            promotion.min_delta,
        )
        return study_winners, decision

    message = (
        f"Study {study_name!r} failed promotion against {baseline_label!r}: "
        f"improvement {improvement!r}, min_delta {promotion.min_delta:g}, "
        f"gates_passed={gates_passed}."
    )
    if promotion.on_fail == "stop":
        raise RuntimeError(message)
    if promotion.on_fail == "skip":
        log.warning("%s Skipping this study for downstream dependencies.", message)
        return None, decision

    log.warning("%s Continuing with baseline winner.", message)
    exposed = dict(study_winners)
    exposed[final_phase] = _clone_winner_from_baseline(
        baseline,
        phase_fingerprint=candidate.phase_fingerprint,
    )
    return exposed, decision
