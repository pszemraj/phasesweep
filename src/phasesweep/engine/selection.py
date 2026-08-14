"""Winner selection and promotion decisions."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import optuna

from phasesweep.config import Experiment, Phase, Promotion, Suite, check_bounds
from phasesweep.engine.errors import PhaseSweepError, PromotionError, TrialEvidenceMissingError
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    FEASIBLE_ATTR,
    GATES_ATTR,
    GENERATION_ID_ATTR,
    OBJECTIVE_PROVENANCE_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_INPUT_ATTR,
    Winner,
    WinnerSource,
    WinnerSourceKind,
    _winner_common_payload,
    _winner_source_or_default,
    constraint_attr,
)


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
    generation_id: str = ""
    attempt_id: str = ""
    # Frozen evidence provenance recorded when the trial's objective was
    # extracted (review v0.5.17 / finding F); None for trials persisted
    # before the record existed.
    objective_provenance: dict[str, Any] | None = None
    # Digest of the trainer environment this trial ran under (review v0.5.18 /
    # finding F3); None for trials persisted before the record existed.
    trainer_env_digest: str | None = None
    # Versioned identity of the exact generated input consumed by the trainer.
    trainer_input: dict[str, Any] | None = None


class NoFeasibleTrialError(PhaseSweepError):
    """Raised when no trial in a completed phase satisfies all constraints."""


def select_winner(
    study: optuna.Study,
    experiment: Experiment,
    *,
    phase_name: str | None = None,
) -> SelectedTrial:
    """Pick the best feasible completed trial from a phase study.

    Rules:
      1. Trial must be COMPLETE (not pruned, not failed).
      2. Trial's metric must be finite.
      3. All constraint values (read from user_attrs) must satisfy bounds.
      4. Among survivors, argmin/argmax on metric.
      5. Ties — exact float equality only — broken by lower trial_number.

    Ordering is exact, never approximate. An earlier absolute epsilon (1e-12)
    folded everything within that distance of the optimum into one "tie" band,
    which silently reordered any objective whose natural scale sits at or below
    1e-12: two genuinely different results at 1e-13 and 3e-13 ranked by trial
    number rather than by value (review v0.5.17 / finding H). PhaseSweep has no
    way to know a config's meaningful resolution, so it does not guess one.

    Comparability is checked but never enforced: if the compared trials span
    more than one trainer-environment digest — a top-up ran under a different
    environment — the ranking mixes environments, and that is reported once
    per selection (review v0.5.18 / finding F3). It is not an error, because
    the environment is deliberately outside the semantic fingerprint.

    Args:
        study: Optuna study for the phase whose winner we want.
        experiment: Parsed experiment config. Provides the optimization goal
            (minimize/maximize) and the constraint definitions used to filter
            trials.
        phase_name: Phase label used in the environment-divergence warning;
            defaults to the study name when the caller has no phase context.

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
        generation_id = t.user_attrs.get(GENERATION_ID_ATTR)
        attempt_id = t.user_attrs.get(ATTEMPT_ID_ATTR)
        if not isinstance(generation_id, str) or not generation_id:
            continue
        if not isinstance(attempt_id, str) or not attempt_id:
            continue
        # Re-verify constraints from user_attrs in case rules changed or stored
        # values are non-finite (defense in depth — review v0.5.2 / item 3).
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

    _warn_mixed_environments(survivors, phase_name=phase_name, study=study)

    best_value = (
        min(_trial_value(t) for t in survivors)
        if minimize
        else max(_trial_value(t) for t in survivors)
    )
    # Exact equality, not a tolerance band (review v0.5.17 / finding H).
    tied = [t for t in survivors if _trial_value(t) == best_value]
    best = min(tied, key=lambda t: t.number)

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

    provenance: dict[str, Any] | None = None
    raw_provenance = best.user_attrs.get(OBJECTIVE_PROVENANCE_ATTR)
    if raw_provenance is not None:
        if not isinstance(raw_provenance, str) or not raw_provenance:
            raise TrialEvidenceMissingError(
                f"Winning trial {best.number} has malformed {OBJECTIVE_PROVENANCE_ATTR!r} evidence."
            )
        try:
            parsed_provenance = json.loads(raw_provenance)
        except json.JSONDecodeError as exc:
            raise TrialEvidenceMissingError(
                f"Winning trial {best.number} has corrupt "
                f"{OBJECTIVE_PROVENANCE_ATTR!r} JSON evidence."
            ) from exc
        if not isinstance(parsed_provenance, dict):
            raise TrialEvidenceMissingError(
                f"Winning trial {best.number} has malformed {OBJECTIVE_PROVENANCE_ATTR!r} evidence."
            )
        provenance = parsed_provenance

    env_digest = best.user_attrs.get(TRAINER_ENV_DIGEST_ATTR)
    raw_trainer_input = best.user_attrs.get(TRAINER_INPUT_ATTR)

    return SelectedTrial(
        trial_number=best.number,
        params=dict(best.params),
        metric=float(selected_value),
        constraints=constraint_vals,
        gates=gates,
        generation_id=str(best.user_attrs[GENERATION_ID_ATTR]),
        attempt_id=str(best.user_attrs[ATTEMPT_ID_ATTR]),
        objective_provenance=provenance,
        trainer_env_digest=env_digest if isinstance(env_digest, str) and env_digest else None,
        trainer_input=(dict(raw_trainer_input) if isinstance(raw_trainer_input, dict) else None),
    )


def _warn_mixed_environments(
    survivors: list[optuna.trial.FrozenTrial],
    *,
    phase_name: str | None,
    study: optuna.Study,
) -> None:
    """Warn once when the compared trials did not all run under one environment.

    :param list[optuna.trial.FrozenTrial] survivors: Feasible completed trials
        that form the comparison set for this selection.
    :param str | None phase_name: Phase label supplied by the caller; the study
        name is used when it is ``None``.
    :param optuna.Study study: Study the survivors came from, read only for its
        name and only when a divergence is being reported.
    """
    digests = {
        digest
        for trial in survivors
        if isinstance(digest := trial.user_attrs.get(TRAINER_ENV_DIGEST_ATTR), str) and digest
    }
    if len(digests) < 2:
        return
    log.warning(
        "[%s] comparing %d candidate trials that ran under %d distinct trainer "
        "environments (digests %s). A top-up ran under a different environment, so this "
        "ranking mixes environments; the environment is outside the study fingerprint by "
        "design. Re-run the phase under one environment if the difference can move the "
        "metric.",
        phase_name or study.study_name,
        len(survivors),
        len(digests),
        ", ".join(sorted(digest[:12] for digest in digests)),
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
    source_kind: WinnerSourceKind,
    source_phase: str,
    source_study: str | None = None,
    phase_fingerprint: str | None,
    completion: dict[str, Any] | None = None,
    promotion: dict[str, Any] | None = None,
) -> Winner:
    """Clone a baseline winner for exposure under another phase/study.

    :param Winner baseline: Winner to copy into the exposed result slot.
    :param WinnerSourceKind source_kind: Why the baseline supplies this exposure slot.
    :param str source_phase: Phase containing the baseline source trial.
    :param str | None source_study: Suite study containing the source trial, if applicable.
    :param str | None phase_fingerprint: Fingerprint to assign to the clone.
    :param dict[str, Any] | None completion: Optional completion payload to
        store instead of the baseline completion.
    :param dict[str, Any] | None promotion: Optional promotion audit payload.
    :return Winner: Cloned winner with copied mutable payloads.
    """
    baseline_source = _winner_source_or_default(baseline, source_phase, study=source_study)
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
        generation_id=baseline.generation_id,
        attempt_id=baseline.attempt_id,
        objective_provenance=(
            dict(baseline.objective_provenance)
            if baseline.objective_provenance is not None
            else None
        ),
        # The exposed result IS the baseline's trial, so it keeps the baseline's
        # environment identity rather than the candidate phase's.
        trainer_env_digest=baseline.trainer_env_digest,
        trainer_inherit_env=(
            list(baseline.trainer_inherit_env)
            if isinstance(baseline.trainer_inherit_env, list)
            else baseline.trainer_inherit_env
        ),
        source=WinnerSource(
            kind=source_kind,
            phase=baseline_source.phase,
            trial_number=baseline_source.trial_number,
            generation_id=baseline_source.generation_id,
            attempt_id=baseline_source.attempt_id,
            study=source_study or baseline_source.study,
        ),
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


def _promotion_decision_payload(
    *,
    phase_name: str,
    baseline_label: str,
    candidate: Winner,
    baseline: Winner,
    promotion: Promotion,
    improvement: float | None,
    gates_passed: bool,
    promoted: bool,
    study_name: str | None = None,
    reason: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Build the shared persisted promotion-decision fields.

    :param str phase_name: Candidate phase exposed by the decision.
    :param str baseline_label: Phase or study selector naming the baseline.
    :param Winner candidate: Candidate winner evaluated for promotion.
    :param Winner baseline: Baseline winner used for comparison.
    :param Promotion promotion: Applied promotion rule.
    :param float | None improvement: Computed metric improvement.
    :param bool gates_passed: Whether the candidate's gates passed.
    :param bool promoted: Whether the candidate met the rule.
    :param str | None study_name: Optional suite study owning the decision.
    :param str | None reason: Optional phase-level decision reason.
    :param str | None message: Optional phase-level diagnostic.
    :return dict[str, Any]: Persistable promotion decision.
    """
    action = "promote" if promoted else promotion.on_fail
    payload: dict[str, Any] = {
        "phase": phase_name,
        "baseline": baseline_label,
        "candidate_trial_number": candidate.trial_number,
        "candidate_generation_id": candidate.generation_id,
        "candidate_attempt_id": candidate.attempt_id,
        "baseline_trial_number": baseline.trial_number,
        "baseline_generation_id": baseline.generation_id,
        "baseline_attempt_id": baseline.attempt_id,
        "exposed_trial_number": (
            candidate.trial_number
            if action == "promote"
            else baseline.trial_number
            if action == "continue_baseline"
            else None
        ),
        "exposed_source": (
            "candidate"
            if action == "promote"
            else "baseline"
            if action == "continue_baseline"
            else None
        ),
        "candidate_metric": candidate.metric,
        "baseline_metric": baseline.metric,
        "min_delta": promotion.min_delta,
        "improvement": improvement,
        "requires_gates": promotion.requires_gates,
        "gates_passed": gates_passed,
        "promoted": promoted,
        "on_fail": promotion.on_fail,
        "action": action,
    }
    if study_name is not None:
        payload["study"] = study_name
    if reason is not None:
        payload["reason"] = reason
    if message:
        payload["message"] = message
    return payload


def _winner_summary_item(name: str, winner: Winner) -> dict[str, Any]:
    """Return the compact winner payload used in run summaries.

    :param str name: Phase or study label for the winner.
    :param Winner winner: Winner to serialize into a summary item.
    :return dict[str, Any]: Compact summary payload for the winner.
    """
    payload = {
        "name": name,
        "metric": winner.metric,
        **_winner_common_payload(winner, name),
    }
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
    :raises RuntimeError: The rule's ``min_delta_vs`` names a phase with no
        winner among ``winners``.
    """
    promotion = phase.promotion
    if promotion is None:
        return candidate, None

    try:
        baseline = winners[promotion.min_delta_vs]
    except KeyError:
        available = ", ".join(repr(name) for name in sorted(winners)) or "none"
        raise RuntimeError(
            f"Phase {phase.name!r} promotion references unknown min_delta_vs baseline "
            f"{promotion.min_delta_vs!r}; available prior phase winners: {available}."
        ) from None
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

    decision = _promotion_decision_payload(
        phase_name=phase.name,
        baseline_label=promotion.min_delta_vs,
        candidate=candidate,
        baseline=baseline,
        promotion=promotion,
        improvement=improvement,
        gates_passed=gates_passed,
        promoted=promoted,
        reason=reason,
        message=message,
    )

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
            source_kind="promotion_baseline",
            source_phase=promotion.min_delta_vs,
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
    :raises PromotionError: The selector names a baseline study or phase that
        was not exposed by the preceding suite decisions.
    :raises RuntimeError: ``results`` contains a baseline study with no winners,
        which violates the suite runner's internal result invariant.
    """
    if "." in selector:
        baseline_study, phase_name = selector.split(".", 1)
    else:
        baseline_study, phase_name = selector, ""
    if baseline_study not in results:
        raise PromotionError(
            f"Study {study_name!r} promotion references unknown baseline study {baseline_study!r}."
        )
    study_winners = results[baseline_study]
    if not study_winners:
        raise RuntimeError(f"Baseline study {baseline_study!r} has no winners.")
    if phase_name:
        if phase_name not in study_winners:
            raise PromotionError(
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
    :raises PromotionError: The rule failed with ``on_fail: stop`` or its
        baseline selector was not exposed by preceding suite decisions.
    :raises RuntimeError: The current study produced no winner, which violates
        the suite runner's internal result invariant.
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

    decision = _promotion_decision_payload(
        phase_name=final_phase,
        baseline_label=baseline_label,
        candidate=candidate,
        baseline=baseline,
        promotion=promotion,
        improvement=improvement,
        gates_passed=gates_passed,
        promoted=promoted,
        study_name=study_name,
    )
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
        raise PromotionError(message)
    if promotion.on_fail == "skip":
        log.warning("%s Skipping this study for downstream dependencies.", message)
        return None, decision

    log.warning("%s Continuing with baseline winner.", message)
    exposed = dict(study_winners)
    exposed[final_phase] = _clone_winner_from_baseline(
        baseline,
        source_kind="suite_baseline",
        source_phase=baseline.source.phase if baseline.source is not None else baseline_label,
        source_study=baseline_label.partition(".")[0],
        phase_fingerprint=candidate.phase_fingerprint,
        completion=candidate.completion,
        promotion=decision,
    )
    return exposed, decision
