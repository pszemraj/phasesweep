"""The one judge of whether a trial's measured values satisfy the configured bounds.

The ledger records what a trial measured: its metric value and each constraint's value.
Whether those satisfy ``metric.min``, ``metric.max`` and each constraint's bounds is a pure
function of the measured values and the current config, applied wherever a trial is read,
so correcting a bound re-judges the recorded trials instead of retraining them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import optuna

from phasesweep.config.models import Constraint, Experiment, Metric
from phasesweep.engine.state import EARLIER_ATTEMPT_RESULT_ATTR, constraint_attr
from phasesweep.runtime.json import JsonObject


def selection_bounds(metric: Metric, constraints: Iterable[Constraint]) -> JsonObject:
    """Capture the bounds that determine which measured trials can win.

    :param Metric metric: Configured objective and bounds.
    :param Iterable[Constraint] constraints: Configured constraints with names and bounds.
    :return JsonObject: Metric bounds and constraint bounds by name.
    """
    return {
        "metric": {"min": metric.min, "max": metric.max},
        "constraints": {c.name: {"min": c.min, "max": c.max} for c in constraints},
    }


def _bound_violations_of(
    label: str, value: float, *, min_value: float | None, max_value: float | None
) -> list[str]:
    """Name the bounds one measured value violates.

    :param str label: How the value is named, such as ``metric 'eval_loss'``.
    :param float value: The measured value.
    :param float | None min_value: Inclusive lower bound, or ``None``.
    :param float | None max_value: Inclusive upper bound, or ``None``.
    :return list[str]: One entry per violated bound; a non-finite value violates every bound.
    """
    if not math.isfinite(value):
        return [f"{label}={value} is not finite"]
    violations = []
    if min_value is not None and value < min_value:
        violations.append(f"{label}={value} < min={min_value}")
    if max_value is not None and value > max_value:
        violations.append(f"{label}={value} > max={max_value}")
    return violations


def bound_violations(
    metric: float, constraints: Mapping[str, float], experiment: Experiment
) -> list[str]:
    """Name each configured metric bound and constraint bound the measured values violate.

    :param float metric: The measured metric value.
    :param Mapping[str, float] constraints: Each measured constraint value by name.
    :param Experiment experiment: Config supplying the current metric and constraint bounds.
    :return list[str]: The violations in config order, a constraint the trial did not measure
        among them; empty when the values satisfy every bound.
    """
    spec = experiment.metric
    violations = _bound_violations_of(
        f"metric {spec.name!r}", metric, min_value=spec.min, max_value=spec.max
    )
    for constraint in experiment.constraints:
        label = f"constraint {constraint.name!r}"
        value = constraints.get(constraint.name)
        if value is None:
            violations.append(f"{label} was not measured")
            continue
        violations.extend(
            _bound_violations_of(label, value, min_value=constraint.min, max_value=constraint.max)
        )
    return violations


def recorded_constraints(
    trial: optuna.trial.FrozenTrial, experiment: Experiment
) -> dict[str, float]:
    """Read the constraint values a trial recorded for the configured constraints.

    :param optuna.trial.FrozenTrial trial: Trial whose constraint attrs are read.
    :param Experiment experiment: Config naming the constraints.
    :return dict[str, float]: Each configured constraint's recorded numeric value by name; a
        constraint without one, or with an integer no float can hold, is absent.
    """
    values: dict[str, float] = {}
    for constraint in experiment.constraints:
        raw = trial.user_attrs.get(constraint_attr(constraint.name))
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            try:
                values[constraint.name] = float(raw)
            except OverflowError:
                continue
    return values


def trial_bound_violations(trial: optuna.trial.FrozenTrial, experiment: Experiment) -> list[str]:
    """Judge a measured trial's recorded values with :func:`bound_violations`.

    :param optuna.trial.FrozenTrial trial: A trial Optuna holds ``COMPLETE``.
    :param Experiment experiment: Config supplying the current bounds.
    :return list[str]: The violations; a trial without a metric value violates the metric.
    """
    if trial.value is None:
        return [f"metric {experiment.metric.name!r} was not measured"]
    return bound_violations(trial.value, recorded_constraints(trial, experiment), experiment)


def measured_reason(trial: optuna.trial.FrozenTrial, experiment: Experiment) -> str | None:
    """State what a measured trial's reason line says, in ``run`` and ``status --trials`` alike.

    That is why it is infeasible under the current bounds, and that it was scored from a
    result an earlier attempt of the trial wrote, which the measured attempt did not rewrite.

    :param optuna.trial.FrozenTrial trial: Trial to describe.
    :param Experiment experiment: Config supplying the current bounds.
    :return str | None: The statements, or ``None`` for a trial that is not ``COMPLETE`` or
        has neither to state.
    """
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return None
    notes = []
    if violations := trial_bound_violations(trial, experiment):
        notes.append(f"outside the metric bounds or constraints: {'; '.join(violations)}")
    if trial.user_attrs.get(EARLIER_ATTEMPT_RESULT_ATTR) is True:
        notes.append(
            "scored from a result an earlier attempt of this trial wrote; this attempt exited 0 "
            "without rewriting it"
        )
    return "; ".join(notes) or None
