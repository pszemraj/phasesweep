"""Pick the winning trial from a completed phase. Deterministic. No LLM."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import optuna

from phasesweep.config import Experiment


@dataclass
class SelectedTrial:
    """The best feasible trial from a completed phase.

    Does not include effective_overrides — the orchestrator adds those.
    """

    trial_number: int
    params: dict[str, Any]
    metric: float
    constraints: dict[str, float] = field(default_factory=dict)


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
        constraint readings).

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
        if not t.user_attrs.get("phasesweep_feasible", False):
            continue
        # Re-verify constraints from user_attrs in case rules changed or stored
        # values are non-finite (defense in depth — review item #3).
        ok = True
        for name, c in constraints_by_name.items():
            v = t.user_attrs.get(f"constraint:{name}")
            if v is None:
                ok = False
                break
            try:
                v_f = float(v)
            except (TypeError, ValueError):
                ok = False
                break
            if not math.isfinite(v_f):
                ok = False
                break
            if c.max is not None and v_f > c.max:
                ok = False
                break
            if c.min is not None and v_f < c.min:
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

    def key(t: optuna.trial.FrozenTrial) -> tuple[float, int]:
        """Sort key for ``min(survivors, ...)``.

        Args:
            t: A survivor trial (already filtered to COMPLETE + finite value).

        Returns:
            ``(signed_metric, trial_number)``. The sign is flipped for maximize
            so ``min`` picks the best in both directions. Trial number is the
            tiebreaker (lower wins).

        """
        # Survivors all passed `t.value is None or not isfinite` filtering above,
        # so t.value is a finite float here. The `assert` narrows for mypy.
        assert t.value is not None
        v = t.value if minimize else -t.value
        return (v, t.number)

    best = min(survivors, key=key)
    assert best.value is not None  # same invariant

    constraint_vals = {
        name: float(best.user_attrs[f"constraint:{name}"]) for name in constraints_by_name
    }

    return SelectedTrial(
        trial_number=best.number,
        params=dict(best.params),
        metric=float(best.value),
        constraints=constraint_vals,
    )
