"""Trainer-side helpers for publishing objective evidence."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from phasesweep.runtime.files import atomic_write_text

_IDENTITY_ENV = (
    "PHASESWEEP_GENERATION_ID",
    "PHASESWEEP_ATTEMPT_ID",
    "PHASESWEEP_OVERRIDES_SHA256",
)
_RESERVED_FIELDS = {
    "schema_version",
    "status",
    "generation_id",
    "attempt_id",
    "overrides_sha256",
    "objective",
    "evaluation",
}


def _required_trial_environment(name: str) -> str:
    """Read one nonempty value from the PhaseSweep-managed trial environment.

    :param str name: Environment variable required by the result-envelope contract.
    :raises RuntimeError: If the helper is called outside a compatible PhaseSweep trial.
    :return str: The environment value.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"report_objective() requires {name}; call it from a PhaseSweep trial "
            "using a json_envelope metric extractor."
        )
    return value


def _nonempty_string(value: str, *, label: str) -> str:
    """Require a nonempty string used to describe an evaluation.

    :param str value: Candidate string.
    :param str label: Argument name used in the error.
    :raises ValueError: If ``value`` is not a nonempty string.
    :return str: The validated value.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string.")
    return value


def _progress_values(progress: Mapping[str, int | float]) -> dict[str, int | float]:
    """Validate named training-progress counters reported beside the step.

    :param Mapping[str, int | float] progress: Axis name to position, such as
        ``{"tokens": 2_000_000}``.
    :raises ValueError: If a name is empty or ``step`` (the envelope's own
        axis), or a position is not a finite, non-negative JSON number.
    :return dict[str, int | float]: The validated counters.
    """
    values: dict[str, int | float] = {}
    for axis, position in progress.items():
        if not isinstance(axis, str) or not axis or axis == "step":
            raise ValueError(
                f"progress names must be nonempty and not 'step' (the envelope's own "
                f"step), got {axis!r}."
            )
        try:
            valid = (
                not isinstance(position, bool)
                and isinstance(position, int | float)
                and math.isfinite(position)
                and position >= 0
            )
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(
                f"progress[{axis!r}] must be a finite non-negative number, got {position!r}."
            )
        values[axis] = position
    return values


def report_objective(
    value: float,
    *,
    name: str,
    split: str,
    policy: str,
    checkpoint: str,
    step: int,
    progress: Mapping[str, int | float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically publish this trial's objective as a PhaseSweep result envelope.

    The destination and attempt identity come from environment variables injected
    for a ``json_envelope`` metric extractor. ``extra`` may add top-level JSON
    fields for constraint extractors or evidence gates, but cannot replace the
    envelope fields managed by this helper. Missing parent directories in a
    nested configured destination are created by the atomic artifact writer.

    :param float value: Finite objective value produced by the evaluation.
    :param str name: Objective name declared by the metric extractor.
    :param str split: Evaluated data split.
    :param str policy: Evaluation policy, such as ``best_checkpoint`` or
        ``final_checkpoint``.
    :param str checkpoint: Nonempty checkpoint identity.
    :param int step: Non-negative evaluation step.
    :param Mapping[str, int | float] | None progress: Optional positions on other
        training-progress axes at this evaluation, such as ``{"tokens": n}``. An
        extractor whose ``evaluation_axis`` names one of them compares trials on
        it instead of ``step``.
    :param Mapping[str, Any] | None extra: Optional additional top-level JSON fields.
    :raises RuntimeError: If called outside a compatible PhaseSweep trial.
    :raises ValueError: If objective metadata is invalid or ``extra`` replaces a
        reserved envelope field.
    :raises TypeError: If ``extra`` cannot be encoded as JSON.
    :raises OSError: If the completed envelope cannot be written atomically.
    :return Path: Configured objective path replaced by the completed envelope.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"value must be a JSON number, got {value!r}.")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"value must be finite, got {value!r}.")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}.")
    evaluation: dict[str, Any] = {
        "policy": _nonempty_string(policy, label="policy"),
        "checkpoint": _nonempty_string(checkpoint, label="checkpoint"),
        "step": step,
    }
    if progress is not None:
        evaluation["progress"] = _progress_values(progress)

    destination = Path(_required_trial_environment("PHASESWEEP_OBJECTIVE_PATH"))
    identity = {key: _required_trial_environment(key) for key in _IDENTITY_ENV}
    additional = dict(extra or {})
    collisions = sorted(_RESERVED_FIELDS.intersection(additional))
    if collisions:
        raise ValueError(f"extra cannot replace reserved result field(s): {collisions!r}.")

    payload: dict[str, Any] = {
        **additional,
        "schema_version": 1,
        "status": "complete",
        "generation_id": identity["PHASESWEEP_GENERATION_ID"],
        "attempt_id": identity["PHASESWEEP_ATTEMPT_ID"],
        "overrides_sha256": identity["PHASESWEEP_OVERRIDES_SHA256"],
        "objective": {
            "name": _nonempty_string(name, label="name"),
            "split": _nonempty_string(split, label="split"),
            "value": numeric_value,
        },
        "evaluation": evaluation,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(destination, text)
    return destination
