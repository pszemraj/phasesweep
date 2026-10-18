"""Trainer-side helpers for publishing objective evidence."""

from __future__ import annotations

import json
import math
import numbers
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from phasesweep.runtime.files import atomic_write_text
from phasesweep.runtime.json import nonempty_text, progress_position

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


def _json_number(value: Any) -> int | float | None:
    """Convert a Python or NumPy real scalar to the JSON number it reports.

    NumPy scalars register as :class:`numbers.Integral` or :class:`numbers.Real`,
    so an ``np.int64`` token count or ``np.float32`` loss reports like a Python
    number. Bools and tensors are not numbers here.

    :param Any value: Candidate number.
    :return int | float | None: The ``int`` or ``float``, or ``None`` if
        ``value`` is not a real number or a float overflows.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    try:
        return float(value)
    except OverflowError:
        return None


def _progress_values(progress: Mapping[str, Any]) -> dict[str, int | float]:
    """Validate named training-progress counters reported beside the step.

    :param Mapping[str, Any] progress: Axis name to position, such as
        ``{"tokens": 2_000_000}``.
    :raises ValueError: If a name is not a string, is empty or ``step`` (the
        envelope's own axis), or a position is not a finite, non-negative number.
    :return dict[str, int | float]: The validated counters as JSON numbers.
    """
    values: dict[str, int | float] = {}
    for axis, position in progress.items():
        if not isinstance(axis, str):
            raise ValueError(f"progress names must be strings, got {axis!r}.")
        if not axis or axis == "step":
            raise ValueError(
                f"progress names must be nonempty and not 'step' (the envelope's own "
                f"step), got {axis!r}."
            )
        number = progress_position(_json_number(position))
        if number is None:
            raise ValueError(
                f"progress[{axis!r}] must be a finite non-negative number, got {position!r}."
            )
        values[axis] = number
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
    """Atomically publish this attempt's objective as a result envelope.

    The managed trial environment supplies destination and attempt identity.
    Extra JSON fields may serve constraints, observations, or gates but cannot
    replace envelope fields. Missing destination parent directories are created.

    :param float value: Finite objective value produced by the evaluation. Here
        and in ``step`` and ``progress``, NumPy scalars count as numbers;
        call ``.item()`` on a tensor.
    :param str name: Objective name declared by the metric extractor.
    :param str split: Evaluated data split.
    :param str policy: Evaluation policy, the exact string the metric extractor's
        ``policy`` names, such as ``final`` or ``best``.
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
    number = _json_number(value)
    if number is None:
        raise ValueError(f"value must be a real number, got {value!r}.")
    try:
        numeric_value = float(number)
    except OverflowError:
        numeric_value = math.inf
    if not math.isfinite(numeric_value):
        raise ValueError(f"value must be finite, got {value!r}.")
    step_number = _json_number(step)
    if not isinstance(step_number, int) or step_number < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}.")
    evaluation: dict[str, Any] = {
        "policy": nonempty_text(policy, label="policy"),
        "checkpoint": nonempty_text(checkpoint, label="checkpoint"),
        "step": step_number,
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
            "name": nonempty_text(name, label="name"),
            "split": nonempty_text(split, label="split"),
            "value": numeric_value,
        },
        "evaluation": evaluation,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(destination, text)
    return destination
