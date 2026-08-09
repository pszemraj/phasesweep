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


def report_objective(
    value: float,
    *,
    name: str,
    split: str,
    policy: str,
    checkpoint: str,
    step: int,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically publish this trial's objective as a PhaseSweep result envelope.

    The destination and attempt identity come from environment variables injected
    for a ``json_envelope`` metric extractor. ``extra`` may add top-level JSON
    fields for constraint extractors or evidence gates, but cannot replace the
    envelope fields managed by this helper.

    :param float value: Finite objective value produced by the evaluation.
    :param str name: Objective name declared by the metric extractor.
    :param str split: Evaluated data split.
    :param str policy: Evaluation policy, such as ``best_checkpoint`` or
        ``final_checkpoint``.
    :param str checkpoint: Nonempty checkpoint identity.
    :param int step: Non-negative evaluation step.
    :param Mapping[str, Any] | None extra: Optional additional top-level JSON fields.
    :raises RuntimeError: If called outside a compatible PhaseSweep trial.
    :raises ValueError: If objective metadata is invalid or ``extra`` replaces a
        reserved envelope field.
    :return Path: Configured objective path replaced by the completed envelope.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"value must be a JSON number, got {value!r}.")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"value must be finite, got {value!r}.")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}.")

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
        "evaluation": {
            "policy": _nonempty_string(policy, label="policy"),
            "checkpoint": _nonempty_string(checkpoint, label="checkpoint"),
            "step": step,
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(destination, text)
    return destination
