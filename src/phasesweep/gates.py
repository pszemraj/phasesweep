"""Evidence gates for completed trials."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from phasesweep._runtime import (
    WandbPollTimeout,
    json_path,
    load_json_file,
    poll_wandb_summary,
    render_trial_run_name,
)
from phasesweep.config import (
    ArtifactSizeGate,
    Gate,
    JsonEqualsGate,
    JsonScalarBoundGate,
    RequiredFileGate,
    Sha256Gate,
    WandbSummaryRequiredGate,
)
from phasesweep.extractors.base import TrialContext


@dataclass(frozen=True)
class GateResult:
    """Result of one evidence gate evaluation."""

    gate_type: str
    passed: bool
    detail: str


def _required_file(ctx: TrialContext, gate: RequiredFileGate) -> GateResult:
    path = ctx.trial_dir / gate.path
    if path.is_file():
        return GateResult(gate.type, True, f"{gate.path} exists")
    return GateResult(gate.type, False, f"{gate.path} is missing")


def _json_equals(ctx: TrialContext, gate: JsonEqualsGate) -> GateResult:
    try:
        actual = json_path(load_json_file(ctx.trial_dir / gate.path), gate.key)
    except Exception as exc:  # noqa: BLE001
        return GateResult(gate.type, False, f"{gate.path}:{gate.key} unavailable: {exc}")
    if actual == gate.value:
        return GateResult(gate.type, True, f"{gate.key} == {gate.value!r}")
    return GateResult(gate.type, False, f"{gate.key} was {actual!r}, expected {gate.value!r}")


def _json_scalar_bound(ctx: TrialContext, gate: JsonScalarBoundGate) -> GateResult:
    try:
        value = float(json_path(load_json_file(ctx.trial_dir / gate.path), gate.key))
    except Exception as exc:  # noqa: BLE001
        return GateResult(gate.type, False, f"{gate.path}:{gate.key} unavailable: {exc}")
    if not math.isfinite(value):
        return GateResult(gate.type, False, f"{gate.key} was non-finite: {value!r}")
    if gate.min is not None and value < gate.min:
        return GateResult(gate.type, False, f"{gate.key}={value:g} < min {gate.min:g}")
    if gate.max is not None and value > gate.max:
        return GateResult(gate.type, False, f"{gate.key}={value:g} > max {gate.max:g}")
    return GateResult(gate.type, True, f"{gate.key}={value:g} within bounds")


def _artifact_size(ctx: TrialContext, gate: ArtifactSizeGate) -> GateResult:
    path = ctx.trial_dir / gate.path
    if gate.source == "file":
        if not path.is_file():
            return GateResult(gate.type, False, f"{gate.path} is not a file")
        size = path.stat().st_size
        label = f"{gate.path} file size"
    elif gate.source == "directory":
        if not path.is_dir():
            return GateResult(gate.type, False, f"{gate.path} is not a directory")
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        label = f"{gate.path} directory size"
    else:
        assert gate.key is not None
        try:
            raw_size = json_path(load_json_file(path), gate.key)
        except Exception as exc:  # noqa: BLE001
            return GateResult(gate.type, False, f"{gate.path}:{gate.key} unavailable: {exc}")
        if not isinstance(raw_size, int) or isinstance(raw_size, bool):
            return GateResult(gate.type, False, f"{gate.key} was not an integer byte count")
        if raw_size < 0:
            return GateResult(gate.type, False, f"{gate.key} was negative: {raw_size}")
        size = raw_size
        label = f"{gate.path}:{gate.key}"
    if gate.min_bytes is not None and size < gate.min_bytes:
        return GateResult(gate.type, False, f"{label} {size} < {gate.min_bytes}")
    if gate.max_bytes is not None and size > gate.max_bytes:
        return GateResult(gate.type, False, f"{label} {size} > {gate.max_bytes}")
    return GateResult(gate.type, True, f"{label} {size} within bounds")


def _sha256(ctx: TrialContext, gate: Sha256Gate) -> GateResult:
    path = ctx.trial_dir / gate.path
    if not path.is_file():
        return GateResult(gate.type, False, f"{gate.path} is missing")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest == gate.sha256:
        return GateResult(gate.type, True, f"{gate.path} sha256 matched")
    return GateResult(gate.type, False, f"{gate.path} sha256 {digest} != {gate.sha256}")


def _wandb_summary_required(ctx: TrialContext, gate: WandbSummaryRequiredGate) -> GateResult:
    target_name = render_trial_run_name(gate.run_name_template, ctx)
    try:
        summary = poll_wandb_summary(
            entity=gate.entity,
            project=gate.project,
            run_name=target_name,
            poll_seconds=gate.poll_seconds,
            timeout_seconds=gate.timeout_seconds,
            wait_for_keys=False,
        )
    except ImportError:
        return GateResult(gate.type, False, "wandb package is not installed")
    except WandbPollTimeout as exc:
        detail = f"W&B run {target_name!r} not ready within {gate.timeout_seconds}s"
        if exc.last_error is not None:
            detail += f"; last error: {exc.last_error}"
        return GateResult(gate.type, False, detail)

    missing = [key for key in gate.keys if key not in summary]
    if not missing:
        return GateResult(gate.type, True, f"W&B summary has {gate.keys}")
    return GateResult(gate.type, False, f"W&B summary missing {missing}")


def evaluate_gates(ctx: TrialContext, gates: list[Gate]) -> list[GateResult]:
    """Evaluate all gates against a completed trial context."""
    results: list[GateResult] = []
    for gate in gates:
        if isinstance(gate, RequiredFileGate):
            result = _required_file(ctx, gate)
        elif isinstance(gate, JsonEqualsGate):
            result = _json_equals(ctx, gate)
        elif isinstance(gate, JsonScalarBoundGate):
            result = _json_scalar_bound(ctx, gate)
        elif isinstance(gate, ArtifactSizeGate):
            result = _artifact_size(ctx, gate)
        elif isinstance(gate, Sha256Gate):
            result = _sha256(ctx, gate)
        elif isinstance(gate, WandbSummaryRequiredGate):
            result = _wandb_summary_required(ctx, gate)
        else:  # pragma: no cover - closed union
            result = GateResult(type(gate).__name__, False, f"unknown gate: {gate!r}")
        results.append(result)
    return results


__all__ = ["GateResult", "evaluate_gates"]
