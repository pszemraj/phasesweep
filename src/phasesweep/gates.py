"""Evidence gates for completed trials."""

from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _json_path(data: Any, key: str) -> Any:
    """Resolve a dotted key in a JSON-like object."""
    cur = data
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(part)
    return cur


def _load_json(trial_dir: Path, relative: str) -> Any:
    """Load JSON under ``trial_dir``."""
    path = trial_dir / relative
    return json.loads(path.read_text())


def _required_file(ctx: TrialContext, gate: RequiredFileGate) -> GateResult:
    path = ctx.trial_dir / gate.path
    if path.is_file():
        return GateResult(gate.type, True, f"{gate.path} exists")
    return GateResult(gate.type, False, f"{gate.path} is missing")


def _json_equals(ctx: TrialContext, gate: JsonEqualsGate) -> GateResult:
    try:
        actual = _json_path(_load_json(ctx.trial_dir, gate.path), gate.key)
    except Exception as exc:  # noqa: BLE001
        return GateResult(gate.type, False, f"{gate.path}:{gate.key} unavailable: {exc}")
    if actual == gate.value:
        return GateResult(gate.type, True, f"{gate.key} == {gate.value!r}")
    return GateResult(gate.type, False, f"{gate.key} was {actual!r}, expected {gate.value!r}")


def _json_scalar_bound(ctx: TrialContext, gate: JsonScalarBoundGate) -> GateResult:
    try:
        value = float(_json_path(_load_json(ctx.trial_dir, gate.path), gate.key))
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
    if not path.is_file():
        return GateResult(gate.type, False, f"{gate.path} is missing")
    size = path.stat().st_size
    if gate.min_bytes is not None and size < gate.min_bytes:
        return GateResult(gate.type, False, f"{gate.path} size {size} < {gate.min_bytes}")
    if gate.max_bytes is not None and size > gate.max_bytes:
        return GateResult(gate.type, False, f"{gate.path} size {size} > {gate.max_bytes}")
    return GateResult(gate.type, True, f"{gate.path} size {size} within bounds")


def _sha256(ctx: TrialContext, gate: Sha256Gate) -> GateResult:
    path = ctx.trial_dir / gate.path
    if not path.is_file():
        return GateResult(gate.type, False, f"{gate.path} is missing")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest == gate.sha256:
        return GateResult(gate.type, True, f"{gate.path} sha256 matched")
    return GateResult(gate.type, False, f"{gate.path} sha256 {digest} != {gate.sha256}")


def _wandb_summary_required(ctx: TrialContext, gate: WandbSummaryRequiredGate) -> GateResult:
    try:
        from wandb.apis.public import Api  # type: ignore[import-not-found]
    except ImportError:
        return GateResult(gate.type, False, "wandb package is not installed")

    api = Api()
    target_name = gate.run_name_template.format(
        experiment=ctx.experiment,
        phase=ctx.phase,
        trial_id=ctx.trial_id,
        run_name=ctx.run_name,
    )
    path = f"{gate.entity}/{gate.project}"
    deadline = time.time() + gate.timeout_seconds
    last_err: Exception | None = None
    while time.time() < deadline:
        remaining = max(0.0, deadline - time.time())
        try:
            runs = _call_with_timeout(
                lambda: api.runs(path, filters={"display_name": target_name}),
                timeout=min(gate.poll_seconds, remaining),
            )
            if len(runs) >= 1:
                run = runs[0]
                if run.state in {"finished", "crashed", "failed"}:
                    summary = dict(run.summary)
                    missing = [key for key in gate.keys if key not in summary]
                    if not missing:
                        return GateResult(gate.type, True, f"W&B summary has {gate.keys}")
                    return GateResult(gate.type, False, f"W&B summary missing {missing}")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(gate.poll_seconds)

    detail = f"W&B run {target_name!r} not ready within {gate.timeout_seconds}s"
    if last_err is not None:
        detail += f"; last error: {last_err}"
    return GateResult(gate.type, False, detail)


def _call_with_timeout(fn: Any, *, timeout: float) -> Any:
    """Run a blocking function in a daemon thread and bound caller wait time."""
    q: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            q.put((True, fn()))
        except Exception as exc:  # noqa: BLE001
            q.put((False, exc))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=max(0.0, timeout))
    if thread.is_alive():
        raise TimeoutError(f"call exceeded {timeout:g}s")
    ok, value = q.get_nowait()
    if ok:
        return value
    raise value


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
