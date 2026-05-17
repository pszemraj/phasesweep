"""Evidence extraction and post-trial gate evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from phasesweep.evidence.models import (
    ArtifactSizeGate,
    Extractor,
    Gate,
    JsonEqualsGate,
    JsonExtractor,
    JsonScalarBoundGate,
    LogRegexExtractor,
    RequiredFileGate,
    Sha256Gate,
    WandbExtractor,
    WandbSummaryRequiredGate,
)
from phasesweep.evidence.wandb import (
    WandbPollTimeout,
    poll_wandb_summary,
    render_trial_run_name,
)


def load_json_value(trial_dir: Path, relative_path: str, key: str) -> tuple[Path, Any]:
    """Load a dotted JSON value from a trial-relative file."""
    target = trial_dir / relative_path
    if not target.is_file():
        raise FileNotFoundError(target)
    cur = json.loads(target.read_text())
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            continue
        raise KeyError(part)
    return target, cur


def json_float(value: Any, *, label: str) -> float:
    """Coerce a JSON value to float with a keyed error message."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value at {label!r} is not numeric: {value!r}") from exc


class ExtractorError(RuntimeError):
    """Raised when an extractor cannot produce a value (file missing, key missing, etc).

    The phase runner catches this and marks the trial as failed.
    """


@dataclass(frozen=True)
class TrialContext:
    """Everything an extractor might need to find a trial's result."""

    experiment: str
    phase: str
    trial_id: int
    trial_dir: Path
    run_name: str  # "{experiment}-{phase}-{trial_id}"
    return_code: int
    duration_seconds: float


def _extract_json(ctx: TrialContext, cfg: JsonExtractor) -> float:
    """Read a JSON file in the trial directory and extract a numeric value.

    Args:
        ctx: Trial context (used for ``trial_dir``).
        cfg: ``JsonExtractor`` config naming the relative file path and the
            dotted lookup key inside it.

    Returns:
        The numeric value at the configured key.

    Raises:
        ExtractorError: File missing, invalid JSON, key not found, or value
            not coercible to ``float``.

    """
    try:
        target, cur = load_json_value(ctx.trial_dir, cfg.path, cfg.key)
    except FileNotFoundError as exc:
        raise ExtractorError(f"JSON file not found: {exc.args[0]}") from exc
    except JSONDecodeError as exc:
        target = ctx.trial_dir / cfg.path
        raise ExtractorError(f"Invalid JSON at {target}: {exc}") from exc
    except KeyError as exc:
        target = ctx.trial_dir / cfg.path
        raise ExtractorError(
            f"Key {cfg.key!r} not found in {target} (failed at {exc.args[0]!r})."
        ) from exc

    try:
        return json_float(cur, label=cfg.key)
    except ValueError as exc:
        raise ExtractorError(str(exc)) from exc


def _extract_log_regex(ctx: TrialContext, cfg: LogRegexExtractor) -> float:
    """Scan a log file line-by-line and return the value of a named regex group.

    Args:
        ctx: Trial context (used for ``trial_dir``).
        cfg: ``LogRegexExtractor`` config; ``pattern`` must contain a named
            group ``(?P<value>...)``, and ``select`` is one of
            ``"first"``/``"last"``/``"min"``/``"max"``.

    Returns:
        The selected numeric value across all matches.

    Raises:
        ExtractorError: Log file missing, invalid regex, missing ``value``
            group, or no lines matched.

    """
    import re

    target = ctx.trial_dir / cfg.file
    if not target.is_file():
        raise ExtractorError(f"Log file not found: {target}")

    try:
        pattern = re.compile(cfg.pattern)
    except re.error as exc:
        raise ExtractorError(f"Invalid regex {cfg.pattern!r}: {exc}") from exc

    if "value" not in pattern.groupindex:
        raise ExtractorError(f"Regex {cfg.pattern!r} must contain a named group 'value'.")

    # Stream line-by-line to avoid 500 MB RSS on large training logs.
    result: float | None = None
    count = 0
    with target.open() as fh:
        for line in fh:
            m = pattern.search(line)
            if m is None:
                continue
            try:
                v = float(m.group("value"))
            except (TypeError, ValueError):
                continue
            count += 1
            if cfg.select == "first":
                return v
            if cfg.select == "last":
                result = v
            elif cfg.select == "min":
                result = v if result is None else min(result, v)
            elif cfg.select == "max":
                result = v if result is None else max(result, v)

    if count == 0:
        raise ExtractorError(f"No matches for {cfg.pattern!r} in {target}.")
    assert result is not None  # count > 0 guarantees this for last/min/max
    return result


def _extract_wandb(ctx: TrialContext, cfg: WandbExtractor) -> float:
    """Poll the W&B public API for a run by display name and return a summary metric.

    Args:
        ctx: Trial context. ``experiment``/``phase``/``trial_id``/``run_name``
            are substituted into ``cfg.run_name_template``.
        cfg: ``WandbExtractor`` config: entity, project, run-name template,
            metric key, poll cadence, and timeout.

    Returns:
        The numeric value of ``cfg.metric_key`` on the finished/crashed/failed run.

    Raises:
        ExtractorError: ``wandb`` not installed, run not found before timeout,
            or metric key missing on the finished run's summary.

    """
    target_name = render_trial_run_name(cfg.run_name_template, ctx)
    try:
        summary = poll_wandb_summary(
            entity=cfg.entity,
            project=cfg.project,
            run_name=target_name,
            poll_seconds=cfg.poll_seconds,
            timeout_seconds=cfg.timeout_seconds,
            required_keys=[cfg.metric_key],
        )
    except ImportError as exc:
        raise ExtractorError(
            "W&B extractor requested but the 'wandb' package is not installed. "
            "Install the wandb extra for the same distribution, for example: "
            'python -m pip install "phasesweep[wandb] @ '
            'git+https://github.com/pszemraj/phasesweep.git"'
        ) from exc
    except WandbPollTimeout as exc:
        msg = (
            f"W&B run {target_name!r} not found or metric {cfg.metric_key!r} "
            f"missing within {cfg.timeout_seconds}s."
        )
        if exc.last_error is not None:
            msg += f" Last error: {exc.last_error}"
        raise ExtractorError(msg) from exc

    try:
        return float(summary[cfg.metric_key])
    except (TypeError, ValueError) as exc:
        raise ExtractorError(
            f"Value at W&B metric {cfg.metric_key!r} is not numeric: {summary[cfg.metric_key]!r}"
        ) from exc


_DISPATCH: dict[type, Callable[[TrialContext, Any], float]] = {
    JsonExtractor: _extract_json,
    LogRegexExtractor: _extract_log_regex,
    WandbExtractor: _extract_wandb,
}


def run_extractor(ctx: TrialContext, cfg: Extractor) -> float:
    """Dispatch to the appropriate extractor for ``cfg``.

    Args:
        ctx: Trial context passed through to the chosen extractor.
        cfg: A concrete extractor config (one of :class:`JsonExtractor`,
            :class:`LogRegexExtractor`, :class:`WandbExtractor`).

    Returns:
        The numeric value the extractor pulled from this trial's outputs.

    Raises:
        ExtractorError: No extractor is registered for the given config type,
            or the chosen extractor failed.

    """
    fn = _DISPATCH.get(type(cfg))
    if fn is None:
        raise ExtractorError(f"No extractor registered for {type(cfg).__name__}.")
    return fn(ctx, cfg)


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
        _, actual = load_json_value(ctx.trial_dir, gate.path, gate.key)
    except Exception as exc:  # noqa: BLE001
        return GateResult(gate.type, False, f"{gate.path}:{gate.key} unavailable: {exc}")
    if type(actual) is type(gate.value) and actual == gate.value:
        return GateResult(gate.type, True, f"{gate.key} == {gate.value!r}")
    return GateResult(
        gate.type,
        False,
        f"{gate.key} was {actual!r} ({type(actual).__name__}), "
        f"expected {gate.value!r} ({type(gate.value).__name__})",
    )


def _json_scalar_bound(ctx: TrialContext, gate: JsonScalarBoundGate) -> GateResult:
    try:
        _, raw_value = load_json_value(ctx.trial_dir, gate.path, gate.key)
        value = json_float(raw_value, label=gate.key)
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
            _, raw_size = load_json_value(ctx.trial_dir, gate.path, gate.key)
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


_GATE_DISPATCH: dict[type, Callable[[TrialContext, Any], GateResult]] = {
    RequiredFileGate: _required_file,
    JsonEqualsGate: _json_equals,
    JsonScalarBoundGate: _json_scalar_bound,
    ArtifactSizeGate: _artifact_size,
    Sha256Gate: _sha256,
    WandbSummaryRequiredGate: _wandb_summary_required,
}


def evaluate_gates(ctx: TrialContext, gates: list[Gate]) -> list[GateResult]:
    """Evaluate all gates against a completed trial context."""
    results: list[GateResult] = []
    for gate in gates:
        fn = _GATE_DISPATCH.get(type(gate))
        if fn is None:  # pragma: no cover - closed union
            results.append(GateResult(type(gate).__name__, False, f"unknown gate: {gate!r}"))
            continue
        results.append(fn(ctx, gate))
    return results
