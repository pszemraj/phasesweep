"""Extractor dispatch.

Adding a new extractor: write a function `extract(ctx, cfg) -> float` and register
it in EXTRACTORS below. That's it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from phasesweep._runtime import (
    WandbPollTimeout,
    json_path,
    load_json_file,
    poll_wandb_summary,
    render_trial_run_name,
)
from phasesweep.config import (
    Extractor,
    JsonExtractor,
    LogRegexExtractor,
    WandbExtractor,
)


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
    target = ctx.trial_dir / cfg.path
    if not target.is_file():
        raise ExtractorError(f"JSON file not found: {target}")
    try:
        data = load_json_file(target)
    except JSONDecodeError as exc:
        raise ExtractorError(f"Invalid JSON at {target}: {exc}") from exc

    try:
        cur = json_path(data, cfg.key)
    except KeyError as exc:
        raise ExtractorError(
            f"Key {cfg.key!r} not found in {target} (failed at {exc.args[0]!r})."
        ) from exc

    try:
        return float(cur)
    except (TypeError, ValueError) as exc:
        raise ExtractorError(f"Value at {cfg.key!r} is not numeric: {cur!r}") from exc


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
            "Install with: pip install phasesweep[wandb]"
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
