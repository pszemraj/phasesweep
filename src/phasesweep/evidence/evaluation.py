"""Evidence extraction and post-trial gate evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from phasesweep.evidence.models import (
    ArtifactSizeGate,
    Extractor,
    Gate,
    JsonEnvelopeExtractor,
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
    WandbRunTerminalError,
    WandbSetupError,
    poll_wandb_summary,
)
from phasesweep.runtime.json import strict_json_loads

# Version of the objective evidence provenance payload frozen alongside a
# metric at extraction time (review v0.5.17 / finding F).
EVIDENCE_PROVENANCE_SCHEMA_VERSION = 1


def extractor_config_fingerprint(cfg: Extractor) -> str:
    """Return the SHA-256 identity of an extractor's exact configuration.

    Frozen into evidence provenance so a forensic review can prove which
    extractor contract produced a published scalar even after the experiment
    config changes (review v0.5.17 / finding F).

    :param Extractor cfg: Concrete extractor config to fingerprint.
    :return str: Hex SHA-256 of the extractor's canonical JSON dump.
    """
    dumped = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _file_digest(raw: bytes) -> dict[str, Any]:
    """Return the digest fields shared by every file-based evidence source.

    :param bytes raw: Exact evidence bytes as read for extraction.
    :return dict[str, Any]: ``sha256`` and ``size_bytes`` of those bytes.
    """
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def load_json_value(
    trial_dir: Path,
    relative_path: str,
    key: str,
    *,
    digest: dict[str, Any] | None = None,
) -> tuple[Path, Any]:
    """Load a dotted JSON value from a trial-relative file.

    :param Path trial_dir: Directory containing the trial outputs.
    :param str relative_path: JSON file path relative to ``trial_dir``.
    :param str key: Dot-separated key path to read from the JSON object.
    :param dict[str, Any] | None digest: Optional sink that receives the
        ``sha256``/``size_bytes`` of the exact bytes parsed, for evidence
        provenance (review v0.5.17 / finding F).
    :return tuple[Path, Any]: Resolved JSON file path and loaded value.
    """
    target = trial_dir / relative_path
    if not target.is_file():
        raise FileNotFoundError(target)
    raw = target.read_bytes()
    if digest is not None:
        digest.update(_file_digest(raw))
    cur = strict_json_loads(raw.decode("utf-8"))
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            continue
        raise KeyError(part)
    return target, cur


def json_float(value: Any, *, label: str) -> float:
    """Require a JSON number and convert it to float with a keyed error message.

    :param Any value: JSON scalar value to coerce.
    :param str label: Human-readable key or metric label for errors.
    :raises ValueError: If ``value`` is not an integer or float, excluding booleans.
    :return float: Coerced numeric value.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Value at {label!r} is not a JSON number: {value!r}")
    return float(value)


class ExtractorError(RuntimeError):
    """Raised when an extractor cannot produce a value (file missing, key missing, etc).

    The phase runner catches this and marks the trial as failed.
    """


class DeadlineExceededError(ExtractorError):
    """Raised when the phase/run deadline directly prevents extraction."""


@dataclass(frozen=True)
class TrialContext:
    """Everything an extractor might need to find a trial's result."""

    experiment: str
    phase: str
    trial_id: int
    generation_id: str
    attempt_id: str
    overrides_sha256: str
    trial_dir: Path
    run_name: str  # "{experiment}-{phase}-{trial_id}-{attempt_id}"
    return_code: int
    duration_seconds: float


def _extract_json(
    ctx: TrialContext, cfg: JsonExtractor, provenance: dict[str, Any] | None = None
) -> float:
    """Read a JSON file in the trial directory and extract a numeric value.

    Args:
        ctx: Trial context (used for ``trial_dir``).
        cfg: ``JsonExtractor`` config naming the relative file path and the
            dotted lookup key inside it.
        provenance: Optional sink that receives the frozen evidence ``source``
            payload on success (review v0.5.17 / finding F).

    Returns:
        The numeric value at the configured key.

    Raises:
        ExtractorError: File missing, invalid JSON, key not found, or value
            not coercible to ``float``.

    """
    digest: dict[str, Any] = {}
    try:
        target, cur = load_json_value(ctx.trial_dir, cfg.path, cfg.key, digest=digest)
    except FileNotFoundError as exc:
        raise ExtractorError(f"JSON file not found: {exc.args[0]}") from exc
    except UnicodeError as exc:
        target = ctx.trial_dir / cfg.path
        raise ExtractorError(f"JSON at {target} is not valid UTF-8: {exc}") from exc
    except ValueError as exc:
        target = ctx.trial_dir / cfg.path
        raise ExtractorError(f"Invalid JSON at {target}: {exc}") from exc
    except OSError as exc:
        target = ctx.trial_dir / cfg.path
        raise ExtractorError(f"Could not read JSON at {target}: {exc}") from exc
    except KeyError as exc:
        target = ctx.trial_dir / cfg.path
        raise ExtractorError(
            f"Key {cfg.key!r} not found in {target} (failed at {exc.args[0]!r})."
        ) from exc

    try:
        value = json_float(cur, label=cfg.key)
    except ValueError as exc:
        raise ExtractorError(str(exc)) from exc
    if provenance is not None:
        provenance["source"] = {"kind": "file", "path": cfg.path, "key": cfg.key, **digest}
    return value


def _extract_json_envelope(
    ctx: TrialContext, cfg: JsonEnvelopeExtractor, provenance: dict[str, Any] | None = None
) -> float:
    """Validate and extract an attempt-bound JSON result envelope.

    :param TrialContext ctx: Current trial identity and resolved-overrides digest.
    :param JsonEnvelopeExtractor cfg: Expected objective and evaluation policy.
    :param dict[str, Any] | None provenance: Optional sink that receives the
        frozen evidence ``source`` payload — envelope digest plus the
        validated evaluation metadata — on success (review v0.5.17 / finding F).
    :raises ExtractorError: If the envelope is missing, malformed, or belongs to
        another execution attempt.
    :return float: Validated objective value from the envelope.
    """
    target = ctx.trial_dir / cfg.path
    try:
        raw = target.read_bytes()
        data = strict_json_loads(raw.decode("utf-8"))
    except FileNotFoundError as exc:
        raise ExtractorError(f"JSON envelope not found: {target}") from exc
    except UnicodeError as exc:
        raise ExtractorError(f"JSON envelope at {target} is not valid UTF-8: {exc}") from exc
    except ValueError as exc:
        raise ExtractorError(f"Invalid JSON envelope at {target}: {exc}") from exc
    except OSError as exc:
        raise ExtractorError(f"Could not read JSON envelope at {target}: {exc}") from exc

    if not isinstance(data, dict):
        raise ExtractorError(f"JSON envelope at {target} must be an object.")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ExtractorError(f"JSON envelope at {target} must use schema_version 1.")
    if data.get("status") != "complete":
        raise ExtractorError(f"JSON envelope at {target} must report status='complete'.")
    if data.get("generation_id") != ctx.generation_id:
        raise ExtractorError(
            f"JSON envelope at {target} does not match generation {ctx.generation_id!r}."
        )
    if data.get("attempt_id") != ctx.attempt_id:
        raise ExtractorError(
            f"JSON envelope at {target} does not match attempt {ctx.attempt_id!r}."
        )
    if data.get("overrides_sha256") != ctx.overrides_sha256:
        raise ExtractorError(
            f"JSON envelope at {target} does not match this attempt's resolved overrides."
        )

    objective = data.get("objective")
    if not isinstance(objective, dict):
        raise ExtractorError(f"JSON envelope at {target} has no objective object.")
    if objective.get("name") != cfg.objective_name:
        raise ExtractorError(
            f"JSON envelope at {target} does not report objective {cfg.objective_name!r}."
        )
    if objective.get("split") != cfg.split:
        raise ExtractorError(f"JSON envelope at {target} does not report split {cfg.split!r}.")

    evaluation = data.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ExtractorError(f"JSON envelope at {target} has no evaluation object.")
    if evaluation.get("policy") != cfg.policy:
        raise ExtractorError(f"JSON envelope at {target} does not report policy {cfg.policy!r}.")
    checkpoint = evaluation.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ExtractorError(f"JSON envelope at {target} has no checkpoint identity.")
    if cfg.checkpoint is not None and checkpoint != cfg.checkpoint:
        raise ExtractorError(
            f"JSON envelope at {target} does not report checkpoint {cfg.checkpoint!r}."
        )
    step = evaluation.get("step")
    if type(step) is not int or step < 0:
        raise ExtractorError(f"JSON envelope at {target} has no non-negative evaluation step.")
    if cfg.expected_step is not None and step != cfg.expected_step:
        raise ExtractorError(f"JSON envelope at {target} does not report step {cfg.expected_step}.")

    try:
        value = json_float(objective.get("value"), label="objective.value")
    except ValueError as exc:
        raise ExtractorError(str(exc)) from exc
    if not math.isfinite(value):
        raise ExtractorError(f"JSON envelope at {target} has a non-finite objective value.")
    if provenance is not None:
        provenance["source"] = {
            "kind": "file",
            "path": cfg.path,
            **_file_digest(raw),
            # Evaluation metadata as validated from the envelope body — the
            # checkpoint and step are the envelope's own reported values.
            "evaluation": {
                "objective_name": cfg.objective_name,
                "split": cfg.split,
                "policy": cfg.policy,
                "checkpoint": checkpoint,
                "step": step,
            },
        }
    return value


def _extract_log_regex(
    ctx: TrialContext, cfg: LogRegexExtractor, provenance: dict[str, Any] | None = None
) -> float:
    """Scan a log file line-by-line and return the value of a named regex group.

    Args:
        ctx: Trial context (used for ``trial_dir``).
        cfg: ``LogRegexExtractor`` config; ``pattern`` must contain a named
            group ``(?P<value>...)``, and ``select`` is one of
            ``"first"``/``"last"``/``"min"``/``"max"``.
        provenance: Optional sink that receives the frozen evidence ``source``
            payload on success: whole-file digest, the 1-based line number
            that supplied the selected value, and how many numeric matches
            were examined (review v0.5.17 / finding F). ``select: "first"``
            stops *matching* at the first hit but still reads the remaining
            bytes so the digest always covers the full file.

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

    # Stream line-by-line to avoid 500 MB RSS on large training logs. Binary
    # iteration feeds the evidence digest with the exact on-disk bytes; regex
    # matching then runs on logical lines split with text-mode universal
    # newline semantics (lone "\r" is a boundary too — tqdm-style progress
    # logs separate updates with bare carriage returns).
    hasher = hashlib.sha256()
    size_bytes = 0
    result: float | None = None
    result_line: int | None = None
    count = 0
    line_no = 0
    try:
        with target.open("rb") as fh:
            for raw_line in fh:
                hasher.update(raw_line)
                size_bytes += len(raw_line)
                if cfg.select == "first" and result is not None:
                    # Value already selected; keep reading only to finish the
                    # whole-file digest.
                    continue
                text = raw_line.decode("utf-8")
                if text.endswith("\r\n"):
                    text = text[:-2] + "\n"
                elif text.endswith("\r"):
                    text = text[:-1] + "\n"
                parts = text.split("\r")
                for line in [part + "\n" for part in parts[:-1]] + parts[-1:]:
                    line_no += 1
                    m = pattern.search(line)
                    if m is None:
                        continue
                    try:
                        v = float(m.group("value"))
                    except (TypeError, ValueError):
                        continue
                    count += 1
                    if cfg.select in ("first", "last"):
                        result = v
                        result_line = line_no
                    elif cfg.select == "min":
                        if result is None or v < result:
                            result, result_line = v, line_no
                    elif cfg.select == "max":
                        if result is None or v > result:
                            result, result_line = v, line_no
                    if cfg.select == "first":
                        break
    except UnicodeError as exc:
        raise ExtractorError(f"Log file is not valid UTF-8 at {target}: {exc}") from exc
    except OSError as exc:
        raise ExtractorError(f"Could not read log file at {target}: {exc}") from exc

    if count == 0:
        raise ExtractorError(f"No matches for {cfg.pattern!r} in {target}.")
    assert result is not None  # count > 0 guarantees this
    if provenance is not None:
        provenance["source"] = {
            "kind": "file",
            "path": cfg.file,
            "sha256": hasher.hexdigest(),
            "size_bytes": size_bytes,
            "matched_line": result_line,
            "match_count": count,
        }
    return result


def _extract_wandb(
    ctx: TrialContext, cfg: WandbExtractor, provenance: dict[str, Any] | None = None
) -> float:
    """Poll the W&B public API for this attempt's run and return a summary metric.

    Args:
        ctx: Trial context containing the immutable attempt id assigned as
            ``WANDB_RUN_ID`` before subprocess launch.
        cfg: ``WandbExtractor`` config: entity, project, metric key, poll
            cadence, and timeout.
        provenance: Optional sink that receives the frozen evidence ``source``
            payload on success: run address, terminal run state, the summary
            subset that justified the metric, and the retrieval timestamp —
            remote summaries are mutable, so the frozen copy is the only
            durable record of what was read (review v0.5.17 / finding F).

    Returns:
        The numeric value of ``cfg.metric_key`` on the finished run.

    Raises:
        ExtractorError: ``wandb`` not installed, the attempt's run failed, the
            run was not ready before timeout, or the metric was missing or invalid.

    """
    try:
        summary = poll_wandb_summary(
            entity=cfg.entity,
            project=cfg.project,
            run_id=ctx.attempt_id,
            poll_seconds=cfg.poll_seconds,
            timeout_seconds=cfg.timeout_seconds,
            required_keys=[cfg.metric_key],
        )
    except ImportError as exc:
        raise ExtractorError(
            "W&B extractor requested but the 'wandb' package is not installed. "
            "Install the wandb extra for the same distribution, for example: "
            'pip install "phasesweep[wandb] @ '
            'git+https://github.com/pszemraj/phasesweep.git"'
        ) from exc
    except WandbSetupError as exc:
        raise ExtractorError(
            f"W&B client setup failed for run {ctx.attempt_id!r}: {exc.cause}. "
            "Fix the W&B credentials/settings on this host; retrying will not help."
        ) from exc
    except WandbRunTerminalError as exc:
        raise ExtractorError(
            f"W&B run {ctx.attempt_id!r} ended in state {exc.state!r}; "
            "only finished runs provide objective evidence."
        ) from exc
    except WandbPollTimeout as exc:
        msg = (
            f"W&B run {ctx.attempt_id!r} not found or metric {cfg.metric_key!r} "
            f"missing within {cfg.timeout_seconds}s."
        )
        if exc.last_error is not None:
            msg += f" Last error: {exc.last_error}"
        raise ExtractorError(msg) from exc

    try:
        value = json_float(summary[cfg.metric_key], label=cfg.metric_key)
    except ValueError as exc:
        raise ExtractorError(
            f"Value at W&B metric {cfg.metric_key!r} is not numeric: {summary[cfg.metric_key]!r}"
        ) from exc
    if provenance is not None:
        provenance["source"] = {
            "kind": "wandb",
            "entity": cfg.entity,
            "project": cfg.project,
            "run_id": ctx.attempt_id,
            # poll_wandb_summary returns only for finished runs; every other
            # terminal state raises WandbRunTerminalError above.
            "run_state": "finished",
            "summary": {cfg.metric_key: summary[cfg.metric_key]},
            "retrieved_at": _utc_now_iso(),
        }
    return value


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    :return str: Second-resolution UTC timestamp for provenance records.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


_DISPATCH: dict[type, Callable[[TrialContext, Any, dict[str, Any] | None], float]] = {
    JsonExtractor: _extract_json,
    JsonEnvelopeExtractor: _extract_json_envelope,
    LogRegexExtractor: _extract_log_regex,
    WandbExtractor: _extract_wandb,
}


def _remaining_budget_seconds(deadline: float | None) -> float | None:
    """Return seconds until ``deadline``, or ``None`` when unbounded.

    :param float | None deadline: Absolute ``time.monotonic()`` deadline.
    :return float | None: Non-negative remaining budget, or ``None``.
    """
    if deadline is None:
        return None
    import time

    return max(0.0, deadline - time.monotonic())


def run_extractor(
    ctx: TrialContext,
    cfg: Extractor,
    *,
    deadline: float | None = None,
    provenance: dict[str, Any] | None = None,
) -> float:
    """Dispatch to the appropriate extractor for ``cfg``.

    Args:
        ctx: Trial context passed through to the chosen extractor.
        cfg: A concrete extractor config (one of :class:`JsonExtractor`,
            :class:`JsonEnvelopeExtractor`, :class:`LogRegexExtractor`,
            :class:`WandbExtractor`).
        deadline: Optional absolute ``time.monotonic()`` phase/run deadline.
            Remote extractors (W&B) cap their own polling timeout to the
            remaining budget so evidence extraction cannot extend a run
            arbitrarily past its configured wallclock bound (review v0.5.17 /
            blocker 8). Local extractors are bounded by the caller's
            stage-boundary deadline checks instead.
        provenance: Optional sink that, on success, is filled with the frozen
            evidence provenance record: schema version, the extractor's kind
            and exact config fingerprint, the per-source evidence identity
            (file digest or frozen remote summary), and the capture timestamp
            (review v0.5.17 / finding F). The extractor fingerprint always
            reflects the *configured* extractor, not any deadline-capped copy.

    Returns:
        The numeric value the extractor pulled from this trial's outputs.

    Raises:
        ExtractorError: No extractor is registered for the given config type,
            or the chosen extractor failed.

    """
    fn = _DISPATCH.get(type(cfg))
    if fn is None:
        raise ExtractorError(f"No extractor registered for {type(cfg).__name__}.")
    if provenance is not None:
        provenance.clear()
        provenance.update(
            {
                "schema_version": EVIDENCE_PROVENANCE_SCHEMA_VERSION,
                "extractor": {
                    "kind": cfg.type,
                    "config_sha256": extractor_config_fingerprint(cfg),
                },
            }
        )
    remaining = _remaining_budget_seconds(deadline)
    if remaining is not None and remaining <= 0.0:
        raise DeadlineExceededError(
            "Phase/run wallclock deadline exceeded before evidence extraction."
        )
    deadline_capped = (
        remaining is not None
        and isinstance(cfg, WandbExtractor)
        and remaining < cfg.timeout_seconds
    )
    if remaining is not None and isinstance(cfg, WandbExtractor):
        cfg = cfg.model_copy(update={"timeout_seconds": min(cfg.timeout_seconds, remaining)})
    try:
        value = fn(ctx, cfg, provenance)
    except ExtractorError as exc:
        if deadline_capped and isinstance(exc.__cause__, WandbPollTimeout):
            raise DeadlineExceededError(
                "Phase/run wallclock deadline exhausted while polling W&B evidence."
            ) from exc
        raise
    if provenance is not None:
        provenance["recorded_at"] = _utc_now_iso()
    return value


@dataclass(frozen=True)
class GateResult:
    """Result of one evidence gate evaluation."""

    gate_type: str
    passed: bool
    detail: str
    deadline_exhausted: bool = False


def _required_file(ctx: TrialContext, gate: RequiredFileGate) -> GateResult:
    """Check that a required trial-relative file exists.

    :param TrialContext ctx: Trial context containing the trial directory.
    :param RequiredFileGate gate: Gate config naming the required file path.
    :return GateResult: Pass/fail result and human-readable detail.
    """
    path = ctx.trial_dir / gate.path
    if path.is_file():
        return GateResult(gate.type, True, f"{gate.path} exists")
    return GateResult(gate.type, False, f"{gate.path} is missing")


def _json_equals(ctx: TrialContext, gate: JsonEqualsGate) -> GateResult:
    """Check that a JSON value exactly equals the expected scalar.

    :param TrialContext ctx: Trial context containing the trial directory.
    :param JsonEqualsGate gate: Gate config naming the JSON path, key, and value.
    :return GateResult: Pass/fail result and human-readable detail.
    """
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
    """Check that a JSON scalar is finite and within configured bounds.

    :param TrialContext ctx: Trial context containing the trial directory.
    :param JsonScalarBoundGate gate: Gate config naming the JSON scalar and bounds.
    :return GateResult: Pass/fail result and human-readable detail.
    """
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
    """Check that an artifact byte size falls within configured bounds.

    :param TrialContext ctx: Trial context containing the trial directory.
    :param ArtifactSizeGate gate: Gate config for file, directory, or JSON byte size.
    :return GateResult: Pass/fail result and human-readable detail.
    """
    path = ctx.trial_dir / gate.path
    if gate.source == "file":
        try:
            if not path.is_file():
                return GateResult(gate.type, False, f"{gate.path} is not a file")
            size = path.stat().st_size
        except OSError as exc:
            return GateResult(gate.type, False, f"could not inspect {gate.path}: {exc}")
        label = f"{gate.path} file size"
    elif gate.source == "directory":
        try:
            if not path.is_dir():
                return GateResult(gate.type, False, f"{gate.path} is not a directory")
            size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        except OSError as exc:
            return GateResult(gate.type, False, f"could not inspect {gate.path}: {exc}")
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
    """Check that a file's SHA-256 digest matches the expected value.

    :param TrialContext ctx: Trial context containing the trial directory.
    :param Sha256Gate gate: Gate config naming the file and expected digest.
    :return GateResult: Pass/fail result and human-readable detail.
    """
    path = ctx.trial_dir / gate.path
    try:
        if not path.is_file():
            return GateResult(gate.type, False, f"{gate.path} is missing")
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError as exc:
        return GateResult(gate.type, False, f"could not read {gate.path}: {exc}")
    digest = hasher.hexdigest()
    if digest == gate.sha256:
        return GateResult(gate.type, True, f"{gate.path} sha256 matched")
    return GateResult(gate.type, False, f"{gate.path} sha256 {digest} != {gate.sha256}")


def _wandb_summary_required(
    ctx: TrialContext,
    gate: WandbSummaryRequiredGate,
    *,
    deadline_capped: bool = False,
) -> GateResult:
    """Check that a finished W&B run summary contains required keys.

    :param TrialContext ctx: Trial context containing the immutable W&B run id.
    :param WandbSummaryRequiredGate gate: Gate config for W&B lookup and keys.
    :return GateResult: Pass/fail result and human-readable detail.
    """
    try:
        summary = poll_wandb_summary(
            entity=gate.entity,
            project=gate.project,
            run_id=ctx.attempt_id,
            poll_seconds=gate.poll_seconds,
            timeout_seconds=gate.timeout_seconds,
            wait_for_keys=False,
        )
    except ImportError:
        return GateResult(gate.type, False, "wandb package is not installed")
    except WandbSetupError as exc:
        return GateResult(gate.type, False, f"W&B client setup failed: {exc.cause}")
    except WandbRunTerminalError as exc:
        return GateResult(
            gate.type,
            False,
            f"W&B run {ctx.attempt_id!r} ended in state {exc.state!r}",
        )
    except WandbPollTimeout as exc:
        detail = f"W&B run {ctx.attempt_id!r} not ready within {gate.timeout_seconds}s"
        if exc.last_error is not None:
            detail += f"; last error: {exc.last_error}"
        return GateResult(
            gate.type,
            False,
            detail,
            deadline_exhausted=deadline_capped,
        )

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


def evaluate_gates(
    ctx: TrialContext,
    gates: list[Gate],
    *,
    deadline: float | None = None,
) -> list[GateResult]:
    """Evaluate all gates against a completed trial context.

    :param TrialContext ctx: Trial context containing outputs and run metadata.
    :param list[Gate] gates: Gate configs to evaluate in order.
    :param float | None deadline: Optional absolute ``time.monotonic()``
        phase/run deadline. An expired deadline fails remaining gates
        immediately, and W&B gates cap their polling to the remaining budget
        (review v0.5.17 / blocker 8).
    :return list[GateResult]: One result for each gate in ``gates``.
    """
    results: list[GateResult] = []
    for gate in gates:
        fn = _GATE_DISPATCH[type(gate)]
        remaining = _remaining_budget_seconds(deadline)
        deadline_capped = False
        if remaining is not None:
            if remaining <= 0.0:
                results.append(
                    GateResult(
                        gate.type,
                        False,
                        "phase/run wallclock deadline exceeded before this gate ran",
                        deadline_exhausted=True,
                    )
                )
                continue
            if isinstance(gate, WandbSummaryRequiredGate):
                deadline_capped = remaining < gate.timeout_seconds
                gate = gate.model_copy(
                    update={"timeout_seconds": min(gate.timeout_seconds, remaining)}
                )
        if isinstance(gate, WandbSummaryRequiredGate):
            results.append(_wandb_summary_required(ctx, gate, deadline_capped=deadline_capped))
        else:
            results.append(fn(ctx, gate))
    return results
