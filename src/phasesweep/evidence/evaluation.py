"""Evidence extraction and post-trial gate evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from phasesweep.errors import UnsafeProcessCleanupError
from phasesweep.evidence.models import (
    ArtifactSizeGate,
    Gate,
    JsonEnvelopeExtractor,
    JsonEqualsGate,
    JsonExtractor,
    JsonScalarBoundGate,
    LogRegexExtractor,
    MlflowExtractor,
    MlflowRequiredGate,
    ObjectiveExtractor,
    RequiredFileGate,
    Sha256Gate,
    WandbExtractor,
    WandbQuery,
    WandbSummaryRequiredGate,
    _WandbSummarySource,
)
from phasesweep.evidence.wandb import (
    WandbPollTimeout,
    WandbRunTerminalError,
    WandbSetupError,
    locate_wandb_position,
    poll_wandb_summary,
)
from phasesweep.runtime.files import file_sha256
from phasesweep.runtime.json import progress_position, strict_json_loads
from phasesweep.runtime.time import utc_now_iso

if TYPE_CHECKING:
    from phasesweep.runtime.json import JsonObject
    from phasesweep.tracking.records import MlflowCapture, MlflowQuery, MlflowTarget

# Objective-evidence provenance schema version.
EVIDENCE_PROVENANCE_SCHEMA_VERSION = 3
# Log-regex evaluation revision included in the extractor contract.
LOG_REGEX_EVALUATION_REVISION = 2


def extractor_config_fingerprint(cfg: ObjectiveExtractor) -> str:
    """Hash the extractor configuration and applicable evaluation revision.

    :param ObjectiveExtractor cfg: Concrete extractor config to fingerprint.
    :return str: Hex SHA-256 of the canonical configuration and, for log-regex,
        its evaluation revision.
    """
    payload = cfg.model_dump(mode="json")
    if isinstance(cfg, LogRegexExtractor):
        payload["evaluation_revision"] = LOG_REGEX_EVALUATION_REVISION
    dumped = json.dumps(payload, sort_keys=True, separators=(",", ":"))
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
        provenance.
    :raises FileNotFoundError: If ``relative_path`` does not resolve to a regular
        file under ``trial_dir``.
    :raises KeyError: If a segment of the dotted ``key`` path is missing, with the
        failing segment as the argument.
    :raises OSError: If the resolved file exists but cannot be read.
    :raises UnicodeDecodeError: If the file bytes are not valid UTF-8.
    :raises ValueError: If the contents are not strict JSON (malformed document,
        duplicate object keys, or a rejected JSON constant).
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
    try:
        return float(value)
    except OverflowError as exc:
        raise ValueError(f"Value at {label!r} is outside the finite scalar range.") from exc


class ExtractorError(RuntimeError):
    """An extraction failure that marks the trial failed."""


class DeadlineExceededError(ExtractorError):
    """Raised when the phase/run deadline directly prevents extraction."""


@contextmanager
def _trial_json_read_errors(target: Path, *, noun: str) -> Iterator[None]:
    """Translate file-read and JSON-decoding failures into extractor diagnostics.

    :param Path target: Trial-relative file being read.
    :param str noun: ``JSON`` or ``JSON envelope`` diagnostic label.
    :raises ExtractorError: The file is missing, unreadable, or malformed.
    :return Iterator[None]: Context covering the read and parse of the same bytes.
    """
    try:
        yield
    except FileNotFoundError as exc:
        missing = (
            f"JSON file not found: {exc.args[0]}"
            if noun == "JSON"
            else f"{noun} not found: {target}"
        )
        raise ExtractorError(missing) from exc
    except UnicodeError as exc:
        raise ExtractorError(f"{noun} at {target} is not valid UTF-8: {exc}") from exc
    except ValueError as exc:
        raise ExtractorError(f"Invalid {noun} at {target}: {exc}") from exc
    except OSError as exc:
        raise ExtractorError(f"Could not read {noun} at {target}: {exc}") from exc


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
    wandb_environment: Mapping[str, str] | None = field(default=None, repr=False, compare=False)
    wandb_query: WandbQuery | None = None
    wandb_capture: dict[str, Any] = field(default_factory=dict, compare=False)
    mlflow_settings: MlflowTarget | None = None
    mlflow_query: MlflowQuery | None = None
    mlflow_environment: Mapping[str, str] | None = field(default=None, repr=False, compare=False)
    mlflow_capture: JsonObject = field(default_factory=dict, compare=False)


def _extract_json(
    ctx: TrialContext,
    cfg: JsonExtractor,
    provenance: dict[str, Any] | None = None,
) -> float:
    """Read a trial-relative JSON file and extract its numeric value.

    :param TrialContext ctx: Trial identity; the file is read from its ``trial_dir``.
    :param JsonExtractor cfg: Relative file path and dotted numeric key.
    :param dict | None provenance: Optional source-evidence sink.
    :raises ExtractorError: The file or key is missing, JSON is invalid, or the value is nonnumeric.
    :return float: Configured numeric value.
    """
    digest: dict[str, Any] = {}
    target = ctx.trial_dir / cfg.path
    try:
        with _trial_json_read_errors(target, noun="JSON"):
            _, cur = load_json_value(ctx.trial_dir, cfg.path, cfg.key, digest=digest)
    except KeyError as exc:
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
    ctx: TrialContext,
    cfg: JsonEnvelopeExtractor,
    provenance: dict[str, Any] | None = None,
) -> float:
    """Validate and extract an attempt-bound JSON result envelope.

    :param TrialContext ctx: Current trial identity and resolved-overrides digest.
    :param JsonEnvelopeExtractor cfg: Expected objective and evaluation policy.
    :param dict[str, Any] | None provenance: Optional sink that receives the
        frozen evidence ``source`` payload — envelope digest plus the
        validated evaluation metadata — on success.
    :raises ExtractorError: If the envelope is missing, malformed, or belongs to
        another execution attempt.
    :return float: Validated objective value from the envelope.
    """
    target = ctx.trial_dir / cfg.path
    with _trial_json_read_errors(target, noun="JSON envelope"):
        raw = target.read_bytes()
        data = strict_json_loads(raw.decode("utf-8"))

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
    if cfg.evaluation_axis == "step":
        position: int | float | None = step
    else:
        reported = evaluation.get("progress")
        position = progress_position(
            reported.get(cfg.evaluation_axis) if isinstance(reported, dict) else None
        )
        if position is None:
            raise ExtractorError(
                f"JSON envelope at {target} reports no finite non-negative "
                f"{cfg.evaluation_axis!r} position in evaluation.progress."
            )

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
            # checkpoint, step, and progress position are the envelope's own
            # reported values; progress is what evaluation_point compares.
            "evaluation": {
                "objective_name": cfg.objective_name,
                "split": cfg.split,
                "policy": cfg.policy,
                "checkpoint": checkpoint,
                "step": step,
                "progress": {"axis": cfg.evaluation_axis, "value": position},
            },
        }
    return value


def _extract_log_regex(
    ctx: TrialContext,
    cfg: LogRegexExtractor,
    provenance: dict[str, Any] | None = None,
) -> float:
    """Select a numeric regex match and digest the complete log file.

    First selection stops matching after its first hit but continues reading for
    the digest. Provenance includes the selected 1-based line and numeric match count.

    :param TrialContext ctx: Trial identity; the log is read from its ``trial_dir``.
    :param LogRegexExtractor cfg: Log path, named value group, and first/last/min/max selector.
    :param dict | None provenance: Optional source-evidence sink.
    :raises ExtractorError: The log is missing, unreadable, has no numeric match,
        or a line eligible for matching is invalid UTF-8.
    :return float: Selected match value.
    """
    import re

    target = ctx.trial_dir / cfg.file
    if not target.is_file():
        raise ExtractorError(f"Log file not found: {target}")

    pattern = re.compile(cfg.pattern)

    # Preserve original bytes for hashing and universal-newline boundaries for matching.
    # Only lines eligible for matching require strict UTF-8 decoding.
    hasher = hashlib.sha256()
    size_bytes = 0
    result: float | None = None
    result_line: int | None = None
    count = 0
    line_no = 0
    try:
        with target.open("r", encoding="utf-8", errors="surrogateescape", newline="") as fh:
            for raw_text in fh:
                raw_line = raw_text.encode("utf-8", errors="surrogateescape")
                hasher.update(raw_line)
                size_bytes += len(raw_line)
                if cfg.select == "first" and result is not None:
                    # Value already selected; keep reading only to finish the
                    # whole-file digest.
                    continue
                line = raw_line.decode("utf-8")
                if line.endswith("\r\n"):
                    line = line[:-2] + "\n"
                elif line.endswith("\r"):
                    line = line[:-1] + "\n"
                line_no += 1
                for match in pattern.finditer(line):
                    try:
                        v = float(match.group("value"))
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


def _capture_wandb(
    ctx: TrialContext, cfg: _WandbSummarySource, *, deadline: float | None = None
) -> dict[str, Any]:
    """Obtain and cache one accepted W&B capture for the attempt's consumers.

    :param TrialContext ctx: Trial identity and optional shared query/capture.
    :param _WandbSummarySource cfg: Source for a one-off query if needed.
    :param float | None deadline: Absolute monotonic deadline, also bounded by the source timeout.
    :raises DeadlineExceededError: The execution deadline prevents capture.
    :raises ExtractorError: Polling, authentication, SDK setup, run state, or evidence is invalid.
    :raises UnsafeProcessCleanupError: Worker cleanup is uncertain.
    :return dict: Finished run identity, numeric values, present keys, and retrieval time.
    """
    if ctx.wandb_capture:
        return ctx.wandb_capture
    query = ctx.wandb_query or WandbQuery(
        cfg,
        (cfg.metric_key,) if isinstance(cfg, WandbExtractor) else (),
        tuple(cfg.keys) if isinstance(cfg, WandbSummaryRequiredGate) else (),
    )
    source = query.source
    poll_deadline = time.monotonic() + source.timeout_seconds
    capped = deadline is not None and deadline <= poll_deadline
    if deadline is not None:
        poll_deadline = min(poll_deadline, deadline)
    target = f"{source.base_url}/{source.entity}/{source.project}/{ctx.attempt_id}"
    try:
        capture = poll_wandb_summary(
            base_url=source.base_url,
            entity=source.entity,
            project=source.project,
            run_id=ctx.attempt_id,
            trial_dir=ctx.trial_dir,
            poll_seconds=source.poll_seconds,
            timeout_seconds=source.timeout_seconds,
            required_keys=query.numeric_keys,
            presence_keys=query.presence_keys,
            environment=ctx.wandb_environment,
            deadline=poll_deadline,
        )
    except WandbPollTimeout as exc:
        error = (
            DeadlineExceededError
            if capped and time.monotonic() >= poll_deadline
            else ExtractorError
        )
        raise error(
            f"W&B evidence deadline expired for {target}; required summary keys: {query.numeric_keys!r}."
            + (f" Last observation: {exc.last_error}" if exc.last_error else "")
        ) from exc
    except WandbSetupError as exc:
        raise ExtractorError(
            f"W&B authentication/setup failed for {target}: {exc.cause}. Check account access and endpoint settings."
        ) from exc
    except WandbRunTerminalError as exc:
        raise ExtractorError(
            f"W&B run {target} ended in {exc.state!r}; only finished runs can provide evidence."
        ) from exc
    except ImportError as exc:
        raise ExtractorError(
            "W&B evidence requires the optional SDK; install phasesweep[wandb]."
        ) from exc
    except ValueError as exc:
        raise ExtractorError(f"Invalid W&B numeric evidence for {target}: {exc}") from exc
    except UnsafeProcessCleanupError:
        raise
    except RuntimeError as exc:
        raise ExtractorError(f"W&B evidence worker failed for {target}.") from exc
    ctx.wandb_capture.update(
        {
            "kind": "wandb",
            "base_url": source.base_url,
            "entity": source.entity,
            "project": source.project,
            "run_id": ctx.attempt_id,
            "run_state": "finished",
            "constraint_keys": dict(query.constraint_keys),
            "gate_keys": {identity: list(keys) for identity, keys in query.gate_keys},
            **capture,
        }
    )
    return ctx.wandb_capture


def _extract_wandb(
    ctx: TrialContext,
    cfg: WandbExtractor,
    provenance: dict[str, Any] | None = None,
    *,
    deadline: float | None = None,
) -> float:
    """Read one finite numeric key from the shared finished W&B capture.

    Objective provenance includes the source and remote capture; its advisory
    evaluation position remains unset until locate_objective_position runs.

    :param TrialContext ctx: Attempt whose shared capture supplies the value.
    :param WandbExtractor cfg: Remote target and exact summary key.
    :param dict | None provenance: Optional objective-evidence sink.
    :param float | None deadline: Absolute monotonic deadline for capture.
    :raises ExtractorError: Capture fails or the selected key is not finite numeric evidence.
    :return float: Selected metric value.
    """
    capture = _capture_wandb(ctx, cfg, deadline=deadline)
    try:
        value = json_float(capture["values"][cfg.metric_key], label=cfg.metric_key)
    except (ValueError, KeyError) as exc:
        raise ExtractorError(f"W&B key {cfg.metric_key!r} has no valid numeric evidence.") from exc
    if not math.isfinite(value):
        raise ExtractorError(f"W&B key {cfg.metric_key!r} is non-finite.")
    if provenance is not None:
        provenance["source"] = {
            "kind": "wandb",
            "metric_key": cfg.metric_key,
            "evaluation": {"progress": {"axis": cfg.evaluation_axis, "value": None}},
        }
        provenance["remote_capture"] = dict(capture)
    return value


def locate_objective_position(
    ctx: TrialContext,
    cfg: ObjectiveExtractor,
    provenance: dict[str, Any],
    *,
    deadline: float | None = None,
) -> None:
    """Add an advisory W&B evaluation position after the objective is accepted.

    Failed or expired history reads leave the position unset and retain the
    accepted objective. Only W&B needs this lookup: envelope and MLflow objectives
    record their position at extraction, and json and log_regex record none.

    :param TrialContext ctx: Accepted attempt.
    :param ObjectiveExtractor cfg: Objective extractor.
    :param dict provenance: Objective evidence updated in place.
    :param float | None deadline: Remaining absolute monotonic execution deadline.
    :raises UnsafeProcessCleanupError: Worker cleanup is uncertain.
    """
    if not isinstance(cfg, WandbExtractor):
        return
    progress = provenance["source"]["evaluation"]["progress"]
    progress["value"] = locate_wandb_position(
        base_url=cfg.base_url,
        entity=cfg.entity,
        project=cfg.project,
        run_id=ctx.attempt_id,
        trial_dir=ctx.trial_dir,
        poll_seconds=cfg.poll_seconds,
        timeout_seconds=cfg.timeout_seconds,
        metric_key=cfg.metric_key,
        axis=cfg.evaluation_axis,
        value=provenance["remote_capture"]["values"][cfg.metric_key],
        environment=ctx.wandb_environment,
        deadline=deadline,
    )


def _capture_mlflow(ctx: TrialContext, *, deadline: float | None) -> MlflowCapture:
    """Read each native history once, after verifying the exact bound run.

    :param TrialContext ctx: Attempt and configured capture contract.
    :param float | None deadline: Existing execution deadline.
    :return MlflowCapture: Shared frozen evidence.
    :raises ExtractorError: Binding or remote evidence cannot be accepted.
    """
    from phasesweep.evidence.mlflow import validate_capture
    from phasesweep.tracking.mlflow import request_operation
    from phasesweep.tracking.session import BINDING_FILE, read_binding, read_record

    if ctx.mlflow_capture:
        return validate_capture(ctx.mlflow_capture)
    if ctx.mlflow_settings is None or ctx.mlflow_query is None:
        raise ExtractorError("MLflow capture requires a configured target and query.")
    try:
        # Managed runs are bound before launch, so only a trainer-owned run can be missing.
        if not (ctx.trial_dir / BINDING_FILE).is_file():
            raise ValueError(
                "The trainer never bound its MLflow run; call "
                "phasesweep.tracking.bind_mlflow_run(run_id) inside its active run."
            )
        binding = read_binding(ctx.trial_dir)
        expected = read_record(ctx.trial_dir / "mlflow-expected.json")
        if (
            any(binding.get(k) != v for k, v in expected.items())
            or binding["attempt_id"] != ctx.attempt_id
            or binding["generation_id"] != ctx.generation_id
            or binding["overrides_sha256"] != ctx.overrides_sha256
        ):
            raise ValueError("MLflow binding belongs to a different attempt or target.")
        stop = time.monotonic() + ctx.mlflow_settings["timeout_seconds"]
        retrieved = request_operation(
            {
                "action": "capture",
                "settings": ctx.mlflow_settings,
                "binding": binding,
                "query": ctx.mlflow_query,
                "timeout_at": stop,
            },
            operation_id=ctx.attempt_id,
            directory=ctx.trial_dir,
            environment=ctx.mlflow_environment or {},
            deadline=min(stop, deadline) if deadline is not None else stop,
        )
        capture = validate_capture(retrieved)
    except TimeoutError as exc:
        if deadline is not None and time.monotonic() >= deadline:
            raise DeadlineExceededError(str(exc)) from exc
        raise ExtractorError(str(exc)) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, UnsafeProcessCleanupError):
            raise
        raise ExtractorError(str(exc)) from exc
    ctx.mlflow_capture.update(retrieved)
    return capture


def _extract_mlflow(
    ctx: TrialContext,
    cfg: MlflowExtractor,
    provenance: JsonObject | None,
    *,
    deadline: float | None = None,
) -> float:
    """Return a selected native metric and retain its native position.

    :param TrialContext ctx: Bound attempt.
    :param MlflowExtractor cfg: Metric selector.
    :param JsonObject | None provenance: Objective evidence sink.
    :param float | None deadline: Existing execution deadline.
    :return float: Selected metric value.
    """
    from phasesweep.evidence.mlflow import observation_id

    capture = _capture_mlflow(ctx, deadline=deadline)
    point = capture["observations"][observation_id(cfg)]
    if provenance is not None:
        provenance["source"] = {
            "kind": "mlflow",
            "metric_key": cfg.metric_key,
            "selection": observation_id(cfg),
            "evaluation": {
                "progress": {"axis": "step", "value": point["step"]},
                "timestamp": point["timestamp"],
            },
        }
        provenance["remote_capture"] = dict(ctx.mlflow_capture)
    return float(point["value"])


def run_extractor(
    ctx: TrialContext,
    cfg: ObjectiveExtractor,
    *,
    deadline: float | None = None,
    provenance: dict[str, Any] | None = None,
) -> float:
    """Extract a trial value and optionally retain its exact source evidence.

    An expired deadline prevents even local reads; remote polling also receives
    the remaining execution deadline.

    :param TrialContext ctx: Trial identity, outputs, and shared remote captures.
    :param ObjectiveExtractor cfg: Configured extractor.
    :param float | None deadline: Absolute monotonic phase/run deadline.
    :param dict | None provenance: Optional sink for the evidence schema,
        extractor kind/fingerprint, file digest or remote capture, and capture timestamp.
    :raises ExtractorError: The kind is unregistered or extraction fails.
    :return float: Extracted numeric value.
    """
    if type(cfg) not in (
        JsonExtractor,
        JsonEnvelopeExtractor,
        LogRegexExtractor,
        WandbExtractor,
        MlflowExtractor,
    ):
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
    if deadline is not None and not (deadline > time.monotonic()):
        raise DeadlineExceededError(
            "Phase/run wallclock deadline exceeded before evidence extraction."
        )
    if isinstance(cfg, JsonExtractor):
        value = _extract_json(ctx, cfg, provenance)
    elif isinstance(cfg, JsonEnvelopeExtractor):
        value = _extract_json_envelope(ctx, cfg, provenance)
    elif isinstance(cfg, LogRegexExtractor):
        value = _extract_log_regex(ctx, cfg, provenance)
    elif isinstance(cfg, WandbExtractor):
        value = _extract_wandb(ctx, cfg, provenance, deadline=deadline)
    else:
        value = _extract_mlflow(ctx, cfg, provenance, deadline=deadline)
    if provenance is not None:
        provenance["recorded_at"] = utc_now_iso(timespec="seconds")
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
    except Exception as exc:
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
    except Exception as exc:
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
            root = path.stat(follow_symlinks=False)
        except OSError as exc:
            return GateResult(gate.type, False, f"could not inspect {gate.path}: {path}: {exc}")
        if not stat.S_ISDIR(root.st_mode):
            return GateResult(gate.type, False, f"{gate.path} is not a directory")
        directories = [path]
        size = 0
        while directories:
            directory = directories.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        entry_path = Path(entry.path)
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                directories.append(entry_path)
                            # Count file-symlink targets without traversing directory symlinks.
                            elif entry.is_file():
                                size += entry_path.stat().st_size
                        except OSError as exc:
                            return GateResult(
                                gate.type,
                                False,
                                f"could not inspect {gate.path}: {entry_path}: {exc}",
                            )
            except OSError as exc:
                return GateResult(
                    gate.type,
                    False,
                    f"could not inspect {gate.path}: {directory}: {exc}",
                )
        label = f"{gate.path} directory size"
    else:
        assert gate.key is not None
        try:
            _, raw_size = load_json_value(ctx.trial_dir, gate.path, gate.key)
        except Exception as exc:
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
        digest = file_sha256(path)
    except OSError as exc:
        return GateResult(gate.type, False, f"could not read {gate.path}: {exc}")
    if digest == gate.sha256:
        return GateResult(gate.type, True, f"{gate.path} sha256 matched")
    return GateResult(gate.type, False, f"{gate.path} sha256 {digest} != {gate.sha256}")


def _mlflow_required(
    ctx: TrialContext, gate: MlflowRequiredGate, *, deadline: float | None = None
) -> GateResult:
    """Check presence from the same verified native metric capture.

    :param TrialContext ctx: Bound attempt.
    :param MlflowRequiredGate gate: Required namespaces.
    :param float | None deadline: Existing execution deadline.
    :return GateResult: Presence result.
    """
    try:
        _capture_mlflow(ctx, deadline=deadline)
    except ExtractorError as exc:
        return GateResult(gate.type, False, str(exc), isinstance(exc, DeadlineExceededError))
    return GateResult(gate.type, True, "MLflow metrics, parameters, and tags are present")


def _wandb_summary_required(
    ctx: TrialContext, gate: WandbSummaryRequiredGate, *, deadline: float | None = None
) -> GateResult:
    """Check key presence in the same capture used by objectives and constraints.

    :param TrialContext ctx: Trial context whose shared W&B capture is inspected.
    :param WandbSummaryRequiredGate gate: Gate config naming the required summary keys.
    :param float | None deadline: Optional absolute ``time.monotonic()`` phase/run
        deadline for obtaining the capture.
    :return GateResult: Pass/fail result and human-readable detail; a failed capture
        fails the gate, marked ``deadline_exhausted`` when the deadline expired.
    """
    try:
        capture = _capture_wandb(ctx, gate, deadline=deadline)
    except DeadlineExceededError as exc:
        return GateResult(gate.type, False, str(exc), deadline_exhausted=True)
    except ExtractorError as exc:
        return GateResult(gate.type, False, str(exc))
    missing = sorted(set(gate.keys) - set(capture["present_keys"]))
    return GateResult(
        gate.type,
        not missing,
        f"W&B summary missing keys: {missing}" if missing else "W&B summary keys present",
    )


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
        immediately.
    :return list[GateResult]: One result for each gate in ``gates``.
    """
    results: list[GateResult] = []
    for gate in gates:
        if type(gate) not in (
            RequiredFileGate,
            JsonEqualsGate,
            JsonScalarBoundGate,
            ArtifactSizeGate,
            Sha256Gate,
            WandbSummaryRequiredGate,
            MlflowRequiredGate,
        ):
            results.append(GateResult(type(gate).__name__, False, f"unknown gate: {gate!r}"))
            continue
        if deadline is not None and not (deadline > time.monotonic()):
            results.append(
                GateResult(
                    gate.type,
                    False,
                    "phase/run wallclock deadline exceeded before this gate ran",
                    deadline_exhausted=True,
                )
            )
            continue
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
            result = _wandb_summary_required(ctx, gate, deadline=deadline)
        else:
            result = _mlflow_required(ctx, gate, deadline=deadline)
        results.append(result)
    return results
