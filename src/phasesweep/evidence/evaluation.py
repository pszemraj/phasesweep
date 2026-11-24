"""Evidence extraction and post-trial gate evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import stat
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from phasesweep.config.common import TRACKER_READ_CONTROLS
from phasesweep.errors import UnsafeProcessCleanupError
from phasesweep.evidence.baseline import EVIDENCE_BASELINE_FILE, EvidenceBaseline, FileStamp
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
    is_captured_stream,
)
from phasesweep.evidence.wandb import (
    WANDB_SDK_MISSING,
    WandbMissingKeysError,
    WandbPollTimeout,
    WandbRunMissingError,
    WandbRunStaleError,
    WandbRunTerminalError,
    WandbSetupError,
    locate_wandb_position,
    poll_wandb_summary,
)
from phasesweep.runtime.files import broken_symlink_detail, file_sha256
from phasesweep.runtime.json import (
    NonFiniteJsonConstantError,
    finite_nonnegative,
    strict_json_loads,
)
from phasesweep.runtime.time import utc_now_iso

if TYPE_CHECKING:
    from phasesweep.runtime.json import JsonObject
    from phasesweep.tracking.records import MlflowCapture, MlflowQuery, MlflowTarget, RunBinding

log = logging.getLogger(__name__)

# Objective-evidence provenance schema version.
EVIDENCE_PROVENANCE_SCHEMA_VERSION = 6
# Log-regex evaluation revision included in the extractor contract.
LOG_REGEX_EVALUATION_REVISION = 3


def extractor_config_fingerprint(cfg: ObjectiveExtractor) -> str:
    """Hash the extractor configuration and applicable evaluation revision.

    A tracker's read controls bound how long the read waits, not what it reads, so they stay out.

    :param ObjectiveExtractor cfg: Concrete extractor config to fingerprint.
    :return str: Hex SHA-256 of the canonical configuration without its tracker read controls
        and, for log-regex, its evaluation revision.
    """
    payload = cfg.model_dump(mode="json", exclude=set(TRACKER_READ_CONTROLS))
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


class TrialPathError(OSError):
    """The configured evidence path cannot supply a file from its owning directory."""


class StaleEvidenceError(OSError):
    """The evidence cannot be attributed to the current attempt."""


def _link_refusal(owner: Path, relative: str | Path, *, directory: bool) -> str | None:
    """Name a link in a configured path whose target is missing or not the kind required.

    :param Path owner: The directory the configured path is relative to.
    :param str | Path relative: Configured path.
    :param bool directory: The path must reach a directory rather than a regular file.
    :return str | None: The diagnostic naming the link and its target, or ``None``.
    """
    detail = broken_symlink_detail(
        owner, relative, require_directory=directory, require_file=not directory
    )
    return None if detail is None else f"Could not read `{relative}` because {detail}."


def attempt_file(trial_dir: Path, relative: str | Path) -> Path:
    """Return a trial-relative evidence path, refusing one the attempt may not be scored from.

    PhaseSweep reads a trial's objective, constraints, observations, gate values and
    digests only from files inside that trial's directory. Config load already refuses
    absolute and ``..`` paths, so a link is the only way out, and several trials whose
    links name one shared file would all be scored from whichever trainer wrote it last.
    A link that stays inside the trial, such as ``latest -> checkpoint-500``, reads as
    usual. A link whose target is missing or is not the kind of file required names the
    link and its target. An ordinary missing file keeps the caller's own not-found handling.

    :param Path trial_dir: The trial directory the file must resolve inside.
    :param str | Path relative: Configured trial-relative path of a regular file.
    :raises TrialPathError: A link is unusable or the path resolves outside ``trial_dir``.
    :return Path: ``trial_dir / relative``, unchanged, for the caller to read.
    """
    path = trial_dir / relative
    if refusal := _link_refusal(trial_dir, relative, directory=False):
        raise TrialPathError(refusal)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError):
            raise TrialPathError(f"Could not resolve {path}: {exc}") from exc
        resolved = path.resolve()
    if not resolved.is_relative_to(trial_dir.resolve()):
        if is_captured_stream(relative):
            raise TrialPathError(
                f"Captured stream `{relative}` resolves to {resolved}, outside its attempt "
                f"directory {trial_dir}. Restore the attempt's captured stream."
            )
        raise TrialPathError(
            f"`{relative}` in trial directory {trial_dir} resolves to {resolved}, outside the "
            "trial. Have the trainer write it under PHASESWEEP_TRIAL_DIR."
        )
    return path


def trial_file(
    trial_dir: Path,
    attempt_dir: Path,
    relative: str | Path,
    *,
    checked: bool = True,
) -> Path:
    """Return where a configured trial path is read from.

    Every extractor, gate and verification of a configured path resolves it here, so they
    agree on the file. A captured stream (:func:`is_captured_stream`) is the attempt's own,
    kept in its attempt directory; any other path is a trainer file inside the trial
    directory, as :func:`attempt_file` resolves and checks it.

    :param Path trial_dir: The trial directory the trainer writes into.
    :param Path attempt_dir: The attempt's directory, holding its captured streams.
    :param str | Path relative: Configured trial-relative path.
    :param bool checked: Check the path against the directory that owns it. ``False``
        only locates it: for a gate that tests an artifact wherever its links lead, or to
        name the file in a diagnostic.
    :raises TrialPathError: A checked path contains an unusable link or resolves outside its
        directory.
    :return Path: The path to read.
    """
    if is_captured_stream(relative):
        return attempt_file(attempt_dir, relative) if checked else attempt_dir / relative
    if not checked:
        return trial_dir / relative
    return attempt_file(trial_dir, relative)


def unusable_link(
    trial_dir: Path, attempt_dir: Path, relative: str | Path, *, directory: bool = False
) -> str | None:
    """Name an unusable link for a gate that locates its path unchecked.

    The same rule :func:`attempt_file` refuses with, applied in the directory
    :func:`trial_file` locates the path in.

    :param Path trial_dir: The trial directory the trainer writes into.
    :param Path attempt_dir: The attempt's directory, holding its captured streams.
    :param str | Path relative: Configured trial-relative path.
    :param bool directory: The path must reach a directory rather than a regular file,
        defaults to False.
    :return str | None: The diagnostic naming the link and its target, or ``None``.
    """
    owner = attempt_dir if is_captured_stream(relative) else trial_dir
    return _link_refusal(owner, relative, directory=directory)


def _require_fresh(
    baseline: EvidenceBaseline | None,
    trial_dir: Path,
    attempt_dir: Path,
    relative: str,
    target: Path,
) -> None:
    """Refuse a value file this attempt did not write.

    A file the baseline recorded is this attempt's once its size or modification time
    differs from the record: that is the file's own earlier stamp, so no clock is compared.
    A file at a target that held none at launch was created during the attempt. A link
    retargeted to a file the baseline did not record may reach one that existed before; it
    counts when modified at or after the baseline record itself was written, just before
    launch. That one comparison assumes the trial and attempt directories share a clock.

    :param EvidenceBaseline | None baseline: Value paths and file stamps recorded at launch.
    :param Path trial_dir: Directory containing trainer files.
    :param Path attempt_dir: The attempt's directory, holding the baseline record.
    :param str relative: Configured trial-relative path.
    :param Path target: Current location of evidence after containment checks.
    :raises StaleEvidenceError: The file was not written by this attempt, or its state at
        launch could not be inspected.
    :raises OSError: The current file or the baseline record could not be inspected.
    """
    if baseline is None or not baseline.reused or is_captured_stream(relative):
        return
    before = baseline.files.get(relative, relative)
    if before is None:
        raise StaleEvidenceError(
            f"`{target}` could not be inspected before this attempt, so its value "
            "cannot be attributed to this attempt."
        )
    info = target.stat()
    resolved = target.resolve()
    current = resolved.relative_to(trial_dir.resolve()).as_posix()
    recorded = baseline.files.values()
    prior = next((s for s in recorded if isinstance(s, FileStamp) and s.path == current), None)
    if prior is not None:
        if prior.unchanged(info):
            raise StaleEvidenceError(
                f"`{target}` was not written by this attempt: its size and modification time "
                "are unchanged since launch, so the trainer exited without writing it."
            )
        return
    if current == before or current in recorded:
        return
    if info.st_mtime_ns >= (attempt_dir / EVIDENCE_BASELINE_FILE).stat().st_mtime_ns:
        return
    raise StaleEvidenceError(
        f"`{target}` now resolves to `{resolved}`, which was last modified before this "
        "attempt started, so the trainer exited without writing it."
    )


def load_json_value(
    trial_dir: Path,
    attempt_dir: Path,
    relative_path: str,
    key: str,
    *,
    digest: dict[str, Any] | None = None,
    baseline: EvidenceBaseline | None = None,
) -> tuple[Path, Any]:
    """Load a dotted JSON value from a configured trial path, as :func:`trial_file` locates it.

    :param Path trial_dir: Directory containing the trial outputs.
    :param Path attempt_dir: The attempt's directory, holding its captured streams.
    :param str relative_path: Configured trial-relative JSON file path.
    :param str key: Dot-separated key path to read from the JSON object.
    :param dict[str, Any] | None digest: Optional sink that receives the
        ``sha256``/``size_bytes`` of the exact bytes parsed, for evidence
        provenance.
    :param EvidenceBaseline | None baseline: Value paths and file stamps recorded at launch.
    :raises TrialPathError: If the path contains a broken link or resolves outside its directory.
    :raises StaleEvidenceError: If the file was not written by this attempt.
    :raises FileNotFoundError: If ``relative_path`` does not resolve to a regular file.
    :raises JsonKeyMissingError: If a segment of the dotted ``key`` path is missing, with the
        failing segment as the argument and what the file holds there.
    :raises OSError: If the resolved file exists but cannot be read.
    :raises UnicodeDecodeError: If the file bytes are not valid UTF-8.
    :raises ValueError: If the contents are not strict JSON (malformed document,
        duplicate object keys, or a rejected JSON constant).
    :return tuple[Path, Any]: Resolved JSON file path and loaded value.
    """
    target = trial_file(trial_dir, attempt_dir, relative_path)
    if not target.is_file():
        raise FileNotFoundError(target)
    _require_fresh(baseline, trial_dir, attempt_dir, relative_path, target)
    raw = target.read_bytes()
    if digest is not None:
        digest.update(_file_digest(raw))
    cur = strict_json_loads(raw.decode("utf-8"))
    walked: list[str] = []
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            walked.append(part)
            continue
        raise JsonKeyMissingError(part, _level_text(".".join(walked), cur))
    return target, cur


#: How many of an object's keys a missing-key diagnostic lists before it counts the rest.
_KEYS_SHOWN = 5


class JsonKeyMissingError(KeyError):
    """A segment of a dotted key is missing from a JSON file, with what the file holds there.

    :param str segment: The segment the lookup failed at, also ``args[0]``.
    :param str found: What the file holds where the lookup failed, a clause for the diagnostic.
    """

    def __init__(self, segment: str, found: str) -> None:
        super().__init__(segment)
        self.segment = segment
        self.found = found

    def __str__(self) -> str:
        """Say which key is missing and what the file holds there."""
        return f"no key {self.segment!r}; {self.found}"


def _level_text(path: str, level: object) -> str:
    """Say what a JSON file holds at the level a key lookup failed.

    :param str path: Dotted path of ``level`` within the file; empty for the file's top level.
    :param object level: The value the lookup could not descend into or find its segment in.
    :return str: ``the file has 'a', 'b' and 3 more``, ``'m' has no keys`` or ``'m' is a list``.
    """
    where = repr(path) if path else "the file"
    if not isinstance(level, dict):
        return f"{where} is {_json_noun(level)}"
    if not level:
        return f"{where} has no keys"
    keys = [repr(name) for name in level]
    more = len(keys) - _KEYS_SHOWN
    return f"{where} has {', '.join(keys[:_KEYS_SHOWN])}" + (
        f" and {more} more" if more > 0 else ""
    )


def _json_noun(value: object) -> str:
    """Name the kind of a parsed JSON value that is not an object.

    :param object value: A null, boolean, number, string or list.
    :return str: ``null``, ``a boolean``, ``a number``, ``a string`` or ``a list``.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int | float):
        return "a number"
    if isinstance(value, str):
        return "a string"
    return "a list"


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


class EvidenceUnreadableError(ExtractorError):
    """Raised when the tracker gave no answer about the trial's remote evidence.

    An outage, rejected credentials, a missing SDK or a failed reader worker says nothing
    about the trial, so its finished training waits for a later read instead of failing.
    """


class MlflowObjectiveMissingError(ExtractorError):
    """Raised when a finished managed MLflow run lacks the configured objective metric."""


def _non_finite_text(constant: str) -> str:
    """Say what a file holding NaN or Infinity means, for a trial's failure reason.

    :param str constant: The refused token.
    :return str: A clause that follows the file's subject.
    """
    return f"holds {constant}, which is not a finite number; a diverged run writes it"


def _gate_unavailable(path: str, key: str, exc: Exception) -> str:
    """Describe a JSON gate whose value could not be read.

    :param str path: Trial-relative file the gate reads.
    :param str key: Dotted key the gate reads.
    :param Exception exc: Failure raised while reading it.
    :return str: The gate result's detail.
    """
    if isinstance(exc, NonFiniteJsonConstantError):
        return f"{path}:{key} unavailable: the file {_non_finite_text(exc.constant)}"
    return f"{path}:{key} unavailable: {exc}"


@contextmanager
def _trial_json_read_errors(target: Path, *, noun: str, trainer_step: str) -> Iterator[None]:
    """Translate file-read and JSON-decoding failures into extractor diagnostics.

    :param Path target: Trial-relative file being read.
    :param str noun: ``JSON`` or ``JSON envelope`` diagnostic label.
    :param str trainer_step: What the trainer must do for the file to exist, a clause without
        a final period that follows a missing-file diagnostic in parentheses, since a trainer
        that exits 0 without it fails here. The path ends the diagnostic's own words, so no
        punctuation sticks to it when a reader copies it.
    :raises ExtractorError: The file is missing, unreadable, malformed, or outside the
        attempt; the last carries the refusal's own text as the trial's failure reason.
    :return Iterator[None]: Context covering the read and parse of the same bytes.
    """
    try:
        yield
    except (TrialPathError, StaleEvidenceError) as exc:
        raise ExtractorError(str(exc)) from exc
    except FileNotFoundError as exc:
        missing = (
            f"JSON file not found: {exc.args[0]}"
            if noun == "JSON"
            else f"{noun} not found: {target}"
        )
        raise ExtractorError(f"{missing} ({trainer_step})") from exc
    except UnicodeError as exc:
        raise ExtractorError(f"{noun} at {target} is not valid UTF-8: {exc}") from exc
    except NonFiniteJsonConstantError as exc:
        raise ExtractorError(f"{noun} at {target} {_non_finite_text(exc.constant)}.") from exc
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
    # The trainer's directory, and beside it the attempt's own records and process slot.
    trial_dir: Path
    attempt_dir: Path
    run_name: str  # "{experiment}-{phase}-{trial_id}-{attempt_id}"
    return_code: int
    duration_seconds: float
    # The trial's W&B run, which every attempt in trial_dir logs to: the attempt that opened
    # the directory.
    wandb_run_id: str
    wandb_environment: Mapping[str, str] | None = field(default=None, repr=False, compare=False)
    wandb_query: WandbQuery | None = None
    wandb_capture: dict[str, Any] = field(default_factory=dict, compare=False)
    # Holds at most one failed capture.
    wandb_capture_failure: list[ExtractorError] = field(default_factory=list, compare=False)
    mlflow_settings: MlflowTarget | None = None
    mlflow_query: MlflowQuery | None = None
    mlflow_environment: Mapping[str, str] | None = field(default=None, repr=False, compare=False)
    mlflow_capture: JsonObject = field(default_factory=dict, compare=False)
    # Value paths and file stamps recorded before the attempt ran.
    baseline: EvidenceBaseline | None = field(default=None, compare=False)
    # (generation ID, attempt ID) of the earlier attempts that trained in trial_dir, oldest
    # first, whose tracker histories are the start of this trial's.
    earlier_attempts: tuple[tuple[str, str], ...] = ()


def _extract_json(
    ctx: TrialContext,
    cfg: JsonExtractor,
    provenance: dict[str, Any] | None = None,
) -> float:
    """Read a trial-relative JSON file and extract its numeric value.

    :param TrialContext ctx: Trial identity; the file is read from its ``trial_dir``, or its
        ``attempt_dir`` for a captured stream.
    :param JsonExtractor cfg: Relative file path and dotted numeric key.
    :param dict | None provenance: Optional source-evidence sink.
    :raises ExtractorError: The file or key is missing, the file resolves outside the attempt,
        JSON is invalid, or the value is nonnumeric.
    :return float: Configured numeric value.
    """
    digest: dict[str, Any] = {}
    target = trial_file(ctx.trial_dir, ctx.attempt_dir, cfg.path, checked=False)
    try:
        with _trial_json_read_errors(
            target,
            noun="JSON",
            trainer_step=(
                f"the trainer must write `{cfg.path}` under PHASESWEEP_TRIAL_DIR before it exits, "
                f"with the numeric key `{cfg.key}`"
            ),
        ):
            _, cur = load_json_value(
                ctx.trial_dir,
                ctx.attempt_dir,
                cfg.path,
                cfg.key,
                digest=digest,
                baseline=ctx.baseline,
            )
    except JsonKeyMissingError as exc:
        raise ExtractorError(
            f"Key {cfg.key!r} not found in {target} (failed at {exc.segment!r}; {exc.found})."
        ) from exc

    try:
        value = json_float(cur, label=cfg.key)
    except ValueError as exc:
        raise ExtractorError(str(exc)) from exc
    if provenance is not None:
        provenance["source"] = {"kind": "file", "path": cfg.path, "key": cfg.key, **digest}
    return value


def _it_reports(value: object) -> str:
    """Show what an envelope holds where it differs from the config, so a mismatch names both sides.

    :param object value: The envelope's own value for the mismatched field.
    :return str: ``; it reports 'x'`` for a string or an integer, and an empty string for a
        missing or malformed value, which is not quoted back as ``None``.
    """
    if isinstance(value, str) or type(value) is int:
        return f"; it reports {value!r}"
    return ""


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
    :raises ExtractorError: If the envelope is missing, resolves outside the attempt
        directory, is malformed, or belongs to another execution attempt.
    :return float: Validated objective value from the envelope.
    """
    target = trial_file(ctx.trial_dir, ctx.attempt_dir, cfg.path, checked=False)
    with _trial_json_read_errors(
        target,
        noun="JSON envelope",
        trainer_step=(
            "the trainer must call `phasesweep.report_objective(...)` before it exits, from rank "
            f"0 only; it writes `{cfg.path}` under PHASESWEEP_TRIAL_DIR"
        ),
    ):
        raw = trial_file(ctx.trial_dir, ctx.attempt_dir, cfg.path).read_bytes()
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
    reported_name = objective.get("name")
    if reported_name != cfg.objective_name:
        raise ExtractorError(
            f"JSON envelope at {target} does not report objective {cfg.objective_name!r}"
            f"{_it_reports(reported_name)}."
        )
    reported_split = objective.get("split")
    if reported_split != cfg.split:
        raise ExtractorError(
            f"JSON envelope at {target} does not report split {cfg.split!r}"
            f"{_it_reports(reported_split)}."
        )

    evaluation = data.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ExtractorError(f"JSON envelope at {target} has no evaluation object.")
    reported_policy = evaluation.get("policy")
    if reported_policy != cfg.policy:
        raise ExtractorError(
            f"JSON envelope at {target} does not report policy {cfg.policy!r}"
            f"{_it_reports(reported_policy)}."
        )
    checkpoint = evaluation.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ExtractorError(f"JSON envelope at {target} has no checkpoint identity.")
    if cfg.checkpoint is not None and checkpoint != cfg.checkpoint:
        raise ExtractorError(
            f"JSON envelope at {target} does not report checkpoint {cfg.checkpoint!r}"
            f"{_it_reports(checkpoint)}."
        )
    step = evaluation.get("step")
    if type(step) is not int or step < 0:
        raise ExtractorError(f"JSON envelope at {target} has no non-negative evaluation step.")
    if cfg.expected_step is not None and step != cfg.expected_step:
        raise ExtractorError(
            f"JSON envelope at {target} does not report step {cfg.expected_step}"
            f"{_it_reports(step)}."
        )
    if cfg.evaluation_axis == "step":
        position: int | float | None = step
    else:
        reported = evaluation.get("progress")
        position = finite_nonnegative(
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

    A trainer log kept across a trial's attempts is the trial's log, so every line counts
    once this attempt has written to it. First selection
    stops matching after its first numeric hit but continues reading for the digest.
    Warn when a numeric result leaves unparseable captures behind.
    Provenance includes the selected 1-based line and numeric match count.

    :param TrialContext ctx: Trial identity; the log is read from its ``trial_dir``, or its
        ``attempt_dir`` for a captured stream.
    :param LogRegexExtractor cfg: Log path, named value group, and first/last/min/max selector.
    :param dict | None provenance: Optional source-evidence sink.
    :raises ExtractorError: The log is missing, outside the attempt, unreadable, unchanged,
        has no numeric match, has a non-finite match under ``min`` or ``max``, or a line
        eligible for matching is invalid UTF-8.
    :return float: Selected match value.
    """
    try:
        target = trial_file(ctx.trial_dir, ctx.attempt_dir, cfg.file)
        if not target.is_file():
            raise ExtractorError(
                f"Log file not found: {target} (the trainer must write `{cfg.file}` under "
                "PHASESWEEP_TRIAL_DIR; `stdout.log` and `stderr.log` are written automatically)"
            )
    except TrialPathError as exc:
        raise ExtractorError(str(exc)) from exc
    except OSError as exc:
        raise ExtractorError(f"Could not inspect log file {cfg.file!r}: {exc}") from exc

    pattern = re.compile(cfg.pattern)

    # Preserve original bytes for hashing and universal-newline boundaries for matching.
    # Only lines eligible for matching require strict UTF-8 decoding.
    hasher = hashlib.sha256()
    size_bytes = 0
    result: float | None = None
    result_line: int | None = None
    count = 0
    first_unparseable: str | None = None
    try:
        _require_fresh(ctx.baseline, ctx.trial_dir, ctx.attempt_dir, cfg.file, target)
        with target.open("r", encoding="utf-8", errors="surrogateescape", newline="") as fh:
            for line_no, raw_text in enumerate(fh, 1):
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
                for match in pattern.finditer(line):
                    captured = match.group("value")
                    try:
                        v = float(captured)
                    except (TypeError, ValueError):
                        if first_unparseable is None:
                            first_unparseable = (
                                f"First unparseable capture {captured!r} on line {line_no}. "
                                "Adjust the capture to exclude delimiters around the number."
                            )
                        continue
                    count += 1
                    if cfg.select in ("first", "last"):
                        result = v
                        result_line = line_no
                    elif not math.isfinite(v):
                        # Float ordering puts nan and inf anywhere, so a diverged
                        # run would be scored on whichever side of them it landed.
                        raise ExtractorError(
                            f"Log file {target} has a non-finite match on line {line_no}, "
                            f"so select: {cfg.select} cannot score this trial."
                        )
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
    except StaleEvidenceError as exc:
        raise ExtractorError(str(exc)) from exc
    except OSError as exc:
        raise ExtractorError(f"Could not read log file at {target}: {exc}") from exc

    sha256 = hasher.hexdigest()
    if first_unparseable is not None:
        if result is None:
            raise ExtractorError(
                f"Pattern {cfg.pattern!r} matched in {target}, but no 'value' capture was "
                f"numeric. {first_unparseable}"
            )
        log.warning(
            "Pattern %r skipped nonnumeric 'value' captures in %s. %s",
            cfg.pattern,
            target,
            first_unparseable,
        )
    if result is None:
        raise ExtractorError(f"No matches for {cfg.pattern!r} in {target}")
    if provenance is not None:
        provenance["source"] = {
            "kind": "file",
            "path": cfg.file,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "matched_line": result_line,
            "match_count": count,
        }
    return result


def _capture_wandb(
    ctx: TrialContext, cfg: _WandbSummarySource, *, deadline: float | None = None
) -> dict[str, Any]:
    """Share one W&B capture or capture failure between the attempt's consumers.

    :param TrialContext ctx: Trial identity and optional shared query/capture.
    :param _WandbSummarySource cfg: Source for a one-off query if needed.
    :param float | None deadline: Absolute monotonic deadline, also bounded by the source timeout.
    :raises DeadlineExceededError: The execution deadline prevents capture.
    :raises EvidenceUnreadableError: W&B gave no answer: authentication or setup failed, the
        SDK is missing, the reader worker failed, or every poll failed in transport.
    :raises ExtractorError: The run is missing, ended unsuccessfully or never finished, shows
        nothing a rerun logged, the finished summary lacks required keys, or evidence is invalid.
    :raises UnsafeProcessCleanupError: Worker cleanup is uncertain.
    :return dict: Finished run identity, numeric values, present keys, and retrieval time.
    """
    if ctx.wandb_capture:
        return ctx.wandb_capture
    if ctx.wandb_capture_failure:
        raise ctx.wandb_capture_failure[0]
    query = ctx.wandb_query or WandbQuery(
        cfg,
        (cfg.metric_key,) if isinstance(cfg, WandbExtractor) else (),
        tuple(cfg.keys) if isinstance(cfg, WandbSummaryRequiredGate) else (),
    )
    source = query.source
    run_id = ctx.wandb_run_id
    poll_deadline = time.monotonic() + source.timeout_seconds
    capped = deadline is not None and deadline <= poll_deadline
    if deadline is not None:
        poll_deadline = min(poll_deadline, deadline)
    target = f"{source.base_url}/{source.entity}/{source.project}/{run_id}"
    try:
        try:
            capture = poll_wandb_summary(
                base_url=source.base_url,
                entity=source.entity,
                project=source.project,
                run_id=run_id,
                attempt_id=ctx.attempt_id,
                attempt_dir=ctx.attempt_dir,
                poll_seconds=source.poll_seconds,
                timeout_seconds=source.timeout_seconds,
                required_keys=query.numeric_keys,
                presence_keys=query.presence_keys,
                logged_after=None if ctx.baseline is None else ctx.baseline.launched_at,
                environment=ctx.wandb_environment,
                deadline=poll_deadline,
            )
        except (WandbPollTimeout, WandbRunMissingError) as exc:
            unanswered = isinstance(exc, WandbPollTimeout) and not exc.reached
            # The worker stops polling a reserve before its deadline so that its report arrives
            # in time, so the clock is usually still short of that deadline here. A poll that
            # ended before the evidence was complete while the execution deadline was the
            # binding limit was cut short by it, whether or not the clock has passed it.
            error: type[ExtractorError] = (
                DeadlineExceededError
                if capped
                else EvidenceUnreadableError
                if unanswered
                else ExtractorError
            )
            if isinstance(exc, WandbRunMissingError):
                message = (
                    f"No W&B run {run_id!r} was found in {source.entity}/{source.project} "
                    "at the evidence deadline. The trainer should call wandb.init() without "
                    "its own project, entity or id."
                )
            else:
                expired = (
                    f"W&B gave no answer for {target} before the evidence deadline"
                    if unanswered
                    else f"W&B evidence deadline expired for {target}"
                )
                message = f"{expired}; required summary keys: {query.numeric_keys!r}." + (
                    f" Last observation: {exc.last_error}" if exc.last_error else ""
                )
            raise error(message) from exc
        except WandbSetupError as exc:
            raise EvidenceUnreadableError(
                f"W&B authentication/setup failed for {target}: {exc.cause}."
            ) from exc
        except WandbRunTerminalError as exc:
            raise ExtractorError(
                f"W&B run {target} ended in {exc.state!r}; only finished runs can provide evidence."
            ) from exc
        except WandbRunStaleError as exc:
            raise ExtractorError(
                f"W&B run {target} was not written by this attempt: its summary records no "
                "wandb.log() call after the attempt launched."
            ) from exc
        except WandbMissingKeysError as exc:
            raise ExtractorError(
                f"W&B run {target} finished without required summary key(s) {exc.missing!r}; "
                f"available summary keys: {exc.available!r}."
            ) from exc
        except ImportError as exc:
            raise EvidenceUnreadableError(WANDB_SDK_MISSING) from exc
        except ValueError as exc:
            raise ExtractorError(f"Invalid W&B numeric evidence for {target}: {exc}") from exc
        except (ExtractorError, UnsafeProcessCleanupError):
            raise
        except RuntimeError as exc:
            raise EvidenceUnreadableError(f"W&B evidence worker failed for {target}.") from exc
    except DeadlineExceededError:
        raise
    except ExtractorError as exc:
        ctx.wandb_capture_failure.append(exc)
        raise
    ctx.wandb_capture.update(
        {
            "kind": "wandb",
            "base_url": source.base_url,
            "entity": source.entity,
            "project": source.project,
            "run_id": run_id,
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
        run_id=ctx.wandb_run_id,
        attempt_id=ctx.attempt_id,
        attempt_dir=ctx.attempt_dir,
        poll_seconds=cfg.poll_seconds,
        timeout_seconds=cfg.timeout_seconds,
        metric_key=cfg.metric_key,
        axis=cfg.evaluation_axis,
        value=provenance["remote_capture"]["values"][cfg.metric_key],
        environment=ctx.wandb_environment,
        deadline=deadline,
    )


def _earlier_mlflow_bindings(ctx: TrialContext, binding: RunBinding) -> list[RunBinding]:
    """Collect the bound runs of the earlier attempts that trained in the trial's directory.

    An earlier attempt that never bound a run logged nothing to merge.

    :param TrialContext ctx: Attempt naming its earlier attempts, oldest first.
    :param RunBinding binding: This attempt's verified binding.
    :raises ValueError: An earlier binding is unreadable or names another target or trial input.
    :return list[RunBinding]: The earlier bindings, oldest first.
    """
    from phasesweep.tracking.session import BINDING_FILE, attempt_tracking_dir, read_binding

    earlier: list[RunBinding] = []
    for generation_id, attempt_id in ctx.earlier_attempts:
        directory = attempt_tracking_dir(ctx.trial_dir.parent.parent, generation_id, attempt_id)
        if not (directory / BINDING_FILE).is_file():
            continue
        prior = read_binding(directory)
        if (
            prior["attempt_id"] != attempt_id
            or prior["tracking_uri"] != binding["tracking_uri"]
            or prior["experiment_id"] != binding["experiment_id"]
            or prior["experiment"] != binding["experiment"]
            or prior["phase"] != binding["phase"]
            or prior["overrides_sha256"] != binding["overrides_sha256"]
        ):
            raise ValueError(
                f"MLflow binding of earlier attempt {attempt_id} names another target or trial."
            )
        earlier.append(prior)
    return earlier


def _capture_mlflow(ctx: TrialContext, *, deadline: float | None) -> MlflowCapture:
    """Read each native history once, after verifying the exact bound run.

    A resumed trial's histories include its earlier attempts' runs, so every selector reads
    the trial's whole history.

    :param TrialContext ctx: Attempt and configured capture contract.
    :param float | None deadline: Existing execution deadline.
    :return MlflowCapture: Shared frozen evidence.
    :raises DeadlineExceededError: The execution deadline stopped the capture.
    :raises EvidenceUnreadableError: MLflow gave no answer: it was unreachable, refused
        access, lacked its SDK, timed out, or the reader's outcome is unknown.
    :raises MlflowObjectiveMissingError: A finished managed run lacks the objective metric.
    :raises ExtractorError: Binding or remote evidence cannot be accepted.
    """
    from phasesweep.evidence.mlflow import validate_capture
    from phasesweep.tracking.mlflow import ManagedObjectiveMissingError, request_operation
    from phasesweep.tracking.session import (
        BINDING_FILE,
        EXPECTED_BINDING_FILE,
        attempt_tracking_dir,
        gave_no_answer,
        read_binding,
        read_record,
    )

    if ctx.mlflow_capture:
        return validate_capture(ctx.mlflow_capture)
    if ctx.mlflow_settings is None or ctx.mlflow_query is None:
        raise ExtractorError("MLflow capture requires a configured target and query.")
    tracking_dir = attempt_tracking_dir(
        ctx.trial_dir.parent.parent, ctx.generation_id, ctx.attempt_id
    )
    try:
        # Managed runs are bound before launch, so only a trainer-owned run can be missing.
        if not (tracking_dir / BINDING_FILE).is_file():
            raise ValueError(
                "The trainer never bound its MLflow run; call "
                "phasesweep.tracking.bind_mlflow_run(run_id) inside its active run."
            )
        binding = read_binding(tracking_dir)
        expected = read_record(tracking_dir / EXPECTED_BINDING_FILE)
        if (
            any(binding.get(k) != v for k, v in expected.items())
            or binding["attempt_id"] != ctx.attempt_id
            or binding["generation_id"] != ctx.generation_id
            or binding["overrides_sha256"] != ctx.overrides_sha256
        ):
            raise ValueError("MLflow binding belongs to a different attempt or target.")
        earlier = _earlier_mlflow_bindings(ctx, binding)
        stop = time.monotonic() + ctx.mlflow_settings["timeout_seconds"]
        retrieved = request_operation(
            {
                "action": "capture",
                "settings": ctx.mlflow_settings,
                "binding": binding,
                "earlier": earlier,
                "query": ctx.mlflow_query,
                "timeout_at": stop,
            },
            operation_id=ctx.attempt_id,
            directory=ctx.attempt_dir,
            environment=ctx.mlflow_environment or {},
            deadline=min(stop, deadline) if deadline is not None else stop,
        )
        capture = validate_capture(retrieved)
    except ManagedObjectiveMissingError as exc:
        raise MlflowObjectiveMissingError(str(exc)) from exc
    except (OSError, ValueError, RuntimeError) as exc:  # every NO_ANSWER error included
        if isinstance(exc, UnsafeProcessCleanupError):
            raise
        if not gave_no_answer(exc):
            raise ExtractorError(str(exc)) from exc
        if isinstance(exc, TimeoutError) and deadline is not None and time.monotonic() >= deadline:
            raise DeadlineExceededError(str(exc)) from exc
        raise EvidenceUnreadableError(str(exc)) from exc
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
    # The tracker gave no answer about the evidence this gate reads; never persisted.
    evidence_unreadable: bool = False


def _required_file(ctx: TrialContext, gate: RequiredFileGate) -> GateResult:
    """Check that a required trial-relative file exists.

    :param TrialContext ctx: Trial context containing the trial directory.
    :param RequiredFileGate gate: Gate config naming the required file path.
    :return GateResult: Pass/fail result and human-readable detail.
    """
    path = trial_file(ctx.trial_dir, ctx.attempt_dir, gate.path, checked=False)
    if path.is_file():
        return GateResult(gate.type, True, f"{gate.path} exists")
    refusal = unusable_link(ctx.trial_dir, ctx.attempt_dir, gate.path)
    return GateResult(gate.type, False, refusal or f"{gate.path} is missing")


def _json_equals(ctx: TrialContext, gate: JsonEqualsGate) -> GateResult:
    """Check that a JSON value exactly equals the expected scalar.

    :param TrialContext ctx: Trial context containing the trial directory.
    :param JsonEqualsGate gate: Gate config naming the JSON path, key, and value.
    :return GateResult: Pass/fail result and human-readable detail.
    """
    try:
        _, actual = load_json_value(
            ctx.trial_dir, ctx.attempt_dir, gate.path, gate.key, baseline=ctx.baseline
        )
    except Exception as exc:
        return GateResult(gate.type, False, _gate_unavailable(gate.path, gate.key, exc))
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
        _, raw_value = load_json_value(
            ctx.trial_dir, ctx.attempt_dir, gate.path, gate.key, baseline=ctx.baseline
        )
        value = json_float(raw_value, label=gate.key)
    except Exception as exc:
        return GateResult(gate.type, False, _gate_unavailable(gate.path, gate.key, exc))
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
    path = trial_file(ctx.trial_dir, ctx.attempt_dir, gate.path, checked=False)
    if gate.source != "json" and (
        refusal := unusable_link(
            ctx.trial_dir, ctx.attempt_dir, gate.path, directory=gate.source == "directory"
        )
    ):
        return GateResult(gate.type, False, refusal)
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
            _, raw_size = load_json_value(
                ctx.trial_dir, ctx.attempt_dir, gate.path, gate.key, baseline=ctx.baseline
            )
        except Exception as exc:
            return GateResult(gate.type, False, _gate_unavailable(gate.path, gate.key, exc))
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
    try:
        path = trial_file(ctx.trial_dir, ctx.attempt_dir, gate.path)
        if not path.is_file():
            return GateResult(gate.type, False, f"{gate.path} is missing")
        digest = file_sha256(path)
    except TrialPathError as exc:
        return GateResult(gate.type, False, str(exc))
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
    :return GateResult: Presence result; a failed capture fails the gate, marked as the
        deadline or an unreadable tracker when either caused it.
    """
    try:
        _capture_mlflow(ctx, deadline=deadline)
    except ExtractorError as exc:
        return GateResult(
            gate.type,
            False,
            str(exc),
            deadline_exhausted=isinstance(exc, DeadlineExceededError),
            evidence_unreadable=isinstance(exc, EvidenceUnreadableError),
        )
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
        fails the gate, marked ``deadline_exhausted`` when the deadline expired and
        ``evidence_unreadable`` when W&B gave no answer.
    """
    try:
        capture = _capture_wandb(ctx, gate, deadline=deadline)
    except DeadlineExceededError as exc:
        return GateResult(gate.type, False, str(exc), deadline_exhausted=True)
    except ExtractorError as exc:
        return GateResult(
            gate.type,
            False,
            str(exc),
            evidence_unreadable=isinstance(exc, EvidenceUnreadableError),
        )
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
