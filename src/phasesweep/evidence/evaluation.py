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

from phasesweep.config.common import TRACKER_READ_CONTROLS, is_sha256_hex
from phasesweep.errors import UnsafeProcessCleanupError
from phasesweep.evidence.baseline import (
    EVIDENCE_BASELINE_FILE,
    EvidenceBaseline,
    FileStamp,
    read_evidence_baseline,
)
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
    WandbNothingToScoreError,
    WandbPollTimeout,
    WandbRunMissingError,
    WandbRunSupersededError,
    WandbRunTerminalError,
    WandbSetupError,
    locate_wandb_position,
    poll_wandb_summary,
)
from phasesweep.runtime.files import broken_symlink_detail, file_sha256, resolved_file_change_ns
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
EVIDENCE_PROVENANCE_SCHEMA_VERSION = 8
# Retained evidence is indexed by configured path and named by its recorded file digest.
RETAINED_EVIDENCE_SCHEMA_VERSION = 1
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


def retained_evidence_dir(attempt_dir: Path, relative: str) -> Path:
    """Locate this attempt's saved measurement bytes for a configured file path.

    :param Path attempt_dir: Producing attempt's audit directory.
    :param str relative: Configured path, relative to the trial directory.
    :return Path: Directory whose filenames are the original file digests.
    """
    path_key = hashlib.sha256(Path(relative).as_posix().encode()).hexdigest()
    return attempt_dir / f"retained-evidence-v{RETAINED_EVIDENCE_SCHEMA_VERSION}" / path_key


def retained_evidence_file(
    attempt_dir: Path, relative: str, *, digest: str | None = None
) -> Path | None:
    """Find the retained score file for this attempt and path, if a rerun saved it.

    :param Path attempt_dir: Producing attempt's audit directory.
    :param str relative: Configured path, relative to the trial directory.
    :param str | None digest: Expected digest for a frozen objective, or None while re-reading.
    :raises TrialPathError: The retained file has no unambiguous, readable original contents.
    :return Path | None: The retained file, or None if this attempt has no capture of that path.
    """
    directory = retained_evidence_dir(attempt_dir, relative)
    if digest is not None:
        path = directory / digest
        try:
            path.lstat()
        except FileNotFoundError:
            return None
    else:
        try:
            paths = [entry for entry in directory.iterdir() if is_sha256_hex(entry.name)]
        except FileNotFoundError:
            return None
        if not paths:
            return None
        if len(paths) != 1:
            raise TrialPathError(f"Several retained files claim {relative!r} in {attempt_dir}.")
        path = paths[0]
    retained = attempt_file(attempt_dir, path.relative_to(attempt_dir))
    if digest is None and file_sha256(retained) != retained.name:
        raise TrialPathError(f"Retained evidence at {retained} no longer has its original bytes.")
    return retained


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


def _require_before_next_launch(
    trial_dir: Path, relative: str, target: Path, evidence_before: float | str | None
) -> None:
    """Refuse a shared artifact whose state cannot be attributed to the earlier attempt.

    :param Path trial_dir: Directory shared by a trial's attempts.
    :param str relative: Configured trial-relative artifact path.
    :param Path target: Location already resolved by the evidence reader.
    :param float | str | None evidence_before: Next launch time, an attribution failure, or None.
    :raises StaleEvidenceError: The artifact can no longer be attributed to this attempt.
    :raises OSError: The artifact or a link to it could not be inspected.
    """
    if evidence_before is None or is_captured_stream(relative):
        return
    if isinstance(evidence_before, str):
        raise StaleEvidenceError(
            f"`{target}` cannot be attributed to this attempt: {evidence_before}."
        )
    if resolved_file_change_ns(trial_dir, target) >= round(evidence_before * 1e9):
        raise StaleEvidenceError(
            f"`{target}` changed after this attempt finished and another attempt reused its "
            "directory. The earlier value cannot be recovered from this path."
        )


def _written_before_launch(
    baseline: EvidenceBaseline | None,
    trial_dir: Path,
    attempt_dir: Path,
    relative: str,
    target: Path,
    *,
    evidence_before: float | str | None = None,
) -> bool:
    """Say whether a value file an attempt reads was written by an earlier attempt of its trial.

    Only the trial's own attempts write in its directory, so a value file there is the trial's
    result whichever of them wrote it; the answer only decides what the trial discloses, and
    what a read that finds no value there implies.
    A recorded file is this attempt's when its size or modification time changes, or when its
    change time changes while its hard-link count stays the same. This compares the file to
    its own earlier stamp and covers copies that preserve modification time.
    A file at a target that held none at launch was created during the attempt. Any other
    target, such as one reached through a retargeted link or one the baseline could not
    inspect, is this attempt's when its modification time is after the baseline write, or its
    change time is at or after that write. A write can share the baseline's clock tick. This
    comparison assumes the trial and attempt directories share a clock.

    :param EvidenceBaseline | None baseline: Value paths and file stamps recorded at launch.
    :param Path trial_dir: Directory containing trainer files.
    :param Path attempt_dir: The attempt's directory, holding the baseline record.
    :param str relative: Configured trial-relative path.
    :param Path target: Current location of evidence after containment checks.
    :param float | str | None evidence_before: Unix time the shared value file must predate,
        or the reason it cannot be attributed to the attempt, defaults to None.
    :raises StaleEvidenceError: The file changed after a later launch in its directory.
    :raises OSError: The current file or the baseline record could not be inspected.
    :return bool: Whether the file predates this attempt in a directory an earlier one used.
    """
    if is_captured_stream(relative):
        return False
    _require_before_next_launch(trial_dir, relative, target, evidence_before)
    if baseline is None or not baseline.reused:
        return False
    info = target.stat()
    current = target.resolve().relative_to(trial_dir.resolve()).as_posix()
    recorded = baseline.files.values()
    prior = next((s for s in recorded if isinstance(s, FileStamp) and s.path == current), None)
    if prior is not None:
        return prior.unchanged(info)
    if current == baseline.files.get(relative) or current in recorded:
        return False
    launch_stamp = (attempt_dir / EVIDENCE_BASELINE_FILE).stat().st_mtime_ns
    return not (info.st_mtime_ns > launch_stamp or info.st_ctime_ns >= launch_stamp)


def load_json_value(
    trial_dir: Path,
    attempt_dir: Path,
    relative_path: str,
    key: str,
    *,
    digest: dict[str, Any] | None = None,
    baseline: EvidenceBaseline | None = None,
    evidence_before: float | str | None = None,
    earlier: list[str] | None = None,
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
    :param float | str | None evidence_before: Unix time the shared JSON file must predate,
        or the reason it cannot be attributed to the attempt, defaults to None.
    :param list[str] | None earlier: Sink that receives ``relative_path`` before the file is
        parsed when an earlier attempt of the trial wrote it (:func:`_written_before_launch`).
    :raises TrialPathError: If the path contains a broken link or resolves outside its directory.
    :raises StaleEvidenceError: If the file changed after a later launch in its directory.
    :raises FileNotFoundError: If ``relative_path`` does not resolve to a regular file.
    :raises JsonKeyMissingError: If a segment of the dotted ``key`` path is missing, with the
        failing segment as the argument and what the file holds there.
    :raises OSError: If the resolved file exists but cannot be read.
    :raises UnicodeDecodeError: If the file bytes are not valid UTF-8.
    :raises ValueError: If the contents are not strict JSON (malformed document,
        duplicate object keys, or a rejected JSON constant).
    :return tuple[Path, Any]: Resolved JSON file path and loaded value.
    """
    retained = retained_evidence_file(attempt_dir, relative_path)
    target = retained if retained is not None else trial_file(trial_dir, attempt_dir, relative_path)
    if not target.is_file():
        raise FileNotFoundError(target)
    if (
        retained is None
        and _written_before_launch(
            baseline, trial_dir, attempt_dir, relative_path, target, evidence_before=evidence_before
        )
        and earlier is not None
    ):
        earlier.append(relative_path)
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


class NothingToScoreError(ExtractorError):
    """Raised when an attempt in a directory an earlier attempt used left nothing to score there.

    The trainer exited 0, but the value file is missing, or an earlier attempt of the trial
    wrote it and it does not parse, or the attempt's own captured stream has no match, or the
    trial's resumed W&B run shows at the deadline no row since the launch and no finished
    summary that is the trial's result. Such
    an attempt can never succeed in that directory, which says nothing about the
    configuration, so it is owed a rerun in a fresh directory instead of failing.
    """


def _in_reused_directory(ctx: TrialContext) -> bool:
    """Say whether an attempt ran in a directory an earlier attempt of its trial used.

    :param TrialContext ctx: The attempt being read.
    :return bool: Whether its baseline records a reused directory.
    """
    return ctx.baseline is not None and ctx.baseline.reused


def _read_failure(ctx: TrialContext, relative: str, exc: BaseException) -> type[ExtractorError]:
    """Choose the extractor error for a value file that gave no value.

    A missing file, or one an earlier attempt of the trial wrote that is not valid UTF-8 or
    JSON, leaves nothing to score in a reused directory. A file that parses is evidence: a
    missing key or a non-number in it fails the trial as in a fresh directory, where a
    corrected reading setting can still re-read it, and so does a refused location.

    :param TrialContext ctx: The attempt being read.
    :param str relative: Configured trial-relative path of the file.
    :param BaseException exc: Why the read gave no value.
    :return type[ExtractorError]: :class:`NothingToScoreError` or :class:`ExtractorError`.
    """
    if not _in_reused_directory(ctx):
        return ExtractorError
    if isinstance(exc, FileNotFoundError):
        return NothingToScoreError
    unparsed = isinstance(exc, ValueError) and not isinstance(exc, NonFiniteJsonConstantError)
    return NothingToScoreError if unparsed and relative in ctx.earlier_files else ExtractorError


class MlflowObjectiveMissingError(ExtractorError):
    """Raised when a finished managed MLflow run lacks the configured objective metric."""


def _non_finite_text(constant: str) -> str:
    """Say what a file holding NaN or Infinity means, for a trial's failure reason.

    :param str constant: The refused token.
    :return str: A clause that follows the file's subject.
    """
    return f"holds {constant}, which is not a finite number; a diverged run writes it"


def _gate_unavailable(
    ctx: TrialContext, gate_type: str, path: str, key: str, exc: Exception
) -> GateResult:
    """Fail a JSON gate whose file could not be read or parsed.

    :param TrialContext ctx: The attempt being read.
    :param str gate_type: The gate's configured type.
    :param str path: Trial-relative file the gate reads.
    :param str key: Dotted key the gate reads.
    :param Exception exc: Failure raised while reading it.
    :return GateResult: The failed result, marked when the attempt left nothing to score.
    """
    if isinstance(exc, NonFiniteJsonConstantError):
        detail = f"{path}:{key} unavailable: the file {_non_finite_text(exc.constant)}"
    else:
        detail = f"{path}:{key} unavailable: {exc}"
    return GateResult(
        gate_type,
        False,
        detail,
        nothing_to_score=_read_failure(ctx, path, exc) is NothingToScoreError,
    )


@contextmanager
def _trial_json_read_errors(
    ctx: TrialContext, relative: str, target: Path, *, noun: str, trainer_step: str
) -> Iterator[None]:
    """Translate file-read and JSON-decoding failures into extractor diagnostics.

    :param TrialContext ctx: The attempt being read.
    :param str relative: Configured trial-relative path of the file.
    :param Path target: Trial-relative file being read.
    :param str noun: ``JSON`` or ``JSON envelope`` diagnostic label.
    :param str trainer_step: What the trainer must do for the file to exist, a clause without
        a final period that follows a missing-file diagnostic in parentheses, since a trainer
        that exits 0 without it fails here. The path ends the diagnostic's own words, so no
        punctuation sticks to it when a reader copies it.
    :raises ExtractorError: The file is missing, unreadable, malformed, or outside the
        attempt; the last carries the refusal's own text as the trial's failure reason. It is
        a :class:`NothingToScoreError` when :func:`_read_failure` says so.
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
        raise _read_failure(ctx, relative, exc)(f"{missing} ({trainer_step})") from exc
    except UnicodeError as exc:
        raise _read_failure(ctx, relative, exc)(
            f"{noun} at {target} is not valid UTF-8: {exc}"
        ) from exc
    except NonFiniteJsonConstantError as exc:
        raise ExtractorError(f"{noun} at {target} {_non_finite_text(exc.constant)}.") from exc
    except ValueError as exc:
        raise _read_failure(ctx, relative, exc)(f"Invalid {noun} at {target}: {exc}") from exc
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
    # Configured value paths read so far from a file an earlier attempt of the trial wrote, and
    # ``wandb:<run id>`` once the W&B capture is a summary an earlier attempt left.
    earlier_files: list[str] = field(default_factory=list, compare=False)
    # End of the attributable period for mutable evidence in a directory another attempt reused.
    evidence_before: float | str | None = None
    managed_envelope_path: str | None = None
    # (generation ID, attempt ID) of the earlier attempts that trained in trial_dir, oldest
    # first, whose tracker histories are the start of this trial's.
    earlier_attempts: tuple[tuple[str, str], ...] = ()
    # Reading a terminal attempt uses a separately registered worker process.
    worker_process: tuple[str, Path] | None = None

    @property
    def process_slot(self) -> tuple[str, Path]:
        """Return the process identity and directory a tracker worker may claim.

        :return tuple[str, Path]: The registered reader slot, or the active attempt's slot.
        """
        return self.worker_process or (self.attempt_id, self.attempt_dir)


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
            ctx,
            cfg.path,
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
                evidence_before=ctx.evidence_before,
                earlier=ctx.earlier_files,
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

    The envelope names the attempt that wrote it: this one, or an earlier attempt that
    trained in the trial's directory, whose envelope is the trial's result and is listed in
    ``ctx.earlier_files``. A missing envelope in a reused directory leaves nothing to score.

    :param TrialContext ctx: Current trial identity and resolved-overrides digest.
    :param JsonEnvelopeExtractor cfg: Expected objective and evaluation policy.
    :param dict[str, Any] | None provenance: Optional sink that receives the
        frozen evidence ``source`` payload — envelope digest plus the
        validated evaluation metadata — on success.
    :raises ExtractorError: If the envelope is missing, resolves outside the attempt
        directory, is malformed, or belongs to an attempt that is not this trial's.
    :return float: Validated objective value from the envelope.
    """
    path = ctx.managed_envelope_path or cfg.path
    target = trial_file(ctx.trial_dir, ctx.attempt_dir, path, checked=False)
    with _trial_json_read_errors(
        ctx,
        path,
        target,
        noun="JSON envelope",
        trainer_step=(
            "the trainer must call `phasesweep.report_objective(...)` before it exits, from rank "
            f"0 only; it writes `{path}` under PHASESWEEP_TRIAL_DIR"
        ),
    ):
        retained = retained_evidence_file(ctx.attempt_dir, path)
        raw = (
            retained if retained is not None else trial_file(ctx.trial_dir, ctx.attempt_dir, path)
        ).read_bytes()
        data = strict_json_loads(raw.decode("utf-8"))

    if not isinstance(data, dict):
        raise ExtractorError(f"JSON envelope at {target} must be an object.")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ExtractorError(f"JSON envelope at {target} must use schema_version 1.")
    if data.get("status") != "complete":
        raise ExtractorError(f"JSON envelope at {target} must report status='complete'.")
    # An earlier attempt of the trial that trained in its directory wrote the trial's result.
    earlier = (data.get("generation_id"), data.get("attempt_id")) in ctx.earlier_attempts
    if not earlier and data.get("generation_id") != ctx.generation_id:
        raise ExtractorError(
            f"JSON envelope at {target} does not match generation {ctx.generation_id!r}."
        )
    if not earlier and data.get("attempt_id") != ctx.attempt_id:
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
            "path": path,
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
    if earlier:
        ctx.earlier_files.append(path)
    return value


def _extract_log_regex(
    ctx: TrialContext,
    cfg: LogRegexExtractor,
    provenance: dict[str, Any] | None = None,
) -> float:
    """Select a numeric regex match and digest the complete log file.

    A trainer log kept across a trial's attempts is the trial's log, so every line counts,
    whichever attempt wrote it. First selection
    stops matching after its first numeric hit but continues reading for the digest.
    Warn when a numeric result leaves unparseable captures behind.
    Provenance includes the selected 1-based line and numeric match count.

    :param TrialContext ctx: Trial identity; the log is read from its ``trial_dir``, or its
        ``attempt_dir`` for a captured stream.
    :param LogRegexExtractor cfg: Log path, named value group, and first/last/min/max selector.
    :param dict | None provenance: Optional source-evidence sink.
    :raises ExtractorError: The log is missing, outside the attempt, unreadable, changed after
        a later launch, has no numeric match, has a non-finite match under ``min`` or ``max``,
        or a line eligible for matching is invalid UTF-8; a :class:`NothingToScoreError` for a
        missing log or a captured stream with no match in a reused directory.
    :return float: Selected match value.
    """
    try:
        retained = retained_evidence_file(ctx.attempt_dir, cfg.file)
        target = (
            retained
            if retained is not None
            else trial_file(ctx.trial_dir, ctx.attempt_dir, cfg.file)
        )
        if not target.is_file():
            raise _read_failure(ctx, cfg.file, FileNotFoundError(target))(
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
        if retained is None and _written_before_launch(
            ctx.baseline,
            ctx.trial_dir,
            ctx.attempt_dir,
            cfg.file,
            target,
            evidence_before=ctx.evidence_before,
        ):
            ctx.earlier_files.append(cfg.file)
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
        # A resumed trainer that skips finished work prints nothing to its own stream. A
        # trainer log in the trial directory is the trial's, which a corrected pattern re-reads.
        unscored = _in_reused_directory(ctx) and is_captured_stream(cfg.file)
        raise (NothingToScoreError if unscored else ExtractorError)(
            f"No matches for {cfg.pattern!r} in {target}"
        )
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
        SDK is missing, the reader worker failed, or the last poll failed in transport, found
        a finished summary still being ingested, or found a rerun's run still showing only an
        earlier attempt's summary.
    :raises ExtractorError: The run is missing, ended unsuccessfully or never finished, was
        reused by a later attempt, the finished summary lacks required keys, or evidence is
        invalid; a :class:`NothingToScoreError` when at the deadline a rerun's resumed run
        shows no row since its launch and no finished summary that is the trial's result. A
        summary judged the trial's result that way is captured and lists the run in
        ``ctx.earlier_files``; either verdict is a :class:`DeadlineExceededError` when the
        execution deadline cut the poll short.
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
    evidence_before = ctx.evidence_before
    try:
        destination = source.destination if ctx.baseline is None else ctx.baseline.wandb_destination
        if destination is None:
            raise ExtractorError(
                f"Attempt {ctx.attempt_id!r} had no managed W&B destination at launch."
            )
        target = f"{destination.base_url}/{destination.entity}/{destination.project}/{run_id}"
        if isinstance(evidence_before, str):
            raise ExtractorError(
                f"W&B run {target} cannot be attributed to this attempt: {evidence_before}."
            )
        try:
            logged_after = None if ctx.baseline is None else ctx.baseline.launched_at
            resume_axis = source.evaluation_axis if isinstance(source, WandbExtractor) else "_step"
            earlier_launches: list[float] = []
            if (
                logged_after is not None
                and resume_axis != "_step"
                and any(key.endswith((".min", ".max")) for key in query.numeric_keys)
            ):
                for _, attempt_id in ctx.earlier_attempts:
                    directory = ctx.attempt_dir.with_name(attempt_id)
                    if attempt_id == run_id or not (directory / EVIDENCE_BASELINE_FILE).is_file():
                        continue
                    launched_at = read_evidence_baseline(directory).launched_at
                    if launched_at is not None:
                        earlier_launches.append(launched_at)
            worker_id, worker_dir = ctx.process_slot
            capture = poll_wandb_summary(
                base_url=destination.base_url,
                entity=destination.entity,
                project=destination.project,
                run_id=run_id,
                attempt_id=worker_id,
                attempt_dir=worker_dir,
                poll_seconds=source.poll_seconds,
                timeout_seconds=source.timeout_seconds,
                required_keys=query.numeric_keys,
                presence_keys=query.presence_keys,
                logged_after=logged_after,
                logged_before=evidence_before,
                earlier_launches=tuple(earlier_launches),
                resume_axis=resume_axis,
                launch_rows=None if ctx.baseline is None else ctx.baseline.wandb_rows,
                launch_state=None if ctx.baseline is None else ctx.baseline.wandb_state,
                environment=ctx.wandb_environment,
                deadline=poll_deadline,
            )
            # A rerun's run judged at the deadline, so an execution deadline that was the
            # binding limit cut the wait short instead.
            if capped and capture.get("earlier_attempt") is True:
                raise DeadlineExceededError(
                    f"W&B run {target} still showed only an earlier attempt's summary when the "
                    "phase or run deadline ended its evidence read."
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
                    f"No W&B run {run_id!r} was found in {destination.entity}/{destination.project} "
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
        except WandbRunSupersededError as exc:
            raise ExtractorError(
                f"W&B run {target} was reused by another attempt and its current summary "
                "cannot be attributed to this attempt."
            ) from exc
        # A verdict reached only at the deadline, so an execution deadline that was the binding
        # limit cut the wait short instead.
        except WandbNothingToScoreError as exc:
            raise (DeadlineExceededError if capped else NothingToScoreError)(
                f"W&B run {target} {exc.cause}."
            ) from exc
        except WandbMissingKeysError as exc:
            raise (DeadlineExceededError if capped else ExtractorError)(
                f"W&B run {target} finished without required summary key(s) {exc.missing!r} "
                f"after {exc.waited_seconds:.0f} s; available summary keys: {exc.available!r}."
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
    # The worker's mark that the summary is an earlier attempt's discloses the trial's result
    # the way an earlier attempt's value file does, and is not part of the stored capture.
    if capture.pop("earlier_attempt", None) is True:
        ctx.earlier_files.append(f"wandb:{run_id}")
    ctx.wandb_capture.update(
        {
            "kind": "wandb",
            **destination._asdict(),
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
    capture = provenance["remote_capture"]
    worker_id, worker_dir = ctx.process_slot
    progress["value"] = locate_wandb_position(
        base_url=capture["base_url"],
        entity=capture["entity"],
        project=capture["project"],
        run_id=ctx.wandb_run_id,
        attempt_id=worker_id,
        attempt_dir=worker_dir,
        poll_seconds=cfg.poll_seconds,
        timeout_seconds=cfg.timeout_seconds,
        metric_key=cfg.metric_key,
        axis=cfg.evaluation_axis,
        value=capture["values"][cfg.metric_key],
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
    :raises MlflowObjectiveMissingError: A finished managed run lacks the objective metric;
        a :class:`NothingToScoreError` instead for an attempt in a reused directory.
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
        worker_id, worker_dir = ctx.process_slot
        retrieved = request_operation(
            {
                "action": "capture",
                "settings": ctx.mlflow_settings,
                "binding": binding,
                "earlier": earlier,
                "query": ctx.mlflow_query,
                "timeout_at": stop,
            },
            operation_id=worker_id,
            directory=worker_dir,
            environment=ctx.mlflow_environment or {},
            deadline=min(stop, deadline) if deadline is not None else stop,
        )
        capture = validate_capture(retrieved)
    except ManagedObjectiveMissingError as exc:
        # In a reused directory the attempt's own run is fresh and empty when its trainer
        # skipped finished work; scoring an earlier attempt's run would move the result
        # delivery to that run, so the attempt reruns once with a fresh run instead.
        if _in_reused_directory(ctx):
            raise NothingToScoreError(str(exc)) from exc
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
    # The gate's file is missing or unparsed in a reused directory (:class:`NothingToScoreError`).
    nothing_to_score: bool = False


def _missing_file(ctx: TrialContext, gate: Gate, detail: str) -> GateResult:
    """Fail a gate whose file is missing, marked when a reused directory leaves nothing to score.

    :param TrialContext ctx: The attempt being read.
    :param Gate gate: The gate.
    :param str detail: What is missing.
    :return GateResult: The failed result.
    """
    return GateResult(gate.type, False, detail, nothing_to_score=_in_reused_directory(ctx))


def _required_file(ctx: TrialContext, gate: RequiredFileGate) -> GateResult:
    """Check that a required trial-relative file exists.

    :param TrialContext ctx: Trial context containing the trial directory.
    :param RequiredFileGate gate: Gate config naming the required file path.
    :return GateResult: Pass/fail result and human-readable detail.
    """
    path = trial_file(ctx.trial_dir, ctx.attempt_dir, gate.path, checked=False)
    if path.is_file():
        try:
            _require_before_next_launch(ctx.trial_dir, gate.path, path, ctx.evidence_before)
        except (OSError, StaleEvidenceError) as exc:
            return GateResult(gate.type, False, str(exc))
        return GateResult(gate.type, True, f"{gate.path} exists")
    refusal = unusable_link(ctx.trial_dir, ctx.attempt_dir, gate.path)
    if refusal is not None:
        return GateResult(gate.type, False, refusal)
    return _missing_file(ctx, gate, f"{gate.path} is missing")


def _json_equals(ctx: TrialContext, gate: JsonEqualsGate) -> GateResult:
    """Check that a JSON value exactly equals the expected scalar.

    :param TrialContext ctx: Trial context containing the trial directory.
    :param JsonEqualsGate gate: Gate config naming the JSON path, key, and value.
    :return GateResult: Pass/fail result and human-readable detail.
    """
    try:
        _, actual = load_json_value(
            ctx.trial_dir,
            ctx.attempt_dir,
            gate.path,
            gate.key,
            baseline=ctx.baseline,
            evidence_before=ctx.evidence_before,
            earlier=ctx.earlier_files,
        )
    except Exception as exc:
        return _gate_unavailable(ctx, gate.type, gate.path, gate.key, exc)
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
            ctx.trial_dir,
            ctx.attempt_dir,
            gate.path,
            gate.key,
            baseline=ctx.baseline,
            evidence_before=ctx.evidence_before,
            earlier=ctx.earlier_files,
        )
    except Exception as exc:
        return _gate_unavailable(ctx, gate.type, gate.path, gate.key, exc)
    try:
        value = json_float(raw_value, label=gate.key)
    except ValueError as exc:
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
                if not path.exists():
                    return _missing_file(ctx, gate, f"{gate.path} is not a file")
                return GateResult(gate.type, False, f"{gate.path} is not a file")
            _require_before_next_launch(ctx.trial_dir, gate.path, path, ctx.evidence_before)
            size = path.stat().st_size
        except StaleEvidenceError as exc:
            return GateResult(gate.type, False, str(exc))
        except OSError as exc:
            return GateResult(gate.type, False, f"could not inspect {gate.path}: {exc}")
        label = f"{gate.path} file size"
    elif gate.source == "directory":
        try:
            root = path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            return _missing_file(ctx, gate, f"could not inspect {gate.path}: {path}: {exc}")
        except OSError as exc:
            return GateResult(gate.type, False, f"could not inspect {gate.path}: {path}: {exc}")
        if not stat.S_ISDIR(root.st_mode):
            return GateResult(gate.type, False, f"{gate.path} is not a directory")
        directories = [path]
        size = 0
        while directories:
            directory = directories.pop()
            try:
                _require_before_next_launch(
                    ctx.trial_dir, gate.path, directory, ctx.evidence_before
                )
                with os.scandir(directory) as entries:
                    for entry in entries:
                        entry_path = Path(entry.path)
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                directories.append(entry_path)
                            # Count file-symlink targets without traversing directory symlinks.
                            elif entry.is_file():
                                _require_before_next_launch(
                                    ctx.trial_dir, gate.path, entry_path, ctx.evidence_before
                                )
                                size += entry_path.stat().st_size
                        except OSError as exc:
                            return GateResult(
                                gate.type,
                                False,
                                f"could not inspect {gate.path}: {entry_path}: {exc}",
                            )
            except StaleEvidenceError as exc:
                return GateResult(gate.type, False, str(exc))
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
                ctx.trial_dir,
                ctx.attempt_dir,
                gate.path,
                gate.key,
                baseline=ctx.baseline,
                evidence_before=ctx.evidence_before,
                earlier=ctx.earlier_files,
            )
        except Exception as exc:
            return _gate_unavailable(ctx, gate.type, gate.path, gate.key, exc)
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
            return _missing_file(ctx, gate, f"{gate.path} is missing")
        _require_before_next_launch(ctx.trial_dir, gate.path, path, ctx.evidence_before)
        digest = file_sha256(path)
    except (TrialPathError, StaleEvidenceError) as exc:
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
            nothing_to_score=isinstance(exc, NothingToScoreError),
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
            nothing_to_score=isinstance(exc, NothingToScoreError),
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
