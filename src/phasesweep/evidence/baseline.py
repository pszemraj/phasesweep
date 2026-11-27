"""Record value file metadata and a rerun's launch time before the trainer starts."""

from __future__ import annotations

import errno
import json
import stat
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, assert_never

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
    WandbDestination,
    WandbExtractor,
    WandbSummaryRequiredGate,
    is_captured_stream,
)
from phasesweep.runtime.files import atomic_write_text
from phasesweep.runtime.json import json_object, json_text, strict_json_loads

if TYPE_CHECKING:
    from os import stat_result

    from phasesweep.config.models import Experiment, Phase

EVIDENCE_BASELINE_SCHEMA_VERSION = 8
EVIDENCE_BASELINE_FILE = "evidence_baseline.json"


class FileStamp(NamedTuple):
    """Where a value file resolved before launch, with its freshness metadata."""

    path: str
    size: int
    mtime_ns: int
    ctime_ns: int
    nlink: int

    def unchanged(self, info: stat_result) -> bool:
        """Check whether a file still carries the stamp it had before launch.

        :param stat_result info: Current metadata of the file at :attr:`path`.
        :return bool: Whether the metadata shows no fresh value file.
        """
        same_size_and_mtime = (info.st_size, info.st_mtime_ns) == (self.size, self.mtime_ns)
        # A link-count change can advance ctime by itself, including when the trainer
        # removes the result path and puts the old file back via a hard link.
        return same_size_and_mtime and (
            info.st_ctime_ns == self.ctime_ns or info.st_nlink != self.nlink
        )


class EvidenceBaseline(NamedTuple):
    """The value paths of a trial directory, and the time, before one attempt starts."""

    reused: bool
    # A path string records a resolved target with no regular file before launch. None
    # records a target whose prior state could not be inspected.
    files: Mapping[str, FileStamp | str | None]
    # Unix time just before a rerun's trainer started, which the trial's W&B run must show
    # a row logged after; None in a directory of the attempt's own, whose run is its own.
    launched_at: float | None = None
    wandb_destination: WandbDestination | None = None
    # The trial's W&B run's history row count and state just before a rerun's trainer
    # started (:func:`phasesweep.evidence.wandb.read_wandb_stamp`); None when unread.
    wandb_rows: int | None = None
    wandb_state: str | None = None


def _extractor_value_path(extractor: ObjectiveExtractor) -> str | None:
    """Return the trial-relative file an extractor reads its value from, if it reads one.

    Every extractor type is named, so a new one is a type error here until it says whether
    it reads a file in the trial directory. ``json_envelope`` is left out: the envelope
    names the attempt that wrote it.

    :param ObjectiveExtractor extractor: A metric, constraint or observation extractor.
    :return str | None: The path, or ``None`` when the value comes from no trial file.
    """
    if isinstance(extractor, JsonExtractor):
        return extractor.path
    if isinstance(extractor, LogRegexExtractor):
        return extractor.file
    if isinstance(extractor, JsonEnvelopeExtractor | WandbExtractor | MlflowExtractor):
        return None
    assert_never(extractor)


def _gate_value_path(gate: Gate) -> str | None:
    """Return the trial-relative file a gate reads a value from, if it reads one.

    Every gate type is named, so a new one is a type error here until it says whether it
    reads a value from a file in the trial directory. ``required_file``, ``sha256`` and file
    or directory ``artifact_size`` gates test the trial's artifacts whichever attempt wrote
    them, so they are left out.

    :param Gate gate: A phase gate.
    :return str | None: The path, or ``None`` when the gate reads no value from a trial file.
    """
    if isinstance(gate, JsonEqualsGate | JsonScalarBoundGate):
        return gate.path
    if isinstance(gate, ArtifactSizeGate):
        return gate.path if gate.source == "json" else None
    if isinstance(
        gate, RequiredFileGate | Sha256Gate | WandbSummaryRequiredGate | MlflowRequiredGate
    ):
        return None
    assert_never(gate)


def _value_paths(experiment: Experiment, phase: Phase) -> set[str]:
    """Collect every trial-directory path a value is read from.

    A captured stream is left out whichever extractor or gate reads it: it is the attempt's
    own, in its attempt directory.

    :param Experiment experiment: Metric, constraints and observations.
    :param Phase phase: Gates of the phase the attempt runs in.
    :return set[str]: The distinct paths.
    """
    paths = [
        _extractor_value_path(experiment.metric.extractor),
        *(_extractor_value_path(constraint.extractor) for constraint in experiment.constraints),
        *(_extractor_value_path(observation.extractor) for observation in experiment.observe),
        *(_gate_value_path(gate) for gate in phase.gates),
    ]
    return {path for path in paths if path is not None and not is_captured_stream(path)}


def write_evidence_baseline(
    experiment: Experiment,
    phase: Phase,
    trial_dir: Path,
    attempt_dir: Path,
    *,
    reused: bool = False,
    wandb_destination: WandbDestination | None = None,
    wandb_stamp: tuple[int | None, str | None] = (None, None),
) -> EvidenceBaseline:
    """Record where each value path resolves and the stamp of any file there before launch.

    Only a directory an earlier attempt used can hold a value file this attempt did not
    write, so a fresh directory records no paths. An I/O error that leaves a file's prior
    state uncertain is recorded for the individual reader to handle after the trainer runs.
    A reused directory also records the current time, which the W&B run the attempt resumes
    must log past, and that run's row count and state.

    :param Experiment experiment: Metric, constraints and observations.
    :param Phase phase: Gates of the phase the attempt runs in.
    :param Path trial_dir: The trial directory the trainer writes into.
    :param Path attempt_dir: The attempt's directory, which receives the record.
    :param bool reused: The trial directory already held an interrupted attempt, defaults to False.
    :param WandbDestination | None wandb_destination: Managed tracker location set for the trainer.
    :param tuple[int | None, str | None] wandb_stamp: The resumed W&B run's row count and
        state, defaults to unread.
    :raises OSError: The record could not be written, so no trainer may start.
    :return EvidenceBaseline: The baseline, for the attempt's own extraction.
    """
    files: dict[str, FileStamp | str | None] = {}
    root = trial_dir.resolve()
    for relative in sorted(_value_paths(experiment, phase)) if reused else ():
        try:
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                continue
            resolved = path.relative_to(root).as_posix()
            files[relative] = resolved
            info = path.stat()
            if stat.S_ISREG(info.st_mode):
                files[relative] = FileStamp(
                    resolved, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink
                )
        except (OSError, RuntimeError) as exc:
            # Python 3.11 and 3.12 raise RuntimeError when resolving a link loop. A missing
            # target or a loop names no prior file; the read after the trainer exits names
            # any link that is still unusable.
            if isinstance(exc, OSError) and exc.errno not in {
                errno.ENOENT,
                errno.ENOTDIR,
                errno.ELOOP,
            }:
                files[relative] = None
    launched_at = time.time() if reused else None
    payload = {
        "schema_version": EVIDENCE_BASELINE_SCHEMA_VERSION,
        "reused": reused,
        "files": {
            relative: before._asdict() if isinstance(before, FileStamp) else before
            for relative, before in files.items()
        },
        "launched_at": launched_at,
        "wandb_destination": None if wandb_destination is None else wandb_destination._asdict(),
        "wandb_rows": wandb_stamp[0],
        "wandb_state": wandb_stamp[1],
    }
    atomic_write_text(attempt_dir / EVIDENCE_BASELINE_FILE, json.dumps(payload) + "\n")
    return EvidenceBaseline(reused, files, launched_at, wandb_destination, *wandb_stamp)


def read_evidence_baseline(attempt_dir: Path) -> EvidenceBaseline:
    """Load an attempt's baseline for recovery's extraction.

    :param Path attempt_dir: The attempt's directory.
    :raises ValueError: The record is missing, unreadable, malformed or in another format;
        without it no value file can be shown to be the attempt's own.
    :return EvidenceBaseline: The baseline.
    """
    path = attempt_dir / EVIDENCE_BASELINE_FILE
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Evidence baseline {path} is missing or unreadable: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != EVIDENCE_BASELINE_SCHEMA_VERSION
    ):
        raise ValueError(f"Evidence baseline {path} is malformed or in another format.")
    reused, files = payload.get("reused"), payload.get("files")
    launched_at = payload.get("launched_at")
    rows, state = payload.get("wandb_rows"), payload.get("wandb_state")
    if (
        type(reused) is not bool
        or not isinstance(files, dict)
        or "wandb_destination" not in payload
        or not {"wandb_rows", "wandb_state"} <= payload.keys()
        or not (rows is None or (type(rows) is int and rows >= 0))
        or not (state is None or isinstance(state, str))
        or (launched_at is None) == reused
        or not (launched_at is None or (type(launched_at) is float and launched_at > 0))
    ):
        raise ValueError(f"Evidence baseline {path} is malformed or in another format.")
    baseline: dict[str, FileStamp | str | None] = {}
    for relative, fields in files.items():
        if fields is None or isinstance(fields, str):
            baseline[relative] = fields
            continue
        if not isinstance(fields, dict) or set(fields) != set(FileStamp._fields):
            raise ValueError(f"Evidence baseline {path} records {relative!r} malformed.")
        resolved, size, mtime_ns = fields["path"], fields["size"], fields["mtime_ns"]
        ctime_ns, nlink = fields["ctime_ns"], fields["nlink"]
        if (
            not isinstance(resolved, str)
            or type(size) is not int
            or size < 0
            or type(mtime_ns) is not int
            or type(ctime_ns) is not int
            or type(nlink) is not int
            or nlink < 1
        ):
            raise ValueError(f"Evidence baseline {path} records {relative!r} malformed.")
        baseline[relative] = FileStamp(resolved, size, mtime_ns, ctime_ns, nlink)
    destination = None
    raw_destination = payload["wandb_destination"]
    if raw_destination is not None:
        try:
            fields = json_object(raw_destination)
            destination = WandbDestination(
                json_text(fields, "base_url"),
                json_text(fields, "entity"),
                json_text(fields, "project"),
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Evidence baseline {path} records a malformed W&B destination."
            ) from exc
    return EvidenceBaseline(
        reused,
        baseline,
        launched_at,
        destination,
        rows if type(rows) is int else None,
        state if isinstance(state, str) else None,
    )
