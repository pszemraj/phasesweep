"""What the value files looked like before an attempt ran, so it is scored only from its own writes.

A rerun continues in its trial's directory, where an earlier attempt may have left a result.
Before the trainer starts, PhaseSweep records the identity of every file a value is read from:
device, inode, size, modification and change time. A file whose identity is unchanged after
the attempt was not written by it and fails the read. The check fails closed: a coarse
timestamp can only make a rewritten file look unchanged and refuse it, never accept a stale
one, because a file's change time cannot be set from user space.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from phasesweep.evidence.models import (
    ArtifactSizeGate,
    JsonEqualsGate,
    JsonExtractor,
    JsonScalarBoundGate,
    LogRegexExtractor,
)
from phasesweep.runtime.files import atomic_write_text
from phasesweep.runtime.json import strict_json_loads

if TYPE_CHECKING:
    from phasesweep.config.models import Experiment, Phase

EVIDENCE_BASELINE_SCHEMA_VERSION = 1
EVIDENCE_BASELINE_FILE = "evidence_baseline.json"


class FileStamp(NamedTuple):
    """The identity of one file: replaced, rewritten or touched, at least one field changes."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


#: Relative value path to the identity its file had before the attempt ran.
EvidenceBaseline = Mapping[str, FileStamp]


def _value_paths(experiment: Experiment, phase: Phase) -> Iterator[str]:
    """Yield every trial-relative path a value is read from.

    The ``json_envelope`` extractor is left out: the envelope names the attempt that wrote
    it. ``required_file``, ``sha256`` and file or directory ``artifact_size`` gates test the
    trial's artifacts whichever attempt wrote them, so they are left out too.

    :param Experiment experiment: Metric, constraints and observations.
    :param Phase phase: Gates of the phase the attempt runs in.
    :return Iterator[str]: The paths, possibly repeated.
    """
    from phasesweep.evidence.evaluation import CAPTURED_STREAMS

    readers = [
        experiment.metric.extractor,
        *(constraint.extractor for constraint in experiment.constraints),
        *(observation.extractor for observation in experiment.observe),
    ]
    for reader in readers:
        if isinstance(reader, JsonExtractor):
            yield reader.path
        elif isinstance(reader, LogRegexExtractor) and reader.file not in CAPTURED_STREAMS:
            yield reader.file
    for gate in phase.gates:
        if isinstance(gate, JsonEqualsGate | JsonScalarBoundGate) or (
            isinstance(gate, ArtifactSizeGate) and gate.source == "json"
        ):
            yield gate.path


def file_stamp(path: Path) -> FileStamp | None:
    """Return a file's identity, or ``None`` when nothing is there.

    :param Path path: The file, followed through links as the readers follow them.
    :return FileStamp | None: Its identity, or ``None`` when it does not exist.
    """
    try:
        stat = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    return FileStamp(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def write_evidence_baseline(
    experiment: Experiment, phase: Phase, trial_dir: Path, attempt_dir: Path
) -> dict[str, FileStamp]:
    """Record the identity of each value file present before the trainer starts.

    :param Experiment experiment: Metric, constraints and observations.
    :param Phase phase: Gates of the phase the attempt runs in.
    :param Path trial_dir: The trial directory the trainer writes into.
    :param Path attempt_dir: The attempt's directory, which receives the record.
    :raises OSError: The record could not be written, so no trainer may start.
    :return dict[str, FileStamp]: The baseline, for the attempt's own extraction.
    """
    baseline: dict[str, FileStamp] = {}
    for relative in _value_paths(experiment, phase):
        stamp = file_stamp(trial_dir / relative)
        if stamp is not None:
            baseline[relative] = stamp
    payload = {
        "schema_version": EVIDENCE_BASELINE_SCHEMA_VERSION,
        "files": {relative: list(stamp) for relative, stamp in sorted(baseline.items())},
    }
    atomic_write_text(attempt_dir / EVIDENCE_BASELINE_FILE, json.dumps(payload) + "\n")
    return baseline


def read_evidence_baseline(attempt_dir: Path) -> dict[str, FileStamp]:
    """Load an attempt's baseline for recovery's extraction.

    :param Path attempt_dir: The attempt's directory.
    :raises ValueError: The record is missing, unreadable, malformed or in another format;
        without it no value file can be shown to be the attempt's own.
    :return dict[str, FileStamp]: The baseline.
    """
    path = attempt_dir / EVIDENCE_BASELINE_FILE
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Evidence baseline {path} is missing or unreadable: {exc}") from exc
    files = payload.get("files") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != EVIDENCE_BASELINE_SCHEMA_VERSION
        or not isinstance(files, dict)
    ):
        raise ValueError(f"Evidence baseline {path} is malformed or in another format.")
    baseline: dict[str, FileStamp] = {}
    for relative, fields in files.items():
        numbers = (
            [field for field in fields if type(field) is int] if isinstance(fields, list) else []
        )
        if not isinstance(fields, list) or len(numbers) != len(fields) or len(numbers) != 5:
            raise ValueError(f"Evidence baseline {path} records {relative!r} malformed.")
        baseline[relative] = FileStamp(*numbers)
    return baseline
