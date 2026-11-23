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
    WandbExtractor,
    WandbSummaryRequiredGate,
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


def _extractor_value_path(extractor: ObjectiveExtractor) -> str | None:
    """Return the trial-relative file an extractor reads its value from, if it reads one.

    Every extractor type is named, so a new one is a type error here until it says whether
    it reads a file in the trial directory. ``json_envelope`` is left out: the envelope
    names the attempt that wrote it. A captured stream is the attempt's own already.

    :param ObjectiveExtractor extractor: A metric, constraint or observation extractor.
    :return str | None: The path, or ``None`` when the value comes from no trial file.
    """
    from phasesweep.evidence.evaluation import CAPTURED_STREAMS

    if isinstance(extractor, JsonExtractor):
        return extractor.path
    if isinstance(extractor, LogRegexExtractor):
        return None if extractor.file in CAPTURED_STREAMS else extractor.file
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


def _value_paths(experiment: Experiment, phase: Phase) -> Iterator[str]:
    """Yield every trial-relative path a value is read from.

    :param Experiment experiment: Metric, constraints and observations.
    :param Phase phase: Gates of the phase the attempt runs in.
    :return Iterator[str]: The paths, possibly repeated.
    """
    paths = [
        _extractor_value_path(experiment.metric.extractor),
        *(_extractor_value_path(constraint.extractor) for constraint in experiment.constraints),
        *(_extractor_value_path(observation.extractor) for observation in experiment.observe),
        *(_gate_value_path(gate) for gate in phase.gates),
    ]
    yield from (path for path in paths if path is not None)


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
