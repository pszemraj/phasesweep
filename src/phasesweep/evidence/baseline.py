"""Identify value files and their contents before an attempt can change them."""

from __future__ import annotations

import errno
import json
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, assert_never

from phasesweep.config.common import is_sha256_hex
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
    is_captured_stream,
)
from phasesweep.runtime.files import atomic_write_text, file_sha256
from phasesweep.runtime.json import strict_json_loads

if TYPE_CHECKING:
    from os import stat_result

    from phasesweep.config.models import Experiment, Phase

EVIDENCE_BASELINE_SCHEMA_VERSION = 2
EVIDENCE_BASELINE_FILE = "evidence_baseline.json"


class FileBaseline(NamedTuple):
    """Identity and contents of a regular file present before launch."""

    device: int
    inode: int
    size: int
    sha256: str

    def same_file(self, info: stat_result) -> bool:
        """Check whether a path still names the same file.

        :param stat_result info: Current file identity.
        :return bool: Whether the device and inode still match.
        """
        return (self.device, self.inode) == (info.st_dev, info.st_ino)


#: Relative paths of old value files, or None where the prior contents could not be read.
EvidenceBaseline = Mapping[str, FileBaseline | None]


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
    experiment: Experiment, phase: Phase, trial_dir: Path, attempt_dir: Path
) -> EvidenceBaseline:
    """Record each existing value file's identity and contents before launch.

    Paths that do not name a regular file in the trial directory cannot supply a value.
    An I/O error that leaves prior contents uncertain is recorded for the individual
    reader to handle after the trainer runs.

    :param Experiment experiment: Metric, constraints and observations.
    :param Phase phase: Gates of the phase the attempt runs in.
    :param Path trial_dir: The trial directory the trainer writes into.
    :param Path attempt_dir: The attempt's directory, which receives the record.
    :raises OSError: The record could not be written, so no trainer may start.
    :return EvidenceBaseline: The baseline, for the attempt's own extraction.
    """
    baseline: dict[str, FileBaseline | None] = {}
    root = trial_dir.resolve()
    for relative in sorted(_value_paths(experiment, phase)):
        try:
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root):
                continue
            info = path.stat()
            if stat.S_ISREG(info.st_mode):
                baseline[relative] = FileBaseline(
                    info.st_dev, info.st_ino, info.st_size, file_sha256(path)
                )
        except (FileNotFoundError, NotADirectoryError, RuntimeError):
            # A missing target or a symlink loop names no file. Python 3.11 and 3.12
            # raise RuntimeError for symlink loops, while newer versions use ELOOP.
            continue
        except OSError as exc:
            if exc.errno != errno.ELOOP:
                baseline[relative] = None
    payload = {
        "schema_version": EVIDENCE_BASELINE_SCHEMA_VERSION,
        "files": {
            relative: None if before is None else list(before)
            for relative, before in baseline.items()
        },
    }
    atomic_write_text(attempt_dir / EVIDENCE_BASELINE_FILE, json.dumps(payload) + "\n")
    return baseline


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
    files = payload.get("files") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != EVIDENCE_BASELINE_SCHEMA_VERSION
        or not isinstance(files, dict)
    ):
        raise ValueError(f"Evidence baseline {path} is malformed or in another format.")
    baseline: dict[str, FileBaseline | None] = {}
    for relative, fields in files.items():
        if fields is None:
            baseline[relative] = None
            continue
        if not isinstance(fields, list) or len(fields) != 4:
            raise ValueError(f"Evidence baseline {path} records {relative!r} malformed.")
        device, inode, size, digest = fields
        if (
            type(device) is not int
            or type(inode) is not int
            or type(size) is not int
            or size < 0
            or not is_sha256_hex(digest)
        ):
            raise ValueError(f"Evidence baseline {path} records {relative!r} malformed.")
        baseline[relative] = FileBaseline(device, inode, size, digest)
    return baseline
