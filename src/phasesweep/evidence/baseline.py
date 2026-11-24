"""Identify value files and their contents before an attempt can change them."""

from __future__ import annotations

import ctypes
import errno
import json
import logging
import os
import stat
import sys
import time
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
from phasesweep.runtime.files import (
    atomic_write_text,
    broken_symlink_detail,
    file_sha256,
    fsync_directory,
)
from phasesweep.runtime.json import strict_json_loads

if TYPE_CHECKING:
    from os import stat_result

    from phasesweep.config.models import Experiment, Phase

EVIDENCE_BASELINE_SCHEMA_VERSION = 3
EVIDENCE_BASELINE_FILE = "evidence_baseline.json"
EVIDENCE_PINS_DIR = "evidence_pins"

_AT_FDCWD = -100
_STATX_BTIME = 0x0800
_NS_PER_SECOND = 1_000_000_000


class _LinuxStatx(ctypes.Structure):
    """Linux's 256-byte statx buffer with the creation timestamp at offset 0x50."""

    mask: int
    birth_seconds: int
    birth_nanoseconds: int
    _fields_ = [
        ("mask", ctypes.c_uint32),
        ("before_birth", ctypes.c_byte * 76),
        ("birth_seconds", ctypes.c_int64),
        ("birth_nanoseconds", ctypes.c_uint32),
        ("after_birth", ctypes.c_byte * 164),
    ]


def file_birth_ns(path: Path, info: stat_result) -> int | None:
    """Read a file's creation time where the filesystem supplies it.

    :param Path path: File whose identity is being compared.
    :param stat_result info: Current file metadata from the same path.
    :return int | None: Creation time in nanoseconds since the epoch.
    """
    native_ns: object = getattr(info, "st_birthtime_ns", None)
    if isinstance(native_ns, int):
        return native_ns
    native_seconds: object = getattr(info, "st_birthtime", None)
    if isinstance(native_seconds, float):
        return int(native_seconds * _NS_PER_SECOND)
    if sys.platform != "linux":
        return None
    try:
        query = ctypes.CDLL(None).statx
    except AttributeError:
        return None
    result = _LinuxStatx()
    if query(_AT_FDCWD, os.fsencode(path), 0, _STATX_BTIME, ctypes.byref(result)) != 0:
        return None
    if not result.mask & _STATX_BTIME:
        return None
    return result.birth_seconds * _NS_PER_SECOND + result.birth_nanoseconds


class FileBaseline(NamedTuple):
    """Identity and contents of a regular file present before launch."""

    path: str
    device: int
    inode: int
    pin: int | None
    birth_ns: int | None
    size: int
    sha256: str

    def same_file(self, info: stat_result) -> bool:
        """Check whether a path still names the same file.

        :param stat_result info: Current file identity.
        :return bool: Whether the device and inode still match.
        """
        return (self.device, self.inode) == (info.st_dev, info.st_ino)


class EvidenceBaseline(NamedTuple):
    """The value paths of a trial directory before one attempt starts."""

    captured_at_ns: int
    reused: bool
    # A path string records a resolved target with no regular file before launch. None
    # records a target whose prior contents could not be inspected.
    files: Mapping[str, FileBaseline | str | None]


def release_evidence_pins(attempt_dir: Path) -> None:
    """Release a settled attempt's links to previously measured files.

    :param Path attempt_dir: The attempt directory that owns the temporary links.
    :raises OSError: The existing links could not be removed.
    """
    root = attempt_dir / EVIDENCE_PINS_DIR
    try:
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except (FileNotFoundError, NotADirectoryError):
        return
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return
        raise
    try:
        for name in os.listdir(directory_fd):
            if name.isascii() and name.isdecimal():
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISREG(info.st_mode):
                    os.unlink(name, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        root.rmdir()
    except OSError as exc:
        if exc.errno in {errno.ENOTEMPTY, errno.EEXIST}:
            fsync_directory(root)
        elif exc.errno not in {errno.ENOENT, errno.ENOTDIR}:
            raise
    fsync_directory(attempt_dir)


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
) -> EvidenceBaseline:
    """Record each existing value file's identity and contents before launch.

    Paths that do not name a regular file in the trial directory cannot supply a value.
    An I/O error that leaves prior contents uncertain is recorded for the individual
    reader to handle after the trainer runs.

    :param Experiment experiment: Metric, constraints and observations.
    :param Phase phase: Gates of the phase the attempt runs in.
    :param Path trial_dir: The trial directory the trainer writes into.
    :param Path attempt_dir: The attempt's directory, which receives the record.
    :param bool reused: The trial directory already held an interrupted attempt, defaults to False.
    :raises OSError: The record could not be written, so no trainer may start.
    :return EvidenceBaseline: The baseline, for the attempt's own extraction.
    """
    files: dict[str, FileBaseline | str | None] = {}
    snapshots: dict[tuple[int, int], FileBaseline] = {}
    pins_dir = attempt_dir / EVIDENCE_PINS_DIR
    root = trial_dir.resolve()
    for relative in sorted(_value_paths(experiment, phase)):
        try:
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                continue
            resolved = path.relative_to(root).as_posix()
            files[relative] = resolved
            info = path.stat()
            if stat.S_ISREG(info.st_mode):
                identity = (info.st_dev, info.st_ino)
                if (prior := snapshots.get(identity)) is not None:
                    files[relative] = prior._replace(path=resolved)
                    continue
                digest = file_sha256(path)
                pin: int | None = None
                birth_ns: int | None = None
                if reused:
                    try:
                        pins_dir.mkdir(exist_ok=True)
                        (pins_dir / str(len(snapshots))).hardlink_to(path)
                    except OSError:
                        birth_ns = file_birth_ns(path, info)
                    else:
                        pin = len(snapshots)
                before = FileBaseline(
                    resolved,
                    info.st_dev,
                    info.st_ino,
                    pin,
                    birth_ns,
                    info.st_size,
                    digest,
                )
                snapshots[identity] = before
                files[relative] = before
        except (OSError, RuntimeError) as exc:
            # Python 3.11 and 3.12 raise RuntimeError when resolving a link loop.
            if isinstance(exc, OSError) and exc.errno not in {
                errno.ENOENT,
                errno.ENOTDIR,
                errno.ELOOP,
            }:
                files[relative] = None
            elif reused and (detail := broken_symlink_detail(root, relative)):
                logging.getLogger(__name__).warning(
                    "Rerunning in %s. Could not read %s before launch because %s. "
                    "The trainer can still use other checkpoints in this directory.",
                    root,
                    relative,
                    detail,
                )
    if pins_dir.is_dir():
        fsync_directory(pins_dir)
    baseline = EvidenceBaseline(time.time_ns(), reused, files)
    payload = {
        "schema_version": EVIDENCE_BASELINE_SCHEMA_VERSION,
        "captured_at_ns": baseline.captured_at_ns,
        "reused": baseline.reused,
        "files": {
            relative: list(before) if isinstance(before, FileBaseline) else before
            for relative, before in files.items()
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
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != EVIDENCE_BASELINE_SCHEMA_VERSION
    ):
        raise ValueError(f"Evidence baseline {path} is malformed or in another format.")
    captured_at_ns, reused, files = (
        payload.get("captured_at_ns"),
        payload.get("reused"),
        payload.get("files"),
    )
    if type(captured_at_ns) is not int or type(reused) is not bool or not isinstance(files, dict):
        raise ValueError(f"Evidence baseline {path} is malformed or in another format.")
    baseline: dict[str, FileBaseline | str | None] = {}
    for relative, fields in files.items():
        if fields is None or isinstance(fields, str):
            baseline[relative] = fields
            continue
        if not isinstance(fields, list) or len(fields) != 7:
            raise ValueError(f"Evidence baseline {path} records {relative!r} malformed.")
        resolved, device, inode, pin, birth_ns, size, digest = fields
        if (
            not isinstance(resolved, str)
            or type(device) is not int
            or type(inode) is not int
            or (pin is not None and (type(pin) is not int or pin < 0))
            or (birth_ns is not None and type(birth_ns) is not int)
            or type(size) is not int
            or size < 0
            or not is_sha256_hex(digest)
        ):
            raise ValueError(f"Evidence baseline {path} records {relative!r} malformed.")
        baseline[relative] = FileBaseline(resolved, device, inode, pin, birth_ns, size, digest)
    return EvidenceBaseline(captured_at_ns, reused, baseline)
