"""Publication manifests and immutable summary validation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import yaml

import phasesweep.engine.paths as path_ops
from phasesweep.config.common import SAFE_NAME_PATTERN, is_sha256_hex
from phasesweep.config.models import Experiment, Metric, _metric_semantics_payload
from phasesweep.config.search import Repeats
from phasesweep.engine.confound import _validate_confound_block
from phasesweep.engine.errors import PublicationAccessError, PublicationIntegrityError
from phasesweep.engine.repeats import RepeatsRecord, _check_repeats_axis, _validate_repeats_block
from phasesweep.engine.state import (
    _GENERATION_FILE_FILENAMES,
    _MANIFEST_GENERATION_FILE_KINDS,
    GENERATION_CONFIG_SNAPSHOT_FILENAME,
    GENERATION_SUMMARY_SCHEMA_VERSION,
    PUBLICATION_POINTER_SCHEMA_VERSION,
    WINNER_FILENAME,
    WinnerSource,
    _parse_winner_source,
    _validate_observed_block,
)
from phasesweep.runtime.files import file_sha256, nofollow_flag
from phasesweep.runtime.json import JsonObject, json_object


class PlannedPhase(NamedTuple):
    """One entry of a generation's recorded phase plan."""

    name: str
    comment: str | None


class SnapshotPhases(NamedTuple):
    """Phase layout and argparse base from a generation's validated config snapshot."""

    inherits: Mapping[str, tuple[str, ...]]
    """Each phase's parents, in plan order."""
    repeats: Mapping[str, Repeats | None]
    """Each phase's configured ``repeats``, or ``None`` for a phase without them."""
    argparse_trainer_config: JsonObject | None
    """Producing argparse base, or ``None`` when the snapshot uses another format."""


@dataclass(frozen=True)
class ValidatedWinner:
    """One published winner artifact exactly as manifest validation read it."""

    content: bytes
    """The exact bytes whose digest matched the manifest."""
    payload: Mapping[str, Any]
    """The mapping decoded from ``content``."""
    metric: float
    source: WinnerSource
    confound: Mapping[str, object]
    observed: dict[str, float]
    repeats: RepeatsRecord | None


@dataclass(frozen=True)
class ValidatedPublication:
    """One manifest validation's captured summary, phase plan, metric, and winner bytes.

    Consumers read or resume from this capture without reopening published artifacts.
    """

    generation_id: str
    summary: Mapping[str, Any]
    """The validated generation summary."""
    phase_plan: tuple[PlannedPhase, ...]
    """The recorded phase plan, whose names are safe path components."""
    inherits: Mapping[str, tuple[str, ...]]
    """Each planned phase's parents, from the validated config snapshot."""
    metric: Mapping[str, Any]
    """The summary's metric semantics, checked against the config snapshot."""
    winners: Mapping[str, ValidatedWinner]
    """Every manifest-listed winner, keyed by the phase that exposes it."""
    argparse_trainer_config: JsonObject | None
    """Producing argparse base captured from the validated snapshot bytes."""


def _parse_phase_plan(raw: object) -> tuple[PlannedPhase, ...] | None:
    """Parse a complete recorded phase plan with safe names and optional string comments.

    Any malformed entry invalidates the whole plan. Publication validation refuses
    it; permissive unpublished reads fall back to the current config.

    :param object raw: The summary's ``phase_plan`` value.
    :return tuple[PlannedPhase, ...] | None: Recorded phases in execution
        order, or ``None`` when the plan is absent, empty, or malformed.
    """
    if not isinstance(raw, list) or not raw:
        return None
    plan: list[PlannedPhase] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return None
        name = item.get("name")
        comment = item.get("comment")
        if not isinstance(name, str) or not SAFE_NAME_PATTERN.fullmatch(name):
            return None
        if comment is not None and not isinstance(comment, str):
            return None
        plan.append(PlannedPhase(name, comment))
    return tuple(plan)


def _read_unlinked_bytes(path: Path, *, root: Path) -> bytes:
    """Read a file only when no path component below ``root`` is a symlink.

    :param Path path: File to read.
    :param Path root: Artifact-tree directory that must lexically contain ``path``.
    :raises OSError: If ``path`` escapes ``root``, any traversed component is a
        symlink, or the file cannot be read.
    :return bytes: Exact file contents.
    """
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise OSError(f"artifact {path} is outside {root}") from exc
    if not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise OSError(f"artifact {path} is not a file below {root}")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | nofollow_flag()
    directory_fd = os.open(root, directory_flags)
    file_fd = -1
    try:
        for part in relative.parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            relative.parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | nofollow_flag(),
            dir_fd=directory_fd,
        )
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise OSError(f"artifact {path} is not a regular file")
        with os.fdopen(file_fd, "rb", closefd=False) as artifact:
            return artifact.read()
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(directory_fd)


def _generation_artifact_manifest(
    experiment: Experiment, generation_id: str
) -> list[dict[str, str]]:
    """List and hash result artifacts present in an exclusively claimed generation.

    Scan the namespace itself. Mandatory claim-time provenance files precede the
    phase-ordered winner entries.

    :param Experiment experiment: Experiment whose generation is summarized.
    :param str generation_id: Immutable generation namespace to scan.
    :return list[dict[str, str]]: One ``{"kind", "path", "sha256"}`` entry per
        namespace-root provenance file, then one ``{"kind", "phase",
        "sha256"}`` entry per winner artifact ordered by phase.
    """
    generation_dir = path_ops._generation_dir(experiment, generation_id)
    items: list[dict[str, str]] = []
    for kind, filename in _GENERATION_FILE_FILENAMES.items():
        artifact = generation_dir / filename
        if artifact.is_file():
            items.append({"kind": kind, "path": filename, "sha256": file_sha256(artifact)})
    phases_dir = generation_dir / "phases"
    if not phases_dir.is_dir():
        return items
    for phase_dir in sorted(phases_dir.iterdir()):
        if not phase_dir.is_dir():
            continue
        artifact = phase_dir / WINNER_FILENAME
        if artifact.is_file():
            items.append(
                {"kind": "winner", "phase": phase_dir.name, "sha256": file_sha256(artifact)}
            )
    return items


def _unreadable_artifact_permission_detail(subject: str) -> str:
    """Describe an unreadable artifact by role, directing validation to its publishing user.

    :param str subject: Artifact that could not be read, named by role
        (e.g. ``"config_snapshot artifact"``), never by path.
    :return str: Bare reason clause for the caller's manifest-validation error.
    """
    return (
        f"{subject} is not readable as this user (permission denied): validation cannot run "
        "without it, and a generation's config.snapshot.yaml is deliberately owner-only, so "
        "only the publishing user can fully validate this tree -- re-read it as that user "
        "before treating this publication as corrupt"
    )


def _validate_generation_manifest(
    generation_dir: Path,
    generation_id: str,
    summary: Mapping[str, Any],
) -> ValidatedPublication:
    """Validate a generation's embedded results against its summary manifest.

    Run before pointer commit and on authoritative reads. Every listed artifact must
    exist, match its hash, parse, and agree with the summary; unlisted artifacts are
    refused. The manifest includes phase winners and both claim-time provenance files.

    :param Path generation_dir: The generation's immutable namespace directory.
    :param str generation_id: Generation id the summary must belong to;
        ownership is checked by the caller.
    :param Mapping[str, Any] summary: Parsed generation summary payload.
    :raises PublicationAccessError: An artifact cannot be read by the current user.
    :raises PublicationIntegrityError: The manifest is missing, malformed, any artifact is
        absent, altered, unparsable, or inconsistent with the summary, or a
        winner's ``repeats`` block differs from its config snapshot's repeat axis.
    :return ValidatedPublication: The summary, its phase plan and metric
        semantics, and every winner exactly as this validation read it.
    """

    def _fail(reason: str) -> PublicationIntegrityError:
        """Build one uniformly labeled manifest-validation error.

        :param str reason: Specific validation failure being reported.
        :return PublicationIntegrityError: Error naming the generation and reason.
        """
        return PublicationIntegrityError(
            f"Generation {generation_id!r} manifest validation failed: {reason}"
        )

    def _permission_fail(reason: str) -> PublicationAccessError:
        """Build a permission-specific manifest-validation error.

        :param str reason: Permission failure being reported.
        :return PublicationAccessError: Error naming the generation and reason.
        """
        return PublicationAccessError(
            f"Generation {generation_id!r} manifest validation could not run: {reason}"
        )

    if summary.get("schema_version") != GENERATION_SUMMARY_SCHEMA_VERSION:
        raise _fail(f"unsupported summary schema_version {summary.get('schema_version')!r}")
    metric = summary.get("metric")
    if (
        not isinstance(metric, Mapping)
        or not isinstance(metric.get("name"), str)
        or metric.get("goal") not in ("minimize", "maximize")
    ):
        raise _fail("summary metric block is malformed")

    raw_artifacts = summary.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise _fail("summary has no artifact manifest")
    listed: dict[str, Mapping[str, object]] = {}
    listed_files: dict[str, Mapping[str, Any]] = {}
    for entry in raw_artifacts:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("sha256"), str):
            raise _fail("summary artifact entry is malformed")
        kind = entry.get("kind")
        if kind in _MANIFEST_GENERATION_FILE_KINDS:
            if entry.get("path") != _GENERATION_FILE_FILENAMES[str(kind)]:
                raise _fail(f"{kind} manifest entry names an unexpected path")
            if str(kind) in listed_files:
                raise _fail(f"duplicate artifact entry for {kind}")
            listed_files[str(kind)] = entry
            continue
        if kind != "winner" or not isinstance(entry.get("phase"), str):
            raise _fail("summary artifact entry is malformed")
        name = str(entry["phase"])
        if name in listed:
            raise _fail(f"duplicate artifact entry for {('winner', name)}")
        listed[name] = entry
    snapshot_phases = _validate_generation_provenance_files(
        generation_dir,
        summary,
        listed_files,
        _fail,
        _permission_fail,
    )
    phase_plan = _parse_phase_plan(summary.get("phase_plan"))
    if phase_plan is None:
        raise _fail("summary phase plan is malformed")

    raw_phases = summary.get("phases")
    if not isinstance(raw_phases, list):
        raise _fail("summary has no phase winner list")
    phase_items: dict[str, Mapping[str, Any]] = {}
    for item in raw_phases:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise _fail("summary phase winner entry is malformed")
        name = str(item["name"])
        if name in phase_items:
            raise _fail(f"duplicate phase winner entry {name!r}")
        phase_items[name] = item
        if name not in listed:
            raise _fail(f"phase {name!r} winner is not listed in the artifact manifest")

    for name in listed:
        if name not in phase_items:
            raise _fail(f"artifact manifest lists a winner for unknown phase {name!r}")

    winners: dict[str, ValidatedWinner] = {}
    for name, entry in listed.items():
        artifact_path = generation_dir / "phases" / name / WINNER_FILENAME
        try:
            content = _read_unlinked_bytes(artifact_path, root=generation_dir.parent)
        except PermissionError as exc:
            raise _permission_fail(
                _unreadable_artifact_permission_detail(f"winner artifact for phase {name!r}")
            ) from exc
        except OSError as exc:
            raise _fail(f"winner artifact for phase {name!r} is missing or unreadable") from exc
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise _fail(f"winner artifact for phase {name!r} does not match its recorded hash")
        try:
            payload = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise _fail(f"winner artifact for phase {name!r} is not parseable") from exc
        if not isinstance(payload, Mapping):
            raise _fail(f"winner artifact for phase {name!r} is not a mapping")
        if payload.get("phase") != name:
            raise _fail(f"winner artifact for phase {name!r} names a different phase")
        item = phase_items[name]
        if payload.get("trial_number") != item.get("trial_number"):
            raise _fail(f"winner for phase {name!r} disagrees with the summary trial number")
        metric_block = payload.get("metric")
        if not isinstance(metric_block, Mapping) or metric_block.get("goal") != metric["goal"]:
            raise _fail(f"winner for phase {name!r} has a mismatched metric goal")
        value = metric_block.get(metric["name"])
        if not isinstance(value, (int, float)) or float(value) != item.get("metric"):
            raise _fail(f"winner for phase {name!r} disagrees with the summary metric value")
        for field in (
            "params",
            "effective_overrides",
            "constraints",
            "gates",
            "completion",
            "generation_id",
            "attempt_id",
            "winner_source",
            "trainer_input",
        ):
            if (field in payload) != (field in item) or payload.get(field) != item.get(field):
                raise _fail(f"winner for phase {name!r} disagrees with the summary {field}")
        # Carried winners retain their original execution identities.
        for id_field in ("generation_id", "attempt_id"):
            recorded = payload.get(id_field)
            if (
                not isinstance(recorded, str)
                or not recorded
                or (id_field == "generation_id" and not SAFE_NAME_PATTERN.fullmatch(recorded))
            ):
                raise _fail(f"winner for phase {name!r} has no valid {id_field}")
        fingerprint = payload.get("phase_fingerprint")
        if not is_sha256_hex(fingerprint):
            raise _fail(f"winner for phase {name!r} has no valid phase_fingerprint")
        try:
            parsed_source = _parse_winner_source(payload.get("winner_source"), expected_phase=name)
        except ValueError as exc:
            raise _fail(f"winner for phase {name!r} has no valid winner_source ({exc})") from exc
        for id_field, parsed_value in (
            ("trial_number", parsed_source.trial_number),
            ("generation_id", parsed_source.generation_id),
            ("attempt_id", parsed_source.attempt_id),
        ):
            if parsed_value != payload.get(id_field):
                raise _fail(
                    f"winner for phase {name!r} has a winner_source "
                    f"that disagrees with its {id_field}"
                )
        if not isinstance(payload.get("completion"), Mapping):
            raise _fail(f"winner for phase {name!r} has no completion metadata")
        try:
            confound = _validate_confound_block(
                payload.get("confound"), winner_trial=parsed_source.trial_number
            )
        except ValueError as exc:
            raise _fail(f"winner for phase {name!r} has no valid confound block ({exc})") from exc
        try:
            observed = _validate_observed_block(payload.get("observed"))
        except ValueError as exc:
            raise _fail(f"winner for phase {name!r} has no valid observed block ({exc})") from exc
        try:
            repeats = _validate_repeats_block(
                payload, metric_value=float(value), ranked_groups=confound["ranked_groups"]
            )
            if name not in snapshot_phases.repeats:
                raise ValueError("the config snapshot plans no such phase")
            _check_repeats_axis(repeats, snapshot_phases.repeats[name])
        except ValueError as exc:
            raise _fail(f"winner for phase {name!r} has no valid repeats block ({exc})") from exc
        winners[name] = ValidatedWinner(
            content=content,
            payload=payload,
            metric=float(value),
            source=parsed_source,
            confound=confound,
            observed=observed,
            repeats=repeats,
        )
    phases_dir = generation_dir / "phases"
    if phases_dir.is_dir():
        for phase_dir in phases_dir.iterdir():
            if not phase_dir.is_dir():
                continue
            if (phase_dir / WINNER_FILENAME).is_file() and phase_dir.name not in listed:
                raise _fail(
                    f"namespace contains an unlisted winner artifact for phase {phase_dir.name!r}"
                )
    return ValidatedPublication(
        generation_id=generation_id,
        summary=summary,
        phase_plan=phase_plan,
        inherits=snapshot_phases.inherits,
        metric=metric,
        winners=winners,
        argparse_trainer_config=snapshot_phases.argparse_trainer_config,
    )


def _validate_generation_provenance_files(
    generation_dir: Path,
    summary: Mapping[str, Any],
    listed_files: Mapping[str, Mapping[str, Any]],
    fail: Callable[[str], PublicationIntegrityError],
    permission_fail: Callable[[str], PublicationAccessError],
) -> SnapshotPhases:
    """Validate claim-time configuration and reproducibility files against the manifest.

    Both files are mandatory and must match their hashes and the summary's identity.
    The reproducibility record also anchors the snapshot bytes. Permission denial
    is reported separately from corruption.

    :param Path generation_dir: The generation's immutable namespace directory.
    :param Mapping[str, Any] summary: Parsed generation summary payload, whose
        identity the reproducibility record must agree with.
    :param Mapping[str, Mapping[str, Any]] listed_files: Manifest entries for
        the namespace-root provenance files, keyed by kind.
    :param Callable[[str], PublicationIntegrityError] fail: Builder for the caller's
        uniformly labeled manifest-validation error.
    :param Callable[[str], PublicationAccessError] permission_fail: Builder for
        a permission-specific validation error.
    :raises PublicationAccessError: A provenance file cannot be read by the current user.
    :raises PublicationIntegrityError: Whatever ``fail`` builds, when a provenance file is
        listed without its partner, is absent, unreadable, altered,
        unparsable, or disagrees with the summary's own identity.
    :return SnapshotPhases: Each snapshot phase's parents, in plan order, its
        configured ``repeats``, and the producing argparse base when applicable.
    """
    for kind, filename in _GENERATION_FILE_FILENAMES.items():
        if (generation_dir / filename).is_file() and kind not in listed_files:
            raise fail(f"namespace contains an unlisted {kind} artifact")
    if set(listed_files) != _MANIFEST_GENERATION_FILE_KINDS:
        raise fail("summary does not list the complete generation provenance record")

    contents: dict[str, bytes] = {}
    digests: dict[str, str] = {}
    for kind, filename in _GENERATION_FILE_FILENAMES.items():
        try:
            content = _read_unlinked_bytes(
                generation_dir / filename,
                root=generation_dir.parent,
            )
        except PermissionError as exc:
            raise permission_fail(
                _unreadable_artifact_permission_detail(f"{kind} artifact")
            ) from exc
        except OSError as exc:
            raise fail(f"{kind} artifact is missing or unreadable") from exc
        digest = hashlib.sha256(content).hexdigest()
        if digest != listed_files[kind]["sha256"]:
            raise fail(f"{kind} artifact does not match its recorded hash")
        contents[kind] = content
        digests[kind] = digest

    try:
        snapshot = yaml.safe_load(contents["config_snapshot"])
    except yaml.YAMLError as exc:
        raise fail("config_snapshot artifact is not parseable") from exc
    if not isinstance(snapshot, Mapping):
        raise fail("config_snapshot artifact is not a mapping")
    if snapshot.get("experiment") != summary.get("experiment"):
        raise fail("config_snapshot artifact names a different experiment")

    try:
        record = json.loads(contents["reproducibility"])
    except ValueError as exc:
        raise fail("reproducibility artifact is not parseable") from exc
    if not isinstance(record, Mapping):
        raise fail("reproducibility artifact is not a mapping")
    recorded_identity = (record.get("experiment"), record.get("generation_id"))
    if recorded_identity != (summary.get("experiment"), summary.get("generation_id")):
        raise fail("reproducibility artifact names a different generation")
    recorded_snapshot = record.get("config_snapshot")
    if (
        not isinstance(recorded_snapshot, Mapping)
        or recorded_snapshot.get("path") != GENERATION_CONFIG_SNAPSHOT_FILENAME
        or recorded_snapshot.get("sha256") != digests["config_snapshot"]
    ):
        raise fail("reproducibility artifact does not anchor the config snapshot it published")

    if summary.get("config_fingerprint") != record.get("config_fingerprint"):
        raise fail("summary config fingerprint disagrees with the reproducibility artifact")

    snapshot_phases = snapshot.get("phases")
    if not isinstance(snapshot_phases, list) or any(
        not isinstance(phase, Mapping) or not isinstance(phase.get("name"), str)
        for phase in snapshot_phases
    ):
        raise fail("config_snapshot artifact has a malformed phase plan")
    expected_phase_plan = [
        {"name": phase["name"], "comment": phase.get("comment")} for phase in snapshot_phases
    ]
    if summary.get("phase_plan") != expected_phase_plan:
        raise fail("summary phase plan disagrees with the config_snapshot artifact")

    try:
        snapshot_metric = Metric.model_validate(snapshot.get("metric"))
    except ValueError as exc:
        raise fail("config_snapshot artifact has malformed metric semantics") from exc
    if summary.get("metric") != _metric_semantics_payload(snapshot_metric):
        raise fail("summary metric semantics disagree with the config_snapshot artifact")

    # Rebuilding a generation's lineage reads this graph, never the current
    # config's, so every parent must name a phase planned before its child.
    inherits: dict[str, tuple[str, ...]] = {}
    repeats: dict[str, Repeats | None] = {}
    for phase in snapshot_phases:
        parents = phase.get("inherits")
        if not isinstance(parents, list) or any(
            not isinstance(parent, str) or parent not in inherits for parent in parents
        ):
            raise fail("config_snapshot artifact has a malformed inheritance graph")
        inherits[phase["name"]] = tuple(parents)
        try:
            raw_repeats = phase.get("repeats")
            repeats[phase["name"]] = (
                None if raw_repeats is None else Repeats.model_validate(raw_repeats)
            )
        except ValueError as exc:
            raise fail(
                f"config_snapshot artifact has a malformed repeats block for {phase['name']!r}"
            ) from exc
    argparse_trainer_config: JsonObject | None = None
    if snapshot.get("override_format") == "argparse":
        try:
            argparse_trainer_config = json_object(snapshot.get("trainer_config"))
        except ValueError as exc:
            raise fail("config_snapshot artifact has a malformed trainer_config") from exc
    return SnapshotPhases(
        inherits=inherits,
        repeats=repeats,
        argparse_trainer_config=argparse_trainer_config,
    )


@dataclass(frozen=True)
class _PointerTarget:
    """Validated summary identity carried by one last-success pointer."""

    generation_id: str
    summary_size_bytes: int
    summary_sha256: str


def _read_pointer_target(pointer_path: Path, *, experiment_name: str) -> _PointerTarget | None:
    """Read one last-success pointer's target id, validating the pointer itself.

    :param Path pointer_path: Pointer YAML file to read.
    :param str experiment_name: Experiment the pointer must name.
    :return _PointerTarget | None: A safe-name target and exact summary-byte identity, or
        ``None`` when the pointer is missing, unreadable, malformed, names
        another experiment, or carries an unsafe id.
    :raises PublicationAccessError: The pointer cannot be read by the current user.
    """
    try:
        payload = yaml.safe_load(
            _read_unlinked_bytes(pointer_path, root=pointer_path.parent).decode("utf-8")
        )
    except PermissionError as exc:
        raise PublicationAccessError(
            _unreadable_artifact_permission_detail("last-success pointer")
        ) from exc
    except (OSError, ValueError, yaml.YAMLError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PUBLICATION_POINTER_SCHEMA_VERSION
        or payload.get("experiment") != experiment_name
    ):
        return None
    target_id = payload.get("generation_id")
    if not isinstance(target_id, str) or not SAFE_NAME_PATTERN.fullmatch(target_id):
        return None
    summary_size_bytes = payload.get("summary_size_bytes")
    summary_sha256 = payload.get("summary_sha256")
    if (
        type(summary_size_bytes) is not int
        or summary_size_bytes < 0
        or not is_sha256_hex(summary_sha256)
    ):
        return None
    return _PointerTarget(
        generation_id=target_id,
        summary_size_bytes=summary_size_bytes,
        summary_sha256=summary_sha256,
    )


def _read_pointer_target_summary(
    summary_path: Path,
    *,
    experiment_name: str,
    target: _PointerTarget,
) -> dict[str, Any] | None:
    """Read and identify a last-success pointer's immutable summary bytes.

    Require the pointer's byte anchors and exact experiment/generation identity.
    The lifecycle record is written after commit and is not publication authority.
    Callers validate the summary's complete manifest next.

    :param Path summary_path: Immutable generation summary YAML file to read.
    :param str experiment_name: Experiment the summary must name.
    :param _PointerTarget target: Pointer-recorded generation id and exact
        summary byte length and SHA-256.
    :return dict[str, Any] | None: The parsed summary when readable and
        correctly named; ``None`` otherwise.
    :raises PublicationAccessError: The summary cannot be read by the current user.
    :raises PublicationIntegrityError: The summary bytes disagree with either pointer anchor.
    """
    try:
        content = _read_unlinked_bytes(summary_path, root=summary_path.parent.parent)
    except PermissionError as exc:
        raise PublicationAccessError(
            _unreadable_artifact_permission_detail("generation summary")
        ) from exc
    except OSError:
        return None
    if len(content) != target.summary_size_bytes:
        raise PublicationIntegrityError(
            "Generation summary byte length does not match the last-success pointer."
        )
    if hashlib.sha256(content).hexdigest() != target.summary_sha256:
        raise PublicationIntegrityError(
            "Generation summary digest does not match the last-success pointer."
        )
    try:
        summary = yaml.safe_load(content)
    except yaml.YAMLError:
        return None
    if (
        isinstance(summary, dict)
        and summary.get("experiment") == experiment_name
        and summary.get("generation_id") == target.generation_id
    ):
        return summary
    return None
