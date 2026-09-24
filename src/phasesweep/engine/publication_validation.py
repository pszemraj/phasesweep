"""Publication manifests and immutable summary validation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import phasesweep.engine.paths as path_ops
from phasesweep.config import Experiment, Metric
from phasesweep.config.common import SAFE_NAME_PATTERN, is_sha256_hex
from phasesweep.config.models import _metric_semantics_payload
from phasesweep.engine.errors import PublicationAccessError, PublicationIntegrityError
from phasesweep.engine.paths import GENERATION_SUMMARY_FILENAME
from phasesweep.engine.state import (
    _GENERATION_FILE_FILENAMES,
    _MANIFEST_GENERATION_FILE_KINDS,
    GENERATION_CONFIG_SNAPSHOT_FILENAME,
    GENERATION_SUMMARY_SCHEMA_VERSION,
    PUBLICATION_POINTER_SCHEMA_VERSION,
    WINNER_FILENAME,
)
from phasesweep.runtime.files import file_sha256, nofollow_flag


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
    """List and hash every result artifact in one generation's namespace.

    Scans the namespace itself rather than reconstructing the set from
    in-memory bookkeeping so the manifest cannot omit an artifact the run
    actually wrote. The namespace is exclusively claimed by this invocation, so
    everything present is this run's own output.

    The namespace-root provenance files written at claim time
    (:func:`phasesweep.engine.provenance._write_generation_provenance`) are listed first, under entries
    that carry ``path`` instead of ``phase``. Both are mandatory for every
    current-format generation.

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
    """Build the manifest-failure detail for an artifact this user may not read.

    A permission denial is not corruption, and reporting it as corruption sends
    the operator to inspect or restore a namespace that is perfectly healthy
    (re-review v0.5.19 / observation N1). Every generation's
    ``config.snapshot.yaml`` is owner-only, so a second operator reading a
    sound tree hits exactly this case; the verdict still fails closed -- an
    unvalidatable publication may not be reported as published -- but the
    remedy named is the publishing user, not the restore procedure.

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
) -> None:
    """Validate one generation's complete result graph against its summary manifest.

    "Published" must mean the immutable result is internally complete
    (review v0.5.16 / blocker 3): every artifact the summary claims exists,
    hashes to the recorded content, parses, and cross-checks against the
    summary's own winner facts — and the namespace holds nothing the
    manifest does not list. Runs both pre-commit (before the last-success
    pointer may advance) and on every read before a pointer target is trusted.

    The manifest covers phase-scoped winners (``kind`` + ``phase``) and the
    namespace-root provenance files frozen at claim time (``kind`` + ``path``; see
    :func:`_validate_generation_provenance_files`, review v0.5.18 / finding
    F6). Every current-format summary lists both provenance files, each of
    which must exist and match its recorded hash.

    A winner carried forward from an earlier generation additionally has that
    generation resolved in this same tree
    (:func:`_validate_winner_source_generation`, PR #5 review / reviewer 2,
    blocker 7): the cited namespace holds the evidence behind the number, so a
    publication whose source generation is absent - or which disagrees with the
    winner that source published - is not a result this tree can stand behind.

    :param Path generation_dir: The generation's immutable namespace directory.
    :param str generation_id: Generation id the summary must belong to (used
        for error text and to tell a carried-forward winner from a local one;
        ownership is checked by the caller).
    :param Mapping[str, Any] summary: Parsed generation summary payload.
    :raises PublicationAccessError: An artifact cannot be read by the current user.
    :raises PublicationIntegrityError: The manifest is missing, malformed, any artifact is
        absent, altered, unparsable, or inconsistent with the summary, or a
        winner cites a source generation this tree does not hold.
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
    if "promotion_decisions" in summary:
        raise _fail("summary contains removed promotion decisions")
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
    listed: dict[tuple[str, str], Mapping[str, Any]] = {}
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
        key = (str(kind), str(entry["phase"]))
        if key in listed:
            raise _fail(f"duplicate artifact entry for {key}")
        listed[key] = entry
    _validate_generation_provenance_files(
        generation_dir,
        summary,
        listed_files,
        _fail,
        _permission_fail,
    )

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
        if ("winner", name) not in listed:
            raise _fail(f"phase {name!r} winner is not listed in the artifact manifest")

    for kind, name in listed:
        if kind == "winner" and name not in phase_items:
            raise _fail(f"artifact manifest lists a winner for unknown phase {name!r}")

    for (kind, name), entry in listed.items():
        artifact_path = generation_dir / "phases" / name / WINNER_FILENAME
        try:
            content = _read_unlinked_bytes(artifact_path, root=generation_dir.parent)
        except PermissionError as exc:
            raise _permission_fail(
                _unreadable_artifact_permission_detail(f"{kind} artifact for phase {name!r}")
            ) from exc
        except OSError as exc:
            raise _fail(f"{kind} artifact for phase {name!r} is missing or unreadable") from exc
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise _fail(f"{kind} artifact for phase {name!r} does not match its recorded hash")
        try:
            payload = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise _fail(f"{kind} artifact for phase {name!r} is not parseable") from exc
        if not isinstance(payload, Mapping):
            raise _fail(f"{kind} artifact for phase {name!r} is not a mapping")
        if payload.get("phase") != name:
            raise _fail(f"{kind} artifact for phase {name!r} names a different phase")
        if kind == "winner":
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
            # A winner is legitimately carried forward from an earlier
            # generation, so its own generation_id may name that earlier
            # generation — the check is well-formedness, not equality.
            for id_field in ("generation_id", "attempt_id"):
                recorded = payload.get(id_field)
                if not isinstance(recorded, str) or not recorded:
                    raise _fail(f"winner for phase {name!r} has no valid {id_field}")
            fingerprint = payload.get("phase_fingerprint")
            if not is_sha256_hex(fingerprint):
                raise _fail(f"winner for phase {name!r} has no valid phase_fingerprint")
            source = payload.get("winner_source")
            if not isinstance(source, Mapping):
                raise _fail(f"winner for phase {name!r} has no valid winner_source")
            if source.get("kind") != "phase_trial":
                raise _fail(f"winner for phase {name!r} has no valid winner_source kind")
            if set(source) != {"kind", "phase", "trial_number", "generation_id", "attempt_id"}:
                raise _fail(f"winner for phase {name!r} has a removed winner_source field")
            source_phase = source.get("phase")
            if not isinstance(source_phase, str) or not SAFE_NAME_PATTERN.fullmatch(source_phase):
                raise _fail(f"winner for phase {name!r} has no valid winner_source phase")
            if source_phase != name:
                raise _fail(f"winner for phase {name!r} names another winner_source phase")
            source_trial = source.get("trial_number")
            if (
                not isinstance(source_trial, int)
                or isinstance(source_trial, bool)
                or source_trial != payload.get("trial_number")
            ):
                raise _fail(f"winner for phase {name!r} has no valid winner_source trial_number")
            for id_field in ("generation_id", "attempt_id"):
                if source.get(id_field) != payload.get(id_field):
                    raise _fail(
                        f"winner for phase {name!r} has a winner_source "
                        f"that disagrees with its {id_field}"
                    )
            _validate_winner_source_generation(
                generation_dir,
                generation_id,
                name,
                source_phase,
                summary.get("experiment"),
                payload,
                _fail,
                _permission_fail,
            )
            if not isinstance(payload.get("completion"), Mapping):
                raise _fail(f"winner for phase {name!r} has no completion metadata")
    phases_dir = generation_dir / "phases"
    if phases_dir.is_dir():
        for phase_dir in phases_dir.iterdir():
            if not phase_dir.is_dir():
                continue
            if (phase_dir / WINNER_FILENAME).is_file() and ("winner", phase_dir.name) not in listed:
                raise _fail(
                    f"namespace contains an unlisted winner artifact for phase {phase_dir.name!r}"
                )
            if (phase_dir / "promotion.yaml").is_file():
                raise _fail(
                    f"namespace contains removed promotion artifact for phase {phase_dir.name!r}"
                )


def _validate_winner_source_generation(
    generation_dir: Path,
    generation_id: str,
    phase_name: str,
    source_phase: object,
    expected_experiment: object,
    payload: Mapping[str, Any],
    fail: Callable[[str], PublicationIntegrityError],
    permission_fail: Callable[[str], PublicationAccessError],
) -> None:
    """Validate a carried winner against its source in this tree.

    A winner's own ``generation_id`` may legitimately name an earlier
    generation - a top-up that reselects an existing trial, or a ``--from-phase``
    resume that reuses a validated parent winner - and the manifest deliberately
    checks that field for well-formedness only. That left the whole
    cross-generation claim unverified: a published winner could cite a
    generation that exists only in some *other* artifact tree, or no tree at
    all, and the publication still validated as ``ok``. The cited namespace
    is where the evidence behind that number lives, so it has to resolve here.

    Deviating deliberately from a blanket "the source must also hold a winner
    record for this phase": a crash between trial completion and publication
    leaves a claim-time namespace - provenance files, no summary - whose trials
    a later top-up legitimately publishes first, citing that crashed
    generation. Requiring a winner record there would make ordinary crash
    recovery permanently unpublishable. So the record is required exactly when
    the source generation itself published (it has a summary); an unpublished
    source needs only to exist. Directory existence alone already enforces the
    same-tree invariant, and the summary carve-out still catches a published
    source that lost its winner record.

    :param Path generation_dir: The publishing generation's namespace directory.
    :param str generation_id: The publishing generation's own id.
    :param str phase_name: Phase exposing the winner being validated.
    :param object source_phase: Recorded phase that owns the source winner artifact.
    :param object expected_experiment: Experiment identity the source summary must retain.
    :param Mapping[str, Any] payload: Parsed winner artifact, whose
        ``generation_id``/``attempt_id``/``trial_number`` the caller has
        already checked for well-formedness and internal agreement.
    :param Callable[[str], PublicationIntegrityError] fail: Builder for the caller's
        uniformly labeled manifest-validation error.
    :param Callable[[str], PublicationAccessError] permission_fail: Builder for
        a permission-specific validation error.
    :raises PublicationAccessError: The source winner cannot be read by the current user.
    :raises PublicationIntegrityError: Whatever ``fail`` builds, when the cited source
        generation is unsafely named, absent from this tree, has a damaged
        published manifest, or recorded a different winning result.
    """
    source_generation = payload["generation_id"]
    same_generation = source_generation == generation_id
    if same_generation and source_phase == phase_name:
        return
    if not isinstance(source_phase, str) or not SAFE_NAME_PATTERN.fullmatch(source_phase):
        raise fail(f"winner for phase {phase_name!r} has no valid winner_source phase")
    if not SAFE_NAME_PATTERN.fullmatch(source_generation):
        # A corrupt or hostile winner could otherwise steer the lookups below
        # out of the generations root with a traversal component.
        raise fail(
            f"winner for phase {phase_name!r} cites source generation "
            f"{source_generation!r}, which is not a valid generation name"
        )
    source_dir = generation_dir if same_generation else generation_dir.parent / source_generation
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise fail(
            f"winner for phase {phase_name!r} cites source generation "
            f"{source_generation!r} which does not exist in this tree"
        )
    source_winner_path = source_dir / "phases" / source_phase / WINNER_FILENAME
    if not source_winner_path.is_file():
        if (source_dir / GENERATION_SUMMARY_FILENAME).is_file():
            raise fail(
                f"winner for phase {phase_name!r} cites source generation "
                f"{source_generation!r}, which published but holds no winner record "
                f"for source phase {source_phase!r}"
            )
        # Unpublished source namespace: the crash-recovery case above.
        return
    try:
        content = _read_unlinked_bytes(source_winner_path, root=generation_dir.parent)
    except PermissionError as exc:
        raise permission_fail(
            _unreadable_artifact_permission_detail(
                f"source generation {source_generation!r} winner artifact for phase "
                f"{source_phase!r}"
            )
        ) from exc
    except OSError as exc:
        raise fail(
            f"source generation {source_generation!r} winner artifact for phase "
            f"{source_phase!r} is missing or unreadable"
        ) from exc
    try:
        source_payload = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise fail(
            f"source generation {source_generation!r} winner artifact for phase "
            f"{source_phase!r} is not parseable"
        ) from exc
    if not isinstance(source_payload, Mapping):
        raise fail(
            f"source generation {source_generation!r} winner artifact for phase "
            f"{source_phase!r} is not a mapping"
        )
    source_summary_path = source_dir / GENERATION_SUMMARY_FILENAME
    if source_summary_path.is_file() and not same_generation:
        try:
            source_summary = yaml.safe_load(
                _read_unlinked_bytes(source_summary_path, root=generation_dir.parent)
            )
        except PermissionError as exc:
            raise permission_fail(
                _unreadable_artifact_permission_detail(
                    f"source generation {source_generation!r} summary"
                )
            ) from exc
        except (OSError, yaml.YAMLError) as exc:
            raise fail(
                f"source generation {source_generation!r} summary is unreadable or invalid"
            ) from exc
        if not isinstance(source_summary, Mapping):
            raise fail(f"source generation {source_generation!r} summary is not a mapping")
        if source_summary.get("experiment") != expected_experiment:
            raise fail(
                f"source generation {source_generation!r} summary names a different experiment"
            )
        if source_summary.get("generation_id") != source_generation:
            raise fail(
                f"source generation {source_generation!r} summary names a different generation"
            )
        try:
            _validate_generation_manifest(source_dir, source_generation, source_summary)
        except PublicationAccessError as exc:
            raise permission_fail(str(exc)) from exc
        except PublicationIntegrityError as exc:
            raise fail(f"source generation {source_generation!r} does not validate: {exc}") from exc
    expected = {
        "phase": source_phase,
        "trial_number": payload.get("trial_number"),
        "generation_id": source_generation,
        "attempt_id": payload.get("attempt_id"),
    }
    for field_name, expected_value in expected.items():
        if source_payload.get(field_name) != expected_value:
            raise fail(
                f"winner for phase {phase_name!r} disagrees with the winner recorded by its "
                f"source generation {source_generation!r} on {field_name}"
            )
    # Completion and phase fingerprint belong to the exposing phase; the
    # fields below come from the cited source trial.
    for field_name in (
        "metric",
        "params",
        "effective_overrides",
        "constraints",
        "gates",
        "objective_provenance",
        "trainer_input",
        "trainer_env_digest",
        "trainer_inherit_env",
    ):
        if source_payload.get(field_name) != payload.get(field_name):
            raise fail(
                f"winner for phase {phase_name!r} disagrees with the result recorded by its "
                f"source generation {source_generation!r} on {field_name}"
            )


def _validate_generation_provenance_files(
    generation_dir: Path,
    summary: Mapping[str, Any],
    listed_files: Mapping[str, Mapping[str, Any]],
    fail: Callable[[str], PublicationIntegrityError],
    permission_fail: Callable[[str], PublicationAccessError],
) -> None:
    """Validate a generation's claim-time provenance files against its manifest.

    Extends the manifest invariant -- every listed artifact exists and hashes
    to its recorded content, and the namespace holds nothing the manifest does
    not list -- onto ``config.snapshot.yaml`` and ``reproducibility.json``
    (review v0.5.18 / finding F6). The two are all-or-nothing: they are
    written together at claim time and every current-format summary must list
    both. A missing file or manifest entry is therefore a format violation.

    Note that the snapshot is owner-only, so a reader who cannot read it
    cannot validate the publication at all -- the same fail-closed outcome as
    any other unreadable manifest-listed artifact. A permission denial is
    reported as its own reason rather than as "missing or unreadable"
    (:func:`_unreadable_artifact_permission_detail`, re-review v0.5.19 /
    observation N1): a second operator on a healthy tree hits it routinely,
    and the remedy is the publishing user, not a namespace restore.

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


@dataclass(frozen=True)
class _PointerTarget:
    """Validated summary identity carried by one last-success pointer."""

    generation_id: str
    summary_size_bytes: int
    summary_sha256: str


def _read_pointer_target(
    pointer_path: Path,
    *,
    id_key: str,
    owner_key: str,
    owner_name: str,
) -> _PointerTarget | None:
    """Read one last-success pointer's target id, validating the pointer itself.

    :param Path pointer_path: Pointer YAML file to read.
    :param str id_key: Payload key holding the target generation id.
    :param str owner_key: Payload key naming the owning experiment.
    :param str owner_name: Expected owner name the pointer must record.
    :return _PointerTarget | None: A safe-name target and exact summary-byte identity, or
        ``None`` when the pointer is missing, unreadable, malformed, names
        another owner, or carries an unsafe id.
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
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PUBLICATION_POINTER_SCHEMA_VERSION
        or payload.get(owner_key) != owner_name
    ):
        return None
    target_id = payload.get(id_key)
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
    id_key: str,
    target_id: str,
    owner_key: str,
    owner_name: str,
    expected_size_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Read a pointer target's own immutable summary and confirm its identity.

    Replaces the old record-state check (``_record_is_complete``): the
    per-generation lifecycle record is now informational and written *after*
    the last-success pointer commit (review v0.5.15 / blocker 3), so requiring
    ``state == "complete"``/``"published"`` on it would create a crash window
    where a just-committed publication reads back as nothing-published. This
    fails closed on the pointer target's own immutable *summary*: it must
    parse as a mapping naming this exact owner and id. Callers holding a
    schema-versioned summary additionally validate the complete artifact
    manifest (:func:`_validate_generation_manifest`; review v0.5.16 /
    blocker 3) — this helper only performs the identity gate common to
    experiment pointers.

    :param Path summary_path: Immutable generation summary YAML file to read.
    :param str id_key: Summary key holding the generation id.
    :param str target_id: Generation id the summary must name.
    :param str owner_key: Summary key naming the owning experiment.
    :param str owner_name: Expected owner name the summary must carry.
    :param int | None expected_size_bytes: Pointer-recorded exact summary byte length.
    :param str | None expected_sha256: Pointer-recorded SHA-256 of the exact summary bytes.
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
    if expected_size_bytes is not None and len(content) != expected_size_bytes:
        raise PublicationIntegrityError(
            "Generation summary byte length does not match the last-success pointer."
        )
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise PublicationIntegrityError(
            "Generation summary digest does not match the last-success pointer."
        )
    try:
        summary = yaml.safe_load(content)
    except yaml.YAMLError:
        return None
    if (
        isinstance(summary, dict)
        and summary.get(owner_key) == owner_name
        and summary.get(id_key) == target_id
    ):
        return summary
    return None
