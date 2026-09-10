"""Publication manifests and immutable summary validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import phasesweep.engine.paths as path_ops
from phasesweep.config import Experiment, Metric
from phasesweep.config.common import SAFE_NAME_PATTERN
from phasesweep.config.models import _metric_semantics_payload
from phasesweep.engine.errors import PublicationAccessError, PublicationIntegrityError
from phasesweep.engine.paths import GENERATION_SUMMARY_FILENAME
from phasesweep.engine.state import (
    _ARTIFACT_FILENAMES,
    _GENERATION_FILE_FILENAMES,
    _MANIFEST_ARTIFACT_KINDS,
    _MANIFEST_GENERATION_FILE_KINDS,
    GENERATION_CONFIG_SNAPSHOT_FILENAME,
    GENERATION_SUMMARY_SCHEMA_VERSION,
    PUBLICATION_POINTER_SCHEMA_VERSION,
)
from phasesweep.runtime.files import file_sha256


def _generation_artifact_manifest(
    experiment: Experiment, generation_id: str
) -> list[dict[str, str]]:
    """List and hash every result artifact in one generation's namespace.

    Scans the namespace itself rather than reconstructing the set from
    in-memory bookkeeping so the manifest cannot omit an artifact the run
    actually wrote (e.g. a prior promotion decision projected for a skipped
    phase). The namespace is exclusively claimed by this invocation, so
    everything present is this run's own output.

    The namespace-root provenance files written at claim time
    (:func:`phasesweep.engine.provenance._write_generation_provenance`) are listed first, under entries
    that carry ``path`` instead of ``phase``. They are absent -- and so
    unlisted -- in generations published before finding F6.

    :param Experiment experiment: Experiment whose generation is summarized.
    :param str generation_id: Immutable generation namespace to scan.
    :return list[dict[str, str]]: One ``{"kind", "path", "sha256"}`` entry per
        namespace-root provenance file, then one ``{"kind", "phase",
        "sha256"}`` entry per winner/promotion artifact ordered by phase then
        kind.
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
        for kind in ("winner", "promotion"):
            artifact = phase_dir / _ARTIFACT_FILENAMES[kind]
            if artifact.is_file():
                items.append(
                    {"kind": kind, "phase": phase_dir.name, "sha256": file_sha256(artifact)}
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

    The manifest covers two kinds of artifact: phase-scoped winners and
    promotion decisions (``kind`` + ``phase``), and the namespace-root
    provenance files frozen at claim time (``kind`` + ``path``; see
    :func:`_validate_generation_provenance_files`, review v0.5.18 / finding
    F6). Both obey the same listed-if-and-only-if-present rule, which is what
    keeps a generation published before either existed valid without a schema
    bump.

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
        if kind not in _MANIFEST_ARTIFACT_KINDS or not isinstance(entry.get("phase"), str):
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

    raw_decisions = summary.get("promotion_decisions")
    if not isinstance(raw_decisions, list):
        raise _fail("summary has no promotion decision list")
    decision_items: dict[str, Mapping[str, Any]] = {}
    for decision in raw_decisions:
        if not isinstance(decision, Mapping) or not isinstance(decision.get("phase"), str):
            raise _fail("summary promotion decision entry is malformed")
        name = str(decision["phase"])
        decision_items[name] = decision
        if ("promotion", name) not in listed:
            raise _fail(f"phase {name!r} promotion decision is not listed in the artifact manifest")

    for kind, name in listed:
        if kind == "winner" and name not in phase_items:
            raise _fail(f"artifact manifest lists a winner for unknown phase {name!r}")

    for (kind, name), entry in listed.items():
        artifact_path = generation_dir / "phases" / name / _ARTIFACT_FILENAMES[kind]
        try:
            content = artifact_path.read_bytes()
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
            # A winner is legitimately carried forward from an earlier
            # generation, so its own generation_id may name that earlier
            # generation — the check is well-formedness, not equality.
            for id_field in ("generation_id", "attempt_id"):
                recorded = payload.get(id_field)
                if not isinstance(recorded, str) or not recorded:
                    raise _fail(f"winner for phase {name!r} has no valid {id_field}")
            source = payload.get("winner_source")
            if not isinstance(source, Mapping):
                raise _fail(f"winner for phase {name!r} has no valid winner_source")
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
                source.get("phase"),
                payload,
                _fail,
                _permission_fail,
            )
            if not isinstance(payload.get("completion"), Mapping):
                raise _fail(f"winner for phase {name!r} has no completion metadata")
        elif name in decision_items:
            if payload.get("action") != decision_items[name].get("action"):
                raise _fail(
                    f"promotion decision for phase {name!r} disagrees with the summary action"
                )
        else:
            # Older schema-2 resumes listed projected promotions only in the
            # manifest. Accept only a copy of the named earlier generation's
            # decision; this generation's decisions must appear in its summary.
            source_generation = payload.get("generation_id")
            if (
                not isinstance(source_generation, str)
                or not SAFE_NAME_PATTERN.fullmatch(source_generation)
                or source_generation == generation_id
            ):
                raise _fail(
                    f"artifact manifest lists a promotion decision absent from the summary for phase {name!r}"
                )
            source_path = (
                generation_dir.parent
                / source_generation
                / "phases"
                / name
                / _ARTIFACT_FILENAMES["promotion"]
            )
            try:
                source_payload = yaml.safe_load(source_path.read_bytes())
            except PermissionError as exc:
                raise _permission_fail(
                    _unreadable_artifact_permission_detail(
                        f"source promotion artifact for phase {name!r}"
                    )
                ) from exc
            except (OSError, yaml.YAMLError) as exc:
                raise _fail(
                    f"promotion decision for phase {name!r} cites source generation "
                    f"{source_generation!r} with no readable promotion artifact"
                ) from exc
            if source_payload != payload:
                raise _fail(
                    f"promotion decision for phase {name!r} disagrees with "
                    f"source generation {source_generation!r}"
                )

    phases_dir = generation_dir / "phases"
    if phases_dir.is_dir():
        for phase_dir in phases_dir.iterdir():
            if not phase_dir.is_dir():
                continue
            for kind, filename in _ARTIFACT_FILENAMES.items():
                if (phase_dir / filename).is_file() and (kind, phase_dir.name) not in listed:
                    raise _fail(
                        f"namespace contains an unlisted {kind} artifact "
                        f"for phase {phase_dir.name!r}"
                    )


def _validate_winner_source_generation(
    generation_dir: Path,
    generation_id: str,
    phase_name: str,
    source_phase: object,
    payload: Mapping[str, Any],
    fail: Callable[[str], PublicationIntegrityError],
    permission_fail: Callable[[str], PublicationAccessError],
) -> None:
    """Require a carried-forward winner's source generation to exist in this tree.

    A winner's own ``generation_id`` may legitimately name an earlier
    generation - a top-up that reselects an existing trial, or a ``--from-phase``
    resume that reuses a validated parent winner - and the manifest deliberately
    checks that field for well-formedness only. That left the whole
    cross-generation claim unverified: a published winner could cite a
    generation that exists only in some *other* artifact tree, or no tree at
    all, and the publication still validated as ``ok`` (PR #5 review /
    reviewer 2, blocker 7). The cited namespace is where the evidence behind
    that number lives, so it has to resolve here.

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
    :param Mapping[str, Any] payload: Parsed winner artifact, whose
        ``generation_id``/``attempt_id``/``trial_number`` the caller has
        already checked for well-formedness and internal agreement.
    :param Callable[[str], PublicationIntegrityError] fail: Builder for the caller's
        uniformly labeled manifest-validation error.
    :param Callable[[str], PublicationAccessError] permission_fail: Builder for
        a permission-specific validation error.
    :raises PublicationAccessError: The source winner cannot be read by the current user.
    :raises PublicationIntegrityError: Whatever ``fail`` builds, when the cited source
        generation is unsafely named, absent from this tree, or published a
        winner record for this phase that disagrees with the carried winner.
    """
    source_generation = payload["generation_id"]
    if source_generation == generation_id:
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
    source_dir = generation_dir.parent / source_generation
    if not source_dir.is_dir():
        raise fail(
            f"winner for phase {phase_name!r} cites source generation "
            f"{source_generation!r} which does not exist in this tree"
        )
    source_winner_path = source_dir / "phases" / source_phase / _ARTIFACT_FILENAMES["winner"]
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
        content = source_winner_path.read_bytes()
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
    written together at claim time, so a manifest that lists one without the
    other has been edited. A generation published before those files existed
    lists neither and holds neither, which passes every check here unchanged.

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
    if not listed_files:
        return
    if set(listed_files) != _MANIFEST_GENERATION_FILE_KINDS:
        raise fail("summary lists only part of the generation provenance record")

    contents: dict[str, bytes] = {}
    digests: dict[str, str] = {}
    for kind, filename in _GENERATION_FILE_FILENAMES.items():
        try:
            content = (generation_dir / filename).read_bytes()
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
    :param str owner_key: Payload key naming the owning experiment or suite.
    :param str owner_name: Expected owner name the pointer must record.
    :return _PointerTarget | None: A safe-name target and exact summary-byte identity, or
        ``None`` when the pointer is missing, unreadable, malformed, names
        another owner, or carries an unsafe id.
    :raises PublicationAccessError: The pointer cannot be read by the current user.
    """
    try:
        payload = yaml.safe_load(pointer_path.read_text())
    except PermissionError as exc:
        raise PublicationAccessError(
            _unreadable_artifact_permission_detail("last-success pointer")
        ) from exc
    except (OSError, yaml.YAMLError):
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
        or not isinstance(summary_sha256, str)
        or len(summary_sha256) != 64
        or any(character not in "0123456789abcdef" for character in summary_sha256)
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
    experiment and suite pointers.

    :param Path summary_path: Immutable generation summary YAML file to read.
    :param str id_key: Summary key holding the generation id.
    :param str target_id: Generation id the summary must name.
    :param str owner_key: Summary key naming the owning experiment or suite.
    :param str owner_name: Expected owner name the summary must carry.
    :param int | None expected_size_bytes: Pointer-recorded exact summary byte length.
    :param str | None expected_sha256: Pointer-recorded SHA-256 of the exact summary bytes.
    :return dict[str, Any] | None: The parsed summary when readable and
        correctly named; ``None`` otherwise.
    :raises PublicationAccessError: The summary cannot be read by the current user.
    :raises PublicationIntegrityError: The summary bytes disagree with either pointer anchor.
    """
    try:
        content = summary_path.read_bytes()
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


def _validate_suite_summary_integrity(
    generation_id: str,
    summary: Mapping[str, Any],
) -> None:
    """Validate a suite summary's winner facts against its recorded components.

    Suite mirror of :func:`_validate_generation_manifest` (review v0.5.17 gap
    hunt): the published suite summary is the authoritative read surface for
    suite winners, so its per-study results must be anchored to hash-covered
    component artifacts rather than trusted as bare text. Each study record
    names its component generation summary by absolute path plus content
    hash; the file must sit at ``generations/<component id>/summary.yaml``,
    parse, and name the recorded experiment and generation — and every
    exposed phase winner in the suite summary must match one of those verified
    component summaries. A study's own winner matches its complete compact
    payload. A promotion-adopted suite baseline matches the recorded source
    phase and trial-derived payload (trial, metric, parameters, effective
    overrides, constraints, gates, generation, and attempt), while its
    completion/source/promotion metadata may legitimately describe the
    candidate slot that exposes the clone. Runs pre-commit (before the suite
    last-success pointer advances) and read-side (before a pointer target is
    trusted).

    :param str generation_id: Suite generation id, used for error text.
    :param Mapping[str, Any] summary: Parsed suite summary payload.
    :raises PublicationAccessError: A component summary cannot be read by the current user.
    :raises PublicationIntegrityError: A study record is malformed, a component summary is
        missing, altered, or misidentified, or an exposed winner is not
        anchored to any verified component summary.
    """

    def _fail(reason: str) -> PublicationIntegrityError:
        """Build one uniformly labeled suite-integrity error.

        :param str reason: Specific validation failure being reported.
        :return PublicationIntegrityError: Error naming the suite generation and reason.
        """
        return PublicationIntegrityError(
            f"Suite generation {generation_id!r} summary integrity validation failed: {reason}"
        )

    def _permission_fail(reason: str) -> PublicationAccessError:
        """Build a permission-specific suite-validation error.

        :param str reason: Permission failure being reported.
        :return PublicationAccessError: Error naming the suite generation and reason.
        """
        return PublicationAccessError(
            f"Suite generation {generation_id!r} summary validation could not run: {reason}"
        )

    records = summary.get("studies")
    if not isinstance(records, list):
        raise _fail("summary has no study records")

    evidence_fields = (
        "trial_number",
        "metric",
        "params",
        "effective_overrides",
        "constraints",
        "gates",
        "generation_id",
        "attempt_id",
    )
    full_fields = (*evidence_fields, "completion", "winner_source", "promotion")

    def _winner_key(
        item: Mapping[str, Any],
        *,
        phase_name: str,
        include_exposure_metadata: bool,
    ) -> tuple[str, str]:
        """Return a type-aware canonical key for one component winner.

        :param Mapping[str, Any] item: One summary phase-winner entry.
        :param str phase_name: Source component phase that produced the winner.
        :param bool include_exposure_metadata: Include completion, source, and
            promotion fields when the suite exposes its own component winner.
        :raises PublicationIntegrityError: The winner payload cannot be represented as safe YAML.
        :return tuple[str, str]: Source phase plus canonical winner payload.
        """
        fields = full_fields if include_exposure_metadata else evidence_fields
        payload = {field: item[field] for field in fields if field in item}
        try:
            encoded = yaml.safe_dump(payload, sort_keys=True)
        except yaml.YAMLError as exc:
            raise _fail("summary contains an invalid winner payload") from exc
        return phase_name, encoded

    component_full_winners: set[tuple[str, str]] = set()
    component_evidence_winners: set[tuple[str, str]] = set()
    legacy_component_studies: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("name"), str):
            raise _fail("summary study record is malformed")
        name = str(record["name"])
        component_generation = record.get("experiment_generation_id")
        recorded_path = record.get("component_summary_path")
        recorded_sha = record.get("component_summary_sha256")
        if (
            not isinstance(component_generation, str)
            or not component_generation
            or not isinstance(recorded_path, str)
            or not recorded_path
            or not isinstance(recorded_sha, str)
        ):
            raise _fail(f"study {name!r} has no component summary reference")
        target = Path(recorded_path)
        expected_suffix = Path("generations") / component_generation / "summary.yaml"
        if not target.is_absolute() or target.parts[-3:] != expected_suffix.parts:
            raise _fail(
                f"study {name!r} component summary path does not name its "
                "recorded generation namespace"
            )
        try:
            content = target.read_bytes()
        except PermissionError as exc:
            raise _permission_fail(
                _unreadable_artifact_permission_detail(f"study {name!r} component summary")
            ) from exc
        except OSError as exc:
            raise _fail(f"study {name!r} component summary is missing or unreadable") from exc
        if hashlib.sha256(content).hexdigest() != recorded_sha:
            raise _fail(f"study {name!r} component summary does not match its recorded hash")
        try:
            payload = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise _fail(f"study {name!r} component summary is not parseable") from exc
        if not isinstance(payload, Mapping):
            raise _fail(f"study {name!r} component summary is not a mapping")
        if (
            payload.get("experiment") != record.get("experiment")
            or payload.get("generation_id") != component_generation
        ):
            raise _fail(f"study {name!r} component summary names a different identity")
        if "schema_version" not in payload:
            # Pre-manifest legacy component summary: identity + hash anchor
            # only, mirroring the legacy rule in the publication-time
            # component-manifest chase. Its winners cannot be indexed for the
            # membership check below.
            legacy_component_studies.add(name)
            continue
        phases = payload.get("phases")
        if not isinstance(phases, list):
            raise _fail(f"study {name!r} component summary has no phase list")
        for item in phases:
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                raise _fail(f"study {name!r} component summary has a malformed phase entry")
            phase_name = str(item["name"])
            component_full_winners.add(
                _winner_key(item, phase_name=phase_name, include_exposure_metadata=True)
            )
            component_evidence_winners.add(
                _winner_key(item, phase_name=phase_name, include_exposure_metadata=False)
            )

    for record in records:
        name = str(record["name"])
        if name in legacy_component_studies:
            continue
        phases = record.get("phases")
        if not isinstance(phases, list):
            raise _fail(f"study {name!r} has no phase list")
        for item in phases:
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                raise _fail(f"study {name!r} has a malformed phase entry")
            if item.get("exposed") is not True:
                continue
            source = item.get("winner_source")
            if isinstance(source, Mapping) and source.get("kind") == "suite_baseline":
                source_phase = source.get("phase")
                if not isinstance(source_phase, str) or not source_phase:
                    raise _fail(
                        f"study {name!r} exposed winner for phase {item['name']!r} "
                        "has no baseline source phase"
                    )
                for identity_field in (
                    "trial_number",
                    "generation_id",
                    "attempt_id",
                ):
                    if type(source.get(identity_field)) is not type(
                        item.get(identity_field)
                    ) or source.get(identity_field) != item.get(identity_field):
                        raise _fail(
                            f"study {name!r} exposed winner for phase {item['name']!r} "
                            "does not match its baseline source identity"
                        )
                key = _winner_key(
                    item,
                    phase_name=source_phase,
                    include_exposure_metadata=False,
                )
                anchored = key in component_evidence_winners
            else:
                key = _winner_key(
                    item,
                    phase_name=str(item["name"]),
                    include_exposure_metadata=True,
                )
                anchored = key in component_full_winners
            if not anchored:
                raise _fail(
                    f"study {name!r} exposed winner for phase {item['name']!r} does "
                    "not match any verified component summary"
                )
