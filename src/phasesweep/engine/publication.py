"""Read and resolve authoritative experiment publication pointers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import phasesweep.engine.paths as path_ops
import phasesweep.engine.publication_validation as validation_ops
from phasesweep.config import Experiment
from phasesweep.engine.errors import PublicationAccessError, PublicationIntegrityError
from phasesweep.engine.state import PublicationState

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublicationPointer:
    """Four-state resolution of one last-success publication pointer.

    "Nothing was ever published" and "the recorded publication no longer
    validates" are different operational facts with opposite remedies, and
    collapsing both into ``None`` made a corrupt result tree indistinguishable
    from a fresh one (review v0.5.18 / finding F4). The reporting surfaces
    resolve through this type; path-construction helpers keep the boolean
    :func:`_last_successful_generation_id` view, since all non-``ok`` states
    mean the same thing to them: nothing may be read as published.

    ``error`` is deliberately path-free: it names artifacts by role rather than
    location, so any reporting surface could quote it safely -- though the MCP
    layer forwards only the ``publication_integrity`` enum and leaves the
    free-text detail to the operator-facing CLI.
    """

    state: PublicationState
    generation_id: str | None
    """Pointer target id: the published generation when ``ok``, the generation
    whose validation failed when ``failed`` or ``permission_denied``, ``None``
    when ``absent`` or when the pointer itself is the unreadable part."""
    error: str | None
    """Validation diagnostic for ``failed`` and ``permission_denied`` states."""
    summary: Mapping[str, Any] | None = field(default=None, compare=False, repr=False)
    """Exact pointer-authenticated parsed summary, for consumers that need its fields."""


def _unresolvable_pointer(pointer_path: Path, owner_label: str) -> PublicationPointer:
    """Classify a pointer whose own payload could not be resolved to a target.

    A pointer file is written once, atomically, after its target validates, so
    a file that exists but does not name a target this owner published is
    tampering or corruption -- not an owner that has published nothing.

    :param Path pointer_path: Last-success pointer whose read just failed.
    :param str owner_label: Human-readable owner for the diagnostic text.
    :return PublicationPointer: ``absent`` when no pointer file exists,
        ``permission_denied`` when this user cannot inspect it, and ``failed``
        when it otherwise exists but cannot be resolved.
    """
    try:
        pointer_path.lstat()
    except FileNotFoundError:
        return PublicationPointer(state="absent", generation_id=None, error=None)
    except PermissionError:
        return PublicationPointer(
            state="permission_denied",
            generation_id=None,
            error=validation_ops._unreadable_artifact_permission_detail("last-success pointer"),
        )
    except OSError:
        pass
    return PublicationPointer(
        state="failed",
        generation_id=None,
        error=(
            f"The last-success pointer for {owner_label} is unreadable, malformed, "
            "or does not name a generation this owner published."
        ),
    )


def _resolve_publication_pointer(
    experiment: Experiment,
    *,
    raise_on_manifest_error: bool = False,
) -> PublicationPointer:
    """Resolve the last-success pointer to its four-state validation verdict.

    The pointer is authoritative only when its target's own immutable summary
    parses and names this exact experiment and generation (review v0.5.15 /
    blocker 3) -- not when the per-generation lifecycle record says so, since
    that record is now written *after* this pointer commits and would
    otherwise create a crash window where a committed publication reads as
    nothing-published.

    The target's current-format complete artifact manifest is always
    validated -- every listed winner artifact plus the
    claim-time provenance files must exist, hash to their recorded content,
    and cross-check against the summary (review v0.5.16 / blocker 3, review
    v0.5.18 / finding F6). Validation runs on every authoritative read:
    generation directories are write-once by PhaseSweep convention, but the
    filesystem does not enforce immutability and a long-lived reader must
    notice later corruption or operator edits. Older summary formats are
    rejected at this breaking-release boundary.

    Every validation refusal resolves to an unusable state, never ``absent``:
    structural or content failures become ``failed``, while an actual
    permission denial becomes ``permission_denied``. The pointer's existence
    is durable evidence that this tree once held a result, and a caller told
    "nothing published" could otherwise re-run over it and advance the pointer
    past evidence that still needs operator attention.

    :param Experiment experiment: Experiment config with artifact root details.
    :param bool raise_on_manifest_error: Re-raise a publication's
        manifest error for an actionable resume failure instead of reporting
        it as an unusable verdict, as read-only status APIs require. Only the
        manifest branch raises; every other failure resolves to ``failed`` or
        ``permission_denied``.
    :raises PublicationAccessError: ``raise_on_manifest_error`` is set and a
        manifest artifact cannot be read as the current user.
    :raises PublicationIntegrityError: ``raise_on_manifest_error`` is set and
        the target's artifact manifest does not validate.
    :return PublicationPointer: The publication verdict for this experiment.
    """
    pointer_path = path_ops._last_successful_generation_path(experiment)
    try:
        target = validation_ops._read_pointer_target(
            pointer_path,
            id_key="generation_id",
            owner_key="experiment",
            owner_name=experiment.experiment,
        )
    except PublicationAccessError as exc:
        return PublicationPointer(state="permission_denied", generation_id=None, error=str(exc))
    if target is None:
        return _unresolvable_pointer(pointer_path, f"experiment {experiment.experiment!r}")
    generation_id = target.generation_id
    try:
        summary = validation_ops._read_pointer_target_summary(
            path_ops._generation_summary_path(experiment, generation_id),
            id_key="generation_id",
            target_id=generation_id,
            owner_key="experiment",
            owner_name=experiment.experiment,
            expected_size_bytes=target.summary_size_bytes,
            expected_sha256=target.summary_sha256,
        )
    except PublicationAccessError as exc:
        return PublicationPointer(
            state="permission_denied", generation_id=generation_id, error=str(exc)
        )
    except PublicationIntegrityError as exc:
        if raise_on_manifest_error:
            raise
        return PublicationPointer(state="failed", generation_id=generation_id, error=str(exc))
    if summary is None:
        return PublicationPointer(
            state="failed",
            generation_id=generation_id,
            error=(
                f"Generation {generation_id!r} summary is missing, unreadable, or does not "
                f"name experiment {experiment.experiment!r} and this generation."
            ),
        )
    generation_dir = path_ops._generation_dir(experiment, generation_id)
    if raise_on_manifest_error:
        validation_ops._validate_generation_manifest(generation_dir, generation_id, summary)
    else:
        try:
            validation_ops._validate_generation_manifest(generation_dir, generation_id, summary)
        except PublicationAccessError as exc:
            log.warning("%s", exc)
            return PublicationPointer(
                state="permission_denied",
                generation_id=generation_id,
                error=str(exc),
            )
        except PublicationIntegrityError as exc:
            log.warning("%s", exc)
            return PublicationPointer(state="failed", generation_id=generation_id, error=str(exc))
    return PublicationPointer(state="ok", generation_id=generation_id, error=None, summary=summary)


def _last_successful_generation_id(
    experiment: Experiment,
    *,
    raise_on_manifest_error: bool = False,
) -> str | None:
    """Return the last-success generation id, failing closed on any invalidity.

    The boolean-blind view of :func:`_resolve_publication_pointer`, kept for
    the path-construction and resume callers to which "never published" and
    "published but corrupt" mean the same thing: nothing here may be read as
    published. Every caller that *reports* publication state to an operator or
    agent must use the four-state resolver instead (review v0.5.18 / finding
    F4).

    :param Experiment experiment: Experiment config with artifact root details.
    :param bool raise_on_manifest_error: Re-raise a versioned publication's
        manifest error for an actionable resume failure instead of returning
        ``None`` as read-only status APIs require.
    :raises PublicationAccessError: ``raise_on_manifest_error`` is set and a
        manifest artifact cannot be read as the current user.
    :raises PublicationIntegrityError: ``raise_on_manifest_error`` is set and
        the target's artifact manifest does not validate.
    :return str | None: The last-successful generation id, or ``None`` if the
        pointer or its target summary is missing, unreadable, malformed,
        unsafely named, owned by another experiment, or fails manifest
        validation.
    """
    pointer = _resolve_publication_pointer(
        experiment,
        raise_on_manifest_error=raise_on_manifest_error,
    )
    return pointer.generation_id if pointer.state == "ok" else None


def _published_winner_path_for(
    experiment: Experiment,
    published_generation_id: str | None,
    phase_name: str,
) -> Path | None:
    """Resolve one authoritative current-format winner path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str | None published_generation_id: Already-resolved last-success id.
    :param str phase_name: Phase name whose published winner path is requested.
    :return Path | None: The immutable generation-scoped winner path, or
        ``None`` when no last-success generation was resolved.
    """
    if published_generation_id is None:
        return None
    return path_ops._generation_winner_path(experiment, published_generation_id, phase_name)


def _published_winner_path(experiment: Experiment, phase_name: str) -> Path | None:
    """Return one authoritative last-success winner path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str phase_name: Phase name whose published winner path is requested.
    :return Path | None: The immutable generation-scoped winner path, or
        ``None`` when no last-success generation was resolved.
    """
    return _published_winner_path_for(
        experiment, _last_successful_generation_id(experiment), phase_name
    )


def _published_summary_path_for(
    experiment: Experiment,
    published_generation_id: str | None,
) -> Path | None:
    """Resolve an authoritative current-format summary path.

    :param Experiment experiment: Experiment config with artifact root details.
    :param str | None published_generation_id: Already-resolved last-success id.
    :return Path | None: The immutable generation-scoped summary path, or
        ``None`` when no last-success generation was resolved.
    """
    if published_generation_id is None:
        return None
    return path_ops._generation_summary_path(experiment, published_generation_id)
