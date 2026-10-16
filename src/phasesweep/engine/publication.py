"""Read and resolve authoritative experiment publication pointers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import phasesweep.engine.paths as path_ops
import phasesweep.engine.publication_validation as validation_ops
from phasesweep.config.models import Experiment
from phasesweep.engine.errors import PublicationAccessError, PublicationIntegrityError
from phasesweep.engine.state import PublicationState

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublicationPointer:
    """Four-state resolution of a last-success publication pointer.

    Only ``ok`` exposes a validated result. ``absent`` means no publication pointer;
    ``failed`` and ``permission_denied`` mean an existing result could not be validated.
    ``error`` is path-free operator detail; MCP exposes the integrity enum only.
    """

    state: PublicationState
    generation_id: str | None
    """Pointer target id: the published generation when ``ok``, the generation
    whose validation failed when ``failed`` or ``permission_denied``, ``None``
    when ``absent`` or when the pointer itself is the unreadable part."""
    error: str | None
    """Validation diagnostic for ``failed`` and ``permission_denied`` states."""
    validated: validation_ops.ValidatedPublication | None = field(
        default=None, compare=False, repr=False
    )
    """Everything the validation read, set exactly when ``state`` is ``ok``.
    Consumers show and resume from it rather than rereading any artifact."""


def _unresolvable_pointer(pointer_path: Path, owner_label: str) -> PublicationPointer:
    """Classify an existing pointer that cannot identify a published target as failed.

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
    """Resolve the last-success pointer and validate its current-format result graph.

    Summary byte anchors and experiment/generation identity establish the pointer's
    target. Lifecycle records follow commit and do not determine publication authority.
    Validate all manifest artifacts on each authoritative read, retaining one captured
    result for consumers.

    A missing pointer is ``absent``. Structural or content failure is ``failed``;
    permission denial is ``permission_denied``. Neither failure exposes published
    results or permits treating existing publication evidence as a fresh tree.

    :param Experiment experiment: Experiment config with artifact root details.
    :param bool raise_on_manifest_error: Re-raise a publication's
        manifest error or summary-anchor mismatch for an actionable resume
        failure instead of reporting it as an unusable verdict, as read-only
        status APIs require. Only those two raise: a manifest artifact that
        cannot be read or fails validation, and a summary whose byte length or
        SHA-256 disagrees with the pointer. Every other failure resolves to a
        verdict: an unreadable or malformed pointer, a dangling target, and a
        missing, unparsable, or misnamed summary are ``failed``, and a
        permission denial on the pointer or the summary is
        ``permission_denied``.
    :raises PublicationAccessError: ``raise_on_manifest_error`` is set and a
        manifest artifact cannot be read as the current user.
    :raises PublicationIntegrityError: ``raise_on_manifest_error`` is set and
        the target's summary disagrees with the pointer's byte length or
        SHA-256, or its artifact manifest does not validate.
    :return PublicationPointer: The publication verdict for this experiment.
    """
    pointer_path = path_ops._last_successful_generation_path(experiment)
    try:
        target = validation_ops._read_pointer_target(
            pointer_path, experiment_name=experiment.experiment
        )
    except PublicationAccessError as exc:
        return PublicationPointer(state="permission_denied", generation_id=None, error=str(exc))
    if target is None:
        return _unresolvable_pointer(pointer_path, f"experiment {experiment.experiment!r}")
    generation_id = target.generation_id
    try:
        summary = validation_ops._read_pointer_target_summary(
            path_ops._generation_summary_path(experiment, generation_id),
            experiment_name=experiment.experiment,
            target=target,
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
        validated = validation_ops._validate_generation_manifest(
            generation_dir, generation_id, summary
        )
    else:
        try:
            validated = validation_ops._validate_generation_manifest(
                generation_dir, generation_id, summary
            )
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
    return PublicationPointer(
        state="ok", generation_id=generation_id, error=None, validated=validated
    )
