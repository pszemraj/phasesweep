"""Experiment generation claims, lifecycle records, and publication writes."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import yaml

import phasesweep.engine.artifacts as artifact_io
import phasesweep.engine.paths as path_ops
import phasesweep.engine.provenance as provenance_ops
import phasesweep.engine.publication_validation as validation_ops
from phasesweep._metadata import __version__
from phasesweep.config.models import Experiment
from phasesweep.engine.errors import PublicationCommitError, RunRequestError
from phasesweep.engine.state import PUBLICATION_POINTER_SCHEMA_VERSION, Winner
from phasesweep.runtime.shutdown import absorb_shutdown_signals

if TYPE_CHECKING:
    from phasesweep.engine.run import PublicationHook

log = logging.getLogger(__name__)


def _claim_generation(experiment: Experiment, requested_id: str | None) -> str:
    """Exclusively claim a generation namespace under the experiment lock.

    Freeze the effective configuration and reproducibility record before returning;
    provenance failure aborts the claim. The record retains whether an external
    launcher supplied the generation id, even if its own state is later lost.

    :param Experiment experiment: Experiment whose generations root is created if missing.
    :param str | None requested_id: Caller-supplied generation id to claim, or
        ``None`` to mint a fresh random id.
    :return str: The claimed generation id (``requested_id`` if supplied and
        free, otherwise a freshly minted UUID4 hex string).
    :raises RunRequestError: ``requested_id`` already exists.
    :raises RuntimeError: No unused random id could be minted after 10 attempts,
        which indicates broken UUID generation rather than an operator conflict.
    :raises OSError: The generation namespace or its provenance files could not
        be created.
    """
    root = path_ops._generations_dir(experiment)
    root.mkdir(parents=True, exist_ok=True)
    if requested_id is not None:
        try:
            path_ops._generation_dir(experiment, requested_id).mkdir()
        except FileExistsError as exc:
            raise RunRequestError(
                f"Generation id {requested_id!r} already exists; refusing to overwrite history."
            ) from exc
        provenance_ops._write_generation_provenance(experiment, requested_id, caller_owned_id=True)
        return requested_id

    for _ in range(10):
        candidate = uuid4().hex
        try:
            path_ops._generation_dir(experiment, candidate).mkdir()
        except FileExistsError:
            continue
        provenance_ops._write_generation_provenance(experiment, candidate, caller_owned_id=False)
        return candidate
    raise RuntimeError("Could not mint an unused generation id after 10 attempts.")


_TERMINAL_GENERATION_STATES = frozenset({"published", "publication_failed", "failed"})


def _persist_terminal_failure(write_state: Callable[[], None]) -> None:
    """Persist failure bookkeeping without replacing the propagating exception.

    Log secondary failures, including control-flow exceptions, while preserving the
    primary error's identity, cleanup semantics, and exit status.

    :param Callable[[], None] write_state: Zero-argument persistence action.
    """
    try:
        write_state()
    except BaseException:  # noqa: BLE001
        log.exception("failed to persist terminal failure state; preserving the original error")


def _persist_failed_state_unless_recorded(
    record_path: Path, write_failed: Callable[[], None]
) -> None:
    """Write generic failed state unless publication failure is already recorded.

    :param Path record_path: Immutable per-generation record path to check.
    :param Callable[[], None] write_failed: Zero-argument "failed" state
        persistence action, run via :func:`_persist_terminal_failure`.
    """
    if not record_path.is_file():
        _persist_terminal_failure(write_failed)


def _log_on_failure(action: Callable[[], None], message: str) -> None:
    """Run post-commit bookkeeping, logging any failure without propagating it.

    Control-flow exceptions must not reclassify a committed result. The surrounding
    publication transaction also absorbs shutdown signals until commit bookkeeping ends.

    :param Callable[[], None] action: Zero-argument best-effort action.
    :param str message: Message logged (with exception info) on failure.
    """
    try:
        action()
    except BaseException:  # noqa: BLE001
        log.exception(message)


def _recorded_generation_state(record_path: Path) -> str | None:
    """Read one lifecycle record's state label, or ``None`` when unreadable.

    :param Path record_path: Immutable lifecycle record YAML path.
    :return str | None: The recorded ``state`` string, or ``None`` when the
        record is missing, unreadable, or malformed.
    """
    try:
        payload = yaml.safe_load(record_path.read_text())
    except (OSError, ValueError, yaml.YAMLError):
        return None
    if not isinstance(payload, dict):
        return None
    state = payload.get("state")
    return state if isinstance(state, str) else None


def _write_generation_record_once(
    *,
    record_path: Path,
    generation_id: str,
    state: str,
    payload: dict[str, Any],
    label: str,
) -> None:
    """Create an immutable terminal lifecycle record once.

    Only ``published``, ``publication_failed``, and ``failed`` states are recorded.
    Subsequent writes, including same-state rewrites, are refused and logged.

    :param Path record_path: Immutable per-generation lifecycle record path.
    :param str generation_id: Immutable generation namespace being recorded;
        used only for the refusal log message.
    :param str state: Terminal lifecycle state label being written.
    :param dict[str, Any] payload: Full record payload to persist.
    :param str label: Human label identifying the record kind in the refusal
        log message.
    """
    if artifact_io._write_yaml_exclusive(record_path, payload):
        return
    existing_state = _recorded_generation_state(record_path)
    log.warning(
        "Refusing to rewrite terminal %s %s record (existing state %r, attempted %r)",
        label,
        generation_id,
        existing_state,
        state,
    )


def _write_generation_state(
    experiment: Experiment,
    *,
    generation_id: str,
    state: str,
    from_phase: str | None,
    publish_current: bool,
    error_class: str | None = None,
    write_record: bool = True,
) -> None:
    """Write mutable current-generation progress and/or its immutable terminal record.

    The current pointer accepts every state and may restart at ``preflighting`` for
    a new invocation. Per-generation records accept terminal states only, once.

    :param Experiment experiment: Experiment whose generation state is written.
    :param str generation_id: Immutable generation namespace being recorded.
    :param str state: Lifecycle state label (``"preflighting"``, ``"running"``,
        ``"published"``, ``"publication_failed"``, or ``"failed"``).
    :param str | None from_phase: Resume point for this invocation, or ``None``.
    :param bool publish_current: If ``True``, also overwrite the experiment's
        current-generation pointer with this state.
    :param str | None error_class: Optional exception class name to record for
        a failed or publication_failed state.
    :param bool write_record: If ``False``, skip the write-once per-generation
        record even for a terminal state. Used by the post-commit
        current-pointer refresh, whose record was already written moments
        earlier -- re-attempting it would trip the write-once refusal warning
        on every successful publication.
    """
    payload = {
        "experiment": experiment.experiment,
        "generation_id": generation_id,
        "state": state,
        "from_phase": from_phase,
        "error_class": error_class,
        "phasesweep_version": __version__,
    }
    if write_record and state in _TERMINAL_GENERATION_STATES:
        _write_generation_record_once(
            record_path=path_ops._generation_record_path(experiment, generation_id),
            generation_id=generation_id,
            state=state,
            payload=payload,
            label="generation",
        )
    if publish_current:
        artifact_io._write_yaml_atomic(path_ops._generation_path(experiment), payload)


def _copy_yaml_projection(source: Path, destination: Path) -> None:
    """Atomically project one immutable YAML artifact to its convenience path.

    :param Path source: Immutable generation-scoped YAML file to read.
    :param Path destination: Convenience projection path to atomically overwrite.
    """
    payload = yaml.safe_load(source.read_text())
    artifact_io._write_yaml_atomic(destination, payload)


def _validate_publishable_summary(
    *,
    summary_path: Path,
    experiment_name: str,
    generation_id: str,
) -> tuple[dict[str, Any], bytes]:
    """Read a generation's summary before committing its last-success pointer.

    Require a mapping naming the expected experiment and generation, and return it
    for manifest validation without rereading. No terminal lifecycle record exists yet.

    :param Path summary_path: Immutable summary YAML path to read back.
    :param str experiment_name: Expected owning experiment name the summary must carry.
    :param str generation_id: Expected generation id the summary must name.
    :raises PublicationCommitError: The summary cannot be read back as a correctly
        named mapping; the last-success pointer must not advance to it.
    :return tuple[dict[str, Any], bytes]: Parsed summary and the exact bytes validated.
    """
    try:
        summary_bytes = summary_path.read_bytes()
        summary = yaml.safe_load(summary_bytes)
    except (OSError, yaml.YAMLError) as exc:
        raise PublicationCommitError(
            f"Generation {generation_id!r} summary could not be read back; "
            "refusing to advance the last-success pointer."
        ) from exc
    if (
        not isinstance(summary, dict)
        or summary.get("experiment") != experiment_name
        or summary.get("generation_id") != generation_id
    ):
        raise PublicationCommitError(
            f"Generation {generation_id!r} summary failed publication validation; "
            "refusing to advance the last-success pointer."
        )
    return summary, summary_bytes


def _validate_generation_publishable(
    experiment: Experiment, generation_id: str
) -> tuple[validation_ops.ValidatedPublication, bytes]:
    """Validate summary identity and the complete artifact manifest before commit.

    Carried winners may name their source generation; their ids need not equal the
    generation exposing them.

    :param Experiment experiment: Experiment whose generation is being published.
    :param str generation_id: Immutable generation namespace to validate.
    :raises PublicationCommitError: The generation summary cannot be read back
        or does not name this generation.
    :raises PublicationAccessError: A manifest artifact cannot be read as the current user.
    :raises PublicationIntegrityError: A manifest-listed artifact fails validation.
    :return tuple[ValidatedPublication, bytes]: Everything the validation read, and the
        exact summary bytes whose manifest was validated.
    """
    summary, summary_bytes = _validate_publishable_summary(
        summary_path=path_ops._generation_summary_path(experiment, generation_id),
        experiment_name=experiment.experiment,
        generation_id=generation_id,
    )
    publication = validation_ops._validate_generation_manifest(
        path_ops._generation_dir(experiment, generation_id),
        generation_id,
        summary,
    )
    return publication, summary_bytes


def _record_superseded_publication(experiment: Experiment, *, superseding_id: str) -> None:
    """Record the old pointer target's publication before replacing its only commit proof.

    A crash may leave a committed pointer without its best-effort lifecycle record.
    Write that missing record first, preserving any existing record. The unavailable
    original resume point is recorded as ``from_phase=None``; artifact validation
    remains the reader's responsibility.

    :param Experiment experiment: Experiment whose pointer is about to move.
    :param str superseding_id: Generation about to replace the pointer's target.
    :raises PublicationAccessError: The current pointer cannot be read by the current user.
    :raises OSError: The record could not be written; the caller must not move the pointer.
    """
    target = validation_ops._read_pointer_target(
        path_ops._last_successful_generation_path(experiment),
        experiment_name=experiment.experiment,
    )
    if target is None or target.generation_id == superseding_id:
        return
    if path_ops._generation_record_path(experiment, target.generation_id).exists():
        return
    _write_generation_state(
        experiment,
        generation_id=target.generation_id,
        state="published",
        from_phase=None,
        publish_current=False,
    )


def _publish_generation(
    experiment: Experiment,
    generation_id: str,
    *,
    from_phase: str | None,
    winners: Mapping[str, Winner],
    publication_hook: PublicationHook | None,
) -> None:
    """Commit a generation as the experiment's last successful publication.

    Under an absorbed-shutdown-signal window:

    1. Validate the already-written immutable summary and artifact manifest.
    2. Record the prior pointer target as published if its lifecycle record is missing.
    3. Require any detached-runner hook to durably prepare the frozen result.
    4. Atomically commit ``last_successful_generation.yaml``.

    A failure before commit preserves the prior pointer, records ``publication_failed``
    best-effort, and reraises the original error. The pointer write is the authoritative
    publication event.

    After commit, independently notify the sidecar, create the published lifecycle
    record, refresh the current pointer, and project convenience root artifacts.
    Each step logs its failures without failing the run or preventing subsequent
    steps. Readers use immutable generation artifacts, not convenience projections.

    A shutdown before this window cancels without publication. One received inside
    is deferred until durable classification, then honored at the next launch or
    runner-terminal checkpoint before further work starts.

    :param Experiment experiment: Experiment whose generation is being published.
    :param str generation_id: Immutable generation namespace to publish.
    :param str | None from_phase: Resume point recorded on the lifecycle state.
    :param Mapping[str, Winner] winners: Engine-selected winners to hand to a
        required publication sidecar.
    :param PublicationHook | None publication_hook: Optional detached-run sidecar.
    :raises Exception: Whatever steps 2 through 5 raised, after best-effort
        "publication_failed" bookkeeping.
    """
    with absorb_shutdown_signals() as absorbed:
        try:
            publication, summary_bytes = _validate_generation_publishable(experiment, generation_id)
            # Before the pointer stops naming its current target, which may
            # have no record yet if its invocation died after the commit.
            _record_superseded_publication(experiment, superseding_id=generation_id)
            if publication_hook is not None:
                publication_hook.prepare(
                    experiment=experiment,
                    generation_id=generation_id,
                    winners=MappingProxyType(dict(winners)),
                    summary=MappingProxyType(publication.summary),
                )
            artifact_io._write_yaml_atomic(
                path_ops._last_successful_generation_path(experiment),
                {
                    "schema_version": PUBLICATION_POINTER_SCHEMA_VERSION,
                    "experiment": experiment.experiment,
                    "generation_id": generation_id,
                    "summary_size_bytes": len(summary_bytes),
                    "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                },
            )
        except BaseException as exc:
            error_class = type(exc).__name__
            _persist_terminal_failure(
                lambda: _write_generation_state(
                    experiment,
                    generation_id=generation_id,
                    state="publication_failed",
                    from_phase=from_phase,
                    publish_current=True,
                    error_class=error_class,
                )
            )
            raise

        if publication_hook is not None:
            _log_on_failure(
                lambda: publication_hook.committed(generation_id=generation_id),
                "failed to record the detached-run publication receipt after the "
                "last-success pointer committed; the prepared snapshot remains recoverable",
            )

        def _write_published_record() -> None:
            """Create the immutable per-generation published lifecycle record once."""
            _write_generation_state(
                experiment,
                generation_id=generation_id,
                state="published",
                from_phase=from_phase,
                publish_current=False,
            )

        _log_on_failure(
            _write_published_record,
            "failed to write the immutable generation record after publication; "
            "the published result is unaffected",
        )

        def _refresh_published_pointer_and_projections() -> None:
            """Drive the current pointer and refresh convenience projections."""
            _write_generation_state(
                experiment,
                generation_id=generation_id,
                state="published",
                from_phase=from_phase,
                publish_current=True,
                write_record=False,
            )
            for phase in experiment.phases:
                source_winner = path_ops._generation_winner_path(
                    experiment, generation_id, phase.name
                )
                projected_winner = path_ops._winner_path(experiment, phase.name)
                if source_winner.is_file():
                    _copy_yaml_projection(source_winner, projected_winner)
                else:
                    projected_winner.unlink(missing_ok=True)

            _copy_yaml_projection(
                path_ops._generation_summary_path(experiment, generation_id),
                path_ops._summary_path(experiment),
            )

        _log_on_failure(
            _refresh_published_pointer_and_projections,
            "failed to refresh the current-generation pointer or convenience projections "
            "after publication; the published result is unaffected",
        )

    if absorbed.signum is not None:
        log.warning(
            "Shutdown signal %d arrived during the publication transaction for "
            "generation %s; the committed publication wins and the shutdown is "
            "honored at the next checkpoint before any new work starts.",
            absorbed.signum,
            generation_id,
        )
