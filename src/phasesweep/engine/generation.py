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
from phasesweep.config import Experiment
from phasesweep.engine.errors import PublicationCommitError, RunRequestError
from phasesweep.engine.state import PUBLICATION_POINTER_SCHEMA_VERSION, Winner
from phasesweep.runtime.process import absorb_shutdown_signals

if TYPE_CHECKING:
    from phasesweep.engine.run import PublicationHook

log = logging.getLogger("phasesweep.engine.run")


def _claim_generation(experiment: Experiment, requested_id: str | None) -> str:
    """Create one exclusively owned generation namespace under the experiment lock.

    The claim is only complete once the namespace records the configuration
    that is about to run: :func:`phasesweep.engine.provenance._write_generation_provenance`
    freezes ``config.snapshot.yaml`` and ``reproducibility.json`` before this
    returns (review v0.5.18 / finding F6), so a generation that later fails
    preflight, execution, or publication still says what search spaces, fixed
    overrides, contracts, env, and trial command produced it. A failure
    writing them fails the claim rather than starting a run whose
    configuration would be unrecoverable. The record also freezes whether the
    id was caller-supplied, so a generation launched under an external
    authority grant (a detached MCP run) stays recognizable as such even if
    the launcher's own state is later lost (PR #5 review / P2 missing-handle
    authority).

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
    """Persist failed-state bookkeeping without replacing the active exception.

    Bookkeeping is never more authoritative than the failure it records: the
    original exception carries safety-critical semantics (signal exit codes,
    cleanup uncertainty, trainer failure identity) that a secondary filesystem
    error must not overwrite (review v0.5.14 / blocker 5). Catching
    ``BaseException`` is intentional in this narrow helper — even a second
    control-flow exception (another SIGINT/SIGTERM, a nested ``SystemExit``)
    delivered during persistence must not supersede the primary failure that
    is already propagating.

    :param Callable[[], None] write_state: Zero-argument persistence action.
    """
    try:
        write_state()
    except BaseException:  # noqa: BLE001 - see docstring: primary exception must survive
        log.exception("failed to persist terminal failure state; preserving the original error")


def _persist_failed_state_unless_recorded(
    record_path: Path, write_failed: Callable[[], None]
) -> None:
    """Write the generic "failed" state, unless a more specific record already exists.

    If a ``publication_failed`` record was already written for this
    generation, the generic ``failed`` state write must not run: the current
    pointer was already driven terminal with the more specific outcome, and
    the write-once record must not be re-attempted on top of it.

    :param Path record_path: Immutable per-generation record path to check.
    :param Callable[[], None] write_failed: Zero-argument "failed" state
        persistence action, run via :func:`_persist_terminal_failure`.
    """
    if not record_path.is_file():
        _persist_terminal_failure(write_failed)


def _log_on_failure(action: Callable[[], None], message: str) -> None:
    """Run a post-commit best-effort action, logging instead of raising on failure.

    Exists for the post-commit best-effort steps of the publication
    transaction (record write, pointer refresh, compatibility projection):
    once the authoritative pointer has committed, nothing after it may fail
    the run, so each such step must log its own failure and never propagate
    it. Catching ``BaseException`` is intentional (review v0.5.16 / blocker
    1): a ``KeyboardInterrupt``/``SystemExit`` escaping one of these steps
    would propagate into the caller's terminal-failure handler and reclassify
    an already-committed publication as failed. Real shutdown *signals* are
    additionally kept out of these steps entirely by the enclosing
    :func:`phasesweep.runtime.process.absorb_shutdown_signals` window.

    :param Callable[[], None] action: Zero-argument best-effort action.
    :param str message: Message logged (with exception info) on failure.
    """
    try:
        action()
    except BaseException:  # noqa: BLE001 - see docstring: commit outcome must survive
        log.exception(message)


def _recorded_generation_state(record_path: Path) -> str | None:
    """Read one lifecycle record's state label, or ``None`` when unreadable.

    :param Path record_path: Immutable lifecycle record YAML path.
    :return str | None: The recorded ``state`` string, or ``None`` when the
        record is missing, unreadable, or malformed.
    """
    try:
        payload = yaml.safe_load(record_path.read_text())
    except (OSError, yaml.YAMLError):
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
    """Create one generation's immutable terminal record exactly once.

    Shared write-once core for :func:`_write_generation_state` and
    :func:`phasesweep.engine.suite._write_suite_generation_state` (review
    v0.5.15 / blocker 3, item B): the per-generation record is written only for a terminal state
    (``published`` / ``publication_failed`` / ``failed``), and only ever
    once. A second attempt for any generation -- even a same-state rewrite --
    is refused and logged; the first content is never touched. Progress
    states (``preflighting`` / ``running``) never reach this function at all;
    see the ``state in _TERMINAL_GENERATION_STATES`` guard in the callers.

    :param Path record_path: Immutable per-generation lifecycle record path.
    :param str generation_id: Immutable generation (or suite generation)
        namespace being recorded; used only for the refusal log message.
    :param str state: Terminal lifecycle state label being written.
    :param dict[str, Any] payload: Full record payload to persist.
    :param str label: Human label identifying the record kind in the refusal
        log message (e.g. ``"generation"``, ``"suite generation"``).
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
    """Write one generation's current-pointer projection and/or immutable record.

    The current pointer (``generation.yaml``) is a mutable progress record --
    every state (``preflighting``, ``running``, ``published``,
    ``publication_failed``, ``failed``) may be written to it, with no
    monotonic guard: a new invocation legitimately overwrites it starting from
    ``preflighting`` (review v0.5.15 / blocker 3). The per-generation record
    under ``generations/<id>/generation.yaml`` is truly immutable: it is only
    ever written for a terminal state, and only once (see
    :func:`_write_generation_record_once`); progress states never touch it.

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
    """Atomically project one immutable YAML artifact to its compatibility path.

    :param Path source: Immutable generation-scoped YAML file to read.
    :param Path destination: Legacy compatibility path to atomically overwrite.
    """
    payload = yaml.safe_load(source.read_text())
    artifact_io._write_yaml_atomic(destination, payload)


def _validate_publishable_summary(
    *,
    summary_path: Path,
    owner_key: str,
    owner_value: str,
    id_key: str,
    id_value: str,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    """Parse back one generation's own immutable summary before its publication commit.

    Shared pre-commit validation core (item A, review v0.5.15) for
    :func:`_validate_generation_publishable` and
    :func:`phasesweep.engine.suite._validate_suite_generation_publishable`. This runs *before* the
    last-success pointer commits and before the per-generation lifecycle
    record is ever written for this generation, so it cannot check that
    record (it does not exist yet); it instead confirms the immutable summary
    itself parses as a mapping naming the expected owner and id, and returns
    it so the caller can validate the complete manifest without re-reading.

    :param Path summary_path: Immutable summary YAML path to read back.
    :param str owner_key: Summary key naming the owning experiment or suite.
    :param str owner_value: Expected owner name the summary must carry.
    :param str id_key: Summary key holding the generation (or suite generation) id.
    :param str id_value: Expected id the summary must name.
    :param str label: Human label for error text (e.g. ``"Generation"``,
        ``"Suite generation"``).
    :raises PublicationCommitError: The summary cannot be read back as a correctly
        named mapping; the last-success pointer must not advance to it.
    :return tuple[dict[str, Any], bytes]: Parsed summary and the exact bytes validated.
    """
    try:
        summary_bytes = summary_path.read_bytes()
        summary = yaml.safe_load(summary_bytes)
    except (OSError, yaml.YAMLError) as exc:
        raise PublicationCommitError(
            f"{label} {id_value!r} summary could not be read back; "
            "refusing to advance the last-success pointer."
        ) from exc
    if (
        not isinstance(summary, dict)
        or summary.get(owner_key) != owner_value
        or summary.get(id_key) != id_value
    ):
        raise PublicationCommitError(
            f"{label} {id_value!r} summary failed publication validation; "
            "refusing to advance the last-success pointer."
        )
    return summary, summary_bytes


def _validate_generation_publishable(experiment: Experiment, generation_id: str) -> bytes:
    """Validate a generation's complete result manifest before its publication commit.

    Checks the generation's immutable summary names this exact experiment and
    generation id, then validates the versioned artifact manifest end to end
    (:func:`phasesweep.engine.publication_validation._validate_generation_manifest`, review
    v0.5.16 / blocker 3): every winner and promotion artifact the summary
    claims must exist, hash to its recorded content, parse, and cross-check
    against the summary's own winner facts (trial number, metric name, value,
    goal, source identity fields, completion metadata) — and the namespace
    must hold nothing the manifest does not list. A winner's own
    ``generation_id`` is deliberately *not* required to equal
    ``generation_id``: a winner is legitimately carried forward from
    whichever generation actually produced the best trial, so the field is
    checked for well-formedness only.

    :param Experiment experiment: Experiment whose generation is being published.
    :param str generation_id: Immutable generation namespace to validate.
    :raises PublicationCommitError: The generation summary cannot be read back
        or does not name this generation.
    :raises PublicationAccessError: A manifest artifact cannot be read as the current user.
    :raises PublicationIntegrityError: A manifest-listed artifact fails validation.
    :return bytes: Exact summary bytes whose manifest was validated.
    """
    summary, summary_bytes = _validate_publishable_summary(
        summary_path=path_ops._generation_summary_path(experiment, generation_id),
        owner_key="experiment",
        owner_value=experiment.experiment,
        id_key="generation_id",
        id_value=generation_id,
        label="Generation",
    )
    validation_ops._validate_generation_manifest(
        path_ops._generation_dir(experiment, generation_id),
        generation_id,
        summary,
    )
    return summary_bytes


def _publish_generation(
    experiment: Experiment,
    generation_id: str,
    *,
    from_phase: str | None,
    winners: Mapping[str, Winner],
    publication_hook: PublicationHook | None,
) -> None:
    """Publish a generation as the experiment's last successful result.

    This is the publication transaction (review v0.5.15 / blocker 3), in
    strict order:

    1. Immutable generation-scoped artifacts (winners/promotions/summary) are
       already written by the time this runs.
    2. Pre-commit validation (:func:`_validate_generation_publishable`, item
       A): parse this generation's own summary and winner files back and
       confirm they name themselves correctly.
    3. When supplied, require the detached runner's sidecar hook to durably
       prepare its frozen result.
    4. Commit ``last_successful_generation.yaml`` atomically -- the single
       authoritative publication event. Failures in either preceding step
       leave the prior pointer untouched.

    If step 2 or 3 raises, the immutable per-generation record is written
    once with state ``"publication_failed"`` (+ ``error_class``), the current
    pointer is driven to ``"publication_failed"``, the prior last-success
    pointer is left untouched (still authoritative), and the original
    exception re-raises. Persisting that bookkeeping is itself best-effort
    (:func:`_persist_terminal_failure`): a secondary failure while writing it
    is logged, never substituted for the primary error.

    Once step 4 has committed, nothing after it may fail the run:

    5. Best-effort, notify the prepared sidecar that the pointer committed.
    6. Write the immutable per-generation record once, state ``"published"``.
    7. Best-effort, diagnostic-only: drive the current pointer to
       ``"published"``, then refresh the legacy compatibility projections
       (root ``winner.yaml`` / ``promotion.yaml`` / ``summary.yaml``; review
       v0.5.15 / item D). These are post-commit caches for humans and legacy
       tooling only -- no reader re-derives them, and once any generation has
       published, reads resolve the generation-scoped artifacts directly (see
       :func:`phasesweep.engine.publication._published_winner_path_for`), so a
       failure projecting them never affects what callers actually see.

    Steps 5 through 7 each independently log and swallow their own failure so one
    cannot prevent the other from running.

    The whole transaction runs inside an
    :func:`phasesweep.runtime.process.absorb_shutdown_signals` window (review
    v0.5.16 / blocker 1): a shutdown signal that arrives after the pointer
    commit must not reclassify the committed publication as failed or
    cancelled. The race has a deterministic winner — a shutdown delivered
    before this window opens cancels the run with nothing published; one
    delivered inside it is absorbed until the publication is durably
    classified and then honored at the next checkpoint (the suite loop, the
    next trial launch, or the MCP runner's terminal status write) before any
    new work starts.

    :param Experiment experiment: Experiment whose generation is being published.
    :param str generation_id: Immutable generation namespace to publish.
    :param str | None from_phase: Resume point recorded on the lifecycle state.
    :param Mapping[str, Winner] winners: Engine-selected winners to hand to a
        required publication sidecar.
    :param PublicationHook | None publication_hook: Optional detached-run sidecar.
    :raises Exception: Whatever steps 2 through 4 raised, after best-effort
        "publication_failed" bookkeeping.
    """
    with absorb_shutdown_signals() as absorbed:
        try:
            summary_bytes = _validate_generation_publishable(experiment, generation_id)
            if publication_hook is not None:
                publication_hook.prepare(
                    experiment=experiment,
                    generation_id=generation_id,
                    winners=MappingProxyType(dict(winners)),
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
            """Step 6: write the immutable per-generation record once, state ``published``."""
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
            """Step 7: drive the pointer to ``published``, refresh legacy projections (best-effort)."""
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

                source_promotion = path_ops._generation_promotion_decision_path(
                    experiment, generation_id, phase.name
                )
                projected_promotion = path_ops._promotion_decision_path(experiment, phase.name)
                if source_promotion.is_file():
                    _copy_yaml_projection(source_promotion, projected_promotion)
                else:
                    projected_promotion.unlink(missing_ok=True)

            _copy_yaml_projection(
                path_ops._generation_summary_path(experiment, generation_id),
                path_ops._summary_path(experiment),
            )

        _log_on_failure(
            _refresh_published_pointer_and_projections,
            "failed to refresh the current-generation pointer or compatibility caches "
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
