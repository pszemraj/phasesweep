"""Experiment generation claims, lifecycle records, and publication writes."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import yaml

import phasesweep.engine.artifacts as artifact_io
import phasesweep.engine.paths as path_ops
import phasesweep.engine.provenance as provenance_ops
import phasesweep.engine.publication as publication_ops
import phasesweep.engine.publication_validation as validation_ops
from phasesweep._metadata import __version__
from phasesweep.config.models import Experiment
from phasesweep.engine.errors import PublicationCommitError, RunRequestError
from phasesweep.engine.fingerprints import _recorded_config_matches
from phasesweep.engine.state import PUBLICATION_POINTER_SCHEMA_VERSION, PhaseCounts, Winner
from phasesweep.runtime.files import atomic_create_text
from phasesweep.runtime.shutdown import absorb_shutdown_signals, signal_name

if TYPE_CHECKING:
    from phasesweep.engine.run import PublicationHook

log = logging.getLogger(__name__)


def _new_generation_id(experiment: Experiment, requested_id: str | None) -> str:
    """Choose an invocation's generation id without creating its namespace.

    :param Experiment experiment: Experiment whose generations root is checked.
    :param str | None requested_id: Caller-supplied id, or ``None`` to mint one.
    :return str: ``requested_id``, or a fresh UUID4 hex string.
    :raises RunRequestError: ``requested_id`` already names a generation.
    """
    if requested_id is None:
        return uuid4().hex
    if path_ops._generation_dir(path_ops._experiment_dir(experiment), requested_id).exists():
        raise RunRequestError(
            f"Generation id {requested_id!r} already exists; refusing to overwrite history."
        )
    return requested_id


def _claim_generation(experiment: Experiment, generation_id: str, *, caller_owned_id: bool) -> None:
    """Exclusively create a generation namespace under the experiment lock.

    Freeze the effective configuration and reproducibility record before returning;
    provenance failure aborts the claim. The record retains whether an external
    launcher supplied the generation id, even if its own state is later lost.

    :param Experiment experiment: Experiment whose generations root is created if missing.
    :param str generation_id: Id from :func:`_new_generation_id`.
    :param bool caller_owned_id: Whether the caller supplied ``generation_id``.
    :raises OSError: The generation namespace already exists, or it or its
        provenance files could not be created.
    """
    artifact_root = path_ops._experiment_dir(experiment)
    path_ops._generations_dir(artifact_root).mkdir(parents=True, exist_ok=True)
    path_ops._generation_dir(artifact_root, generation_id).mkdir()
    provenance_ops._write_generation_provenance(
        experiment, generation_id, caller_owned_id=caller_owned_id
    )


class InvocationGeneration:
    """One invocation's generation: an id from the start, a namespace only once work needs it.

    :meth:`claim` creates the namespace and points the current-generation pointer at it.
    The engine calls it before the first trial allocation, or before publishing winners
    the current publication does not already represent under the current config
    (:func:`_publication_holds`). Winners saved before the claim stay in memory and are
    written by it, so an invocation that launches nothing and publishes nothing leaves no
    generation. A matching replay makes the published generation current.
    """

    def __init__(
        self,
        experiment: Experiment,
        *,
        generation_id: str,
        caller_owned_id: bool,
        from_phase: str | None,
    ) -> None:
        """Hold an unclaimed generation.

        :param Experiment experiment: Experiment the generation belongs to.
        :param str generation_id: Id from :func:`_new_generation_id`.
        :param bool caller_owned_id: Whether the caller supplied ``generation_id``.
        :param str | None from_phase: Resume point recorded on the lifecycle state.
        """
        self.experiment = experiment
        self.generation_id = generation_id
        self.claimed = False
        self.replayed_generation_id: str | None = None
        #: Whether the invocation claimed its generation only to publish, launching no trial.
        self.published_without_launch = False
        #: Trial counts of each phase this invocation ran, recorded as the phase ends.
        self.phase_counts: dict[str, PhaseCounts] = {}
        self._caller_owned_id = caller_owned_id
        self._from_phase = from_phase
        self._pending_winners: dict[str, Winner] = {}

    def claim(self) -> str:
        """Create the namespace, mark it running and write held winners, on first call only.

        :return str: The generation id.
        :raises OSError: The namespace already exists, or it, its provenance, its
            running state or a held winner could not be written.
        """
        if not self.claimed:
            _claim_generation(
                self.experiment, self.generation_id, caller_owned_id=self._caller_owned_id
            )
            self.claimed = True
            _write_generation_state(
                self.experiment,
                generation_id=self.generation_id,
                state="running",
                from_phase=self._from_phase,
                publish_current=True,
            )
            for phase_name, winner in self._pending_winners.items():
                artifact_io._save_winner(
                    self.experiment, phase_name, winner, generation_id=self.generation_id
                )
            self._pending_winners.clear()
        return self.generation_id

    def replay(self, published_id: str) -> None:
        """Make the verified publication current after a replay of all its winners.

        :param str published_id: Matching publication from :func:`_publication_holds`.
        """
        current = artifact_io.read_yaml_or_none(path_ops._generation_path(self.experiment))
        if not (
            isinstance(current, dict)
            and current.get("generation_id") == published_id
            and current.get("state") == "published"
        ):
            # The immutable lifecycle record can be missing after a post-publication crash.
            record = artifact_io.read_yaml_or_none(
                path_ops._generation_record_path(self.experiment, published_id)
            )
            from_phase = record.get("from_phase") if isinstance(record, dict) else None
            _write_generation_state(
                self.experiment,
                generation_id=published_id,
                state="published",
                from_phase=from_phase if isinstance(from_phase, str) else None,
                publish_current=True,
                write_record=False,
            )
        self.replayed_generation_id = published_id

    def save_winner(self, phase_name: str, winner: Winner) -> None:
        """Write a phase winner into the claimed namespace, or hold it until the claim.

        :param str phase_name: Phase the winner belongs to.
        :param Winner winner: The phase's winner.
        """
        if self.claimed:
            artifact_io._save_winner(
                self.experiment, phase_name, winner, generation_id=self.generation_id
            )
        else:
            self._pending_winners[phase_name] = winner


def _publication_holds(experiment: Experiment, winners: Mapping[str, Winner]) -> str | None:
    """Return the published generation that already represents these winners under this config.

    The config is judged by :func:`_recorded_config_matches`, the comparison every reader of
    the publication makes, so a run replays exactly when ``status`` and ``show-winners`` would
    call the publication current. Equal winners under an edit the experiment fingerprint
    covers, such as a bound that selects the same trials, are published again instead.

    :param Experiment experiment: Experiment whose last-success pointer is resolved.
    :param Mapping[str, Winner] winners: Winners an invocation would publish.
    :return str | None: The published generation id when its publication validates, its
        recorded fingerprint matches ``experiment`` and its winner files hold exactly these
        phases with these payloads, else ``None``.
    """
    validated = publication_ops._resolve_publication_pointer(experiment).validated
    if validated is None or set(validated.winners) != set(winners):
        return None
    if not _recorded_config_matches(
        experiment,
        validated.summary.get("config_fingerprint"),
        phase_count=len(validated.phase_plan),
    ):
        return None
    for phase_name, winner in winners.items():
        payload = artifact_io._winner_payload(experiment, phase_name, winner)
        if yaml.safe_load(yaml.safe_dump(payload)) != validated.winners[phase_name].payload:
            return None
    return validated.generation_id


_TERMINAL_GENERATION_STATES = frozenset({"published", "publication_failed", "failed"})


def _persist_failed_state_unless_recorded(
    record_path: Path, write_failed: Callable[[], None]
) -> None:
    """Write generic failed state unless publication failure is already recorded.

    :param Path record_path: Immutable per-generation record path to check.
    :param Callable[[], None] write_failed: Zero-argument "failed" state
        persistence action, run via :func:`_log_on_failure`.
    """
    if not record_path.is_file():
        _log_on_failure(
            write_failed,
            "failed to persist terminal failure state; preserving the original error",
        )


def _log_on_failure(action: Callable[[], None], message: str) -> None:
    """Run best-effort bookkeeping, logging any failure without propagating it.

    Preserve the primary error or committed outcome even for control-flow exceptions.

    :param Callable[[], None] action: Zero-argument best-effort action.
    :param str message: Message logged (with exception info) on failure.
    """
    try:
        action()
    except BaseException:
        log.exception(message)


@dataclass(frozen=True)
class _RecordedGeneration:
    """The terminal lifecycle state, or why a record that exists could not be read."""

    state: str | None
    error_class: str | None
    problem: str | None = None


def _recorded_generation(record_path: Path) -> _RecordedGeneration:
    """Read one lifecycle record's state label and failure class.

    :param Path record_path: Immutable lifecycle record YAML path.
    :return _RecordedGeneration: The state and error class, or a result with no state
        if the file does not exist or it cannot supply a valid terminal state. A missing
        file has no problem recorded.
    """
    try:
        payload: object = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _RecordedGeneration(None, None)
    except OSError:
        return _RecordedGeneration(None, None, "unreadable")
    except (ValueError, yaml.YAMLError):
        return _RecordedGeneration(None, None, "malformed")
    if isinstance(payload, dict) and "error_class" in payload:
        state, error_class = payload.get("state"), payload["error_class"]
        if (
            isinstance(state, str)
            and state in _TERMINAL_GENERATION_STATES
            and (error_class is None or isinstance(error_class, str))
        ):
            return _RecordedGeneration(state, error_class)
    return _RecordedGeneration(None, None, "malformed")


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

    The current pointer accepts every state and restarts at ``running`` for a newly
    claimed generation. Per-generation records accept terminal states only, once.
    Subsequent record writes, including same-state rewrites, are refused and logged.

    :param Experiment experiment: Experiment whose generation state is written.
    :param str generation_id: Immutable generation namespace being recorded.
    :param str state: Lifecycle state label (``"running"``, ``"published"``,
        ``"publication_failed"``, or ``"failed"``).
    :param str | None from_phase: Resume point for this invocation, or ``None``.
    :param bool publish_current: If ``True``, also overwrite the experiment's
        current-generation pointer with this state.
    :param str | None error_class: Optional exception class name to record for
        a failed or publication_failed state.
    :param bool write_record: Skip the terminal record when ``False``, as during
        post-commit current-pointer refresh.
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
        record_path = path_ops._generation_record_path(experiment, generation_id)
        if not atomic_create_text(record_path, yaml.safe_dump(payload, sort_keys=False)):
            log.warning(
                "Refusing to rewrite terminal generation %s record (existing state %r, attempted %r)",
                generation_id,
                _recorded_generation(record_path).state,
                state,
            )
    if publish_current:
        artifact_io._write_yaml_atomic(path_ops._generation_path(experiment), payload)


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
        path_ops._generation_dir(path_ops._experiment_dir(experiment), generation_id),
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

    Validate immutable artifacts, save any missing prior publication record and prepare
    the runner hook before atomically replacing ``last_successful_generation.yaml``.
    This write commits publication; earlier failures retain the old pointer and record
    ``publication_failed`` best-effort. Later sidecar, lifecycle and current-pointer
    updates are best-effort with logged failures; readers use immutable artifacts.

    Shutdown before this window cancels publication. Signals absorbed inside it are
    serviced after durable classification, at the next launch or runner checkpoint.

    :param Experiment experiment: Experiment whose generation is being published.
    :param str generation_id: Immutable generation namespace to publish.
    :param str | None from_phase: Resume point recorded on the lifecycle state.
    :param Mapping[str, Winner] winners: Engine-selected winners for the publication sidecar.
    :param PublicationHook | None publication_hook: Optional detached-run sidecar.
    :raises Exception: Pre-commit failure, after best-effort failure bookkeeping.
    """
    with absorb_shutdown_signals() as absorbed:
        try:
            publication, summary_bytes = _validate_generation_publishable(experiment, generation_id)
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
            _log_on_failure(
                lambda: _write_generation_state(
                    experiment,
                    generation_id=generation_id,
                    state="publication_failed",
                    from_phase=from_phase,
                    publish_current=True,
                    error_class=error_class,
                ),
                "failed to persist terminal failure state; preserving the original error",
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

        def _refresh_published_pointer() -> None:
            """Drive the current pointer to the committed publication."""
            _write_generation_state(
                experiment,
                generation_id=generation_id,
                state="published",
                from_phase=from_phase,
                publish_current=True,
                write_record=False,
            )

        _log_on_failure(
            _refresh_published_pointer,
            "failed to refresh the current-generation pointer "
            "after publication; the published result is unaffected",
        )

    if absorbed.signum is not None:
        log.warning(
            "Shutdown signal %s arrived while publishing generation %s; the publication "
            "committed and the run stops at the next checkpoint.",
            signal_name(absorbed.signum),
            generation_id,
        )
