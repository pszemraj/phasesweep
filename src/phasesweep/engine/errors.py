"""Typed engine failures used for safe orchestration decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from phasesweep.errors import (
    FRESH_STUDY,
    PERSISTENT_STUDY_KEEPS_TRIALS,
    STORAGE_NULL_KEEPS_NOTHING,
    OperatorAction,
    PhaseSweepError,
)

if TYPE_CHECKING:
    from phasesweep.config.models import Experiment


class StudyFingerprintMismatchError(PhaseSweepError):
    """Raised when stored study or winner semantics do not match the current config."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FRESH_NAMESPACE


class ResumeBoundsChangedError(StudyFingerprintMismatchError):
    """Raised when skipping a phase would reuse a winner selected under different bounds."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


class StudySchemaMismatchError(PhaseSweepError):
    """Raised when populated storage uses an unsupported PhaseSweep schema."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FRESH_NAMESPACE


class ArtifactRootConflictError(PhaseSweepError):
    """Raised when a study or artifact tree names a different publication owner."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FRESH_NAMESPACE


class ArtifactRootSchemaMismatchError(ArtifactRootConflictError):
    """Raised when an artifact root records an unsupported PhaseSweep schema."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FRESH_NAMESPACE


class LaunchEnvironmentError(PhaseSweepError):
    """Raised when no trial could start in the configured environment, before one is allocated.

    ``execution.cwd`` is not a directory, the trial command's program cannot be found, or the
    trainer environment cannot be composed. The configuration or the environment it runs in
    holds the fix, so every raise routes to ``FIX_CONFIG``.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


class PublicationIntegrityError(PhaseSweepError):
    """Raised when a recorded publication exists but fails validation.

    This is distinct from an absent publication.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class PublicationAccessError(PhaseSweepError):
    """Raised when the current user cannot validate a publication.

    Permission denial does not establish corruption; no published result is exposed.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class PublicationCommitError(PhaseSweepError):
    """Raised when a completed generation fails its pre-commit publication validation."""

    default_action: ClassVar[OperatorAction] = OperatorAction.INSPECT_LOGS


class WinnerIntegrityError(PhaseSweepError):
    """Raised when a saved result is unreadable, malformed, or cannot be reused.

    Includes ambiguous source identity and disallowed partial-result replay.
    Fingerprint drift uses :class:`StudyFingerprintMismatchError`.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class ResumeWinnerMissingError(PhaseSweepError):
    """Raised when a resume skips a phase whose winner the publication does not hold."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


class RunRequestError(PhaseSweepError):
    """Raised when a caller requests an unsupported or conflicting run identity."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


class TrialEvidenceMissingError(PhaseSweepError):
    """Raised to refuse selection of a trial whose recorded evidence is missing."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class StudyStorageUnavailableError(PhaseSweepError):
    """Raised when required persistent study state cannot be inspected or persisted."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_LEDGER


class IncompleteJournalRecordError(StudyStorageUnavailableError):
    """Raised when a partial final journal record cannot be repaired before writing.

    Causes include lock timeout or replacement, backup or truncation failure,
    and journal changes during repair. Intact records remain;
    a lock still held after the repair's wait, lock replacement or concurrent changes carry
    ``OperatorAction.RETRY``.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_LEDGER


class PublishedStudyMissingError(PhaseSweepError):
    """Raised before launch when a published phase's local trial is absent or replaced."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_LEDGER


class ActiveAttemptPersistenceError(PhaseSweepError):
    """Raised when required recovery state cannot persist before launch or GPU acquisition."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class ExperimentLockBusyError(PhaseSweepError):
    """Raised when another orchestrator owns an experiment consistency lock."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RETRY


class PhaseTimeoutError(PhaseSweepError, TimeoutError):
    """A phase or whole-run wallclock budget stopped phase work."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


def phase_timeout(message: str, *, scope: str | None, experiment: Experiment) -> PhaseTimeoutError:
    """Attach the budget edit and the storage-appropriate rerun behavior.

    :param str message: Diagnostic describing where the deadline stopped work.
    :param str | None scope: ``run`` for the run guard, otherwise the phase guard.
    :param Experiment experiment: Experiment whose storage decides whether completed trials
        persist for the next invocation.
    :return PhaseTimeoutError: Routed timeout shared by CLI, run logs, and MCP failures.
    """
    key = "timeout_seconds_per_run" if scope == "run" else "timeout_seconds_per_phase"
    # The step names neither the config nor `--set`: an MCP launch has no `--set`, and a CLI
    # rerun carries the override that set the key.
    kept = PERSISTENT_STUDY_KEEPS_TRIALS if experiment.persistent else STORAGE_NULL_KEEPS_NOTHING
    return PhaseTimeoutError(message, remediation=f"Raise {key} and rerun; {kept}.")


class TrialTargetRefusedError(PhaseSweepError):
    """Raised when a phase target contradicts its durable study state.

    ``n_trials`` cannot fall below the accepted target. A fixed configuration may
    retry failures only until one attempt completes, feasible or not. A grid
    applies that limit per cell when its target exceeds the matrix size.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


class TrackingTargetUnavailableError(PhaseSweepError):
    """Raised when a tracker gives no answer that new work or completed training needs.

    Before a phase allocates trials, its MLflow target must resolve. In a persistent
    study, a trial that finished training while its tracker could not be read stays
    RUNNING, its training kept, and the run stops until the tracker answers.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


TRACKER_UNREADABLE_STEP = (
    "Restore access to the tracker or install its SDK, then run the same config again; the "
    f"finished training is kept. If the tracker is gone for good, {FRESH_STUDY}."
)


def tracker_unreadable(message: str) -> TrackingTargetUnavailableError:
    """Attach the step that fixes the tracker read to a stop that keeps finished training pending.

    The stop is a retry: the tracker answering again is all the same command needs.

    :param str message: Diagnostic naming the pending trials and the tracker's failure.
    :return TrackingTargetUnavailableError: Refusal shared by the live run and recovery.
    """
    return TrackingTargetUnavailableError(
        message, action=OperatorAction.RETRY, remediation=TRACKER_UNREADABLE_STEP
    )


class TrackingSyncRefusedError(PhaseSweepError):
    """Raised when ``sync-tracking`` has no matching MLflow deliveries it may retry."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG
