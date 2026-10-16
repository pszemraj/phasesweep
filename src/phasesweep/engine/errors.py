"""Typed engine failures used for safe orchestration decisions."""

from typing import ClassVar

from phasesweep.errors import OperatorAction as OperatorAction
from phasesweep.errors import PhaseSweepError as PhaseSweepError


class StudyFingerprintMismatchError(PhaseSweepError):
    """Raised when stored study or winner semantics do not match the current config."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FRESH_NAMESPACE


class StudySchemaMismatchError(PhaseSweepError):
    """Raised when populated storage uses an unsupported PhaseSweep schema."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FRESH_NAMESPACE


class StudyContextConflictError(PhaseSweepError):
    """Raised when an upstream top-up would invalidate a bound descendant study."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FRESH_NAMESPACE


class ArtifactRootConflictError(PhaseSweepError):
    """Raised when a study or artifact tree names a different publication owner.

    Reuse the original workdir or choose a fresh artifact root and local storage.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.FRESH_NAMESPACE


class PublicationIntegrityError(PhaseSweepError):
    """Raised when a recorded publication exists but fails validation.

    Readers distinguish this from an absent publication. Resume and forward runs
    refuse it before replacing the last-success pointer.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class PublicationAccessError(PhaseSweepError):
    """Raised when the current user cannot validate a publication.

    Expose no published result without validation. Permission denial does not prove
    corruption; the publishing user must validate the artifacts.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class PublicationCommitError(PhaseSweepError):
    """Raised when a completed generation fails its pre-commit publication validation.

    The new generation remains unpublished; read-side access and integrity verdicts
    use their separate exception types.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.INSPECT_LOGS


class WinnerIntegrityError(PhaseSweepError):
    """Raised when a saved result is unreadable, malformed, or cannot be reused.

    Includes ambiguous source identity and disallowed partial-result replay.
    Fingerprint drift uses :class:`StudyFingerprintMismatchError`.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class RunRequestError(PhaseSweepError):
    """Raised when a caller requests an unsupported or conflicting run identity."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


class TrialEvidenceMissingError(PhaseSweepError):
    """Raised when a selection-eligible trial lacks its recorded on-disk evidence.

    Refuse selection rather than republishing unsupported results or dropping eligible
    trials, which would change the ranked population.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class StudyStorageUnavailableError(PhaseSweepError):
    """Raised when required persistent study state cannot be inspected or persisted."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_LEDGER


class IncompleteJournalRecordError(StudyStorageUnavailableError):
    """Raised when a partial final journal record cannot be repaired before writing.

    Causes include lock timeout or replacement, multiple journal hard links, backup
    or truncation failure, and journal changes during repair. Intact records remain;
    lock replacement or concurrent changes carry ``OperatorAction.RETRY``.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_LEDGER


class PublishedStudyMissingError(PhaseSweepError):
    """Raised before launch when a published phase's local trial is absent or replaced."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_LEDGER


class ActiveAttemptPersistenceError(PhaseSweepError):
    """Raised when required recovery state cannot persist before launch.

    The allocated lifecycle marker proves no process started; the experiment registry
    retains attempts across phase or storage changes. Failure to write either prevents
    trainer launch and GPU acquisition. Restore access to the experiment tree.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class ExperimentLockBusyError(PhaseSweepError):
    """Raised when another orchestrator owns an experiment consistency lock."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RETRY


class TrialTargetRefusedError(PhaseSweepError):
    """Raised when a phase target contradicts its durable study state.

    ``n_trials`` cannot fall below the accepted target. A fixed configuration may
    retry failures only until one attempt completes, feasible or not.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


class TrackingTargetUnavailableError(PhaseSweepError):
    """Raised before a phase allocates trials when its MLflow target cannot be reached."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG
