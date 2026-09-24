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
    """Raised when a persistent study is bound to a different artifact root.

    Deliberately a sibling of :class:`StudyContextConflictError` rather than a
    reuse of it: that error means one specific thing (an upstream top-up would
    invalidate a bound descendant study) and its remedy is a new experiment
    name, while this one means the offered ``workdir`` is not the publication
    root this study already claimed and its remedy is the original ``workdir``
    or a fresh artifact root and local storage.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.FRESH_NAMESPACE


class PublicationIntegrityError(PhaseSweepError):
    """Raised when a recorded publication exists but no longer validates.

    Read-only surfaces use this type to distinguish corruption from a tree that
    never published, and resume/publication paths use it when a manifest they
    must trust is malformed or no longer matches its artifacts. A corrupt
    last-success pointer also blocks a forward run so it cannot mask the
    damaged publication by advancing that pointer.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class PublicationAccessError(PhaseSweepError):
    """Raised when publication validation cannot run under the current user.

    Distinct from :class:`PublicationIntegrityError`: a permission denial does
    not show that the publication is corrupt, only that this user cannot prove
    it sound. Read surfaces still fail closed and expose no result, but the
    remedy is validation by the publishing user rather than restoration.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class PublicationCommitError(PhaseSweepError):
    """Raised when a completed generation cannot pass its publication commit.

    This is a write-side failure: the generation remains unpublished because
    its newly written summary cannot be read back or does not identify the
    generation being committed. It is distinct from read-side integrity and
    access verdicts about a previously published result.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.INSPECT_LOGS


class WinnerIntegrityError(PhaseSweepError):
    """Raised when a saved winner cannot safely be used for a skipped phase.

    The winner may be unreadable, structurally incomplete, ambiguously scoped,
    or incompatible with the current phase's partial-result policy. These are
    operator-visible artifact/configuration refusals; fingerprint drift keeps
    its more specific :class:`StudyFingerprintMismatchError` type.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class RunRequestError(PhaseSweepError):
    """Raised when a caller requests an unsupported or conflicting run identity."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG


class TrialEvidenceMissingError(PhaseSweepError):
    """Raised when a selection-eligible trial's on-disk evidence is gone.

    A completed trial that the study still records as eligible to win no
    longer has the evidence directory and audit artifacts its own record
    names. Winner selection reads Optuna alone, so without this check the
    orchestrator would republish that trial's number, metric, and provenance
    from a tree that no longer contains a single byte behind them.

    Fails the run closed rather than either silently selecting such a trial
    or silently skipping it: skipping is not the safe fallback here, because
    dropping the trials whose evidence happens to be missing changes which
    trial wins and therefore biases the published result (PR #5 review /
    reviewer 2, blocker 7).
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class StudyStorageUnavailableError(PhaseSweepError):
    """Raised when required persistent study state cannot be inspected or persisted."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_LEDGER


class LedgerTransactionInterruptedError(StudyStorageUnavailableError):
    """Raised when a SQLite ledger holds a transaction a crash interrupted.

    The crash left a hot rollback journal beside the database. SQLite rolls
    it back on the next read-write open, but every read PhaseSweep makes
    before the experiment lock opens the ledger ``mode=ro``, which cannot. The
    committed state is intact, so restoring the ledger is the wrong remedy:
    a command that holds the lock lets SQLite finish its own recovery first.
    Routed to recovery because the one read that raises this to an operator
    is ``recover-run`` inspection, whose confirmed form holds that lock. A
    locked open whose rollback SQLite refuses raises it routed to restoring
    the ledger's write access instead.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RUN_RECOVER_RUN


class IncompleteJournalRecordError(StudyStorageUnavailableError):
    """Raised before a write when a journal ledger ends with a partial record.

    Reads skip that record as Optuna does, but an append after it would be
    glued onto it and corrupt the journal for good. The experiment lock does
    not exclude another experiment's append to a shared journal, so the record
    may be an append still in flight, and nothing repairs it automatically.
    The message has the operator stop every writer and retry, and, for a
    refusal that persists, names a truncation command that does nothing once
    the journal has changed: the ledger repair this routes to.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_LEDGER


class PublishedStudyMissingError(PhaseSweepError):
    """Raised before launch when a published phase's local trial is absent or replaced."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_LEDGER


class ActiveAttemptPersistenceError(PhaseSweepError):
    """Raised when required attempt recovery state cannot persist before launch.

    The allocated lifecycle marker proves a process was never created, and the
    experiment-level ``attempts/`` registry discovers work after a phase rename,
    removal, or storage change. Losing either record before launch creates an
    attempt a later run cannot safely account for.

    Both writes are consequently pre-launch durability requirements. This
    failure is raised before any trainer starts or any GPU lease is consumed,
    so nothing needs undoing. The condition is environmental (the experiment
    workdir is not writable), so the remedy is restoring that tree, not a
    config change.
    """

    default_action: ClassVar[OperatorAction] = OperatorAction.RESTORE_TREE


class ExperimentLockBusyError(PhaseSweepError):
    """Raised when another orchestrator owns an experiment consistency lock."""

    default_action: ClassVar[OperatorAction] = OperatorAction.RETRY


class SamplerContinuationUnsupportedError(PhaseSweepError):
    """Raised when a stateful sampler cannot safely continue across invocations."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FRESH_NAMESPACE


class TrialTargetRegressionError(PhaseSweepError):
    """Raised when a persistent study requests less than its accepted trial target."""

    default_action: ClassVar[OperatorAction] = OperatorAction.FIX_CONFIG
