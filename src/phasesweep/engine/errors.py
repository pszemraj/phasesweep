"""Typed engine failures used for safe orchestration decisions."""

from phasesweep.errors import PhaseSweepError


class StudyFingerprintMismatchError(PhaseSweepError):
    """Raised when stored study or winner semantics do not match the current config."""


class StudySchemaMismatchError(PhaseSweepError):
    """Raised when populated storage uses an unsupported PhaseSweep schema."""


class StudyContextConflictError(PhaseSweepError):
    """Raised when an upstream top-up would invalidate a bound descendant study."""


class ArtifactRootConflictError(PhaseSweepError):
    """Raised when a persistent study is bound to a different artifact root.

    Deliberately a sibling of :class:`StudyContextConflictError` rather than a
    reuse of it: that error means one specific thing (an upstream top-up would
    invalidate a bound descendant study) and its remedy is a new experiment
    name, while this one means the offered ``workdir`` is not the publication
    root this study already claimed and its remedy is either the original
    ``workdir`` or an explicit ``phasesweep rebind-workdir``.
    """


class LegacyArtifactRootMigrationRequiredError(ArtifactRootConflictError):
    """Raised when a populated study predates artifact-root binding.

    A subclass of :class:`ArtifactRootConflictError` because it is the same
    operator problem - the offered ``workdir`` is not provably the one that
    owns this study's evidence - so every consumer that already classifies
    that conflict keeps classifying this the same way. It is a distinct type
    because the remedy differs: nothing is bound yet, so the operator names
    the *original* tree with ``phasesweep rebind-workdir`` instead of
    restoring a root the study already recorded (re-review v0.5.19 / blocker
    B1).
    """


class ArtifactRootRebindError(PhaseSweepError):
    """Raised when ``phasesweep rebind-workdir`` refuses to move a binding.

    Covers every refusal of that command: nothing is bound to move, storage is
    in-memory so no binding exists, or the destination cannot be validated as
    the experiment's relocated artifact tree. Validation refusals precede
    writes; an apply-time failure in a multi-study suite can follow an earlier
    plan that was already applied, because suite rebind is intentionally not
    one cross-storage transaction.
    """


class PublicationIntegrityError(PhaseSweepError):
    """Raised when a recorded publication exists but no longer validates.

    Read-only surfaces use this type to distinguish corruption from a tree that
    never published, and resume/publication paths use it when a manifest they
    must trust is malformed or no longer matches its artifacts. A fresh forward
    run is not blocked merely because an older publication is corrupt;
    generation namespaces are immutable, so it cannot overwrite that evidence.
    """


class PublicationAccessError(PhaseSweepError):
    """Raised when publication validation cannot run under the current user.

    Distinct from :class:`PublicationIntegrityError`: a permission denial does
    not show that the publication is corrupt, only that this user cannot prove
    it sound. Read surfaces still fail closed and expose no result, but the
    remedy is validation by the publishing user rather than restoration.
    """


class PublicationCommitError(PhaseSweepError):
    """Raised when a completed generation cannot pass its publication commit.

    This is a write-side failure: the generation remains unpublished because
    its newly written summary cannot be read back or does not identify the
    generation being committed. It is distinct from read-side integrity and
    access verdicts about a previously published result.
    """


class PromotionError(PhaseSweepError):
    """Raised when a configured promotion decision cannot expose a winner.

    Covers an unavailable prior suite result (for example, one deliberately
    omitted by an earlier ``on_fail: skip`` decision) and a failed promotion
    whose configured action is ``on_fail: stop``. Both are ordinary run
    outcomes selected by the suite policy, not PhaseSweep implementation bugs.
    """


class WinnerIntegrityError(PhaseSweepError):
    """Raised when a saved winner cannot safely be used for a skipped phase.

    The winner may be unreadable, structurally incomplete, ambiguously scoped,
    or incompatible with the current phase's partial-result policy. These are
    operator-visible artifact/configuration refusals; fingerprint drift keeps
    its more specific :class:`StudyFingerprintMismatchError` type.
    """


class RunRequestError(PhaseSweepError):
    """Raised when a caller requests an unsupported or conflicting run identity."""


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


class StudyStorageUnavailableError(PhaseSweepError):
    """Raised when required persistent study state cannot be inspected or was lost."""


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


class ExperimentLockBusyError(PhaseSweepError):
    """Raised when another orchestrator owns an experiment consistency lock."""


class SamplerContinuationUnsupportedError(PhaseSweepError):
    """Raised when a stateful sampler cannot safely continue across invocations."""


class TrialTargetRegressionError(PhaseSweepError):
    """Raised when a persistent study requests less than its accepted trial target."""
