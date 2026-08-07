"""Typed engine failures used for safe orchestration decisions."""


class PhaseSweepError(RuntimeError):
    """Base for expected operational failures an operator is meant to act on.

    Every subclass carries a message that is the complete diagnostic, so the
    CLI reports it as one line and exits without a traceback: the stack adds
    nothing the operator can use. Exceptions that indicate a PhaseSweep bug
    must not inherit from this class - the CLI boundary reports those with
    their traceback instead. Inheriting from ``RuntimeError`` keeps every
    existing ``except RuntimeError`` handler behaving as before.
    """


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
    the experiment's relocated artifact tree. Nothing has been written when
    this is raised.
    """


class PublicationIntegrityError(PhaseSweepError):
    """Raised when a recorded publication exists but no longer validates.

    Strictly a *reporting* failure, raised by the read-only surfaces
    (``phasesweep status``, ``phasesweep show-winners``) so a corrupt result
    tree exits non-zero instead of reading like a tree that never published
    (review v0.5.18 / finding F4). It never blocks ``phasesweep run``:
    generation namespaces are immutable, so a forward run cannot overwrite the
    corrupt one, and the resume path already raises the same manifest error
    when it loads winners. The message must therefore carry both the
    validation error and the reason not to re-run - a successful re-run
    advances the last-success pointer past the corrupt generation, after which
    nothing reports it at all.
    """


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
    """Raised when persistent study storage cannot be inspected during preflight."""


class ExperimentLockBusyError(PhaseSweepError):
    """Raised when another orchestrator owns an experiment consistency lock."""


class SamplerContinuationUnsupportedError(PhaseSweepError):
    """Raised when a stateful sampler cannot safely continue across invocations."""


class TrialTargetRegressionError(PhaseSweepError):
    """Raised when a persistent study requests less than its accepted trial target."""
