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


class StudyStorageUnavailableError(PhaseSweepError):
    """Raised when persistent study storage cannot be inspected during preflight."""


class ExperimentLockBusyError(PhaseSweepError):
    """Raised when another orchestrator owns an experiment consistency lock."""


class SamplerContinuationUnsupportedError(PhaseSweepError):
    """Raised when a stateful sampler cannot safely continue across invocations."""


class TrialTargetRegressionError(PhaseSweepError):
    """Raised when a persistent study requests less than its accepted trial target."""
