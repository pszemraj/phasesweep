"""Typed engine failures used for safe orchestration decisions."""


class StudyFingerprintMismatchError(RuntimeError):
    """Raised when stored study or winner semantics do not match the current config."""


class StudySchemaMismatchError(RuntimeError):
    """Raised when populated storage uses an unsupported PhaseSweep schema."""


class StudyContextConflictError(RuntimeError):
    """Raised when an upstream top-up would invalidate a bound descendant study."""


class StudyStorageUnavailableError(RuntimeError):
    """Raised when persistent study storage cannot be inspected during preflight."""


class SamplerContinuationUnsupportedError(RuntimeError):
    """Raised when a stateful sampler cannot safely continue across invocations."""


class TrialTargetRegressionError(RuntimeError):
    """Raised when a persistent study requests less than its accepted trial target."""
