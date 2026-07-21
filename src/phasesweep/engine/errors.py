"""Typed engine failures used for safe orchestration decisions."""


class StudyFingerprintMismatchError(RuntimeError):
    """Raised when stored study or winner semantics do not match the current config."""


class StudySchemaMismatchError(RuntimeError):
    """Raised when populated storage uses an unsupported PhaseSweep schema."""


class StudyContextConflictError(RuntimeError):
    """Raised when an upstream top-up would invalidate a bound descendant study."""


class StudyStorageUnavailableError(RuntimeError):
    """Raised when persistent study storage cannot be inspected during preflight."""
