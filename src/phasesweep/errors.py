"""Shared operator-facing error taxonomy."""


class PhaseSweepError(RuntimeError):
    """Base for expected operational failures an operator is meant to act on.

    Every subclass carries a complete diagnostic, so the CLI reports it as one
    line without an internal-bug traceback. Exceptions that indicate a
    PhaseSweep defect must not inherit from this class.
    """


class LockBusyError(PhaseSweepError):
    """Raised when a required same-host lock is already held."""


class GpuConfigurationError(PhaseSweepError):
    """Raised when configured GPU isolation cannot be honored safely."""
