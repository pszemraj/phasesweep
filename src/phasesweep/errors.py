"""Shared operator-facing error taxonomy."""


class PhaseSweepError(RuntimeError):
    """Base for expected operational failures an operator is meant to act on.

    Every subclass carries a complete diagnostic, so the CLI reports it as one
    line without an internal-bug traceback. Exceptions that indicate a
    PhaseSweep defect must not inherit from this class.
    """


class ProcessCleanupUncertainError(PhaseSweepError):
    """Base class for failures where a subprocess group may still be alive."""


class UnsafeProcessCleanupError(ProcessCleanupUncertainError):
    """Raised when a trial's trainer or evidence worker may still be alive.

    This must not be included in Optuna's ``catch`` tuple: uncertain cleanup
    aborts the phase and run until recovery confirms the process group is gone.
    """


class LockBusyError(PhaseSweepError):
    """Raised when a required same-host lock is already held."""


class GpuConfigurationError(PhaseSweepError):
    """Raised when configured GPU isolation cannot be honored safely."""
