"""Sweep execution engine API."""

from phasesweep.engine.errors import (
    ExperimentLockBusyError,
    SamplerContinuationUnsupportedError,
    StudyContextConflictError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrialTargetRegressionError,
)
from phasesweep.engine.read import (
    PhaseWinnerView,
    read_status,
    read_winner,
    read_winners,
)
from phasesweep.engine.run import (
    TerminalReport,
    config_status,
    run_config,
    run_experiment,
    run_suite,
)
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import Winner
from phasesweep.engine.trial import ProcessCleanupUncertainError, UnsafeProcessCleanupError

__all__ = [
    "ExperimentLockBusyError",
    "NoFeasibleTrialError",
    "PhaseWinnerView",
    "ProcessCleanupUncertainError",
    "SamplerContinuationUnsupportedError",
    "StudyContextConflictError",
    "StudyFingerprintMismatchError",
    "StudySchemaMismatchError",
    "StudyStorageUnavailableError",
    "TerminalReport",
    "TrialTargetRegressionError",
    "UnsafeProcessCleanupError",
    "Winner",
    "config_status",
    "read_status",
    "read_winner",
    "read_winners",
    "run_config",
    "run_experiment",
    "run_suite",
]
