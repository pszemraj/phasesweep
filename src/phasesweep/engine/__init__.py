"""Sweep execution engine API."""

from phasesweep.engine.errors import (
    ArtifactRootConflictError,
    ArtifactRootRebindError,
    ExperimentLockBusyError,
    LegacyArtifactRootMigrationRequiredError,
    PhaseSweepError,
    PublicationIntegrityError,
    SamplerContinuationUnsupportedError,
    StudyContextConflictError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrialEvidenceMissingError,
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
from phasesweep.engine.state import Winner, generation_id_source
from phasesweep.engine.trial import ProcessCleanupUncertainError, UnsafeProcessCleanupError

__all__ = [
    "ArtifactRootConflictError",
    "ArtifactRootRebindError",
    "ExperimentLockBusyError",
    "LegacyArtifactRootMigrationRequiredError",
    "NoFeasibleTrialError",
    "PhaseSweepError",
    "PhaseWinnerView",
    "ProcessCleanupUncertainError",
    "PublicationIntegrityError",
    "SamplerContinuationUnsupportedError",
    "StudyContextConflictError",
    "StudyFingerprintMismatchError",
    "StudySchemaMismatchError",
    "StudyStorageUnavailableError",
    "TerminalReport",
    "TrialEvidenceMissingError",
    "TrialTargetRegressionError",
    "UnsafeProcessCleanupError",
    "Winner",
    "config_status",
    "generation_id_source",
    "read_status",
    "read_winner",
    "read_winners",
    "run_config",
    "run_experiment",
    "run_suite",
]
