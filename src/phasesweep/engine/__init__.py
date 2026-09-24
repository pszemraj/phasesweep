"""Sweep execution engine API."""

from phasesweep.engine.errors import (
    ActiveAttemptPersistenceError,
    ArtifactRootConflictError,
    ExperimentLockBusyError,
    IncompleteJournalRecordError,
    PhaseSweepError,
    PublicationAccessError,
    PublicationCommitError,
    PublicationIntegrityError,
    PublishedStudyMissingError,
    RunRequestError,
    SamplerContinuationUnsupportedError,
    StudyContextConflictError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrialEvidenceMissingError,
    TrialTargetRegressionError,
    WinnerIntegrityError,
)
from phasesweep.engine.provenance import generation_id_source
from phasesweep.engine.read import (
    PhaseWinnerView,
    read_status,
    read_winner,
    read_winners,
)
from phasesweep.engine.run import (
    PublicationHook,
    TerminalReport,
    config_status,
    run_config,
    run_experiment,
)
from phasesweep.engine.selection import NoFeasibleTrialError
from phasesweep.engine.state import Winner
from phasesweep.engine.trial import ProcessCleanupUncertainError, UnsafeProcessCleanupError

__all__ = [
    "ActiveAttemptPersistenceError",
    "ArtifactRootConflictError",
    "ExperimentLockBusyError",
    "IncompleteJournalRecordError",
    "NoFeasibleTrialError",
    "PhaseSweepError",
    "PhaseWinnerView",
    "ProcessCleanupUncertainError",
    "PublicationHook",
    "PublicationAccessError",
    "PublicationCommitError",
    "PublicationIntegrityError",
    "PublishedStudyMissingError",
    "RunRequestError",
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
    "WinnerIntegrityError",
    "config_status",
    "generation_id_source",
    "read_status",
    "read_winner",
    "read_winners",
    "run_config",
    "run_experiment",
]
