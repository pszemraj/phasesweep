"""Public engine export checks."""

from __future__ import annotations

import phasesweep.engine as engine
from phasesweep.engine import errors as engine_errors


def test_engine_exports_all_typed_preflight_errors() -> None:
    """Engine callers can catch every typed preflight failure from the public API."""
    names = (
        "ExperimentLockBusyError",
        "SamplerContinuationUnsupportedError",
        "StudyContextConflictError",
        "StudyFingerprintMismatchError",
        "StudySchemaMismatchError",
        "StudyStorageUnavailableError",
        "TrialTargetRegressionError",
    )

    assert set(names).issubset(engine.__all__)
    assert all(getattr(engine, name) is getattr(engine_errors, name) for name in names)
