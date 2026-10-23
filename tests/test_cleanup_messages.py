"""Successful cleanup summaries describe confirmed recovery without alarm."""

from __future__ import annotations

import logging
from pathlib import Path

import optuna
import pytest

from phasesweep.engine.attempts import _preflight_active_attempts, _PreflightCleanupReport
from phasesweep.engine.cleanup import _reap_stale_trials
from phasesweep.engine.state import CLEANUP_RECOVERED_TRIALS_ATTR, TRIAL_OUTCOME_ATTR
from tests.conftest import make_experiment, reaped_pid
from tests.recovery_helpers import (
    fabricate_registered_attempt,
    fabricate_stale_trial,
    write_trial_identity,
)


@pytest.mark.parametrize("via_registry", [False, True], ids=["study", "registry"])
def test_successful_cleanup_summary_is_informational(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, via_registry: bool
) -> None:
    experiment = make_experiment(persistent=tmp_path)
    fabricate = fabricate_registered_attempt if via_registry else fabricate_stale_trial
    study, trial_dir, number = fabricate(experiment, "p", attempt_id="stopped-attempt")
    write_trial_identity(
        trial_dir,
        attempt_id="stopped-attempt",
        pid=reaped_pid(),
        starttime=222,
    )

    with caplog.at_level(logging.INFO):
        if via_registry:
            report = _PreflightCleanupReport()
            _preflight_active_attempts(experiment, report, confirm=True)
            reaped = report.reaped_running_trials
            assert report.cleanup_confirmed is True
            assert report.recovered_attempt_ids == {"stopped-attempt"}
        else:
            reaped = _reap_stale_trials(study, experiment, "p", confirm=True)

    assert reaped == 1
    trial = study.get_trials(deepcopy=False)[number]
    assert trial.state == optuna.trial.TrialState.FAIL
    assert trial.user_attrs[TRIAL_OUTCOME_ATTR]["outcome"] == "interrupted"
    assert study.user_attrs[CLEANUP_RECOVERED_TRIALS_ATTR] == [number]

    suffix = " via the attempt registry" if via_registry else ""
    logger = "phasesweep.engine.attempts" if via_registry else "phasesweep.engine.cleanup"
    summaries = [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == logger and record.getMessage().endswith(f"{study.study_name}{suffix}")
    ]
    assert summaries == [
        (
            logging.INFO,
            f"Confirmed process cleanup for RUNNING trial {number} "
            f"in study {study.study_name}{suffix}",
        )
    ]
