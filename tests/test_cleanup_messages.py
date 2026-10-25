"""Successful cleanup summaries describe confirmed recovery without alarm."""

from __future__ import annotations

import logging
from pathlib import Path

import optuna
import pytest

from phasesweep.config.io import load_experiment
from phasesweep.engine.attempts import _preflight_active_attempts, _PreflightCleanupReport
from phasesweep.engine.cleanup import _reap_stale_trials
from phasesweep.engine.state import (
    CLEANUP_CONFIRMED_ATTR,
    CLEANUP_RECOVERED_TRIALS_ATTR,
    TRIAL_OUTCOME_ATTR,
)
from phasesweep.runtime.process import write_attempt_lifecycle
from tests.conftest import make_experiment, reaped_pid
from tests.mcp_helpers import (
    ALLOW_SIDE_EFFECTS,
    make_mcp_app,
    stage_dead_run,
    write_srv_catalog,
    write_srv_config,
    write_unsafe_cleanup_status,
)
from tests.recovery_helpers import (
    fabricate_registered_attempt,
    fabricate_stale_trial,
    recover_run_cli,
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


def test_confirm_does_not_count_an_already_complete_trial_as_a_recovered_result(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Confirm counts terminal cleanup once and names only newly recovered results as kept."""
    config = write_srv_config(tmp_path)
    experiment = load_experiment(config)
    study, directory, number = fabricate_registered_attempt(
        experiment, "p", attempt_id="completed-attempt"
    )
    trial = optuna.Trial(study, study.get_trials()[number]._trial_id)
    trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(number, 0.5)
    write_trial_identity(directory, attempt_id="completed-attempt", pid=reaped_pid(), starttime=111)
    write_attempt_lifecycle(
        directory,
        attempt_id="completed-attempt",
        state="exited",
        return_code=0,
        cleanup_confirmed=True,
    )
    _app, registry, store = make_mcp_app(
        write_srv_catalog(tmp_path, config, allow=ALLOW_SIDE_EFFECTS)
    )
    handle = stage_dead_run(store, "old-generation", config, "srv", cleanup_uncertain=False)
    write_unsafe_cleanup_status(store, handle.run_id)

    confirmed = recover_run_cli(registry.state_dir, handle.run_id, confirm=True)

    assert confirmed.exit_code == 0, confirmed.output
    assert "kept 0 completed trial result(s)" in confirmed.output
    assert "confirmed 1 cleanup-uncertain trial(s)" in confirmed.output
    assert "Kept completed trial result" not in caplog.text
    retained = study.get_trials()[number]
    assert retained.state == optuna.trial.TrialState.COMPLETE
    assert retained.value == 0.5
