"""Durable state that stale-trial recovery acts on, built the way a crashed run leaves it."""

from __future__ import annotations

from pathlib import Path

import optuna

from phasesweep.config import Experiment, load_config
from phasesweep.engine.paths import _trial_dir_for
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    GENERATION_ID_ATTR,
    TRIAL_DIR_ATTR,
)
from phasesweep.runtime.process import _write_process_identity
from phasesweep.runtime.reaper import (
    PROCESS_IDENTITY_FILE,
    PROCESS_IDENTITY_SCHEMA_VERSION,
    StaleProcessIdentity,
    read_boot_id,
)
from tests.conftest import mark_current_format, reaped_pid


def write_trial_identity(
    trial_dir: Path,
    *,
    attempt_id: str,
    pid: int,
    starttime: int | None,
    pgid: int | None = None,
    boot_id: str | None = None,
) -> None:
    """Leave the process identity file a launched trainer records, with pgid defaulting to pid and boot_id to this boot."""
    _write_process_identity(
        trial_dir / PROCESS_IDENTITY_FILE,
        StaleProcessIdentity(
            schema_version=PROCESS_IDENTITY_SCHEMA_VERSION,
            attempt_id=attempt_id,
            pid=pid,
            pgid=pid if pgid is None else pgid,
            proc_starttime=starttime,
            boot_id=read_boot_id() if boot_id is None else boot_id,
        ),
    )


def _new_phase_study(experiment: Experiment, phase_name: str) -> optuna.Study:
    """Create one phase's minimizing study, stamped as this release's format."""
    study = optuna.create_study(
        study_name=f"{experiment.experiment}::{phase_name}",
        storage=experiment.storage,
        direction="minimize",
    )
    mark_current_format(experiment, study)
    return study


def write_launched_stale_trial(
    config: Path,
    *,
    cleanup_confirmed: bool | None = None,
    generation_id: str = "stale-generation",
    persist_trial_attrs: bool = True,
    boot_id: str | None = None,
    pid: int | None = None,
) -> int:
    """Leave the config's first phase holding a RUNNING trial whose trainer was launched and never reaped."""
    exp = load_config(config)
    assert isinstance(exp, Experiment)
    phase = exp.phases[0]
    study = _new_phase_study(exp, phase.name)
    trial = study.ask()
    attempt_id = f"stale-attempt-{trial.number}"
    trial_dir = _trial_dir_for(
        exp,
        phase.name,
        trial.number,
        generation_id=generation_id,
        attempt_id=attempt_id,
    )
    trial_dir.mkdir(parents=True)
    write_trial_identity(
        trial_dir,
        attempt_id=attempt_id,
        pid=pid or reaped_pid(),
        starttime=222,
        boot_id=boot_id,
    )
    if persist_trial_attrs:
        trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
        trial.set_user_attr(GENERATION_ID_ATTR, generation_id)
        trial.set_user_attr(ATTEMPT_ID_ATTR, attempt_id)
        if cleanup_confirmed is not None:
            trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, cleanup_confirmed)
    return trial.number


def write_uncertain_failed_trial(
    config: Path, *, generation_id: str = "stale-generation", pid: int | None = None
) -> int:
    """Leave the config's first phase holding a FAIL trial whose process cleanup was never confirmed."""
    exp = load_config(config)
    assert isinstance(exp, Experiment)
    phase = exp.phases[0]
    study = _new_phase_study(exp, phase.name)
    trial = study.ask()
    attempt_id = f"stale-attempt-{trial.number}"
    trial_dir = _trial_dir_for(
        exp,
        phase.name,
        trial.number,
        generation_id=generation_id,
        attempt_id=attempt_id,
    )
    trial_dir.mkdir(parents=True)
    write_trial_identity(
        trial_dir,
        attempt_id=attempt_id,
        pid=pid or reaped_pid(),
        starttime=111,
    )
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    trial.set_user_attr(GENERATION_ID_ATTR, generation_id)
    trial.set_user_attr(ATTEMPT_ID_ATTR, attempt_id)
    trial.set_user_attr(CLEANUP_CONFIRMED_ATTR, False)
    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    return trial.number
