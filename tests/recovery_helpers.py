"""Durable state that stale-trial recovery acts on, built the way a crashed run leaves it."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import optuna

from phasesweep.config import Experiment, load_config
from phasesweep.engine.attempts import _register_active_attempt
from phasesweep.engine.paths import _experiment_dir, _trial_dir_for
from phasesweep.engine.state import (
    ARTIFACT_ROOT_ATTR,
    ATTEMPT_ID_ATTR,
    CLEANUP_CONFIRMED_ATTR,
    GENERATION_ID_ATTR,
    TRAINER_ENV_DIGEST_ATTR,
    TRAINER_ENV_NAMES_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_TARGET_ATTR,
)
from phasesweep.engine.trial import _environment_identity
from phasesweep.mcp.recovery import _RecoveryNeeds
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


class StaleTrial(NamedTuple):
    """A RUNNING trial an orchestrator abandoned: its study, directory and number."""

    study: optuna.Study
    trial_dir: Path
    number: int


def stamp_artifact_root(study: optuna.Study, experiment: Experiment) -> None:
    """Bind a hand-built study to the artifact root a prior run would have claimed."""
    # A study an earlier invocation left behind always carries this binding.
    # Without it, preflight refuses the run as pre-cutover state, which
    # preempts the current-format recovery behavior under test.
    study.set_user_attr(ARTIFACT_ROOT_ATTR, str(_experiment_dir(experiment)))


def fabricate_stale_trial(
    experiment: Experiment,
    phase_name: str,
    *,
    attempt_id: str,
    generation_id: str = "old-generation",
    persist_attempt_id: bool = True,
    persist_generation_id: bool = True,
) -> StaleTrial:
    """Create the phase study, as a crashed engine run leaves it, holding one attribute-complete RUNNING trial."""
    study = _new_phase_study(experiment, phase_name)
    phase = next(phase for phase in experiment.phases if phase.name == phase_name)
    study.set_user_attr(TRIAL_TARGET_ATTR, phase.n_trials)
    stamp_artifact_root(study, experiment)
    trial = study.ask()
    identity = _environment_identity(experiment, phase_name)
    trial.set_user_attr(TRAINER_ENV_DIGEST_ATTR, identity.digest)
    trial.set_user_attr(TRAINER_ENV_NAMES_ATTR, list(identity.names))
    trial_dir = _trial_dir_for(
        experiment,
        phase_name,
        trial.number,
        generation_id=generation_id,
        attempt_id=attempt_id,
    )
    trial_dir.mkdir(parents=True)
    if persist_generation_id:
        trial.set_user_attr(GENERATION_ID_ATTR, generation_id)
    if persist_attempt_id:
        trial.set_user_attr(ATTEMPT_ID_ATTR, attempt_id)
    trial.set_user_attr(TRIAL_DIR_ATTR, str(trial_dir))
    return StaleTrial(study, trial_dir, trial.number)


def fabricate_registered_attempt(
    experiment: Experiment,
    phase_name: str,
    *,
    attempt_id: str,
    persist_trial_attempt_id: bool = True,
    persist_trial_generation_id: bool = True,
) -> StaleTrial:
    """Leave the stale trial :func:`fabricate_stale_trial` does, plus its attempt-registry entry."""
    stale = fabricate_stale_trial(
        experiment,
        phase_name,
        attempt_id=attempt_id,
        persist_attempt_id=persist_trial_attempt_id,
        persist_generation_id=persist_trial_generation_id,
    )
    _register_active_attempt(
        experiment,
        attempt_id=attempt_id,
        phase_name=phase_name,
        study_name=stale.study.study_name,
        trial_number=stale.number,
        trial_dir=stale.trial_dir,
        generation_id="old-generation",
    )
    return stale


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


def load_only_recovery_needs(*, ownership_storage_unavailable: bool = False) -> _RecoveryNeeds:
    """Return recovery decisions that load the phase studies and nothing else."""
    return _RecoveryNeeds(
        terminal_status=None,
        stored_snapshot=None,
        prepared_publication_generation=None,
        cleanup_needed=False,
        terminal_cleanup_uncertain=False,
        ownership_storage_unavailable=ownership_storage_unavailable,
        snapshot_recovery_required=False,
        snapshot_unavailable=False,
        snapshot_finalize_needed=False,
    )
