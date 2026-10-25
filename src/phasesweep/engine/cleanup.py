"""Stale-trial and uncertain-cleanup recovery."""

from __future__ import annotations

import logging
from functools import partial

import optuna

from phasesweep.config.models import Experiment
from phasesweep.engine.attempts import (
    _AttemptRecord,
    _cleanup_recovered_trial_numbers,
    _fail_stale_running_trial,
    _record_cleanup_recovery,
    _record_recovery_interruption,
    _require_trial_attempt,
    _resolve_attempt,
    _trial_attempt_record,
    _trial_dir_from_attr,
    _trial_requires_cleanup_recovery,
)
from phasesweep.engine.paths import _experiment_dir, _trial_dir_for
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GENERATION_ID_ATTR,
    TRIAL_DIR_ATTR,
)
from phasesweep.engine.study_policy import _restore_prelaunch_environment_identity
from phasesweep.errors import OperatorAction, ProcessCleanupUncertainError

log = logging.getLogger(__name__)


def _reap_stale_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
    *,
    confirm: bool,
    recovered_attempts: dict[str, _AttemptRecord] | None = None,
    uncertain_attempt_ids: set[str] | None = None,
    process_actions: dict[_AttemptRecord, bool] | None = None,
    inspection_messages: dict[_AttemptRecord, str] | None = None,
) -> int:
    """Resolve stale RUNNING attempts and recover completed evidence or interruptions.

    Inspection validates evidence without signalling or writing. Confirmed recovery
    must still find the same RUNNING trials before recording cleanup evidence.

    :param optuna.Study study: Study whose stale RUNNING trials should be reaped.
    :param Experiment experiment: Experiment used to locate trial directories.
    :param str phase_name: Name of the phase containing the stale trials.
    :param bool confirm: Clean orphaned process groups and mark each stale
        trial terminal when true; otherwise only validate and count.
    :param dict[str, _AttemptRecord] | None recovered_attempts: Optional
        mapping from each attempt id whose trial became terminal (or, when
        inspecting, would be) to its record.
    :param set[str] | None uncertain_attempt_ids: Optional collector for exact
        attempt identities whose cleanup could not be proven.
    :param dict[_AttemptRecord, bool] | None process_actions: Collector of attempts whose live group requires termination.
    :param dict[_AttemptRecord, str] | None inspection_messages: Deferred preview messages, replaced by each source's fresh inspection.
    :return int: Number of stale RUNNING trials settled, or found when inspecting.
    :raises ProcessCleanupUncertainError: The study's trials cannot be
        inspected, or a stale trial's directory, attempt identity, or process
        cleanup could not be proven safe.
    :raises StudyStorageUnavailableError: Cleanup succeeded but the terminal result or durable recovery state could not be committed.
    """
    count = 0
    try:
        trials = study.get_trials(deepcopy=False)
    except Exception as exc:
        # recover-run reaps through here too and cannot read the ledger either,
        # so restoring it comes first.
        raise ProcessCleanupUncertainError(
            f"Could not inspect study {study.study_name!r} for stale RUNNING trials.",
            action=OperatorAction.RESTORE_LEDGER,
        ) from exc
    for trial in trials:
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue
        attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
        record: _AttemptRecord | None
        try:
            trial_dir = _trial_dir_from_attr(
                trial,
                study.study_name,
                fallback=partial(_trial_dir_for, experiment, phase_name, trial.number),
            )
            if TRIAL_DIR_ATTR in trial.user_attrs:
                record = _require_trial_attempt(trial, phase_name, trial_dir, study.study_name)
                terminated = _resolve_attempt(
                    record,
                    context=f"RUNNING trial {trial.number} in study {study.study_name!r}",
                    evidence_action=OperatorAction.INSPECT_LOGS,
                    lifecycle_settles=True,
                    confirm=confirm,
                    before_terminate=partial(_record_recovery_interruption, study, trial),
                    inspection_messages=inspection_messages,
                )
                if process_actions is not None:
                    process_actions[record] = terminated
            else:
                # The launch path records the directory before any process can
                # start, so this allocation launched nothing to resolve.
                record = _trial_attempt_record(trial, phase_name, trial_dir)
                if confirm:
                    _restore_prelaunch_environment_identity(study, trial)
        except ProcessCleanupUncertainError:
            if uncertain_attempt_ids is not None and isinstance(attempt_id, str) and attempt_id:
                uncertain_attempt_ids.add(attempt_id)
            raise

        if confirm:
            settled = _fail_stale_running_trial(
                study,
                trial,
                record=record,
                artifact_root=_experiment_dir(experiment),
                failure_message=(
                    f"Stale process cleanup completed for RUNNING trial {trial.number}, "
                    f"but Optuna state could not be updated to FAIL. Refusing to continue "
                    f"with an inconsistent study. trial_dir={trial_dir}"
                ),
            )
            if not settled:
                continue
            log.info(
                "Confirmed process cleanup for RUNNING trial %d in study %s",
                trial.number,
                study.study_name,
            )
        if recovered_attempts is not None and record is not None:
            recovered_attempts[record.attempt_id] = record
        count += 1
    return count


def _previously_recovered_attempt_locations(
    study: optuna.Study,
    phase_name: str,
    generation_id: str,
    *,
    causal_attempt_ids: set[str] | None = None,
) -> dict[str, tuple[str, int, str]]:
    """Return this run's cleanup evidence retained by a prior interrupted recovery.

    Study-level confirmation precedes the run-level recovery record and uncertainty
    clearance. Recover that evidence on retry, scoped to ``generation_id`` and any
    explicitly causal attempts; another run's evidence cannot clear this run.

    :param optuna.Study study: Study whose ledger and trials are inspected.
    :param str phase_name: Configured phase whose study is being inspected.
    :param str generation_id: Generation identity of the run being recovered.
    :param set[str] | None causal_attempt_ids: Older-generation attempts the
        run's own terminal report explicitly named as cleanup-uncertain.
    :return dict[str, tuple[str, int, str]]: Recovered attempt ids causally
        bound to this run, mapped to phase, trial number, and generation.
    """
    recovered = _cleanup_recovered_trial_numbers(study)
    if not recovered:
        return {}
    attempts: dict[str, tuple[str, int, str]] = {}
    for trial in study.get_trials(deepcopy=False):
        attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
        attempt_generation = trial.user_attrs.get(GENERATION_ID_ATTR)
        if not (
            trial.state.is_finished()
            and trial.number in recovered
            and isinstance(attempt_id, str)
            and attempt_id
            and isinstance(attempt_generation, str)
            and attempt_generation
        ):
            continue
        if attempt_generation == generation_id or (
            causal_attempt_ids is not None and attempt_id in causal_attempt_ids
        ):
            attempts[attempt_id] = (phase_name, trial.number, attempt_generation)
    return attempts


def _recover_cleanup_uncertain_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
    *,
    confirm: bool,
    recovered_attempts: dict[str, _AttemptRecord] | None = None,
    process_actions: dict[_AttemptRecord, bool] | None = None,
    inspection_messages: dict[_AttemptRecord, str] | None = None,
) -> int:
    """Confirm cleanup for terminal trials that explicitly recorded uncertainty.

    The stale reaper visits RUNNING trials only; these FAIL trials retain
    ``phasesweep_cleanup_confirmed=false``. Inspection validates persisted identity
    without signalling or writing.

    :param optuna.Study study: Existing Optuna study for the phase being recovered.
    :param Experiment experiment: Parsed experiment, used for diagnostics.
    :param str phase_name: Name of the phase being recovered.
    :param bool confirm: Clean each recorded process group and consume its
        evidence in the study's recovery ledger when true; otherwise only
        validate and count.
    :param dict[str, _AttemptRecord] | None recovered_attempts: Optional
        mapping from each attempt id whose cleanup was confirmed (or, when
        inspecting, would be) to its record.
    :param dict[_AttemptRecord, bool] | None process_actions: Collector of attempts whose live group requires termination.
    :param dict[_AttemptRecord, str] | None inspection_messages: Deferred preview messages, replaced by each source's fresh inspection.
    :return int: Number of cleanup-uncertain terminal trials confirmed clean,
        or found when inspecting.
    :raises ProcessCleanupUncertainError: A recorded trial cannot be inspected or cleaned.
    """
    recovered_trial_numbers = _cleanup_recovered_trial_numbers(study)
    count = 0
    for trial in study.get_trials(deepcopy=False):
        if not (
            trial.state.is_finished()
            and trial.number not in recovered_trial_numbers
            and _trial_requires_cleanup_recovery(trial)
        ):
            continue
        trial_dir = _trial_dir_from_attr(trial, study.study_name)
        record = _require_trial_attempt(trial, phase_name, trial_dir, study.study_name)
        terminated = _resolve_attempt(
            record,
            context=(
                f"Cleanup-uncertain terminal trial {trial.number} in study {study.study_name!r} "
                f"experiment={experiment.experiment}"
            ),
            evidence_action=OperatorAction.INSPECT_LOGS,
            lifecycle_settles=False,
            confirm=confirm,
            inspection_messages=inspection_messages,
        )
        if process_actions is not None:
            process_actions[record] = terminated
        if confirm:
            _record_cleanup_recovery(study, trial)
        if recovered_attempts is not None:
            recovered_attempts[record.attempt_id] = record
        count += 1
    return count
