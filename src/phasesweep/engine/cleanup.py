"""Stale-trial and uncertain-cleanup recovery."""

from __future__ import annotations

import logging
from pathlib import Path

import optuna

from phasesweep.config import Experiment
from phasesweep.engine.attempts import (
    _AttemptRecord,
    _cleanup_recovered_trial_numbers,
    _CleanupUncertainTrialDiscovery,
    _fail_stale_running_trial,
    _record_cleanup_recovery,
    _require_trial_attempt,
    _resolve_attempt,
    _trial_attempt_record,
    _trial_dir_for_reaping,
    _trial_requires_cleanup_recovery,
    _TrialDiscovery,
)
from phasesweep.engine.errors import OperatorAction
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GENERATION_ID_ATTR,
    TRIAL_DIR_ATTR,
)
from phasesweep.engine.study_policy import _restore_prelaunch_environment_identity
from phasesweep.engine.trial import ProcessCleanupUncertainError

log = logging.getLogger(__name__)


def _reap_stale_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
    *,
    confirm: bool,
    recovered_attempts: dict[str, _AttemptRecord] | None = None,
    uncertain_attempt_ids: set[str] | None = None,
) -> int:
    """Resolve each stale RUNNING trial's attempt, then mark the trial FAIL.

    Inspection (``mcp recover-run`` preflight) validates the same evidence
    without signalling a process or writing state. The follow-up confirmed
    call must still find the same RUNNING trials so it can reap them and
    persist recovery evidence atomically with clearing MCP cleanup
    uncertainty.

    :param optuna.Study study: Study whose stale RUNNING trials should be reaped.
    :param Experiment experiment: Experiment used to locate trial directories.
    :param str phase_name: Name of the phase containing the stale trials.
    :param bool confirm: Clean orphaned process groups and mark each stale
        trial FAIL when true; otherwise only validate and count.
    :param dict[str, _AttemptRecord] | None recovered_attempts: Optional
        mapping from each attempt id whose trial was marked FAIL (or, when
        inspecting, would be) to its record.
    :param set[str] | None uncertain_attempt_ids: Optional collector for exact
        attempt identities whose cleanup could not be proven.
    :return int: Number of stale RUNNING trials marked FAIL, or found when inspecting.
    :raises ProcessCleanupUncertainError: The study's trials cannot be
        inspected, or a stale trial's directory, attempt identity, or process
        cleanup could not be proven safe.
    :raises StudyStorageUnavailableError: Cleanup succeeded but Optuna could
        not be updated to ``FAIL``; refusing to continue with an inconsistent
        study.
    """
    count = 0
    try:
        trials = study.get_trials(deepcopy=False)
    except Exception as exc:
        # recover-run reaps through here too and cannot read the ledger either,
        # so restoring it comes first.
        raise ProcessCleanupUncertainError(
            f"Could not inspect study {study.study_name!r} for stale RUNNING trials. "
            "Restore the original complete storage ledger and access to it before retrying.",
            action=OperatorAction.RESTORE_LEDGER,
        ) from exc
    discovery = _TrialDiscovery(study.study_name)
    for trial in trials:
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue
        attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
        record: _AttemptRecord | None
        try:
            trial_dir = _trial_dir_for_reaping(trial, experiment, phase_name, study.study_name)
            if TRIAL_DIR_ATTR in trial.user_attrs:
                record = _require_trial_attempt(trial, phase_name, trial_dir, study.study_name)
                _resolve_attempt(record, discovery, confirm=confirm)
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
            _fail_stale_running_trial(
                study,
                trial,
                failure_message=(
                    f"Stale process cleanup completed for RUNNING trial {trial.number}, "
                    f"but Optuna state could not be updated to FAIL. Refusing to continue "
                    f"with an inconsistent study. trial_dir={trial_dir}"
                ),
            )
            log.warning("Reaped stale RUNNING trial %d in study %s", trial.number, study.study_name)
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
    """Return this run's trial cleanup evidence from a prior interrupted pass.

    Recovery durably records each confirmed trial in the study-level ledger
    before the CLI can persist its run-level recovery record or clear the
    cleanup-uncertainty marker. A crash in that window must not erase the
    evidence: the retry skips these trials as already recovered, and without
    this mapping the trial-level-evidence guard would refuse to clear cleanup
    uncertainty forever (review v0.5.17 gap hunt).

    The mapping is scoped to ``generation_id`` — detached MCP runs use their run
    id as the generation id — so a *different* run's already-consumed evidence
    cannot clear this run's uncertainty; that cross-run refusal stays
    fail-closed.

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


def _trial_dir_for_cleanup_recovery(
    trial: optuna.trial.FrozenTrial,
    study_name: str,
) -> Path:
    """Return the persisted trial directory for terminal cleanup recovery.

    :param optuna.trial.FrozenTrial trial: Terminal trial with uncertain cleanup.
    :param str study_name: Study name for diagnostics.
    :return Path: Persisted trial directory containing process identity files.
    :raises ProcessCleanupUncertainError: The trial has no safe persisted trial directory.
    """
    stored = trial.user_attrs.get(TRIAL_DIR_ATTR)
    # A launch records the directory before its process can leak, so only a
    # damaged ledger lacks it here, and recover-run reads the same attribute.
    if not isinstance(stored, str) or not stored:
        raise ProcessCleanupUncertainError(
            f"Refusing to recover cleanup-uncertain trial {trial.number} in study "
            f"{study_name}: missing or invalid {TRIAL_DIR_ATTR!r} user attribute "
            f"{stored!r}. The leaked process group cannot be tied to identity files safely. "
            "Restore the original storage ledger before retrying recovery.",
            action=OperatorAction.RESTORE_LEDGER,
        )
    return Path(stored)


def _recover_cleanup_uncertain_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
    *,
    confirm: bool,
    recovered_attempts: dict[str, _AttemptRecord] | None = None,
) -> int:
    """Confirm cleanup for terminal trials that explicitly recorded uncertainty.

    ``UnsafeProcessCleanupError`` can leave an Optuna trial in a terminal FAIL state with
    ``phasesweep_cleanup_confirmed=false``. The normal stale reaper intentionally visits
    only RUNNING trials, so operator recovery needs this separate fail-closed inspection
    before clearing MCP cleanup uncertainty. Inspection (``mcp recover-run``
    preflight) validates each trial's persisted identity without signalling or writing.

    :param optuna.Study study: Existing Optuna study for the phase being recovered.
    :param Experiment experiment: Parsed experiment, used for diagnostics.
    :param str phase_name: Name of the phase being recovered.
    :param bool confirm: Clean each recorded process group and consume its
        evidence in the study's recovery ledger when true; otherwise only
        validate and count.
    :param dict[str, _AttemptRecord] | None recovered_attempts: Optional
        mapping from each attempt id whose cleanup was confirmed (or, when
        inspecting, would be) to its record.
    :return int: Number of cleanup-uncertain terminal trials confirmed clean,
        or found when inspecting.
    :raises ProcessCleanupUncertainError: A recorded trial cannot be inspected or cleaned.
    """
    recovered_trial_numbers = _cleanup_recovered_trial_numbers(study)
    discovery = _CleanupUncertainTrialDiscovery(study.study_name, experiment.experiment)
    count = 0
    for trial in study.get_trials(deepcopy=False):
        if not (
            trial.state.is_finished()
            and trial.number not in recovered_trial_numbers
            and _trial_requires_cleanup_recovery(trial)
        ):
            continue
        trial_dir = _trial_dir_for_cleanup_recovery(trial, study.study_name)
        record = _require_trial_attempt(trial, phase_name, trial_dir, study.study_name)
        _resolve_attempt(record, discovery, confirm=confirm)
        if confirm:
            _record_cleanup_recovery(study, trial)
        if recovered_attempts is not None:
            recovered_attempts[record.attempt_id] = record
        count += 1
    return count
