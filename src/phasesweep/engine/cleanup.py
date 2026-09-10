"""Stale-trial and uncertain-cleanup recovery."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import optuna

from phasesweep.config import Experiment
from phasesweep.engine.attempts import (
    _cleanup_recovered_trial_numbers,
    _collect_attempt_generation,
    _read_trial_process_identity,
    _record_cleanup_recovery,
    _record_stale_trial_failure,
    _resolve_attempt_for_reaping,
    _trial_dir_for_reaping,
    _trial_requires_cleanup_recovery,
)
from phasesweep.engine.errors import (
    StudyStorageUnavailableError,
)
from phasesweep.engine.state import (
    ATTEMPT_ID_ATTR,
    GENERATION_ID_ATTR,
    TRIAL_DIR_ATTR,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError
from phasesweep.runtime.process import (
    StaleProcessIdentity,
    cleanup_stale_trial_process,
)

log = logging.getLogger(__name__)


def _reap_stale_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
    *,
    recovered_attempt_ids: set[str] | None = None,
    recovered_attempt_generations: dict[str, str] | None = None,
    recovered_attempt_locations: dict[str, tuple[str, int, str]] | None = None,
    uncertain_attempt_ids: set[str] | None = None,
) -> int:
    """Mark RUNNING trials as FAIL after killing orphaned process groups.

    :param optuna.Study study: Study whose stale RUNNING trials should be reaped.
    :param Experiment experiment: Experiment used to locate trial directories.
    :param str phase_name: Name of the phase containing the stale trials.
    :param set[str] | None recovered_attempt_ids: Optional collector for exact
        attempt identities whose durable state was changed to FAIL.
    :param dict[str, str] | None recovered_attempt_generations: Optional mapping
        from each recovered attempt id to its producing generation id.
    :param dict[str, tuple[str, int, str]] | None recovered_attempt_locations:
        Optional mapping from attempt id to phase, trial number, and generation.
    :param set[str] | None uncertain_attempt_ids: Optional collector for exact
        attempt identities whose cleanup could not be proven.
    :return int: Number of stale RUNNING trials marked as failed.
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
        raise ProcessCleanupUncertainError(
            f"Could not inspect study {study.study_name!r} for stale RUNNING trials."
        ) from exc
    for trial in trials:
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue
        attempt_id = trial.user_attrs.get(ATTEMPT_ID_ATTR)
        try:
            trial_dir = _trial_dir_for_reaping(trial, experiment, phase_name, study.study_name)

            if TRIAL_DIR_ATTR in trial.user_attrs:
                _resolve_attempt_for_reaping(trial, trial_dir, study.study_name)
        except ProcessCleanupUncertainError:
            if uncertain_attempt_ids is not None and isinstance(attempt_id, str) and attempt_id:
                uncertain_attempt_ids.add(attempt_id)
            raise

        _record_stale_trial_failure(study, trial)
        _record_cleanup_recovery(study, trial)
        try:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        except Exception as exc:
            raise StudyStorageUnavailableError(
                f"Stale process cleanup completed for RUNNING trial {trial.number}, "
                f"but Optuna state could not be updated to FAIL. Refusing to continue "
                f"with an inconsistent study. trial_dir={trial_dir}"
            ) from exc

        _collect_attempt_generation(
            trial,
            phase_name,
            recovered_attempt_ids,
            recovered_attempt_generations,
            recovered_attempt_locations,
        )

        log.warning("Reaped stale RUNNING trial %d in study %s", trial.number, study.study_name)
        count += 1
    return count


def _inspect_stale_running_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
    *,
    recovered_attempt_ids: set[str] | None = None,
    recovered_attempt_generations: dict[str, str] | None = None,
    recovered_attempt_locations: dict[str, tuple[str, int, str]] | None = None,
) -> int:
    """Count stale RUNNING trials without signaling processes or writing state.

    Used by ``mcp recover-run`` preflight mode. The follow-up ``--confirm`` call
    must still find the same RUNNING trials so it can reap them and persist
    recovery evidence atomically with clearing MCP cleanup uncertainty.

    :param optuna.Study study: Study whose stale RUNNING trials should be inspected.
    :param Experiment experiment: Experiment used to locate trial directories.
    :param str phase_name: Name of the phase containing the stale trials.
    :param set[str] | None recovered_attempt_ids: Optional collector for
        attempts a confirmed pass would reap.
    :param dict[str, str] | None recovered_attempt_generations: Optional mapping
        from each collected attempt id to its producing generation id.
    :param dict[str, tuple[str, int, str]] | None recovered_attempt_locations:
        Optional mapping from attempt id to phase, trial number, and generation.
    :return int: Number of stale RUNNING trials found.
    """
    count = 0
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue
        trial_dir = _trial_dir_for_reaping(trial, experiment, phase_name, study.study_name)
        if TRIAL_DIR_ATTR in trial.user_attrs:
            _resolve_attempt_for_reaping(trial, trial_dir, study.study_name, inspect_only=True)
        _collect_attempt_generation(
            trial,
            phase_name,
            recovered_attempt_ids,
            recovered_attempt_generations,
            recovered_attempt_locations,
        )
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
    if not isinstance(stored, str) or not stored:
        raise ProcessCleanupUncertainError(
            f"Refusing to recover cleanup-uncertain trial {trial.number} in study "
            f"{study_name}: missing or invalid {TRIAL_DIR_ATTR!r} user attribute "
            f"{stored!r}. The leaked process group cannot be tied to identity files safely."
        )
    return Path(stored)


def _iter_cleanup_uncertain_trials(
    study: optuna.Study,
) -> Iterator[tuple[optuna.trial.FrozenTrial, Path, StaleProcessIdentity]]:
    """Yield unconsumed terminal trials with their validated process identity.

    :param optuna.Study study: Study whose cleanup evidence should be inspected.
    :return Iterator: Eligible trial, persisted trial directory, and process identity.
    """
    recovered_trial_numbers = _cleanup_recovered_trial_numbers(study)
    for trial in study.get_trials(deepcopy=False):
        if (
            trial.state.is_finished()
            and trial.number not in recovered_trial_numbers
            and _trial_requires_cleanup_recovery(trial)
        ):
            trial_dir = _trial_dir_for_cleanup_recovery(trial, study.study_name)
            identity = _read_trial_process_identity(trial, trial_dir, study.study_name)
            yield trial, trial_dir, identity


def _recover_cleanup_uncertain_trials(
    study: optuna.Study,
    experiment: Experiment,
    phase_name: str,
    *,
    recovered_attempt_ids: set[str] | None = None,
    recovered_attempt_generations: dict[str, str] | None = None,
    recovered_attempt_locations: dict[str, tuple[str, int, str]] | None = None,
) -> int:
    """Confirm cleanup for terminal trials that explicitly recorded uncertainty.

    ``UnsafeProcessCleanupError`` can leave an Optuna trial in a terminal FAIL state with
    ``phasesweep_cleanup_confirmed=false``. The normal stale reaper intentionally visits
    only RUNNING trials, so operator recovery needs this separate fail-closed inspection
    before clearing MCP cleanup uncertainty.

    :param optuna.Study study: Existing Optuna study for the phase being recovered.
    :param Experiment experiment: Parsed experiment, used for diagnostics.
    :param str phase_name: Name of the phase being recovered.
    :param set[str] | None recovered_attempt_ids: Optional collector for exact
        attempts whose cleanup was confirmed.
    :param dict[str, str] | None recovered_attempt_generations: Optional mapping
        from each collected attempt id to its producing generation id.
    :param dict[str, tuple[str, int, str]] | None recovered_attempt_locations:
        Optional mapping from attempt id to phase, trial number, and generation.
    :return int: Number of cleanup-uncertain terminal trials confirmed clean.
    :raises ProcessCleanupUncertainError: A recorded trial cannot be inspected or cleaned.
    """
    recovered = 0
    for trial, trial_dir, identity in _iter_cleanup_uncertain_trials(study):
        safe_to_clear = cleanup_stale_trial_process(identity)
        if not safe_to_clear:
            raise ProcessCleanupUncertainError(
                f"Refusing to clear cleanup uncertainty for trial {trial.number} in "
                f"study {study.study_name}: process cleanup could not be confirmed. "
                f"experiment={experiment.experiment} phase={phase_name} "
                f"trial_dir={trial_dir} pid={identity.pid} pgid={identity.pgid}."
            )
        _record_cleanup_recovery(study, trial)
        _collect_attempt_generation(
            trial,
            phase_name,
            recovered_attempt_ids,
            recovered_attempt_generations,
            recovered_attempt_locations,
        )
        recovered += 1
        log.warning(
            "Confirmed cleanup for terminal cleanup-uncertain trial %d in study %s "
            "(pid=%s pgid=%s)",
            trial.number,
            study.study_name,
            identity.pid,
            identity.pgid,
        )
    return recovered


def _inspect_cleanup_uncertain_trials(
    study: optuna.Study,
    phase_name: str,
    *,
    recovered_attempt_ids: set[str] | None = None,
    recovered_attempt_generations: dict[str, str] | None = None,
    recovered_attempt_locations: dict[str, tuple[str, int, str]] | None = None,
) -> int:
    """Count recoverable terminal cleanup evidence without signals or writes.

    :param optuna.Study study: Existing study inspected by recovery preflight.
    :param str phase_name: Name of the phase containing the recovered trials.
    :param set[str] | None recovered_attempt_ids: Optional collector for
        attempts a confirmed pass would recover.
    :param dict[str, str] | None recovered_attempt_generations: Optional mapping
        from each collected attempt id to its producing generation id.
    :param dict[str, tuple[str, int, str]] | None recovered_attempt_locations:
        Optional mapping from attempt id to phase, trial number, and generation.
    :return int: Number of unconsumed terminal trials that record cleanup uncertainty.
    :raises ProcessCleanupUncertainError: A trial lacks the persisted identity
        required for a safe confirmed recovery.
    """
    count = 0
    for trial, _trial_dir, _identity in _iter_cleanup_uncertain_trials(study):
        _collect_attempt_generation(
            trial,
            phase_name,
            recovered_attempt_ids,
            recovered_attempt_generations,
            recovered_attempt_locations,
        )
        count += 1
    return count
