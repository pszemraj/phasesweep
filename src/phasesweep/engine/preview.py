"""Dry-run preview of one phase against the trials its ledger already records.

A dry run reads the ledger through the write-nothing door and judges each
phase by the rules a launch applies, so it refuses and replays as a launch
would while it creates no study and changes nothing. The rules live in
:mod:`phasesweep.engine.phase`; this module only feeds them read-only
snapshots and reports the verdict.
"""

from __future__ import annotations

import logging

import optuna

from phasesweep.config import Experiment, Phase
from phasesweep.config.search import sampler_search_space
from phasesweep.engine.fingerprints import _phase_fingerprint
from phasesweep.engine.ledger import RecordedPhase, open_preview_study
from phasesweep.engine.optuna import _phase_study_name, _suggest
from phasesweep.engine.paths import _phase_dir
from phasesweep.engine.phase import (
    _composed_overrides,
    _is_placeholder_winner,
    _judge_recorded_phase,
    _PhaseResume,
    _placeholder_winner,
    _replay_met_target,
)
from phasesweep.engine.state import Winner
from phasesweep.engine.study_policy import _load_phase_policy_state
from phasesweep.runtime.commands import render_command

log = logging.getLogger("phasesweep.engine.preview")


def _preview_resume_phase(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    study: optuna.Study,
) -> _PhaseResume | Winner:
    """Judge a phase's recorded state as a launch would, without recovering or writing anything.

    This is the live resume applied to a read-only snapshot: the same target,
    partial-decision and abort rules, minus the steps that change the study
    (recovering stale trials, stamping the schema, binding the fingerprint) and
    minus the fingerprint comparison, whose inherited winners a preview may
    only have stood in for. A replayed winner is selected from the recorded
    trials but not evidence-verified, which reads the trial directories and may
    query W&B or MLflow.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: The phase being previewed.
    :param dict[str, Winner] inherited_winners: Winners, or stand-ins for them,
        for phases earlier in the chain.
    :param optuna.Study study: The phase's read-only snapshot study.
    :raises TrialTargetRefusedError: ``phase.n_trials`` is below the accepted
        target, or a fixed configuration asks for more attempts than it may retry.
    :raises NoFeasibleTrialError: The phase aborted, or its streak meets the
        current limit with work remaining, at an accepted target its current
        ``n_trials`` does not exceed.
    :raises WinnerIntegrityError: An accepted partial-timeout decision would
        replay while the current config disables ``allow_incomplete_on_timeout``.
    :raises TimeoutError: An accepted partial-timeout decision replays with no
        feasible trial.
    :raises StudySchemaMismatchError: Durable phase state is malformed or
        contradicts the study's trials.
    :return _PhaseResume | Winner: The validated state, or the winner an
        accepted partial-timeout decision replays.
    """
    return _judge_recorded_phase(
        experiment,
        phase,
        inherited_winners,
        study,
        policy_state=_load_phase_policy_state(study),
        phase_fingerprint=_phase_fingerprint(experiment, phase, inherited_winners),
        verify_evidence=False,
    )


def _preview_phase(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    recorded: RecordedPhase | None,
) -> Winner:
    """Log what a launch would do with the phase's recorded trials, and return its winner.

    The recorded trials are judged by the rules a launch applies
    (:func:`_preview_resume_phase`), so the preview refuses what a launch
    refuses. A phase that would replay returns the winner a launch would
    republish, which its children inherit. A phase with trials still to run
    returns a placeholder (:func:`_dry_run_phase`). A phase whose recorded
    state could not be read, or whose RUNNING attempts a launch recovers
    first, is previewed from its counts alone, as a placeholder.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: The phase being previewed.
    :param dict[str, Winner] inherited_winners: Winners, or stand-ins for them,
        for phases earlier in the chain.
    :param RecordedPhase | None recorded: What the ledger holds for the phase,
        or ``None`` when that could not be read and the phase previews as fresh.
    :raises PhaseSweepError: A launch would refuse the phase's recorded state.
    :return Winner: The winner a replay would publish, or a placeholder.
    """
    completed = 0 if recorded is None else recorded.finished
    remaining = max(0, phase.n_trials - completed)
    log.info(
        "phase=%s study=%s completed=%d remaining=%d n_jobs=%d",
        phase.name,
        _phase_study_name(experiment, phase),
        completed,
        remaining,
        phase.n_jobs,
    )
    study: optuna.Study | None = None
    resume: _PhaseResume | None = None
    if recorded is None:
        log.warning(
            "DRY RUN warning: phase=%s recorded trial counts could not be read, so it is "
            "previewed as a fresh phase and a real run may launch fewer trials",
            phase.name,
        )
    elif recorded.running:
        log.warning(
            "DRY RUN warning: phase=%s has %d RUNNING attempt(s) that a real run recovers "
            "as failed before it checks the phase, so its recorded target, sampler and abort "
            "state are not previewed",
            phase.name,
            recorded.running,
        )
    elif recorded.study is not None:
        study = recorded.study
        resumed = _preview_resume_phase(experiment, phase, inherited_winners, study)
        if isinstance(resumed, Winner):
            log.info(
                "DRY RUN phase=%s accepted a partial result at %d of %d trials: would "
                "replay it and launch no trials",
                phase.name,
                completed,
                phase.n_trials,
            )
            return resumed
        resume = resumed
    if remaining == 0:
        log.info(
            "DRY RUN phase=%s target met (%d of %d trials recorded): would replay its "
            "existing result and launch no trials",
            phase.name,
            completed,
            phase.n_trials,
        )
        if study is None or resume is None:
            return _placeholder_winner(experiment, phase, inherited_winners)
        return _replay_met_target(
            experiment, phase, inherited_winners, study, resume, verify_evidence=False
        )
    return _dry_run_phase(
        experiment, phase, inherited_winners, open_preview_study(experiment, phase), remaining
    )


def _dry_run_phase(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    study: optuna.Study,
    remaining: int,
) -> Winner:
    """Render and log one example trial command for the phase without launching anything.

    Args:
        experiment: Parsed experiment config.
        phase: The phase being previewed.
        inherited_winners: Winners loaded for phases earlier in the chain.
        study: An in-memory Optuna study used to ``ask`` for one sample.
        remaining: Number of trials that *would* run; logged for the user.

    Returns:
        A :class:`Winner` placeholder built from the previewed sample, or from
        each parameter's low bound or first choice, so downstream dry-run
        previews see consistent inherited context.

    """
    log.info(
        "DRY RUN phase=%s would launch %d %s",
        phase.name,
        remaining,
        "trial" if remaining == 1 else "trials",
    )
    sampled: dict[str, object] | None = None
    if remaining > 0:
        # The preview study is discarded unfinished: telling it would run
        # GridSampler.after_trial, whose study.stop() is illegal outside
        # optimize once a one-point grid is exhausted.
        sample_trial = study.ask()
        sampled = {
            name: _suggest(sample_trial, name, p) for name, p in sampler_search_space(phase).items()
        }
        overrides = _composed_overrides(experiment, phase, sampled, inherited_winners)
        preview_dir = _phase_dir(experiment, phase.name) / "trial_dryrun"
        cmd = render_command(
            experiment.trial_command,
            overrides,
            experiment.override_format,
            trial_dir=preview_dir,
            trial_id=-1,
            phase=phase.name,
            run_name=f"{experiment.experiment}-{phase.name}-DRYRUN",
            trainer_config=experiment.trainer_config,
        )
        placeholders = [
            name for name, winner in inherited_winners.items() if _is_placeholder_winner(winner)
        ]
        note = ""
        if placeholders:
            names = ", ".join(repr(name) for name in placeholders)
            publishes = "publishes a winner" if len(placeholders) == 1 else "publish winners"
            note = f" (inherited values are placeholders until {names} {publishes})"
        log.info("DRY RUN example command%s:\n  %s", note, cmd)

    return _placeholder_winner(
        experiment,
        phase,
        inherited_winners,
        sampled_params=sampled,
    )
