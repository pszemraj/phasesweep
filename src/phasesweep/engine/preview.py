"""Dry-run preview of an experiment against the trials its ledger already records.

A dry run reads the ledger through the write-nothing door and applies the
rules a launch applies to what it read, so it refuses and replays as a launch
would while it creates no study and changes nothing. The rules live where the
launch calls them (:mod:`phasesweep.engine.ledger`,
:mod:`phasesweep.engine.guards`, :mod:`phasesweep.engine.resume`,
:mod:`phasesweep.engine.study_policy`, :mod:`phasesweep.engine.phase`); this
module only reads the studies, feeds the rules read-only snapshots in the
launch's order, and reports the verdict.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import optuna

import phasesweep.engine.resume as resume_ops
from phasesweep.config import Experiment, Phase
from phasesweep.config.search import sampler_search_space
from phasesweep.engine.artifact_roots import (
    _artifact_root_binding_applies,
    _check_published_phase_studies,
)
from phasesweep.engine.errors import StudyStorageUnavailableError
from phasesweep.engine.fingerprints import _phase_fingerprint
from phasesweep.engine.guards import _raise_collected_errors
from phasesweep.engine.ledger import open_preview_study, read_phase_studies, validate_ledger
from phasesweep.engine.optuna import (
    _completed_trial_count,
    _finished_trial_count,
    _phase_study_name,
    _suggest,
)
from phasesweep.engine.paths import _phase_dir
from phasesweep.engine.phase import (
    _composed_overrides,
    _inherited_placeholder_keys,
    _judge_recorded_phase,
    _PhaseResume,
    _placeholder_winner,
    _replay_met_target,
)
from phasesweep.engine.state import (
    STUDY_SCHEMA_ATTR,
    Winner,
    _outcome_sequence_counts,
    _recovered_trial_state,
)
from phasesweep.engine.study_policy import (
    _load_phase_policy_state,
    _pending_rerun_count,
    _validate_fixed_configuration_target,
    _validate_study_direction,
    _validate_study_schema,
    _validate_trial_target,
)
from phasesweep.runtime.commands import render_command

log = logging.getLogger("phasesweep.engine.preview")


@dataclass(frozen=True, slots=True)
class RecordedPhase:
    """What a ledger already holds for one phase, counted as a launch would count it."""

    #: Attempts that count toward ``n_trials``: terminal ones that were not interrupted,
    #: plus each RUNNING one whose objective recorded a failure, which a launch recovers
    #: as failed first. Any other RUNNING attempt recovers as interrupted and reruns.
    finished: int
    #: How many of those are COMPLETE.
    completed: int
    #: How many attempts are RUNNING, which a launch recovers before it checks anything.
    running: int
    #: Reruns a launch would run before any new draw: interrupted trials without one,
    #: WAITING reruns, and RUNNING attempts that recover as interrupted.
    reruns: int
    #: The phase's study as a read-only snapshot, or ``None`` when the ledger holds none.
    study: optuna.Study | None


@dataclass(frozen=True, slots=True)
class PhasePreview:
    """What a dry run says one previewed phase would do."""

    #: The winner the phase hands its children: a replayed one, or a placeholder.
    winner: Winner
    #: Trials a launch would run, the count the phase's "would launch" line prints.
    launches: int
    #: How many of those launches rerun interrupted trials.
    reruns: int
    #: Whether the phase would replay a recorded result and launch no trials.
    replays: bool


def _reruns_phrase(reruns: int) -> str:
    """Name a count of reruns of interrupted trials in the singular or plural.

    :param int reruns: Reruns counted.
    :return str: For example ``"1 rerun of an interrupted trial"``.
    """
    if reruns == 1:
        return "1 rerun of an interrupted trial"
    return f"{reruns} reruns of interrupted trials"


def _dry_run_total_line(previews: Sequence[PhasePreview]) -> str:
    """Total the trials a dry run would launch from the counts its phase lines print.

    :param Sequence[PhasePreview] previews: The previewed phases, in order. A phase that
        ``--from-phase`` skips is not previewed and so is not counted.
    :return str: The line, without the ``DRY RUN`` prefix.
    """
    trials = sum(preview.launches for preview in previews)
    reruns = sum(preview.reruns for preview in previews)
    launching = sum(1 for preview in previews if preview.launches)
    replaying = sum(1 for preview in previews if preview.replays)
    line = f"total: would launch {trials} {'trial' if trials == 1 else 'trials'} "
    if reruns:
        line += f"({_reruns_phrase(reruns)}) "
    line += f"across {launching} {'phase' if launching == 1 else 'phases'}"
    if replaying:
        line += f" ({replaying} other {'phase replays' if replaying == 1 else 'phases replay'})"
    return line


def _recorded_phase(study: optuna.Study | None) -> RecordedPhase:
    """Count one phase's snapshot study the way a launch counts it.

    :param optuna.Study | None study: The phase's snapshot study, or ``None`` when
        the ledger holds no such study, which counts zero attempts.
    :return RecordedPhase: The counts and the study.
    """
    if study is None:
        return RecordedPhase(finished=0, completed=0, running=0, reruns=0, study=None)
    trials = study.get_trials(deepcopy=False)
    running = [trial for trial in trials if trial.state == optuna.trial.TrialState.RUNNING]
    sequence_counts = _outcome_sequence_counts(trials)
    recovered_as_failed = sum(
        1 for trial in running if _recovered_trial_state(trial, sequence_counts) == "FAIL"
    )
    return RecordedPhase(
        finished=_finished_trial_count(trials) + recovered_as_failed,
        completed=_completed_trial_count(trials),
        running=len(running),
        reruns=_pending_rerun_count(trials) + len(running) - recovered_as_failed,
        study=study,
    )


def _read_recorded_phases(experiment: Experiment) -> dict[str, RecordedPhase | None]:
    """Read every phase's study from one capture of the ledger, creating and writing nothing.

    The read goes through the write-nothing door ``status`` uses, so it
    neither creates a study nor constructs file-backed storage. A missing
    ledger or study counts zero attempts. A ledger that cannot be inspected
    leaves every phase unread, so the preview falls back to a fresh study and
    says so.

    :param Experiment experiment: Parsed experiment being previewed.
    :raises ArtifactRootConflictError: The tree is bound to a different owner.
    :raises StudySchemaMismatchError: A ledger study uses an unsupported schema.
    :return dict[str, RecordedPhase | None]: One entry per phase, by phase name;
        ``None`` for every phase when the ledger could not be read.
    """
    try:
        studies = read_phase_studies(validate_ledger(experiment))
    except StudyStorageUnavailableError as exc:
        log.warning(
            "DRY RUN warning: the ledger could not be inspected, and a real run refuses "
            "a ledger it cannot read: %s",
            exc,
        )
        return {phase.name: None for phase in experiment.phases}
    return {name: _recorded_phase(study) for name, study in studies.items()}


def _refuse_as_a_launch_would(
    experiment: Experiment,
    recorded: Mapping[str, RecordedPhase],
    *,
    from_phase: str | None,
) -> None:
    """Apply a launch's claim and preflight refusals to the recorded studies, in its order.

    A launch checks what the ledger holds before it runs anything, so the
    preview refuses before it previews anything. The order is the launch's
    (:func:`phasesweep.engine.ledger.claim_ledger`, then
    :func:`phasesweep.engine.guards._preflight_existing_studies`, then the
    top-up checks of :mod:`phasesweep.engine.resume`), so the first refusal is
    the one a launch gives, and several studies' refusals are combined as it
    combines them. Every check is the launch's own function, run on the
    read-only snapshots.

    A launch recovers each RUNNING attempt before it checks that attempt's
    study (as interrupted, or as failed when its objective recorded a failure),
    and a snapshot cannot. Such a study gets the checks that do not depend on
    that recovery (direction and schema) and a fixed configuration's attempt
    limit from counts that treat each attempt as recovery would record it;
    it is left out of the target, bound-descendant and sampler checks. A launch
    also checks the trainer environment, the stored fingerprints, the trial
    evidence and each study's artifact-root claim, which a preview does not.

    :param Experiment experiment: Parsed experiment being previewed.
    :param Mapping[str, RecordedPhase] recorded: What the ledger holds for each phase.
    :param str | None from_phase: Optional first phase a launch would execute.
    :raises PublishedStudyMissingError: A reached phase has a published winner
        whose trial is missing from, or below the history boundary of, its study.
    :raises PublicationAccessError: The last publication cannot be read.
    :raises PublicationIntegrityError: The last publication no longer validates.
    :raises StudySchemaMismatchError: A study's direction differs from the
        config's, or its failure-policy state is malformed.
    :raises TrialTargetRefusedError: A study already accepted a higher target
        than its phase requests, or a fixed configuration asks for more
        attempts than it may retry.
    :raises StudyContextConflictError: An upstream top-up would invalidate a
        descendant study already bound to its winner.
    """
    studies = {name: record.study for name, record in recorded.items() if record.study is not None}
    if _artifact_root_binding_applies(experiment):
        _check_published_phase_studies(experiment, studies, from_phase=from_phase)
    errors: list[Exception] = []
    reached = from_phase is None
    for phase in experiment.phases:
        if phase.name == from_phase:
            reached = True
        study = studies.get(phase.name)
        if study is None:
            continue
        record = recorded[phase.name]
        try:
            _validate_study_direction(study, experiment.metric.goal)
            # The only write ``_validate_study_schema`` makes stamps an empty,
            # unstamped study, which a snapshot cannot take and which cannot
            # fail the check.
            if study.get_trials(deepcopy=False) or STUDY_SCHEMA_ATTR in study.user_attrs:
                _validate_study_schema(study)
            if reached and record.running:
                _validate_fixed_configuration_target(
                    phase, finished=record.finished, completed=record.completed
                )
            elif reached:
                _validate_trial_target(study, phase)
        except Exception as exc:
            errors.append(exc)
    _raise_collected_errors(errors)
    settled = {name: study for name, study in studies.items() if not recorded[name].running}
    resume_ops._reject_bound_descendant_topups(
        experiment, from_phase=from_phase, existing_studies=settled
    )
    resume_ops._reject_fixed_targets_without_study(
        experiment, from_phase=from_phase, existing_studies=studies
    )


def _preview_ledger(
    experiment: Experiment, *, from_phase: str | None
) -> dict[str, RecordedPhase | None]:
    """Read what the ledger holds and refuse what a launch would refuse over it.

    :param Experiment experiment: Parsed experiment being previewed.
    :param str | None from_phase: Optional first phase a launch would execute.
    :raises ArtifactRootConflictError: The tree is bound to a different owner.
    :raises PhaseSweepError: A launch would refuse the recorded studies, as
        :func:`_refuse_as_a_launch_would` documents. A ledger that cannot be
        read is not refused here: the preview cannot tell what it holds, so it
        warns and previews every phase as fresh, where a launch refuses it.
    :return dict[str, RecordedPhase | None]: One entry per phase, by phase name;
        ``None`` for every phase when the ledger could not be read.
    """
    recorded = _read_recorded_phases(experiment)
    readable = {name: record for name, record in recorded.items() if record is not None}
    if len(readable) == len(recorded):
        _refuse_as_a_launch_would(experiment, readable, from_phase=from_phase)
    return recorded


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
    trials but not evidence-verified: that proof reads the trial directories,
    may query W&B or MLflow, and records the environment-drift warning once per
    process, so a preview leaves it, and that warning, to the launch.

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
) -> PhasePreview:
    """Log what a launch would do with the phase's recorded trials, and return the verdict.

    The recorded trials are judged by the rules a launch applies
    (:func:`_preview_resume_phase`), so the preview refuses what a launch
    refuses. A phase that would replay carries the winner a launch would
    republish, which its children inherit, and launches no trials. A phase with
    trials still to run carries a placeholder (:func:`_dry_run_phase`) and the
    ``remaining`` count its line prints. A phase whose recorded
    state could not be read is previewed as a fresh phase, and one whose
    RUNNING attempts a launch recovers first is previewed from its counts,
    with those attempts counted as recovery would record them and its recorded
    state unjudged; both carry a placeholder.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: The phase being previewed.
    :param dict[str, Winner] inherited_winners: Winners, or stand-ins for them,
        for phases earlier in the chain.
    :param RecordedPhase | None recorded: What the ledger holds for the phase,
        or ``None`` when that could not be read and the phase previews as fresh.
    :raises PhaseSweepError: A launch would refuse the phase's recorded state.
    :return PhasePreview: The winner a replay would publish or a placeholder, the
        trials a launch would run, and whether the phase replays.
    """
    finished = 0 if recorded is None else recorded.finished
    reruns = 0 if recorded is None else recorded.reruns
    remaining = max(0, phase.n_trials - finished)
    log.info(
        "phase=%s study=%s finished=%d remaining=%d reruns=%d n_jobs=%d",
        phase.name,
        _phase_study_name(experiment, phase),
        finished,
        remaining,
        reruns,
        phase.n_jobs,
    )
    study: optuna.Study | None = None
    resume: _PhaseResume | None = None
    if recorded is None:
        log.warning(
            "DRY RUN warning: phase=%s recorded trials could not be read, so it is "
            "previewed as a fresh phase; a real run refuses a ledger it cannot read",
            phase.name,
        )
    elif recorded.running:
        log.warning(
            "DRY RUN warning: phase=%s has %d RUNNING attempt(s) that a real run recovers "
            "before it checks the phase, as failed where the trial recorded a failure and "
            "otherwise as interrupted, which reruns; they are counted that way, and its "
            "accepted target and abort state are not checked",
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
                finished,
                phase.n_trials,
            )
            return PhasePreview(winner=resumed, launches=0, reruns=0, replays=True)
        resume = resumed
    if remaining == 0:
        log.info(
            "DRY RUN phase=%s target met (%d of %d trials recorded): would replay its "
            "existing result and launch no trials",
            phase.name,
            finished,
            phase.n_trials,
        )
        if study is None or resume is None:
            winner = _placeholder_winner(experiment, phase, inherited_winners)
        else:
            winner = _replay_met_target(
                experiment, phase, inherited_winners, study, resume, verify_evidence=False
            )
        return PhasePreview(winner=winner, launches=0, reruns=0, replays=True)
    winner = _dry_run_phase(
        experiment,
        phase,
        inherited_winners,
        open_preview_study(experiment, phase),
        remaining,
        reruns=reruns,
    )
    return PhasePreview(winner=winner, launches=remaining, reruns=reruns, replays=False)


def _dry_run_phase(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    study: optuna.Study,
    remaining: int,
    *,
    reruns: int,
) -> Winner:
    """Render and log one example trial command for the phase without launching anything.

    The log line names, by parent, the inherited keys whose values are stand-ins
    until that parent publishes a winner.

    Args:
        experiment: Parsed experiment config.
        phase: The phase being previewed.
        inherited_winners: Winners loaded for phases earlier in the chain.
        study: An in-memory Optuna study used to ``ask`` for one sample.
        remaining: Number of trials that *would* run; logged for the user.
        reruns: How many of them rerun interrupted trials before any new draw.

    Returns:
        A :class:`Winner` placeholder built from the previewed sample, or from
        each parameter's low bound or first choice, so downstream dry-run
        previews see consistent inherited context.

    """
    of_them = ""
    if reruns == 1:
        of_them = ", 1 of them a rerun of an interrupted trial"
    elif reruns:
        of_them = f", {reruns} of them reruns of interrupted trials"
    log.info(
        "DRY RUN phase=%s would launch %d %s%s",
        phase.name,
        remaining,
        "trial" if remaining == 1 else "trials",
        of_them,
    )
    sampled: dict[str, object] | None = None
    if remaining > 0:
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
        stand_ins = _inherited_placeholder_keys(
            phase, inherited_winners, replaced=[*phase.fixed_overrides, *sampled]
        )
        note = ""
        if stand_ins:
            groups = "; ".join(
                f"until {parent!r} publishes a winner: {', '.join(keys)}"
                for parent, keys in stand_ins.items()
            )
            note = f" (placeholders {groups})"
        log.info("DRY RUN example command%s:\n  %s", note, cmd)

    return _placeholder_winner(
        experiment,
        phase,
        inherited_winners,
        sampled_params=sampled,
    )
