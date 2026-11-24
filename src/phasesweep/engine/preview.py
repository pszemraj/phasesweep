"""Read-only dry-run planning from captured ledger studies.

Apply launch checks in launch order using snapshots and in-memory sample studies.
Preview reports replay, refusal, rerun, and remaining-trial decisions without
launching trainers or changing the configured ledger or artifact tree.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import optuna

import phasesweep.engine.resume as resume_ops
from phasesweep.config.models import Experiment, Phase
from phasesweep.config.search import (
    _placeholder_values_for,
    retry_target_ceiling,
)
from phasesweep.engine.artifact_roots import (
    _artifact_root_claim_needed,
    _check_published_phase_studies,
)
from phasesweep.engine.errors import StudyStorageUnavailableError
from phasesweep.engine.fingerprints import (
    _inherited_override_payload,
    _inherited_override_suppliers,
    _judge_identity,
    _local_keys,
    _phase_fingerprint,
    new_study_note,
    phase_study_name,
)
from phasesweep.engine.growth import describe_gains
from phasesweep.engine.guards import (
    _raise_collected_errors,
    _validate_known_study_direction,
    known_studies,
)
from phasesweep.engine.ledger import open_preview_study, read_every_phase_study, validate_ledger
from phasesweep.engine.optuna import (
    _finished_trial_count,
    _measured_draw_count,
    _phase_namespace,
    _suggest_all,
)
from phasesweep.engine.paths import _phase_dir
from phasesweep.engine.phase import (
    _composed_overrides,
    _judge_recorded_phase,
    _PhaseResume,
    _replay_met_target,
    _require_partial_replay_allowed,
    _winner_completion,
    _winner_overrides,
)
from phasesweep.engine.progress import _log_phase_header
from phasesweep.engine.reread import FillSource, describe_changes, fill_source
from phasesweep.engine.state import (
    PHASE_DECISION_ATTR,
    RecoveredTrialState,
    Winner,
    _recorded_identity,
    _recovered_trial_state,
    _unique_trial_outcomes,
)
from phasesweep.engine.study_policy import (
    ReachedStudy,
    _at_accepted_partial_target,
    _launch_preflight,
    _load_phase_policy_state,
    _parse_accepted_partial_decision,
    _pending_rerun_count,
    _reached_study_refusals,
    _validate_fixed_configuration_target,
    _validate_study_direction,
    _validate_study_schema,
)
from phasesweep.errors import OperatorAction, PhaseSweepError
from phasesweep.run_report import rerun_clause
from phasesweep.runtime.commands import render_command
from phasesweep.runtime.gpu import GpuPool
from phasesweep.runtime.terminal import REPORT_LOGGER, plural

# Every line of a preview is for the person reading the dry run, so all of it is report.
log = logging.getLogger(REPORT_LOGGER)


def warn_dry_run(message: str, *args: object) -> None:
    """Log a dry-run warning under the one label that readers and tests scan for.

    :param str message: Logging format string, without the label.
    :param object args: Values substituted into the format string.
    """
    log.warning(f"DRY RUN warning: {message}", *args)


@dataclass(frozen=True, slots=True)
class RecordedPhase:
    """What a ledger already holds for one phase, counted as a launch would count it."""

    #: Attempts that count toward ``n_trials``: terminal ones that were not interrupted,
    #: plus each RUNNING one whose objective recorded a failure, which a launch recovers
    #: as failed first. Any other RUNNING attempt recovers as interrupted and reruns.
    finished: int
    #: Draws with a COMPLETE attempt, feasible or not.
    measured: int
    #: How many attempts are RUNNING, which a launch recovers before it checks anything.
    running: int
    #: Predicted reruns while the study still needs recovery: interrupted and already queued
    #: attempts plus RUNNING attempts that recover as interrupted. With no recovery pending,
    #: preview instead counts reruns, including grid retries, after the phase passes preflight.
    recovery_reruns: int
    #: The phase's study as a read-only snapshot, or ``None`` when the ledger holds none.
    study: optuna.Study | None
    #: The source of a pending fill, whose old scores cannot supply a winner.
    fill: FillSource | None = None
    #: Whether the next run can launch work only if the re-read leaves enough retry capacity.
    fill_retry_uncertain: bool = False


@dataclass(frozen=True, slots=True)
class PhasePreview:
    """What a dry run says one previewed phase would do."""

    #: The winner the phase hands its children: a replayed one, or a placeholder.
    winner: Winner
    #: Trials a launch would run, or None while an inheritance or re-read verdict is pending.
    launches: int | None
    #: How many of those launches rerun an interrupted trial or retry a failed grid cell.
    reruns: int
    #: Whether the phase would replay a result already read under the current settings.
    replays: bool = False


def _dry_run_total_line(previews: Sequence[PhasePreview]) -> str:
    """Total the launch counts the phase lines establish and report how many remain open.

    :param Sequence[PhasePreview] previews: The previewed phases, in order. A phase that
        ``--from-phase`` skips is not previewed and so is not counted.
    :return str: The line, without the ``DRY RUN`` prefix.
    """
    trials = sum(preview.launches for preview in previews if preview.launches is not None)
    reruns = sum(preview.reruns for preview in previews)
    launching = sum(1 for preview in previews if preview.launches)
    replaying = sum(1 for preview in previews if preview.replays)
    unknown = sum(1 for preview in previews if preview.launches is None)
    line = f"total: would launch {plural(trials, 'trial')} "
    if reruns:
        line += f"({plural(reruns, 'rerun')}) "
    line += f"across {plural(launching, 'phase')}"
    if unknown and not launching:
        line = "total: no launches for phases with known counts"
    if replaying:
        line += f" ({replaying} other {'phase replays' if replaying == 1 else 'phases replay'})"
    if unknown:
        line += f". Launch counts are unresolved for {plural(unknown, 'phase')}"
    return line


def _recovered_running_states(
    study_name: str, trials: Sequence[optuna.trial.FrozenTrial]
) -> list[RecoveredTrialState]:
    """Predict what a launch's recovery settles each RUNNING attempt as.

    :param str study_name: Phase study being previewed.
    :param Sequence[optuna.trial.FrozenTrial] trials: The same study's trials.
    :return list[RecoveredTrialState]: One prediction per RUNNING trial, in trial order.
    """
    outcomes = _unique_trial_outcomes(study_name, trials)
    return [
        _recovered_trial_state(trial, outcomes.get(trial.number))
        for trial in trials
        if trial.state == optuna.trial.TrialState.RUNNING
    ]


def _recorded_phase(
    phase: Phase,
    study: optuna.Study | None,
    *,
    trials: Sequence[optuna.trial.FrozenTrial] | None = None,
) -> RecordedPhase:
    """Count one phase's snapshot study the way a launch counts it.

    :param Phase phase: The configured phase the study belongs to.
    :param optuna.Study | None study: The phase's snapshot study, or ``None`` when
        the ledger holds no such study, which counts zero attempts.
    :param Sequence[optuna.trial.FrozenTrial] | None trials: The source trials of a pending fill,
        or None to count the full study.
    :raises PhaseSweepError: A RUNNING attempt's completed trainer evidence awaits recovery,
        whose extraction decides the counts.
    :return RecordedPhase: The counts and the study.
    """
    if study is None:
        return RecordedPhase(finished=0, measured=0, running=0, recovery_reruns=0, study=None)
    if trials is None:
        trials = study.get_trials(deepcopy=False)
    recovered = _recovered_running_states(study.study_name, trials)
    if "EVIDENCE_PENDING" in recovered:
        raise PhaseSweepError(
            f"Phase {phase.name!r} has completed trainer evidence awaiting recovery, so the "
            "dry run cannot count its trials.",
            action=OperatorAction.RUN_RECOVER_RUN,
            remediation=(
                "Run `phasesweep run` with the same config, without --dry-run, to recover the "
                f"pending trial of `{phase.name}`."
            ),
        )
    recovered_as_failed = recovered.count("FAIL")
    return RecordedPhase(
        finished=_finished_trial_count(trials) + recovered_as_failed,
        measured=_measured_draw_count(trials),
        running=len(recovered),
        recovery_reruns=(
            _pending_rerun_count(study) + len(recovered) - recovered_as_failed if recovered else 0
        ),
        study=study,
    )


#: Every study of each phase as the ledger holds it, oldest first, by phase name.
RecordedStudies = Mapping[str, Sequence[RecordedPhase]]


def _read_recorded_phases(experiment: Experiment) -> RecordedStudies | None:
    """Read every study of every phase from one write-free ledger capture.

    A phase that inherits has a study for each set of values it inherited; the preview picks
    the one its parents' winners address as it reaches the phase. An unreadable ledger leaves
    phases unknown; preview warns and uses fresh in-memory studies.

    :param Experiment experiment: Parsed experiment being previewed.
    :raises ArtifactRootConflictError: The tree is bound to a different owner.
    :raises StudySchemaMismatchError: A ledger study uses an unsupported schema.
    :raises PhaseSweepError: A RUNNING attempt's completed trainer evidence awaits recovery.
    :return RecordedStudies | None: Each phase's studies, a phase without one mapping to an
        empty list; ``None`` when the ledger could not be read.
    """
    try:
        studies = read_every_phase_study(
            validate_ledger(experiment), [phase.name for phase in experiment.phases]
        )
    except StudyStorageUnavailableError as exc:
        warn_dry_run(
            "the ledger could not be inspected, and a real run refuses a ledger it cannot read: %s",
            exc,
        )
        return None
    return {
        phase.name: [_recorded_phase(phase, study) for study in studies[phase.name]]
        for phase in experiment.phases
    }


def _addressed_record(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    recorded: Sequence[RecordedPhase],
) -> tuple[str | None, RecordedPhase | None]:
    """Pick what the ledger holds for the study a phase addresses under its parents' winners.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: The phase being previewed.
    :param dict[str, Winner] inherited_winners: Winners, or stand-ins for them.
    :param Sequence[RecordedPhase] recorded: Every study the phase has.
    :return tuple[str | None, RecordedPhase | None]: The study name, or ``None`` while a parent's
        winner is a stand-in, and its record. The record is ``None`` when a stand-in prevents
        picking among recorded studies, or empty when the phase is known to have no study.
    """
    fresh = RecordedPhase(finished=0, measured=0, running=0, recovery_reruns=0, study=None)
    stand_ins = _inherited_placeholder_keys(phase, inherited_winners)
    if stand_ins and not any(record.study is not None for record in recorded):
        # Nothing recorded to reuse, so the study it reaches is new whatever the winner.
        return None, fresh
    if stand_ins:
        groups = "; ".join(f"{parent!r} ({', '.join(keys)})" for parent, keys in stand_ins.items())
        warn_dry_run(
            "phase=%s study and launch count depend on the inherited values selected by %s. "
            "Earlier trials may be reused",
            phase.name,
            groups,
        )
        return None, None
    inherited = _inherited_override_payload(phase, inherited_winners)
    name = phase_study_name(experiment, phase, inherited)
    addressed = next(
        (r for r in recorded if r.study is not None and r.study.study_name == name), None
    )
    if addressed is not None and addressed.study is not None:
        _validate_study_direction(addressed.study, experiment.metric.goal)
    if addressed is not None and (
        addressed.study is None or _recorded_identity(addressed.study) is not None
    ):
        return name, addressed
    others = [r.study for r in recorded if r.study is not None and r.study.study_name != name]
    source = fill_source(
        experiment, phase, inherited, others, target=None if addressed is None else addressed.study
    )
    if source is not None:
        # A fill keeps the results already copied and the source's remaining trainer attempts.
        log.info(
            "DRY RUN phase=%s reading settings changed (%s): would re-read %s from kept "
            "evidence; nothing would retrain",
            phase.name,
            describe_changes(source.changes),
            plural(source.rereads, "finished trial"),
        )
        copied = (
            []
            if addressed is None or addressed.study is None
            else addressed.study.get_trials(deepcopy=False)
        )
        trials = [*copied, *source.trials[len(copied) :]]
        stand_in = _recorded_phase(phase, source.study, trials=trials)
        uncertain = False
        if phase.n_trials > stand_in.finished:
            # Each draw that might measure can remove one retry from the available budget.
            ceiling = retry_target_ceiling(
                phase,
                finished=stand_in.finished,
                measured=_measured_draw_count(trials, assumed_complete=source.reread_trials),
            )
            uncertain = ceiling is not None and phase.n_trials > ceiling
        if uncertain:
            _validate_fixed_configuration_target(
                phase,
                finished=stand_in.finished,
                measured=_measured_draw_count(trials, excluded=source.reread_trials),
            )
        return name, replace(stand_in, study=None, fill=source, fill_retry_uncertain=uncertain)
    note = new_study_note(
        phase,
        inherited,
        [record.study for record in recorded if record.study is not None],
    )
    if note is not None:
        log.info("DRY RUN phase=%s %s", phase.name, note)
    return name, fresh


def validate_trial_targets(experiment: Experiment) -> None:
    """Refuse a fixed-configuration or grid target that ``run`` refuses over the recorded attempts.

    Only a target above what a fresh study allows (one attempt for a fixed configuration,
    the matrix size for a grid) needs the ledger, since only recorded failures can justify it.
    The attempts are counted as a launch counts them after its recovery: a RUNNING attempt
    that recovery settles as failed counts as finished. One whose completed trainer evidence
    awaits recovery also counts as a failure, the verdict that allows the most retries,
    because validation recovers nothing and a launch's extraction may still record one. A
    study waiting for a re-read uses its source's counts, and a trial still to be re-read also
    counts as a failure for this check. A launch enforces the real ceiling once it has read
    the evidence.

    A phase that inherits is judged here only when it has no study at all: which of its
    studies a launch addresses depends on winners its parents have yet to select, and the
    launch judges that one when it reaches the phase.

    :param Experiment experiment: Config whose targets may retry recorded failures.
    :raises TrialTargetRefusedError: A fixed configuration or grid requests more attempts
        than its recorded attempts let it retry.
    :raises StudyStorageUnavailableError: The ledger cannot be read to justify a raised target.
    """
    phases = [
        phase
        for phase in experiment.phases
        if (fresh := retry_target_ceiling(phase, finished=0, measured=0)) is not None
        and phase.n_trials > fresh
    ]
    if not phases:
        return
    every = read_every_phase_study(validate_ledger(experiment), [phase.name for phase in phases])
    known = known_studies(experiment, every)
    for phase in phases:
        if phase.inherits and every[phase.name]:
            continue
        study = known.get(phase.name)
        trials = [] if study is None else study.get_trials(deepcopy=False)
        unread: frozenset[int] = frozenset()
        if study is None or _recorded_identity(study) is None:
            source = fill_source(
                experiment,
                phase,
                {},
                [other for other in every[phase.name] if other is not study],
                target=study,
            )
            if source is not None:
                study = source.study
                unread = source.reread_trials
                trials = [*trials, *source.trials[len(trials) :]]
        recovered = [] if study is None else _recovered_running_states(study.study_name, trials)
        _validate_fixed_configuration_target(
            phase,
            finished=_finished_trial_count(trials)
            + recovered.count("FAIL")
            + recovered.count("EVIDENCE_PENDING"),
            measured=_measured_draw_count(trials, excluded=unread),
        )


def _refuse_as_a_launch_would(
    experiment: Experiment,
    recorded: RecordedStudies,
    *,
    from_phase: str | None,
) -> None:
    """Apply claim and preflight refusals to snapshots in launch order.

    The reached studies get the checks a launch gives them after its recovery, from the same
    function (:func:`phasesweep.engine.study_policy._reached_study_refusals`): recorded
    identity, then target, then the launch preflight of a phase that needs new trials. A
    RUNNING study gets the fixed-target check on predicted recovery counts in place of its
    accepted target. Preview inspects each study's artifact-root claim without making it,
    and does not verify evidence.

    :param Experiment experiment: Parsed experiment being previewed.
    :param RecordedStudies recorded: Every study the ledger holds for each phase.
    :param str | None from_phase: Optional first phase a launch would execute.
    :raises PublishedStudyMissingError: A reached phase has a published winner
        whose trial is missing from, or below the history boundary of, its study.
    :raises PublicationAccessError: The last publication cannot be read.
    :raises PublicationIntegrityError: The last publication no longer validates.
    :raises ArtifactRootConflictError: A study publishes into another artifact root, or
        holds trials without recording one.
    :raises LaunchEnvironmentError: A reached phase cannot start its trials.
    :raises StudySchemaMismatchError: A study's direction differs from the
        config's, or its failure-policy state is malformed.
    :raises StudyFingerprintMismatchError: A reached study recorded another config, or its
        trials ran under another trainer environment.
    :raises TrialTargetRefusedError: A study already accepted a higher target
        than its phase requests, or a fixed configuration asks for more
        attempts than it may retry.
    """
    studies = {
        name: tuple(record.study for record in records if record.study is not None)
        for name, records in recorded.items()
        if records
    }
    _check_published_phase_studies(experiment, studies, from_phase=from_phase)
    for bound_study in (study for every in studies.values() for study in every):
        _artifact_root_claim_needed(bound_study, experiment)
    resume_ops._preflight_missing_reached_phase_environments(
        experiment, studies, from_phase=from_phase
    )
    errors: list[Exception] = []
    checked: dict[str, tuple[optuna.Study, ...]] = {}
    records = {
        record.study.study_name: record
        for every in recorded.values()
        for record in every
        if record.study is not None
    }
    for phase in experiment.phases:
        for study in studies.get(phase.name, ()):
            try:
                _validate_known_study_direction(experiment, phase, study)
                _validate_study_schema(study)
            except Exception as exc:
                errors.append(exc)
            else:
                checked[phase.name] = (*checked.get(phase.name, ()), study)
    known = known_studies(experiment, checked)
    reached: list[ReachedStudy] = []
    for _index, phase in experiment.phases_from(from_phase):
        known_study = known.get(phase.name)
        if known_study is None:
            continue
        record = records[known_study.study_name]
        predicted = (record.finished, record.measured) if record.running else None
        reached.append(ReachedStudy(phase, known_study, predicted=predicted))
    errors.extend(_reached_study_refusals(experiment, reached))
    _raise_collected_errors(errors)
    resume_ops._reject_fixed_targets_without_study(
        experiment, from_phase=from_phase, existing_studies=studies
    )


def _preview_ledger(experiment: Experiment, *, from_phase: str | None) -> RecordedStudies | None:
    """Read what the ledger holds and refuse what a launch would refuse over it.

    :param Experiment experiment: Parsed experiment being previewed.
    :param str | None from_phase: Optional first phase a launch would execute.
    :raises ArtifactRootConflictError: The tree is bound to a different owner.
    :raises PhaseSweepError: A launch would refuse the recorded studies, as
        :func:`_refuse_as_a_launch_would` documents. A ledger that cannot be
        read is not refused here: the preview cannot tell what it holds, so it
        warns and previews every phase as fresh, where a launch refuses it.
    :return RecordedStudies | None: Every study of each phase, or ``None`` when the ledger
        could not be read.
    """
    recorded = _read_recorded_phases(experiment)
    if recorded is not None:
        _refuse_as_a_launch_would(experiment, recorded, from_phase=from_phase)
    return recorded


def _preview_phase(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    recorded_studies: Sequence[RecordedPhase] | None,
) -> PhasePreview:
    """Log and return the phase's predicted replay or trial work.

    The phase previews the study its parents' winners address, as a run would open it. A
    replay supplies its recorded winner to children. Remaining work supplies a
    sampled placeholder. Unreadable studies and RUNNING studies needing recovery
    also use placeholders; the latter retain predicted counts without judging
    unrecovered state. A phase with recorded studies that still needs its inherited
    values, or whose retry capacity depends on a reread, shows an example command
    and leaves its launch count unresolved.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: The phase being previewed.
    :param dict[str, Winner] inherited_winners: Winners, or stand-ins for them,
        for phases earlier in the chain.
    :param Sequence[RecordedPhase] | None recorded_studies: Every study the ledger holds for
        the phase, or ``None`` when that could not be read and the phase previews as fresh.
    :raises PhaseSweepError: A launch would refuse the phase's recorded state, or, for a phase
        previewed fresh from an unreadable ledger, its launch preflight
        (:func:`phasesweep.engine.study_policy._launch_preflight`) refuses.
    :return PhasePreview: The winner a replay would publish or a placeholder, the
        trials a launch would run, and whether the phase replays.
    """
    recorded: RecordedPhase | None = None
    study_name = _phase_namespace(experiment.experiment, phase.name)
    if recorded_studies is not None:
        addressed, recorded = _addressed_record(
            experiment, phase, inherited_winners, recorded_studies
        )
        if recorded is None:
            winner = _preview_trial_command(experiment, phase, inherited_winners)
            return PhasePreview(winner=winner, launches=None, reruns=0)
        study_name = study_name if addressed is None else addressed
    fill = None if recorded is None else recorded.fill
    if fill is not None:
        partial = _parse_accepted_partial_decision(
            fill.attrs.get(PHASE_DECISION_ATTR), fill.study.study_name
        )
        if _at_accepted_partial_target(partial, phase):
            assert partial is not None
            _require_partial_replay_allowed(phase, partial)
            log.info(
                "DRY RUN phase=%s accepted a partial result at %d of %d trials: would select "
                "from the re-read evidence and launch no trials",
                phase.name,
                partial.finished_trials,
                partial.trial_target,
            )
            winner = _placeholder_winner(experiment, phase, inherited_winners)
            return PhasePreview(winner=winner, launches=0, reruns=0)
    finished = 0 if recorded is None else recorded.finished
    reruns = 0 if recorded is None else recorded.recovery_reruns
    remaining = max(0, phase.n_trials - finished)
    if recorded is not None and recorded.fill_retry_uncertain:
        warn_dry_run(
            "phase=%s has %d of %d trials recorded. Its launch count depends on how much retry "
            "capacity remains after re-reading the evidence",
            phase.name,
            finished,
            phase.n_trials,
        )
        winner = _preview_trial_command(experiment, phase, inherited_winners)
        return PhasePreview(winner=winner, launches=None, reruns=0)
    study: optuna.Study | None = None
    resume: _PhaseResume | None = None
    phase_fingerprint = _phase_fingerprint(experiment, phase, inherited_winners)
    if (
        recorded is not None
        and recorded.study is not None
        and recorded.study.get_trials(deepcopy=False)
    ):
        # Judged only: the preview reads a snapshot and records nothing.
        verdict = _judge_identity(
            recorded.study,
            experiment,
            phase,
            _inherited_override_payload(phase, inherited_winners),
        )
        phase_fingerprint = verdict.fingerprint
        if verdict.gained:
            log.info(
                "DRY RUN phase=%s search_space gains %s: keeps its %s.",
                phase.name,
                describe_gains(verdict.gained, sampler=phase.sampler),
                plural(finished, "finished trial"),
            )
    if recorded is not None and not recorded.running and recorded.study is not None:
        study = recorded.study
        reruns = _pending_rerun_count(study, phase)
        # Empty snapshots need no fingerprint stamp or comparison.
        resumed = _judge_recorded_phase(
            experiment,
            phase,
            inherited_winners,
            study,
            policy_state=_load_phase_policy_state(study, experiment),
            phase_fingerprint=phase_fingerprint,
            verify_evidence=False,
        )
        if isinstance(resumed, Winner):
            # A run replays an accepted partial result before it logs a header.
            log.info(
                "DRY RUN phase=%s accepted a partial result at %d of %d trials: would "
                "replay it and launch no trials",
                phase.name,
                finished,
                phase.n_trials,
            )
            return _replayed_phase_preview(phase, inherited_winners, resumed)
        resume = resumed
    _log_phase_header(
        phase,
        study_name,
        finished=finished,
        remaining=remaining,
        reruns=reruns,
        logger=log,
    )
    if recorded is None:
        warn_dry_run(
            "phase=%s recorded trials could not be read, so it is "
            "previewed as a fresh phase; a real run refuses a ledger it cannot read",
            phase.name,
        )
    elif recorded.running:
        warn_dry_run(
            "phase=%s has %s, counted the way a real run recovers RUNNING attempts (failed "
            "if the trial recorded a failure, otherwise interrupted and rerun); its accepted "
            "target, abort state and partial decision are not checked",
            phase.name,
            plural(recorded.running, "RUNNING attempt"),
        )
    if remaining == 0:
        selection = (
            "replay its existing result" if fill is None else "select from the re-read evidence"
        )
        log.info(
            "DRY RUN phase=%s target met (%d of %d trials recorded): would %s and launch no trials",
            phase.name,
            finished,
            phase.n_trials,
            selection,
        )
        if study is None or resume is None:
            winner = _placeholder_winner(experiment, phase, inherited_winners)
        else:
            winner = _replay_met_target(
                experiment, phase, inherited_winners, study, resume, verify_evidence=False
            )
        if fill is not None:
            return PhasePreview(winner=winner, launches=0, reruns=0)
        return _replayed_phase_preview(phase, inherited_winners, winner)
    if recorded is None or phase.inherits:
        # A readable ledger had the launch of every reached phase whose study is known before
        # any phase runs preflighted before any phase previewed (_refuse_as_a_launch_would).
        # One previewed fresh from an unreadable ledger, or one whose study its parents'
        # winners pick, gets it here, as a run's does when it reaches the phase.
        _launch_preflight(experiment, phase.name, None if recorded is None else recorded.study)
    GpuPool.for_phase(phase, cuda_visible_devices=experiment.env.get("CUDA_VISIBLE_DEVICES"))
    of_them = f", {rerun_clause(reruns)}" if reruns else ""
    log.info("DRY RUN phase=%s would launch %s%s", phase.name, plural(remaining, "trial"), of_them)
    winner = _preview_trial_command(experiment, phase, inherited_winners)
    return PhasePreview(winner=winner, launches=remaining, reruns=reruns)


def _inherited_placeholder_keys(
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> dict[str, list[str]]:
    """Name stand-ins among the values the current phase actually inherits.

    :param Phase phase: The phase whose inheritance order decides which parent supplies a key.
    :param dict[str, Winner] inherited_winners: Parent-phase winners, or stand-ins for them.
    :return dict[str, list[str]]: Sorted stand-in keys per parent, in inheritance order, without
        the parents that supply none.
    """
    suppliers = _inherited_override_suppliers(phase, inherited_winners)
    inherited = suppliers.keys() - _local_keys(phase)
    return {
        parent: sorted(keys)
        for parent in phase.inherits
        if (
            keys := {
                key
                for key in inherited_winners[parent].placeholder_keys & inherited
                if suppliers[key] == parent
            }
        )
    }


def _replayed_phase_preview(
    phase: Phase, inherited_winners: dict[str, Winner], winner: Winner
) -> PhasePreview:
    """Carry unresolved inherited values through a replay to later phases.

    :param Phase phase: Replayed phase.
    :param dict[str, Winner] inherited_winners: Parent winners, including any stand-ins.
    :param Winner winner: Replayed or placeholder winner with composed overrides.
    :return PhasePreview: Conditional replay and the values it passes on.
    """
    carried = _inherited_placeholder_keys(phase, inherited_winners)
    stand_ins = winner.placeholder_keys.union(*carried.values()) & winner.effective_overrides.keys()
    return PhasePreview(
        winner=replace(winner, placeholder_keys=stand_ins), launches=0, reruns=0, replays=True
    )


def _placeholder_winner(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    *,
    sampled_params: dict[str, object] | None = None,
) -> Winner:
    """Build a dry-run winner from preview samples or deterministic parameter defaults.

    Carry inherited overrides, omit the repeat key, and mark searched or inherited
    stand-in keys unless fixed overrides replace them.

    :param Experiment experiment: Experiment defining the phase graph.
    :param Phase phase: Phase needing a placeholder.
    :param dict[str, Winner] inherited_winners: Winners from earlier phases.
    :param dict[str, object] | None sampled_params: Preview samples, or None for low bounds and first choices.
    :return Winner: Placeholder with trial number -1 and a NaN metric.
    """
    placeholder_params = (
        _placeholder_values_for(phase.search_space)
        if sampled_params is None
        else {name: value for name, value in sampled_params.items() if name in phase.search_space}
    )
    effective = _winner_overrides(experiment, phase, placeholder_params, inherited_winners)
    carried = _inherited_placeholder_keys(phase, inherited_winners)
    stand_ins = set(placeholder_params).union(*carried.values())
    return Winner(
        trial_number=-1,
        params=placeholder_params,
        effective_overrides=effective,
        placeholder_keys=frozenset(stand_ins & effective.keys()),
        metric=float("nan"),
        completion=_winner_completion(
            requested_trials=phase.n_trials,
            finished_trials=0,
            measured_trials=0,
            incomplete=True,
            reason="dry_run",
            timeout_scope=None,
        ),
    )


def _preview_trial_command(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
) -> Winner:
    """Render and log one example trial command for the phase without launching anything.

    The log line names, by parent, the inherited keys whose values are stand-ins
    until that parent publishes a winner.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: The phase being previewed.
    :param dict[str, Winner] inherited_winners: Winners loaded for phases earlier in the chain.
    :return Winner: A placeholder built from the previewed sample, with the repeat key
        omitted, so downstream dry-run previews see consistent inherited context.
    """
    sample_trial = open_preview_study(experiment, phase).ask()
    sampled = _suggest_all(sample_trial, phase)
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
    stand_ins = _inherited_placeholder_keys(phase, inherited_winners)
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
