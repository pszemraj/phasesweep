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
from phasesweep.engine.cleanup import project_stale_trials
from phasesweep.engine.errors import (
    StudyStorageUnavailableError,
    TrialEvidenceMissingError,
    TrialTargetRefusedError,
)
from phasesweep.engine.evidence import _validate_selection_evidence
from phasesweep.engine.fingerprints import (
    CarriesOverrides,
    _inherited_override_payload,
    _inherited_override_suppliers,
    _judge_identity,
    _local_keys,
    _phase_fingerprint,
    new_study_note,
    phase_study_name,
    reading_settings,
)
from phasesweep.engine.growth import describe_gains
from phasesweep.engine.guards import (
    _raise_collected_errors,
    _validate_known_study_direction,
    addressed_study,
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
from phasesweep.engine.reread import AdoptionPlan, plan_adoption
from phasesweep.engine.selection import PhaseAbortRefusedError, ReplayWithoutFeasibleTrialError
from phasesweep.engine.state import Winner
from phasesweep.engine.study_policy import (
    ReachedStudy,
    _accepted_trial_target,
    _launch_preflight,
    _load_accepted_partial_decision,
    _load_phase_policy_state,
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

    #: Attempts that count toward ``n_trials`` after projected recovery.
    finished: int
    #: Attempts that were RUNNING in the capture, before projected recovery.
    running: int
    #: The phase's study as a read-only snapshot, or ``None`` when the ledger holds none.
    study: optuna.Study | None
    #: What the study would take from the phase's other studies, when it would take anything.
    plan: AdoptionPlan | None = None


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
    recorded = {}
    for phase in experiment.phases:
        projected = project_stale_trials(experiment, phase.name, studies[phase.name])
        if projected.evidence_pending:
            raise PhaseSweepError(
                f"Phase {phase.name!r} has completed trainer evidence awaiting recovery, so the "
                "dry run cannot count its trials.",
                action=OperatorAction.RUN_RECOVER_RUN,
                remediation=(
                    "Run `phasesweep run` with the same config, without --dry-run, to recover the "
                    f"pending trial of `{phase.name}`."
                ),
            )
        recorded[phase.name] = [
            RecordedPhase(
                _finished_trial_count(study.get_trials(deepcopy=False)),
                projected.running.get(study.study_name, 0),
                study,
            )
            for study in projected.studies
        ]
    return recorded


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
    fresh = RecordedPhase(finished=0, running=0, study=None)
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
    record = fresh if addressed is None else addressed
    plan = plan_adoption(
        experiment,
        phase,
        inherited,
        record.study,
        [r.study for r in recorded if r.study is not None],
        study_name=name,
    )
    remaining = max(0, phase.n_trials - record.finished)
    if plan is not None and (
        plan.adoption.total or plan.partial or plan.launches != (remaining, remaining)
    ):
        return name, replace(record, plan=plan)
    if record.study is None or not record.study.get_trials(deepcopy=False):
        note = new_study_note(
            experiment,
            phase,
            inherited,
            [r.study for r in recorded if r.study is not None and r.study.study_name != name],
        )
        if note is not None:
            log.info("DRY RUN phase=%s %s", phase.name, note)
    return name, record


def validate_trial_targets(experiment: Experiment) -> None:
    """Refuse a fixed-configuration or grid target that ``run`` refuses over the recorded attempts.

    Only a target above what a fresh study allows (one attempt for a fixed configuration,
    the matrix size for a grid) needs the ledger, since only recorded failures can justify it.
    The attempts are counted as a launch counts them after its recovery: a RUNNING attempt
    that recovery settles as failed counts as finished. One whose completed trainer evidence
    awaits recovery also counts as a failure, the verdict that allows the most retries,
    because validation recovers nothing and a launch's extraction may still record one. The
    kept attempts a run would adopt before its first launch count too (:func:`plan_adoption`),
    each re-read as a failure for this check. A launch enforces the real ceiling once it has
    read the evidence.

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
    for phase in phases:
        if phase.inherits and every[phase.name]:
            continue
        projected = project_stale_trials(
            experiment, phase.name, every[phase.name], assume_pending_failures=True
        )
        name = phase_study_name(experiment, phase, {})
        study = next(
            (candidate for candidate in projected.studies if candidate.study_name == name), None
        )
        plan = plan_adoption(experiment, phase, {}, study, projected.studies, study_name=name)
        judged = study if plan is None else plan.failed
        trials = [] if judged is None else judged.get_trials(deepcopy=False)
        _validate_fixed_configuration_target(
            phase,
            finished=_finished_trial_count(trials),
            measured=_measured_draw_count(trials),
            accepted=0 if study is None else _accepted_trial_target(study),
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
    identity, then target, then the launch preflight of a phase that needs new trials, then
    the selection evidence. Preview inspects each study's artifact-root claim
    without making it, and leaves the winner's digest check to the run's selection.

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
    :raises TrialEvidenceMissingError: A candidate's evidence changed, or the trial a reached
        phase would select has no directory.
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
        reached.append(ReachedStudy(phase, known_study, siblings=studies[phase.name]))
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
    sampled placeholder. A phase with recorded studies that still needs its inherited
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
    if recorded is not None and recorded.plan is not None:
        return _preview_adoption(experiment, phase, inherited_winners, recorded, study_name)
    finished = 0 if recorded is None else recorded.finished
    reruns = 0
    remaining = max(0, phase.n_trials - finished)
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
    if recorded is not None and recorded.study is not None:
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
        if phase.inherits:
            _validate_selection_evidence(experiment, {phase.name: study})
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
            "phase=%s has %s; the preview projects recovery before planning its trials",
            phase.name,
            plural(recorded.running, "RUNNING attempt"),
        )
    if remaining == 0:
        log.info(
            "DRY RUN phase=%s target met (%d of %d trials recorded): would replay its existing "
            "result and launch no trials",
            phase.name,
            finished,
            phase.n_trials,
        )
        if study is None or resume is None or (recorded is not None and recorded.running):
            winner = _placeholder_winner(experiment, phase, inherited_winners)
        else:
            winner = _replay_met_target(
                experiment, phase, inherited_winners, study, resume, verify_evidence=False
            )
        return _replayed_phase_preview(phase, inherited_winners, winner)
    if recorded is None or phase.inherits:
        # A readable ledger had the launch of every reached phase whose study is known before
        # any phase runs preflighted before any phase previewed (_refuse_as_a_launch_would).
        # One previewed fresh from an unreadable ledger, or one whose study its parents'
        # winners pick, gets it here, as a run's does when it reaches the phase.
        _launch_preflight(experiment, phase.name, None if recorded is None else recorded.study)
    GpuPool.for_phase(phase, cuda_visible_devices=experiment.env.get("CUDA_VISIBLE_DEVICES"))
    return _preview_launches(experiment, phase, inherited_winners, remaining, reruns)


def _refusal_over_both_verdicts(
    phase: Phase, plan: AdoptionPlan, refusal: PhaseSweepError
) -> PhaseSweepError:
    """Name both retry ceilings when every verdict of the re-reads refuses the target.

    Which kept attempts measure decides how many retries a fixed configuration or grid has
    left, so the run's ceiling lies between the two projections'.

    :param Phase phase: The phase.
    :param AdoptionPlan plan: The projections.
    :param PhaseSweepError refusal: The refusal when every re-read measures.
    :return PhaseSweepError: A refusal naming both ceilings, or ``refusal`` when they agree or
        it is not a ceiling refusal.
    """
    ceilings = [
        retry_target_ceiling(
            phase,
            finished=_finished_trial_count(trials := projected.get_trials(deepcopy=False)),
            measured=_measured_draw_count(trials),
        )
        for projected in (plan.complete, plan.failed)
    ]
    measured, failed = ceilings
    if (
        not isinstance(refusal, TrialTargetRefusedError)
        or measured is None
        or failed is None
        or measured == failed
        or max(measured, failed) >= phase.n_trials
    ):
        return refusal
    return TrialTargetRefusedError(
        f"Phase {phase.name!r}: n_trials={phase.n_trials} exceeds what its kept attempts allow: "
        f"at most {measured} if every re-read measures, at most {failed} if every one fails.",
        remediation=f"Set phases[{phase.name}].n_trials to at most {min(measured, failed)}.",
    )


def _preview_adoption(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    recorded: RecordedPhase,
    study_name: str,
) -> PhasePreview:
    """Preview a phase whose study would take kept attempts from the phase's other studies.

    The run judges the study once adoption has written its rows, so the preview judges the
    projection at its first launch. A re-read's verdict is unknown before the run reads it: a
    projection whose re-reads are all assumed to measure and one whose re-reads all fail bound
    the launches, a refusal is certain only when both meet one of the same kind, and a winner
    selected over re-read rows is a placeholder.

    :param Experiment experiment: Parsed experiment config.
    :param Phase phase: The phase being previewed.
    :param dict[str, Winner] inherited_winners: Winners, or stand-ins, for earlier phases.
    :param RecordedPhase recorded: The addressed study's record, with its plan.
    :param str study_name: The addressed study's name.
    :raises PhaseSweepError: A launch would refuse the projected study, or its launch preflight
        (:func:`phasesweep.engine.study_policy._launch_preflight`) refuses.
    :return PhasePreview: The winner or a placeholder, and the launches when they are known.
    """
    plan = recorded.plan
    assert plan is not None
    if plan.adoption.total:
        log.info(
            "DRY RUN phase=%s %swould %s; no kept trial would retrain",
            phase.name,
            plan.adoption.changed_reading(reading_settings(experiment, phase)),
            plan.adoption.describe(planned=True),
        )
    study = plan.complete
    low, high = plan.launches
    high_or_unknown: int | None = high
    placeholder = plan.adoption.rereads > 0
    if plan.partial:
        decision = _load_accepted_partial_decision(study)
        assert decision is not None
        _require_partial_replay_allowed(phase, decision)
    else:
        # A refusal only one verdict of the re-reads would meet is not one the run is sure to.
        verdicts: list[_PhaseResume | Winner | PhaseSweepError] = []
        for projected in (plan.complete, plan.failed):
            try:
                verdict = _judge_recorded_phase(
                    experiment,
                    phase,
                    inherited_winners,
                    projected,
                    policy_state=_load_phase_policy_state(projected, experiment),
                    phase_fingerprint=_phase_fingerprint(experiment, phase, inherited_winners),
                    verify_evidence=False,
                )
                trials = projected.get_trials(deepcopy=False)
                if isinstance(verdict, _PhaseResume) and _finished_trial_count(trials) >= (
                    phase.n_trials
                ):
                    # The run selects from adopted rows that meet the target, as it would here.
                    _replay_met_target(
                        experiment,
                        phase,
                        inherited_winners,
                        projected,
                        verdict,
                        verify_evidence=False,
                    )
                verdicts.append(verdict)
            except PhaseSweepError as exc:
                verdicts.append(exc)
        refusals = [verdict for verdict in verdicts if isinstance(verdict, PhaseSweepError)]
        # Refusals of one kind bracket every mix of re-read outcomes; a ceiling refusal one way
        # and a streak abort the other leave a mix that meets neither, as the run may find.
        if len(refusals) == len(verdicts) and type(refusals[0]) is type(refusals[1]):
            raise _refusal_over_both_verdicts(phase, plan, refusals[0])
        if len(refusals) == len(verdicts):
            low, high_or_unknown = 0, None
        elif refusals:
            # Mixed verdicts land between the projections. All failing reads give the most
            # launches, all measuring ones the fewest; a failure adds retries only to a fixed
            # configuration or a grid, so elsewhere the measured count is the run's.
            kept = 1 if isinstance(verdicts[0], PhaseSweepError) else 0
            study = (plan.complete, plan.failed)[kept]
            if kept:
                low, high_or_unknown = 0, plan.by_verdict[1]
            elif isinstance(refusals[0], ReplayWithoutFeasibleTrialError):
                # The failing reads met the target before selection refused, so both counts hold.
                low, high_or_unknown = plan.launches
            else:
                low = plan.by_verdict[0]
                retries = retry_target_ceiling(phase, finished=0, measured=0) is not None
                high_or_unknown = None if retries else low
        for outcome, judged in zip(("measures", "fails"), verdicts, strict=True):
            if isinstance(judged, ReplayWithoutFeasibleTrialError) and len(refusals) == 1:
                # Every changed reading of a finished phase has this outcome if no read measures.
                continue
            if isinstance(judged, PhaseAbortRefusedError):
                warn_dry_run(
                    "phase=%s aborts if every re-read %s (max_consecutive_failures=%d)",
                    phase.name,
                    outcome,
                    phase.max_consecutive_failures,
                )
            elif isinstance(judged, PhaseSweepError):
                warn_dry_run(
                    "phase=%s is refused if every re-read %s: %s", phase.name, outcome, judged
                )
    finished = _finished_trial_count(study.get_trials(deepcopy=False))
    if plan.adoption.carried:
        # Every re-read and launch fails in this projection, so its only candidates are the rows
        # copied as recorded, which the run checks once it has adopted them.
        try:
            _validate_selection_evidence(experiment, {phase.name: plan.failed})
        except TrialEvidenceMissingError as exc:
            if not placeholder:
                raise
            warn_dry_run("phase=%s is refused if every re-read fails: %s", phase.name, exc)
    elif recorded.study is not None and recorded.study.get_trials(deepcopy=False):
        _validate_selection_evidence(experiment, {phase.name: recorded.study})
    if plan.partial or high_or_unknown == 0:
        log.info(
            "DRY RUN phase=%s %s: would select from the %s evidence and launch no trials",
            phase.name,
            f"accepted a partial result at {finished} of {phase.n_trials} trials"
            if plan.partial
            else f"target met ({finished} of {phase.n_trials} trials recorded)",
            "re-read" if placeholder else "recorded",
        )
        winner = _placeholder_winner(experiment, phase, inherited_winners)
        return PhasePreview(winner=winner, launches=0, reruns=0)
    if low > 0:
        if phase.inherits or recorded.study is None:
            _launch_preflight(experiment, phase.name, study)
        GpuPool.for_phase(phase, cuda_visible_devices=experiment.env.get("CUDA_VISIBLE_DEVICES"))
    if low == high_or_unknown:
        reruns = min(low, _pending_rerun_count(study, phase))
        _log_phase_header(
            phase,
            study_name,
            finished=finished,
            remaining=max(0, phase.n_trials - finished),
            reruns=reruns,
            logger=log,
        )
        return _preview_launches(experiment, phase, inherited_winners, low, reruns)
    if low > 0:
        log.info(
            "DRY RUN phase=%s would launch at least %s to fill the target",
            phase.name,
            plural(low, "trial"),
        )
    if high_or_unknown is not None:
        warn_dry_run(
            "phase=%s may need to launch up to %s depending on the re-read results",
            phase.name,
            plural(high_or_unknown, "trial"),
        )
    winner = _preview_trial_command(experiment, phase, inherited_winners)
    return PhasePreview(winner=winner, launches=None, reruns=0)


def replay_study_names(
    experiment: Experiment,
    every: Mapping[str, Sequence[optuna.Study]],
    latest: Mapping[str, CarriesOverrides],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Resolve phases whose parents can already replay under the inspected config.

    Just as a dry run would, select from the captured trials before using a winner's inherited
    values to address a child. If a parent needs more work, callers use the latest winner's
    address for counts and can still describe a conditional re-read.

    :param Experiment experiment: Config the next run would use.
    :param Mapping[str, Sequence[optuna.Study]] every: Phase studies from one journal capture.
    :param Mapping[str, CarriesOverrides] latest: Parent values from the captured winner files.
    :return tuple[dict[str, str], dict[str, tuple[str, ...]]]: Addressable studies, including historical forecasts, and the parents whose next choices could change those forecasts.
    """
    needed = {parent for phase in experiment.phases for parent in phase.inherits}
    winners: dict[str, Winner] = {}
    names: dict[str, str] = {}
    unresolved: dict[str, tuple[str, ...]] = {}
    missing: dict[str, set[str]] = {}
    for phase in experiment.phases:
        inherited_winners = {parent: winners[parent] for parent in phase.inherits}
        stand_ins = _inherited_placeholder_keys(phase, inherited_winners)
        missing_values = {
            key for parent, keys in stand_ins.items() for key in keys if key in missing[parent]
        }
        if phase.name in needed:
            previous = latest.get(phase.name)
            observed = {} if previous is None else previous.effective_overrides
            samples = _placeholder_values_for(phase.search_space)
            samples.update({key: observed[key] for key in samples.keys() & observed.keys()})
            winners[phase.name] = _placeholder_winner(
                experiment, phase, inherited_winners, sampled_params=samples
            )
            missing[phase.name] = missing_values | (phase.search_space.keys() - observed.keys())
        if stand_ins:
            unresolved[phase.name] = tuple(stand_ins)
        # A display can forecast from the latest measured values. Fabricated defaults have no study.
        if missing_values:
            continue
        inherited = _inherited_override_payload(phase, inherited_winners)
        name = phase_study_name(experiment, phase, inherited)
        names[phase.name] = name
        if stand_ins or phase.name not in needed:
            continue
        study = addressed_study(every, phase.name, name)
        if study is None:
            continue
        trials = study.get_trials(deepcopy=False)
        if any(trial.state == optuna.trial.TrialState.RUNNING for trial in trials):
            continue
        try:
            resumed = _judge_recorded_phase(
                experiment,
                phase,
                inherited_winners,
                study,
                policy_state=_load_phase_policy_state(study, experiment),
                phase_fingerprint=_judge_identity(study, experiment, phase, inherited).fingerprint,
                verify_evidence=False,
                emit_advisories=False,
            )
            if isinstance(resumed, Winner):
                winners[phase.name] = resumed
            elif _finished_trial_count(trials) >= phase.n_trials:
                winners[phase.name] = _replay_met_target(
                    experiment, phase, inherited_winners, study, resumed, verify_evidence=False
                )
        except PhaseSweepError:
            # Normal status diagnostics handle the refusal using this study's captured rows.
            continue
    return names, unresolved


def _preview_launches(
    experiment: Experiment,
    phase: Phase,
    inherited_winners: dict[str, Winner],
    launches: int,
    reruns: int,
) -> PhasePreview:
    """Use the same exact counts in the phase message and the run total.

    :param Experiment experiment: Configuration whose command is being previewed.
    :param Phase phase: Phase that needs the launches.
    :param dict[str, Winner] inherited_winners: Winners that supply the phase's inherited values.
    :param int launches: Number of physical launches still required.
    :param int reruns: Required launches that resume or retry an earlier trial.
    :return PhasePreview: Projected winner and the reported counts.
    """
    of_them = f", {rerun_clause(reruns)}" if reruns else ""
    log.info("DRY RUN phase=%s would launch %s%s", phase.name, plural(launches, "trial"), of_them)
    winner = _preview_trial_command(experiment, phase, inherited_winners)
    return PhasePreview(winner=winner, launches=launches, reruns=reruns)


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
