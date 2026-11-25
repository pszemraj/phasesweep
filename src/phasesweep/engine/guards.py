"""Existing-study preflight orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import optuna

from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.artifacts import _refresh_recovered_trials_csv
from phasesweep.engine.attempts import (
    _AttemptRecord,
    _preflight_active_attempts,
    _PreflightCleanupReport,
    _retire_active_attempt,
)
from phasesweep.engine.cleanup import (
    _reap_stale_trials,
    _recover_cleanup_uncertain_trials,
)
from phasesweep.engine.errors import (
    LaunchEnvironmentError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrialTargetRefusedError,
)
from phasesweep.engine.fingerprints import _recorded_goal, phase_study_name
from phasesweep.engine.ledger import ClaimedLedger, claim_ledger, validate_ledger
from phasesweep.engine.study_policy import (
    ReachedStudy,
    _initialize_study_schema,
    _reached_study_refusals,
    _validate_study_direction,
)
from phasesweep.errors import (
    OperatorAction,
    PhaseSweepError,
    ProcessCleanupUncertainError,
    operator_step,
)


def _validate_known_study_direction(
    experiment: Experiment, phase: Phase, study: optuna.Study
) -> None:
    """Check a saved study's direction when its goal is known before phase selection.

    :param Experiment experiment: Config that determines the reached root study.
    :param Phase phase: Config for the phase that owns the study.
    :param optuna.Study study: Existing study, which may be waiting for its identity.
    :raises StudySchemaMismatchError: The study's stored direction contradicts its known goal.
    """
    goal = _recorded_goal(study)
    if (
        goal is None
        and not phase.inherits
        and study.study_name == phase_study_name(experiment, phase, {})
    ):
        goal = experiment.metric.goal
    if goal is not None:
        _validate_study_direction(study, goal)


def _reconcile_existing_studies(
    experiment: Experiment,
    *,
    cleanup_report: _PreflightCleanupReport,
    from_phase: str | None = None,
) -> dict[str, tuple[optuna.Study, ...]]:
    """Rediscover and reconcile every existing study after a run has ended.

    Validate and claim a fresh ledger so post-run preflight includes studies
    created during execution.

    :param Experiment experiment: Parsed experiment whose declared phases are inspected.
    :param _PreflightCleanupReport cleanup_report: Report the reconciliation's
        cleanup evidence accumulates into.
    :param str | None from_phase: Optional resume point, as for
        :func:`_preflight_existing_studies`.
    :return dict[str, tuple[optuna.Study, ...]]: Every existing study of each phase, by phase
        name.
    :raises ArtifactRootConflictError: The tree or a phase's persistent study
        is bound to a different owner; raised before any reaping.
    :raises StudyStorageUnavailableError: A phase's persistent storage could not
        be inspected; raised before any reaping or registry recovery.
    :raises PublishedStudyMissingError: A reached phase has a published winner
        but its persistent study is absent or empty.
    :raises PhaseSweepError: Preflight over the discovered studies refused, as
        documented on :func:`_preflight_existing_studies`.
    """
    try:
        claimed = claim_ledger(validate_ledger(experiment), from_phase=from_phase)
    except Exception as exc:
        # Any discovery failure (including unreadable or conflicting roots)
        # aborts before the attempt registry can be inspected, so cleanup
        # cannot be reported as confirmed. Post-reap validation is separate.
        cleanup_report.mark_uncertain(exc)
        raise
    return _preflight_existing_studies(
        claimed, cleanup_report=cleanup_report, from_phase=from_phase
    )


def _preflight_existing_studies(
    ledger: ClaimedLedger,
    *,
    cleanup_report: _PreflightCleanupReport | None = None,
    from_phase: str | None = None,
) -> dict[str, tuple[optuna.Study, ...]]:
    """Validate and reap every existing study of every declared phase before launch.

    A phase that inherits has a study for each set of values it inherited, and a trial a
    crashed run left RUNNING in any of them is settled here, with its process confirmed gone,
    even when the current config addresses another one.

    :param ClaimedLedger ledger: Handle from :func:`phasesweep.engine.ledger.claim_ledger`;
        its ``studies`` are the only studies inspected.
    :param _PreflightCleanupReport | None cleanup_report: Optional shared report to
        accumulate cleanup evidence into; a fresh one is created if omitted.
    :param str | None from_phase: Optional resume point. Recovery and schema checks
        still cover every phase; trial-target validation starts at this reached phase.
    :return dict[str, tuple[optuna.Study, ...]]: Every checked study of each phase, oldest
        first, by phase name (phases with no durable study yet are omitted).
    :raises StudyStorageUnavailableError: A phase's persistent storage could not
        be read while its stale trials were recovered.
    :raises StudySchemaMismatchError: A phase's study uses an incompatible
        storage schema.
    :raises TrialTargetRefusedError: A phase's study already accepted a
        higher trial target than the current config requests, or a fixed
        configuration asks for more attempts than its trials let it retry.
    :raises ProcessCleanupUncertainError: Stale-trial cleanup could not be
        confirmed safe for a phase's study.
    :raises StudyFingerprintMismatchError: A reached study recorded another experiment or
        phase config, or a reached phase that needs new trials holds trials from another
        trainer-environment cohort; see
        :func:`phasesweep.engine.study_policy._reached_study_refusals` for the order.
    :raises PhaseSweepError: A reached phase that needs new trials could not start them
        (see :func:`phasesweep.engine.study_policy._launch_preflight`), or multiple studies
        failed preflight for mixed expected reasons not covered by a more specific common
        exception type.
    :raises RuntimeError: Multiple studies failed and at least one error is an
        unexpected implementation failure that must retain traceback reporting.
    """
    experiment = ledger.experiment
    report = cleanup_report or _PreflightCleanupReport()
    loaded = ledger.studies
    studies: dict[str, tuple[optuna.Study, ...]] = {}
    errors: list[Exception] = []
    # The registry scan runs FIRST and is independent of the declared phase
    # list, so attempts from renamed/removed phases or changed storage URLs
    # are recovered before any current-config validation or launch.
    try:
        _preflight_active_attempts(experiment, report, confirm=True)
    except Exception as exc:
        if isinstance(exc, (ProcessCleanupUncertainError, StudyStorageUnavailableError)):
            report.mark_uncertain(exc)
        errors.append(exc)
    every = [(each, study) for each in experiment.phases for study in loaded.get(each.name, ())]
    for phase, study in every:
        try:
            # Each attempt is collected as soon as its trial's durable state
            # changes, so a later trial's refusal must not drop the evidence
            # already gathered; hence the ``finally`` blocks.
            recovered_terminal_attempts: dict[str, _AttemptRecord] = {}
            try:
                _recover_cleanup_uncertain_trials(
                    study,
                    experiment,
                    phase.name,
                    confirm=True,
                    recovered_attempts=recovered_terminal_attempts,
                )
            finally:
                report.record_generations(recovered_terminal_attempts)
            report.recovered_attempt_ids.update(recovered_terminal_attempts)
            for attempt_id in recovered_terminal_attempts:
                _retire_active_attempt(experiment, attempt_id)
            reaped_attempts: dict[str, _AttemptRecord] = {}
            try:
                _reap_stale_trials(
                    study,
                    experiment,
                    phase.name,
                    confirm=True,
                    recovered_attempts=reaped_attempts,
                    uncertain_attempt_ids=report.uncertain_attempt_ids,
                    report=report,
                )
            finally:
                report.recovered_attempt_ids.update(reaped_attempts)
                report.record_generations(reaped_attempts)
        except Exception as exc:
            if isinstance(exc, (ProcessCleanupUncertainError, StudyStorageUnavailableError)):
                report.mark_uncertain(exc)
            errors.append(exc)
            continue
        try:
            _validate_known_study_direction(experiment, phase, study)
            _initialize_study_schema(study)
        except Exception as exc:
            errors.append(exc)
        else:
            studies[phase.name] = (*studies.get(phase.name, ()), study)
    _refresh_recovered_trials_csv(loaded, experiment)
    # An unreadable tracker leaves finished training pending. Until it answers, neither the
    # retry ceiling nor the need for a trainer is known, and the run reports pending evidence.
    if not report.unreadable_evidence:
        known = known_studies(experiment, studies)
        errors.extend(
            _reached_study_refusals(
                experiment,
                [
                    ReachedStudy(phase, known[phase.name])
                    for _, phase in experiment.phases_from(from_phase)
                    if phase.name in known
                ],
            )
        )
    _raise_collected_errors(errors)
    return studies


def known_studies(
    experiment: Experiment, studies: Mapping[str, Sequence[optuna.Study]]
) -> dict[str, optuna.Study]:
    """Return the studies a launch can name before any phase runs.

    A phase that inherits addresses the study its parents' winners pick
    (:func:`phasesweep.engine.fingerprints.phase_study_name`), which is known only once they
    are selected, so the run checks its study when it reaches the phase. A phase that inherits
    nothing has one study.

    :param Experiment experiment: Current experiment config.
    :param Mapping[str, Sequence[optuna.Study]] studies: Every study of each phase.
    :return dict[str, optuna.Study]: The study of each phase that inherits nothing and has one,
        by phase name.
    """
    known: dict[str, optuna.Study] = {}
    for phase in experiment.phases:
        if phase.inherits:
            continue
        study = addressed_study(studies, phase.name, phase_study_name(experiment, phase, {}))
        if study is not None:
            known[phase.name] = study
    return known


def addressed_study(
    studies: Mapping[str, Sequence[optuna.Study]], phase_name: str, study_name: str
) -> optuna.Study | None:
    """Pick the study a phase addresses out of every study it has.

    :param Mapping[str, Sequence[optuna.Study]] studies: Every study of each phase.
    :param str phase_name: The phase.
    :param str study_name: The study it addresses
        (:func:`phasesweep.engine.fingerprints.phase_study_name`).
    :return optuna.Study | None: That study, or ``None`` when the phase has none by that name.
    """
    return next(
        (study for study in studies.get(phase_name, ()) if study.study_name == study_name), None
    )


def _collected_remediation(errors: Sequence[Exception]) -> str | None:
    """Join the steps the collected refusals call for, each distinct step once.

    :param Sequence[Exception] errors: Refusals collected across the studies, in phase order.
    :return str | None: The steps in order of first appearance, a refusal with no remediation
        of its own contributing its action's step; ``None`` when no refusal carries a
        remediation, so the aggregate's action speaks for it.
    """
    routed = [error for error in errors if isinstance(error, PhaseSweepError)]
    if not any(error.remediation for error in routed):
        return None
    steps = [operator_step(error) for error in routed]
    return " ".join(dict.fromkeys(step[:1].upper() + step[1:] for step in steps))


def _raise_collected_errors(errors: Sequence[Exception]) -> None:
    """Raise one refusal, or an aggregate preserving the collected studies and remedies.

    A refusal every study reports alike, with the same type and message, as an experiment-wide
    refusal does, is raised once and unchanged, so its fix line survives. An aggregate carries
    the remediation every refusal shares, or each distinct step once when they differ.

    :param Sequence[Exception] errors: Refusals collected across the studies, in
        phase order; nothing is raised when it is empty.
    :raises StudySchemaMismatchError: Every collected refusal is one, or the only one.
    :raises StudyFingerprintMismatchError: Every collected refusal is one.
    :raises StudyStorageUnavailableError: Every collected refusal is one.
    :raises TrialTargetRefusedError: Every collected refusal is one.
    :raises LaunchEnvironmentError: Every collected refusal is one.
    :raises ProcessCleanupUncertainError: Any collected refusal is one.
    :raises PhaseSweepError: The collected refusals are expected but of mixed kinds.
    :raises RuntimeError: At least one collected error is an unexpected
        implementation failure that must retain traceback reporting.
    """
    if not errors:
        return
    first = errors[0]
    # An experiment-wide refusal comes once per study; one copy says it all, with its fix line.
    if all(type(error) is type(first) and str(error) == str(first) for error in errors):
        raise first
    message = "Experiment recovery preflight found multiple unsafe studies: " + "; ".join(
        str(error) for error in errors
    )
    # No single error speaks for an aggregate, whatever its types: it
    # routes to a remediation only when every collected error names the
    # same one, and otherwise to reading the refusals it lists. An error that
    # names no action of its own counts as reading them.
    remediations = {
        error.action if isinstance(error, PhaseSweepError) else OperatorAction.INSPECT_LOGS
        for error in errors
    }
    shared = remediations.pop() if len(remediations) == 1 else OperatorAction.INSPECT_LOGS
    remediation = _collected_remediation(errors)
    for error_type in (
        StudySchemaMismatchError,
        StudyFingerprintMismatchError,
        StudyStorageUnavailableError,
        TrialTargetRefusedError,
        LaunchEnvironmentError,
    ):
        if all(isinstance(error, error_type) for error in errors):
            raise error_type(message, action=shared, remediation=remediation) from first
    cleanup_error = next(
        (error for error in errors if isinstance(error, ProcessCleanupUncertainError)),
        None,
    )
    if cleanup_error is not None:
        raise ProcessCleanupUncertainError(
            message, action=shared, remediation=remediation
        ) from cleanup_error
    unexpected = next(
        (error for error in errors if not isinstance(error, PhaseSweepError)),
        None,
    )
    if unexpected is not None:
        raise RuntimeError(message) from unexpected
    raise PhaseSweepError(message, action=shared, remediation=remediation) from first
