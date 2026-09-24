"""Existing-study preflight orchestration."""

from __future__ import annotations

import optuna

from phasesweep.config import Experiment
from phasesweep.engine.attempts import (
    _preflight_active_attempts,
    _PreflightCleanupReport,
    _retire_active_attempt,
)
from phasesweep.engine.cleanup import (
    _reap_stale_trials,
    _recover_cleanup_uncertain_trials,
)
from phasesweep.engine.errors import (
    OperatorAction,
    PhaseSweepError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrialTargetRegressionError,
)
from phasesweep.engine.ledger import ClaimedLedger, claim_ledger, validate_ledger
from phasesweep.engine.study_policy import (
    _load_accepted_partial_decision,
    _validate_environment_cohort,
    _validate_study_direction,
    _validate_study_schema,
    _validate_trial_target,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError, _environment_identity


def _reconcile_existing_studies(
    experiment: Experiment,
    *,
    cleanup_report: _PreflightCleanupReport,
    from_phase: str | None = None,
) -> dict[str, optuna.Study]:
    """Rediscover and reconcile every existing study after a run has ended.

    Post-execution reconciliation cannot reuse the handle the run claimed
    before it started: phases created studies since then, and those must be
    reaped too. It therefore validates and claims the ledger afresh, then runs
    the same preflight over what that discovery found.

    :param Experiment experiment: Parsed experiment whose declared phases are inspected.
    :param _PreflightCleanupReport cleanup_report: Report the reconciliation's
        cleanup evidence accumulates into.
    :param str | None from_phase: Optional resume point, as for
        :func:`_preflight_existing_studies`.
    :return dict[str, optuna.Study]: Existing studies keyed by phase name.
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
) -> dict[str, optuna.Study]:
    """Validate and reap every existing declared phase study before launch.

    :param ClaimedLedger ledger: Handle from :func:`phasesweep.engine.ledger.claim_ledger`;
        its ``studies`` are the only studies inspected.
    :param _PreflightCleanupReport | None cleanup_report: Optional shared report to
        accumulate cleanup evidence into; a fresh one is created if omitted.
    :param str | None from_phase: Optional resume point. Recovery and schema checks
        still cover every phase; trial-target validation starts at this reached phase.
    :return dict[str, optuna.Study]: Existing studies keyed by phase name (phases
        with no durable study yet are omitted).
    :raises StudyStorageUnavailableError: A phase's persistent storage could not
        be read while its stale trials were recovered.
    :raises StudySchemaMismatchError: A phase's study uses an incompatible
        storage schema.
    :raises TrialTargetRegressionError: A phase's study already accepted a
        higher trial target than the current config requests.
    :raises ProcessCleanupUncertainError: Stale-trial cleanup could not be
        confirmed safe for a phase's study.
    :raises PhaseSweepError: Multiple studies failed preflight for mixed
        expected reasons not covered by a more specific common exception type.
    :raises RuntimeError: Multiple studies failed and at least one error is an
        unexpected implementation failure that must retain traceback reporting.
    """
    experiment = ledger.experiment
    report = cleanup_report or _PreflightCleanupReport()
    # Discovery, root checks, and claims already happened in ONE strict pass
    # (claim_ledger), and its study objects are the ones every later step
    # operates on: an invocation offering a second publication root must not
    # reap, inspect, or claim anything in either tree (review v0.5.19 /
    # finding F5), and a storage read that fails must abort rather than let a
    # second, luckier read hand recovery a study whose root was never checked
    # (PR #5 review / reviewer 2, issue 1).
    loaded = ledger.studies
    studies: dict[str, optuna.Study] = {}
    errors: list[Exception] = []
    # The registry scan runs FIRST and is independent of the declared phase
    # list, so attempts from renamed/removed phases or changed storage URLs
    # are recovered before any current-config validation or launch (review
    # v0.5.17 / blocker 3).
    try:
        _preflight_active_attempts(experiment, report)
    except Exception as exc:
        if isinstance(exc, (ProcessCleanupUncertainError, StudyStorageUnavailableError)):
            report.mark_uncertain(exc)
        errors.append(exc)
    reached = from_phase is None
    for phase in experiment.phases:
        if phase.name == from_phase:
            reached = True
        study = loaded.get(phase.name)
        if study is None:
            continue
        studies[phase.name] = study
        try:
            recovered_terminal_attempts: set[str] = set()
            _recover_cleanup_uncertain_trials(
                study,
                experiment,
                phase.name,
                recovered_attempt_ids=recovered_terminal_attempts,
                recovered_attempt_generations=report.recovered_attempt_generations,
            )
            report.recovered_attempt_ids.update(recovered_terminal_attempts)
            for attempt_id in recovered_terminal_attempts:
                _retire_active_attempt(experiment, attempt_id)
            _reap_stale_trials(
                study,
                experiment,
                phase.name,
                recovered_attempt_ids=report.recovered_attempt_ids,
                recovered_attempt_generations=report.recovered_attempt_generations,
                uncertain_attempt_ids=report.uncertain_attempt_ids,
            )
        except Exception as exc:
            if isinstance(exc, (ProcessCleanupUncertainError, StudyStorageUnavailableError)):
                report.mark_uncertain(exc)
            errors.append(exc)
            continue
        try:
            _validate_study_direction(study, experiment.metric.goal)
            _validate_study_schema(study)
            if reached:
                _validate_trial_target(study, phase)
                partial_decision = _load_accepted_partial_decision(study)
                finished_trials = sum(
                    trial.state.is_finished() for trial in study.get_trials(deepcopy=False)
                )
                needs_new_trials = phase.n_trials > finished_trials and not (
                    partial_decision is not None and phase.n_trials == partial_decision.trial_target
                )
                if needs_new_trials:
                    _validate_environment_cohort(
                        study, _environment_identity(experiment, phase.name).digest
                    )
        except Exception as exc:
            errors.append(exc)
    if errors:
        first = errors[0]
        if len(errors) == 1:
            raise first
        message = "Experiment recovery preflight found multiple unsafe studies: " + "; ".join(
            str(error) for error in errors
        )
        # No single error speaks for an aggregate, whatever its types: it
        # routes to a remediation only when every collected error names the
        # same one, and otherwise to reading the refusals it lists.
        remediations = {error.action for error in errors if isinstance(error, PhaseSweepError)}
        shared = remediations.pop() if len(remediations) == 1 else OperatorAction.INSPECT_LOGS
        for error_type in (
            StudySchemaMismatchError,
            StudyFingerprintMismatchError,
            StudyStorageUnavailableError,
            TrialTargetRegressionError,
        ):
            if all(isinstance(error, error_type) for error in errors):
                raise error_type(message, action=shared) from first
        cleanup_error = next(
            (error for error in errors if isinstance(error, ProcessCleanupUncertainError)),
            None,
        )
        if cleanup_error is not None:
            raise ProcessCleanupUncertainError(message, action=shared) from cleanup_error
        unexpected = next(
            (error for error in errors if not isinstance(error, PhaseSweepError)),
            None,
        )
        if unexpected is not None:
            raise RuntimeError(message) from unexpected
        raise PhaseSweepError(message, action=shared) from first
    return studies
