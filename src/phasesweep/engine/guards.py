"""Existing-study preflight orchestration."""

from __future__ import annotations

from collections.abc import Mapping

import optuna

from phasesweep.config import Experiment
from phasesweep.engine.artifact_roots import _load_and_check_artifact_roots
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
    PhaseSweepError,
    PublishedStudyMissingError,
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    StudyStorageUnavailableError,
    TrialTargetRegressionError,
)
from phasesweep.engine.study_policy import (
    _validate_environment_cohort,
    _validate_study_direction,
    _validate_study_schema,
    _validate_trial_target,
)
from phasesweep.engine.trial import ProcessCleanupUncertainError, _environment_identity


def _preflight_existing_studies(
    experiment: Experiment,
    *,
    cleanup_report: _PreflightCleanupReport | None = None,
    from_phase: str | None = None,
    preloaded_studies: Mapping[str, optuna.Study] | None = None,
) -> dict[str, optuna.Study]:
    """Validate and reap every existing declared phase study before launch.

    :param Experiment experiment: Parsed experiment whose declared phases are inspected.
    :param _PreflightCleanupReport | None cleanup_report: Optional shared report to
        accumulate cleanup evidence into; a fresh one is created if omitted.
    :param str | None from_phase: Optional resume point. Recovery and schema checks
        still cover every phase; trial-target validation starts at this reached phase.
    :param Mapping[str, optuna.Study] | None preloaded_studies: Studies already
        discovered and ownership-checked under the experiment lock before the
        generation claim. Direct callers omit this and perform the same strict
        discovery here.
    :return dict[str, optuna.Study]: Existing studies keyed by phase name (phases
        with no durable study yet are omitted).
    :raises ArtifactRootConflictError: A phase's persistent study is already
        bound to a different artifact root than this config's workdir offers;
        raised before any inspection, reaping, or trial work.
    :raises LegacyArtifactRootMigrationRequiredError: A populated phase study
        predates artifact-root binding, so which workdir owns its evidence
        cannot be inferred; raised on the same terms.
    :raises StudyStorageUnavailableError: A phase's persistent storage could not
        be inspected; raised before any claim, reaping, or registry recovery.
    :raises PublishedStudyMissingError: A reached phase has a published winner
        but its persistent study is absent or empty.
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
    current_environment_digest = _environment_identity(experiment).digest
    report = cleanup_report or _PreflightCleanupReport()
    # Discovery, root checks, and claims happen in ONE strict pass, and its
    # study objects are the ones every later step operates on: an invocation
    # offering a second publication root must not reap, inspect, or claim
    # anything in either tree (review v0.5.19 / finding F5), and a storage
    # read that fails must abort rather than let a second, luckier read hand
    # recovery a study whose root was never checked (PR #5 review /
    # reviewer 2, issue 1). The storage error still marks cleanup uncertain:
    # an unreadable ledger cannot prove its attempts are resolved.
    if preloaded_studies is None:
        try:
            loaded = _load_and_check_artifact_roots(experiment, from_phase=from_phase)
        except (StudyStorageUnavailableError, PublishedStudyMissingError) as exc:
            # This discovery also runs during post-execution reconciliation.
            # A lost ledger then aborts recovery before the attempt registry
            # can be inspected, so cleanup cannot be reported as confirmed.
            report.mark_uncertain(exc)
            raise
    else:
        loaded = dict(preloaded_studies)
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
                _validate_environment_cohort(study, current_environment_digest)
                _validate_trial_target(study, phase)
        except Exception as exc:
            errors.append(exc)
    if errors:
        first = errors[0]
        if len(errors) == 1:
            raise first
        message = "Experiment recovery preflight found multiple unsafe studies: " + "; ".join(
            str(error) for error in errors
        )
        if all(isinstance(error, StudySchemaMismatchError) for error in errors):
            raise StudySchemaMismatchError(message) from first
        if all(isinstance(error, StudyFingerprintMismatchError) for error in errors):
            raise StudyFingerprintMismatchError(message) from first
        if all(isinstance(error, StudyStorageUnavailableError) for error in errors):
            raise StudyStorageUnavailableError(message) from first
        if all(isinstance(error, TrialTargetRegressionError) for error in errors):
            raise TrialTargetRegressionError(message) from first
        cleanup_error = next(
            (error for error in errors if isinstance(error, ProcessCleanupUncertainError)),
            None,
        )
        if cleanup_error is not None:
            raise ProcessCleanupUncertainError(message) from cleanup_error
        unexpected = next(
            (error for error in errors if not isinstance(error, PhaseSweepError)),
            None,
        )
        if unexpected is not None:
            raise RuntimeError(message) from unexpected
        raise PhaseSweepError(message) from first
    return studies
