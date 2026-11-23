"""Resume winner and persistent-study top-up preflight."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import optuna

import phasesweep.engine.artifacts as artifact_io
import phasesweep.engine.evidence as evidence_ops
import phasesweep.engine.fingerprints as fingerprint_ops
import phasesweep.engine.publication as publication_ops
import phasesweep.engine.study_policy as study_policy_ops
from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.errors import PhaseTimeoutError, StudyContextConflictError, phase_timeout
from phasesweep.engine.optuna import _finished_trial_count
from phasesweep.engine.paths import _experiment_dir
from phasesweep.engine.publication_validation import ValidatedPublication
from phasesweep.engine.state import Winner, _recorded_identity
from phasesweep.errors import phase_names, rename_step
from phasesweep.runtime.terminal import plural


def _run_deadline_refusal(experiment: Experiment, phase_name: str) -> PhaseTimeoutError:
    """Refuse a phase that the whole-run deadline passed before it could start.

    :param Experiment experiment: Experiment whose storage decides the rerun step.
    :param str phase_name: Phase the deadline reached before it started.
    :return PhaseTimeoutError: The routed refusal, for the caller to raise.
    """
    return phase_timeout(
        f"Run wallclock deadline reached before phase {phase_name!r} could start.",
        scope="run",
        experiment=experiment,
    )


def _preflight_skipped_winners(
    experiment: Experiment,
    *,
    from_phase: str | None,
    run_deadline: float | None,
    error_mapper: Callable[[str, Exception], Exception] | None = None,
) -> dict[str, Winner]:
    """Validate required ancestors and carried published winners before any phase runs.

    :param Experiment experiment: Experiment whose resume prefix is checked.
    :param str | None from_phase: Validated first phase that this invocation will execute.
    :param float | None run_deadline: Whole-run monotonic deadline, when configured.
    :param Callable[[str, Exception], Exception] | None error_mapper: Optional
        frontend adapter receiving the phase whose winner was being checked
        and its refusal. Publication-resolution failures name the first skipped
        phase. Adapted refusals suppress the internal cause; without an adapter
        every exception is re-raised unchanged.
    :return dict[str, Winner]: Compatible required or published winners in declaration order.
    :raises TimeoutError: The whole-run wallclock deadline expired before a
        skipped phase could be validated.
    :raises ResumeWinnerMissingError: A required skipped phase has no persisted ``winner.yaml``.
    :raises WinnerIntegrityError: A skipped phase's persisted winner is invalid
        or incomplete.
    :raises StudyFingerprintMismatchError: A skipped phase's winner fingerprint
        disagrees with the current config.
    :raises TrialEvidenceMissingError: A skipped winner's source trial evidence
        is missing or changed.
    :raises PublicationAccessError: The recorded publication cannot be read as
        the current user.
    :raises PublicationIntegrityError: The recorded publication is unreadable,
        malformed, or fails validation.
    """
    if from_phase is None:
        return {}

    required = {phase.name for _index, phase in experiment.phases_from(from_phase)}
    # One validated immutable generation supplies every skipped winner. The
    # engine holds _experiment_lock; the MCP gate retains the same publication
    # snapshot while checking skipped winners before spawning its runner.
    phase = experiment.phases[0]
    try:
        publication = _published_for_resume(experiment)
        if publication is not None:
            required.update(publication.winners)
        for candidate in reversed(experiment.phases):
            if candidate.name in required:
                required.update(candidate.inherits)
        winners: dict[str, Winner] = {}
        for phase in experiment.phases:
            if run_deadline is not None and time.monotonic() >= run_deadline:
                raise _run_deadline_refusal(experiment, phase.name)
            if phase.name == from_phase:
                break
            if phase.name not in required:
                continue
            winners[phase.name] = _accept_skipped_winner(experiment, phase, winners, publication)

        return winners
    except Exception as exc:
        if error_mapper is None:
            raise
        mapped = error_mapper(phase.name, exc)
        if mapped is exc:
            raise
        raise mapped from None


def _published_for_resume(experiment: Experiment) -> ValidatedPublication | None:
    """Resolve the publication a resume builds on, refusing one that does not validate.

    An unusable publication never reads as absent: a pointer, summary, or
    manifest that cannot be read or fails validation is a refusal, so the
    callers' "no winner yet" answers only a tree that published nothing.

    :param Experiment experiment: Experiment being resumed.
    :raises PublicationAccessError: The pointer, summary, or published manifest
        cannot be read as the current user.
    :raises PublicationIntegrityError: The pointer, summary, or published
        manifest is unreadable, malformed, or fails validation.
    :return ValidatedPublication | None: Everything the validation read, or
        ``None`` when nothing was ever published.
    """
    pointer = publication_ops._resolve_publication_pointer(
        experiment,
        raise_on_manifest_error=True,
    )
    publication_ops.raise_for_unusable_publication(
        experiment.experiment, pointer.state, pointer.error, _experiment_dir(experiment)
    )
    return pointer.validated


def _accept_skipped_winner(
    experiment: Experiment,
    phase: Phase,
    winners: Mapping[str, Winner],
    publication: ValidatedPublication | None,
) -> Winner:
    """Load one skipped phase's published winner and prove the resume may build on it.

    :param Experiment experiment: Experiment being resumed.
    :param Phase phase: Skipped phase whose winner is accepted.
    :param Mapping[str, Winner] winners: Winners already accepted for earlier
        phases, which supply ``phase``'s inherited context.
    :param ValidatedPublication | None publication: The publication resolved
        once for the whole resume.
    :raises ResumeWinnerMissingError: The publication holds no winner for the phase.
    :raises WinnerIntegrityError: The winner is incompatible with the current
        partial-result policy.
    :raises StudyFingerprintMismatchError: The winner was produced by a
        different config.
    :raises TrialEvidenceMissingError: The winner's source trial evidence is
        missing or changed.
    :return Winner: The accepted winner.
    """
    inherited = {parent: winners[parent] for parent in phase.inherits}
    winner = artifact_io._load_winner(experiment, phase, inherited, publication=publication)
    evidence_ops._verify_winner_evidence(experiment, phase, winner)
    return winner


def _reject_bound_descendant_topups(
    experiment: Experiment,
    *,
    from_phase: str | None,
    existing_studies: dict[str, optuna.Study],
) -> None:
    """Reject upstream top-ups that could invalidate a bound descendant study.

    Reachability follows explicit inheritance edges. A new winner from an
    upstream top-up changes the inherited effective overrides that every
    transitive descendant has already hashed into its bound study.

    :param Experiment experiment: Parsed experiment whose phase chain is scanned
        from ``from_phase`` (or the start) onward.
    :param str | None from_phase: Optional resume point; phases before it are skipped.
    :param dict[str, optuna.Study] existing_studies: Existing Optuna studies keyed by
        phase name, as returned by :func:`phasesweep.engine.guards._preflight_existing_studies`.
    :raises StudyContextConflictError: An upstream phase still has unfinished
        top-up trials remaining while an inheriting descendant's study is bound
        to its published winner: it holds a stored fingerprint and at least
        one trial.
    """
    for _, phase in experiment.phases_from(from_phase):
        study = existing_studies.get(phase.name)
        if study is None:
            continue
        finished = _finished_trial_count(study.get_trials(deepcopy=False))
        if finished >= phase.n_trials:
            continue
        partial_decision = study_policy_ops._load_accepted_partial_decision(study)
        if partial_decision is not None and phase.n_trials == partial_decision.trial_target:
            # The phase has already committed a terminal accepted-timeout
            # decision at this exact target. Identical replay performs only
            # deterministic selection; its unused slots are not a top-up.
            continue

        # Only a populated descendant has consumed this winner. The fingerprint
        # is stamped before resource preflight, so a child that failed
        # preflight holds a fingerprint and no trials, and _verify_fingerprint
        # rebinds such an empty study to the new inherited config.
        bound = [
            name
            for name in sorted(experiment.phases_inheriting(phase.name))
            if (dependent := existing_studies.get(name)) is not None
            and _recorded_identity(dependent) is not None
            and dependent.get_trials(deepcopy=False)
        ]
        if bound:
            # The target is no part of a study's identity, so the enlarged phase keeps its
            # trials; its dependents' studies hold trials bound to its old winner.
            rename = rename_step(experiment.phases_inheriting(phase.name))
            raise StudyContextConflictError(
                f"Phase {phase.name!r} has {plural(phase.n_trials - finished, 'top-up trial')} "
                f"remaining, but {plural(len(bound), 'dependent phase')} ({phase_names(bound)}) "
                f"{'is' if len(bound) == 1 else 'are'} bound to its published winner.",
                remediation=(
                    f"{rename[0].upper()}{rename[1:]}; `{phase.name}` keeps its trials and runs "
                    "the rest."
                ),
            )


def _preflight_missing_reached_phase_environments(
    experiment: Experiment,
    existing_studies: Mapping[str, object],
    *,
    from_phase: str | None,
) -> None:
    """Check that each reached phase with no durable study could start its trials.

    A phase absent from an otherwise bound tree has no completed trial to replay, so it must
    be able to launch before this invocation tops up an earlier phase. A run calls this before
    its recovery and its dry run before its study checks, so both refuse it first. Existing
    studies are checked after recovery
    (:func:`phasesweep.engine.study_policy._reached_study_refusals`), which can tell new work
    from an offline-safe no-op.

    :param Experiment experiment: Parsed experiment supplying phase environments.
    :param Mapping[str, object] existing_studies: Durable studies found under the lock, or in the
        dry run's capture.
    :param str | None from_phase: Optional first phase this invocation can execute.
    :raises LaunchEnvironmentError: A missing reached phase has an invalid launch environment, a
        missing trainer directory or a trial program its trainer cannot find; see
        :func:`phasesweep.engine.study_policy._launch_preflight`.
    """
    for _index, phase in experiment.phases_from(from_phase):
        if phase.name not in existing_studies:
            study_policy_ops._launch_preflight(experiment, phase.name, None)


def _reject_fixed_targets_without_study(
    experiment: Experiment,
    *,
    from_phase: str | None,
    existing_studies: dict[str, optuna.Study],
) -> None:
    """Reject a fixed configuration's oversized target before any phase launches.

    A reached phase with no durable study has no failed attempt to replace, so
    it counts zero finished and zero completed trials. Reached phases that do
    have a study were already checked against their trials by the recovery
    preflight (:func:`phasesweep.engine.study_policy._validate_trial_target`).
    Checking the missing ones here keeps a later phase's refusal from arriving
    after an earlier phase has already run.

    :param Experiment experiment: Parsed experiment whose phase chain is scanned
        from ``from_phase`` (or the start) onward.
    :param str | None from_phase: Optional resume point; phases before it are skipped.
    :param dict[str, optuna.Study] existing_studies: Existing Optuna studies keyed by
        phase name, as returned by :func:`phasesweep.engine.guards._preflight_existing_studies`.
    :raises TrialTargetRefusedError: A reached fixed configuration without a
        study asks for more than one trial; delegated to
        :func:`phasesweep.engine.study_policy._validate_fixed_configuration_target`.
    """
    for _index, phase in experiment.phases_from(from_phase):
        if phase.name not in existing_studies:
            study_policy_ops._validate_fixed_configuration_target(phase, finished=0, measured=0)


def _preflight_reached_fingerprint(
    experiment: Experiment,
    *,
    from_phase: str | None,
    preloaded_winners: dict[str, Winner],
    existing_studies: dict[str, optuna.Study],
) -> None:
    """Verify the first reached study before any phase runs.

    :param Experiment experiment: Parsed experiment whose first reached phase
        (``from_phase``, or the first declared phase) is checked.
    :param str | None from_phase: Optional resume point identifying the first
        phase that will actually execute.
    :param dict[str, Winner] preloaded_winners: Validated skipped-phase winners,
        used to resolve the reached phase's inherited context.
    :param dict[str, optuna.Study] existing_studies: Existing Optuna studies keyed by
        phase name; a no-op if the reached phase has none yet.
    :raises StudyFingerprintMismatchError: The reached study's stored
        fingerprint does not match the current config.
    """
    phase = experiment.phases[0]
    if from_phase is not None:
        phase = experiment.phase(from_phase)
    study = existing_studies.get(phase.name)
    if study is None:
        return
    inherited = {name: preloaded_winners[name] for name in phase.inherits}
    fingerprint_ops._verify_fingerprint(study, experiment, phase, inherited)
