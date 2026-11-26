"""Why a recorded phase would refuse a run or wait to re-read kept evidence, as status reports it."""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import optuna

from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.errors import (
    StudyFingerprintMismatchError,
    StudySchemaMismatchError,
    TrialEvidenceMissingError,
    TrialTargetRefusedError,
)
from phasesweep.engine.evidence import _selection_evidence
from phasesweep.engine.fingerprints import _recorded_inheritance, _verify_recorded_identity
from phasesweep.engine.ledger import phase_studies_for
from phasesweep.engine.optuna import _finished_trial_count
from phasesweep.engine.repeats import repeat_axis
from phasesweep.engine.reread import PendingRereadNotice, pending_reread, plan_adoption
from phasesweep.engine.selection import (
    PhaseAbortRefusedError,
    _eligible,
    _no_feasible_remedy,
    _phase_refusal,
    _rank_groups,
)
from phasesweep.engine.state import _recorded_identity
from phasesweep.engine.study_policy import (
    _at_accepted_partial_target,
    _load_accepted_partial_decision,
    _load_phase_policy_state,
    _validate_trial_target,
)
from phasesweep.errors import (
    PERSISTENT_STUDY_KEEPS_TRIALS,
    START_AGAIN_STEP,
    OperatorAction,
    PhaseSweepError,
    operator_step,
)


@dataclass(frozen=True)
class UnpublishedWinnerStep:
    """How a retained winner gets published: the recorded cause when one blocks it, then the fix."""

    cause: str | None
    fix: str


def _status_phase_refusal(
    experiment: Experiment, phase: Phase, study: optuna.Study
) -> PhaseAbortRefusedError | None:
    """Give status the failure-policy refusal when a partial decision does not take precedence.

    :param Experiment experiment: Config whose bounds judge the recorded trials.
    :param Phase phase: Phase whose target and failure limit apply.
    :param optuna.Study study: Read-only phase study containing the recorded decisions.
    :return PhaseAbortRefusedError | None: Failure-policy refusal to report, if any.
    """
    policy = _load_phase_policy_state(study, experiment)
    if _at_accepted_partial_target(_load_accepted_partial_decision(study), phase):
        return None
    return _phase_refusal(experiment, phase, study, policy)


class PendingParentNotice(PhaseSweepError):
    """The displayed child's rows depend on a parent whose winner may change."""

    default_action: typing.ClassVar[OperatorAction] = OperatorAction.RETRY
    #: The parents whose next winner the child waits for.
    parents: tuple[str, ...] = ()


PhaseRefusal: TypeAlias = (
    PhaseAbortRefusedError
    | StudySchemaMismatchError
    | StudyFingerprintMismatchError
    | PendingRereadNotice
    | PendingParentNotice
    | TrialEvidenceMissingError
    | TrialTargetRefusedError
)


def _study_diagnostics(
    experiment: Experiment, phase: Phase, study: optuna.Study
) -> tuple[PhaseRefusal | None, tuple[int, ...]]:
    """Report the identity, target, failure policy or evidence refusal for a known study.

    The checks run in the order of a run's preflight and phase start, and the target is judged
    by the run's own check (:func:`~phasesweep.engine.study_policy._validate_trial_target`).

    :param Experiment experiment: Current config.
    :param Phase phase: The phase the study belongs to.
    :param optuna.Study study: The study the phase addresses.
    :return tuple[PhaseRefusal | None, tuple[int, ...]]: The refusal a run would
        meet, and the finished trials whose directory is gone
        (:func:`phasesweep.engine.evidence._selection_evidence`).
    """
    try:
        _verify_recorded_identity(study, experiment, phase)
        # The run judges the target after adoption, so status judges the projection whose
        # re-reads all fail, the one a launch preflight checks (study_policy._reached_study_refusals).
        record = _recorded_identity(study)
        plan = plan_adoption(
            experiment,
            phase,
            {} if record is None else _recorded_inheritance(record),
            study,
            phase_studies_for(study),
            study_name=study.study_name,
        )
        _validate_trial_target(study if plan is None else plan.failed, phase)
        refusal = _status_phase_refusal(experiment, phase, study)
        if refusal is not None:
            return refusal, ()
        return None, _selection_evidence(experiment, phase, study)
    except (
        StudyFingerprintMismatchError,
        StudySchemaMismatchError,
        TrialEvidenceMissingError,
        TrialTargetRefusedError,
    ) as exc:
        return exc, ()


def _pending_study_notices(
    experiment: Experiment,
    every: Mapping[str, Sequence[optuna.Study]],
    study_names: Mapping[str, str],
    unresolved: Mapping[str, tuple[str, ...]],
) -> dict[str, PhaseRefusal]:
    """Describe pending evidence reads and child studies that depend on a parent's next result.

    Diagnostics for a historical fallback cannot refuse a run until the parent selects the
    values that would send the child back to that study.

    :param Experiment experiment: Current config.
    :param Mapping[str, Sequence[optuna.Study]] every: Every study of each phase, oldest first,
        from the caller's one capture.
    :param Mapping[str, str] study_names: The study each phase addresses.
    :param Mapping[str, tuple[str, ...]] unresolved: Parents that may change the displayed address.
    :return dict[str, PhaseRefusal]: The notice, or the refusal of a study with the same
        inherited values and another training identity or a malformed one, by phase name.
    """
    pending: dict[str, PhaseRefusal] = {}
    for phase in experiment.phases:
        parents = unresolved.get(phase.name, ())
        name = study_names.get(phase.name)
        if name is None:
            continue
        if parents:
            names = ", ".join(repr(parent) for parent in parents)
            waiting = PendingParentNotice(
                f"Phase {phase.name!r} shows counts using the last recorded values from {names}. "
                "Its study depends on the values selected by the next run.",
                remediation="Run the current configuration to select the parent winners.",
            )
            waiting.parents = parents
            pending[phase.name] = waiting
        try:
            notice = pending_reread(
                experiment, phase, name, every.get(phase.name, []), unsettled=parents
            )
        except (StudyFingerprintMismatchError, StudySchemaMismatchError) as exc:
            if not parents:
                pending[phase.name] = exc
            continue
        if notice is not None:
            pending[phase.name] = notice
    return pending


def _phase_refusals(
    experiment: Experiment,
    studies: Mapping[str, optuna.Study],
    pending: Mapping[str, PhaseRefusal],
) -> tuple[dict[str, PhaseRefusal], dict[str, tuple[int, ...]]]:
    """Give the refusal a run would meet at each phase, in the study it would address.

    A phase whose addressed study holds no rows yet and would re-read the kept evidence of its
    other studies reports that, with the command that runs it (:mod:`phasesweep.engine.reread`).
    Otherwise a phase's recorded failure policy is judged first, as a run's preflight does,
    then a populated study is judged against its recorded identity, whose inherited values and
    reading settings are the ones that address it, so only a change to what was trained
    refuses, then its selection evidence (:func:`_study_diagnostics`). Invalid policy state is
    reported for its phase alongside other refusals. It does not prevent status from reporting
    trial counts and publication facts.

    :param Experiment experiment: Current config, which decides whether a recorded abort is
        retired.
    :param Mapping[str, optuna.Study] studies: Addressed studies from the status read's capture.
    :param Mapping[str, PhaseRefusal] pending: :func:`_pending_study_notices` from the same read.
    :return tuple[dict[str, PhaseRefusal], dict[str, tuple[int, ...]]]: Refusals or invalid
        policy diagnostics by phase name, and the finished trials whose directory is gone in
        each phase that has any; both empty without a readable ledger.
    """
    if not experiment.persistent:
        return {}, {}
    refusals: dict[str, PhaseRefusal] = dict(pending)
    pruned: dict[str, tuple[int, ...]] = {}
    for phase in experiment.phases:
        study = studies.get(phase.name)
        if study is None or phase.name in refusals:
            continue
        refusal, gone = _study_diagnostics(experiment, phase, study)
        if gone:
            pruned[phase.name] = gone
        if refusal is not None:
            refusals[phase.name] = refusal
    return _point_children_at_refused_parents(experiment, refusals), pruned


def _point_children_at_refused_parents(
    experiment: Experiment, refusals: dict[str, PhaseRefusal]
) -> dict[str, PhaseRefusal]:
    """Give a child waiting on a refused parent the parent's fix, not a run that would be refused.

    :param Experiment experiment: Config whose phase order and inheritance are followed.
    :param dict[str, PhaseRefusal] refusals: Refusals and notices by phase name.
    :return dict[str, PhaseRefusal]: The same, each pending-parent notice of a child behind a
        refused parent carrying that parent's operator step.
    """
    blockers: dict[str, PhaseRefusal] = {}
    resolved = dict(refusals)
    for phase in experiment.phases:
        current = resolved.get(phase.name)
        if isinstance(current, PendingParentNotice):
            blocker = next((blockers[name] for name in current.parents if name in blockers), None)
            if blocker is not None:
                blockers[phase.name] = blocker
                redirected = PendingParentNotice(str(current), remediation=operator_step(blocker))
                redirected.parents = current.parents
                resolved[phase.name] = redirected
        elif current is not None and not isinstance(current, PendingRereadNotice):
            blockers[phase.name] = current
    return resolved


def unpublished_winner_step(
    experiment: Experiment, studies: Mapping[str, optuna.Study]
) -> UnpublishedWinnerStep:
    """Give one publication remedy from a phase without a feasible winner, for an idle experiment.

    A recorded abort or failure streak, or a study whose own config changed, is
    :func:`inspect_status`' refusals to report.

    :param Experiment experiment: Current config used to inspect durable phase policy.
    :param Mapping[str, optuna.Study] studies: Addressed studies from the status read's capture.
    :return UnpublishedWinnerStep: The recorded cause and remedy or the rerun instruction.
    """
    if not experiment.persistent:
        return UnpublishedWinnerStep(None, START_AGAIN_STEP)
    for phase in experiment.phases:
        study = studies.get(phase.name)
        if study is None:
            continue
        trials = study.get_trials(deepcopy=False)
        if _finished_trial_count(trials) < phase.n_trials:
            continue
        axis = repeat_axis(phase)
        eligible = (
            any(_eligible(trial, experiment) for trial in trials)
            if axis is None
            else bool(_rank_groups(trials, experiment, axis).groups)
        )
        if not eligible:
            return UnpublishedWinnerStep(
                f"Phase {phase.name!r} has no feasible winner.",
                _no_feasible_remedy(experiment, trials, phase.name),
            )
    return UnpublishedWinnerStep(
        None, f"Rerun the same config to resume and publish; {PERSISTENT_STUDY_KEEPS_TRIALS}."
    )
