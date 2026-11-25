"""Engine progress lines and process-scoped warnings.

A live phase reports one rerun line per rerun, and per trial a line as its trainer starts and
one as it finishes; recovery reports the trials it settles as trial lines marked recovered. A
dry run previews its phases with the plain ``phase=… study=…`` header on the same report channel.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping, Sequence

import optuna

from phasesweep.config.models import Experiment, Phase
from phasesweep.engine.bounds import infeasible_reason
from phasesweep.engine.optuna import _trial_count_state
from phasesweep.engine.repeats import repeat_axis
from phasesweep.engine.state import (
    DURATION_ATTR,
    RERUN_OF_ATTR,
    TRIAL_DIR_ATTR,
    TRIAL_OUTCOME_ATTR,
    _parsed_trial_outcome,
)
from phasesweep.run_report import report, rerun_line, trial_line, trial_start_line

# Warnings keep the phase channel that run.log readers match on.
log = logging.getLogger("phasesweep.engine.phase")
_WARNED: set[Hashable] = set()


def _warn_once(logger: logging.Logger, key: Hashable, message: str, *args: object) -> None:
    """Log a warning once per process for its namespaced identity.

    Callers include the warning kind in ``key`` so separate diagnostics never
    suppress one another. The supplied logger retains each diagnostic's channel.

    :param logging.Logger logger: Logger that emits the warning.
    :param Hashable key: Warning kind and the identities defining its scope.
    :param str message: Logging format string.
    :param object args: Values substituted into the format string.
    """
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message, *args)


def _warn_deferred_extraction(
    attempt_id: str, phase_name: str, trial_number: int, *, persistent: bool
) -> None:
    """Explain once across shutdown and reconciliation what becomes of unextracted training.

    Only a persistent study outlives the run to extract it later; ``storage: null`` loses it.

    :param str attempt_id: Attempt whose completed training was not extracted.
    :param str phase_name: Owning phase.
    :param int trial_number: Trial number within that phase.
    :param bool persistent: Whether the trial's study outlives this run.
    """
    if persistent:
        message = (
            "Trial %d in phase %s finished training; its result is retained. The next run "
            "extracts it without retraining."
        )
    else:
        message = (
            "Trial %d in phase %s finished training, but its result was not extracted and is lost."
        )
    _warn_once(log, ("deferred-extraction", attempt_id), message, trial_number, phase_name)


def _log_phase_header(
    phase: Phase,
    study_name: str,
    *,
    finished: int,
    remaining: int,
    reruns: int,
    logger: logging.Logger,
) -> None:
    """Log the line that opens a phase's preview in a dry run.

    :param Phase phase: The phase about to run or replay.
    :param str study_name: Its Optuna study name.
    :param int finished: Trials that count toward its target.
    :param int remaining: Trials still needed to reach ``phase.n_trials``.
    :param int reruns: Reruns it schedules, for interrupted trials and for the failed grid
        cells a raised ``n_trials`` retries.
    :param logging.Logger logger: The dry run's report channel.
    """
    logger.info(
        "phase=%s study=%s finished=%d remaining=%d reruns=%d n_jobs=%d",
        phase.name,
        study_name,
        finished,
        remaining,
        reruns,
        phase.n_jobs,
    )


def _log_finished_trial(
    experiment: Experiment, phase: Phase, trial: optuna.trial.FrozenTrial, finished: int
) -> None:
    """Report one finished trial's state, objective, duration, parameters and first cause line.

    The state is the one :func:`~phasesweep.engine.optuna._trial_count_state` reports, which
    fixes the trial's recorded outcome: ``COMPLETE`` is a success, ``INFEASIBLE`` and
    ``FAIL`` are failures, and ``INTERRUPTED`` is an interruption. Repeat values and rerun
    sources are noted beside the configuration parameters.

    :param Experiment experiment: The experiment the phase belongs to.
    :param Phase phase: The phase the trial belongs to.
    :param optuna.trial.FrozenTrial trial: The just-finished trial.
    :param int finished: Trials of the phase's study that counted toward its target when
        this trial's outcome was recorded, this one included unless it was interrupted.
    """
    seconds = None if trial.duration is None else trial.duration.total_seconds()
    _report_trial(experiment, phase, trial, finished, seconds=seconds, recovered=False)


def _log_recovered_trial(
    experiment: Experiment, phase: Phase, trial: optuna.trial.FrozenTrial, finished: int
) -> None:
    """Report a trial a later run's recovery settled, as its own run would have, marked recovered.

    The seconds are the trainer's recorded duration: Optuna's own spans the recovery too.

    :param Experiment experiment: The producing configuration of the trial's attempt.
    :param Phase phase: That configuration's phase.
    :param optuna.trial.FrozenTrial trial: The trial, reloaded after its terminal transition.
    :param int finished: Trials of the phase's study that count toward its target now.
    """
    duration = trial.user_attrs.get(DURATION_ATTR)
    seconds = float(duration) if isinstance(duration, int | float) else None
    _report_trial(experiment, phase, trial, finished, seconds=seconds, recovered=True)


def _repeat_split(
    phase: Phase, params: Mapping[str, object]
) -> tuple[dict[str, object], tuple[str, object] | None]:
    """Split a trial's parameters into its configuration and the repeat value it runs.

    :param Phase phase: The trial's phase.
    :param Mapping[str, object] params: The trial's parameters.
    :return tuple[dict[str, object], tuple[str, object] | None]: The parameters without the
        repeat key, and that key and value when the phase repeats.
    """
    axis = repeat_axis(phase)
    repeat = None
    if axis is not None and axis.key in params:
        repeat = (axis.key, params[axis.key])
    return {key: value for key, value in params.items() if axis is None or key != axis.key}, repeat


def _report_trial(
    experiment: Experiment,
    phase: Phase,
    trial: optuna.trial.FrozenTrial,
    finished: int,
    *,
    seconds: float | None,
    recovered: bool,
) -> None:
    """Report one finished trial's line.

    :param Experiment experiment: The experiment the phase belongs to.
    :param Phase phase: The phase the trial belongs to.
    :param optuna.trial.FrozenTrial trial: The finished trial.
    :param int finished: The trial's place in the phase's completion order.
    :param float | None seconds: Trainer seconds, ``None`` when none were recorded.
    :param bool recovered: Whether recovery settled the trial.
    """
    params, repeat = _repeat_split(phase, trial.params)
    report.info(
        trial_line(
            experiment,
            phase,
            number=trial.number,
            state=_trial_count_state(trial, experiment),
            value=trial.value,
            seconds=seconds,
            finished=finished,
            params=params,
            repeat=repeat,
            rerun_of=trial.user_attrs.get(RERUN_OF_ATTR),
            reason=_first_cause_line(trial, experiment),
            recovered=recovered,
        )
    )


def _log_trial_start(
    experiment: Experiment,
    phase: Phase,
    trial: optuna.Trial,
    *,
    gpu: str | None,
    in_place: bool,
) -> None:
    """Report a trial whose trainer is about to start, and the trial a rerun in place resumes.

    :param Experiment experiment: The experiment the phase belongs to.
    :param Phase phase: The phase the trial belongs to.
    :param optuna.Trial trial: The trial about to launch.
    :param str | None gpu: The devices its trainer sees, ``None`` or empty when it leases none.
    :param bool in_place: Whether it continues its source's trial directory.
    """
    params, repeat = _repeat_split(phase, trial.params)
    if in_place:
        log.warning(
            "[%s/trial_%d] Retrying in shared checkpoint directory %s. The trainer can overwrite "
            "an earlier attempt's model. Include PHASESWEEP_ATTEMPT_ID in checkpoint names you want to keep.",
            phase.name,
            trial.number,
            trial.user_attrs.get(TRIAL_DIR_ATTR),
        )
    report.info(
        trial_start_line(
            experiment,
            phase,
            number=trial.number,
            gpu=gpu,
            params=params,
            repeat=repeat,
            resumes=trial.user_attrs.get(RERUN_OF_ATTR) if in_place else None,
        )
    )


def _first_cause_line(trial: optuna.trial.FrozenTrial, experiment: Experiment) -> str | None:
    """Return the first line of a trial's recorded outcome cause, or of its bound violations.

    :param optuna.trial.FrozenTrial trial: Trial whose outcome record is read.
    :param Experiment experiment: Config whose bounds judge a measured trial.
    :return str | None: The cause's first nonblank line, else the bounds a measured trial
        violates, else None.
    """
    parsed = _parsed_trial_outcome(trial.user_attrs.get(TRIAL_OUTCOME_ATTR))
    cause = parsed.cause if parsed and parsed.cause else infeasible_reason(trial, experiment)
    lines = cause.strip().splitlines() if cause else []
    return lines[0] if lines else None


def _log_reruns(
    experiment: Experiment,
    phase: Phase,
    study: optuna.Study,
    reruns: Sequence[tuple[int, int]],
) -> None:
    """Report each interrupted rerun or failed grid-cell retry and its source's cause.

    :param Experiment experiment: The experiment the phase belongs to.
    :param Phase phase: The phase whose study scheduled the reruns.
    :param optuna.Study study: That study.
    :param Sequence[tuple[int, int]] reruns: ``(source, rerun)`` trial numbers, as
        :func:`~phasesweep.engine.study_policy._record_reruns` returns them.
    """
    if not reruns:
        return
    trials = {trial.number: trial for trial in study.get_trials(deepcopy=False)}
    for source, rerun in reruns:
        report.info(
            rerun_line(
                experiment,
                phase,
                number=rerun,
                source=source,
                reason=_first_cause_line(trials[source], experiment),
            )
        )
